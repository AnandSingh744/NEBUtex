"""The only module that talks to the LLM. The rest of the project calls chat().

Two providers, chosen by config.LLM_PROVIDER:
- "ollama": local Ollama server
- "openai_compatible": any hosted API using the OpenAI request format (e.g. Groq)
Both return the same LLMResponse and raise the same LLMError, so callers never know which is in use.
"""

import hashlib
import json
import logging
import os
import random
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import ollama
import openai

from research_agent import config

logger = logging.getLogger(__name__)

# HTTP status codes worth retrying: rate limited, or the server is temporarily failing.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRY_AFTER = 30.0  # never wait longer than this, even if the server asks us to


class LLMError(Exception):
    """A failed LLM call, with a message that is safe to show the user."""


@dataclass
class LLMResponse:
    text: str
    input_tokens: int       # tokens in the prompt we sent (system + messages)
    output_tokens: int      # tokens the model generated
    total_seconds: float    # wall time including loading the prompt
    tokens_per_second: float  # generation speed
    time_to_first_token: float | None = None  # seconds until the first piece arrived (streaming only)
    truncated: bool = False  # True if the reply was cut off by MAX_OUTPUT_TOKENS
    tool_calls: list["ToolCall"] = field(default_factory=list)  # tools the model asked us to run


@dataclass
class ToolCall:
    """The model asking US to run a function. The model never runs anything itself."""
    id: str
    name: str
    arguments: dict


_ollama_client = ollama.Client(host=config.OLLAMA_HOST, timeout=config.REQUEST_TIMEOUT)
_openai_client: openai.OpenAI | None = None


def _get_openai_client() -> openai.OpenAI:
    global _openai_client
    if _openai_client is None:
        if not config.LLM_API_KEY:
            raise LLMError("LLM_API_KEY is not set. Add your API key to the environment (or app secrets).")
        # max_retries=0: chat() already retries, and retrying in two places would multiply the attempts.
        _openai_client = openai.OpenAI(
            base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY,
            timeout=config.REQUEST_TIMEOUT, max_retries=0,
        )
    return _openai_client


def is_available() -> bool:
    """Is the LLM reachable, and does it have our model? Used by health checks."""
    return check_llm()[0]


def check_llm() -> tuple[bool, str]:
    """Like is_available(), but also says WHY it isn't available, so problems can be fixed quickly."""
    try:
        if config.LLM_PROVIDER == "ollama":
            names = [m.model for m in _ollama_client.list().models]
        else:
            names = [m.id for m in _get_openai_client().models.list()]
    except LLMError as e:  # e.g. no API key set
        return False, str(e)
    except openai.AuthenticationError:
        return False, "The API key was rejected (401). Re-copy it into LLM_API_KEY, with no extra spaces."
    except openai.APIStatusError as e:
        return False, f"The LLM service returned an error ({e.status_code})."
    except (ConnectionError, httpx.TransportError, ollama.ResponseError, openai.APIConnectionError):
        return False, _unreachable_message()

    if config.MODEL not in names:
        similar = [n for n in names if "llama" in n.lower()] or names
        return False, f"Model '{config.MODEL}' isn't offered. Set LLM_MODEL to one of: {', '.join(sorted(similar)[:8])}"
    return True, "ok"


# ---------- Optional response cache (used by evaluation) ----------
# With temperature 0 the same request always gets the same reply, so it can be remembered.
# Evaluation turns this on so re-runs only call the model for prompts that actually changed.

_cache: dict | None = None
_cache_path: Path | None = None
_cache_lock = threading.Lock()
_unsaved = 0
SAVE_EVERY = 10  # write the cache to disk after this many new replies, so an interrupted run loses little


def enable_cache(path: str | Path) -> None:
    """Remember replies to temperature-0 requests in a JSON file (loaded now, written by save_cache())."""
    global _cache, _cache_path
    _cache_path = Path(path)
    _cache = json.loads(_cache_path.read_text(encoding="utf-8")) if _cache_path.exists() else {}


def save_cache() -> None:
    global _unsaved
    if _cache is None or _cache_path is None:
        return
    with _cache_lock:
        _unsaved = 0
        tmp = _cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_cache), encoding="utf-8")
        os.replace(tmp, _cache_path)


def _cache_key(messages: list[dict], model: str, max_tokens: int, tools: list[dict] | None = None) -> str:
    # Everything that can change the reply goes into the key.
    request = [config.LLM_PROVIDER, config.LLM_BASE_URL, model, max_tokens, config.LLM_REASONING_EFFORT, messages, tools]
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def chat(
    messages: list[dict],
    model: str = config.MODEL,
    on_token: Callable[[str], None] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
) -> LLMResponse:
    """Send a list of {"role", "content"} messages and return the model's reply.

    The model is stateless: it sees only the messages passed in this call.
    If on_token is given, the reply is streamed and on_token is called with
    each piece of text as soon as it is generated.
    temperature overrides config.TEMPERATURE for this call (e.g. 0 for consistent grading).
    max_tokens overrides config.MAX_OUTPUT_TOKENS (e.g. a small cap for short JSON replies, so a
    model stuck repeating itself is cut off early).
    tools: functions the model may ask to call (JSON Schema, OpenAI format). If it does, the reply has
    tool_calls instead of (or as well as) text. Messages may include the neutral forms
    {"role": "assistant", "content", "tool_calls": [ToolCall dicts]} and
    {"role": "tool", "tool_call_id", "name", "content"}; they're converted for each provider.

    Transient failures are retried with exponential backoff. Anything that
    still fails is raised as LLMError.
    """
    effective_temperature = config.TEMPERATURE if temperature is None else temperature
    max_tokens = max_tokens or config.MAX_OUTPUT_TOKENS
    if _cache is not None and effective_temperature == 0:
        key = _cache_key(messages, model, max_tokens, tools)
        with _cache_lock:
            hit = _cache.get(key)
        if hit is not None:
            response = LLMResponse(**{**hit, "tool_calls": [ToolCall(**t) for t in hit.get("tool_calls", [])]})
            if on_token:
                on_token(response.text)  # streaming callers still receive the text
            return response
        response = _chat_with_retries(messages, model, on_token, temperature, max_tokens, tools)
        global _unsaved
        with _cache_lock:
            _cache[key] = asdict(response)
            _unsaved += 1
            save_now = _unsaved >= SAVE_EVERY
        if save_now:
            save_cache()
        return response
    return _chat_with_retries(messages, model, on_token, temperature, max_tokens, tools)


def _chat_with_retries(
    messages: list[dict], model: str, on_token: Callable[[str], None] | None,
    temperature: float | None, max_tokens: int, tools: list[dict] | None = None,
) -> LLMResponse:
    for attempt in range(1, config.MAX_ATTEMPTS + 1):
        stream_started = False

        def forward(piece: str) -> None:
            nonlocal stream_started
            stream_started = True
            on_token(piece)

        try:
            # Tool calls come back as one structured reply, so no streaming when tools are offered.
            stream_to = forward if on_token and not tools else None
            return _chat_once(messages, model, stream_to, temperature, max_tokens, tools)
        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise LLMError(f"Model '{model}' not found. Download it with: ollama pull {model}") from e
            if e.status_code not in RETRYABLE_STATUS_CODES:
                raise LLMError(f"The LLM rejected the request ({e.status_code}): {e.error}") from e
            error = e
        except openai.APIStatusError as e:
            if e.status_code == 401:
                raise LLMError("The LLM API key is invalid. Check LLM_API_KEY.") from e
            if e.status_code == 404:
                raise LLMError(f"Model '{model}' not found at {config.LLM_BASE_URL}.") from e
            if e.status_code not in RETRYABLE_STATUS_CODES:
                raise LLMError(f"The LLM rejected the request ({e.status_code}): {e.message}") from e
            error = e
        except (ConnectionError, httpx.TransportError, openai.APIConnectionError) as e:
            # Network problems: connect failures, timeouts, dropped connections.
            error = e

        if stream_started:
            # Part of the reply was already shown; retrying would repeat it.
            raise LLMError("The connection dropped partway through the reply. Please try again.") from error
        if attempt == config.MAX_ATTEMPTS:
            if isinstance(error, openai.RateLimitError):
                raise LLMError("The LLM service's rate limit was reached. Wait a minute and try again.") from error
            if isinstance(error, (ConnectionError, httpx.ConnectError, openai.APIConnectionError)):
                raise LLMError(_unreachable_message()) from error
            raise LLMError(f"LLM call failed after {attempt} attempts: {error}") from error

        # Exponential backoff (1s, 2s, 4s, ...) plus random jitter, or longer if the server says so.
        delay = config.RETRY_BASE_DELAY * 2 ** (attempt - 1) + random.uniform(0, 0.5)
        delay = max(delay, _retry_after_seconds(error))
        logger.warning(
            "LLM call failed (%s). Retrying in %.1fs (attempt %d/%d)...",
            error, delay, attempt + 1, config.MAX_ATTEMPTS,
        )
        time.sleep(delay)


def _retry_after_seconds(error: Exception) -> float:
    """How long the server asked us to wait (its Retry-After header), or 0 if it didn't say."""
    if not isinstance(error, openai.APIStatusError):
        return 0.0
    try:
        return min(float(error.response.headers.get("retry-after", 0)), MAX_RETRY_AFTER)
    except ValueError:  # the header can also be a date; ignore that form
        return 0.0


def _unreachable_message() -> str:
    if config.LLM_PROVIDER == "ollama":
        return f"Can't reach Ollama at {config.OLLAMA_HOST}. Is it running? Start it with: ollama serve"
    return f"Can't reach the LLM API at {config.LLM_BASE_URL}. Check the internet connection."


def _chat_once(
    messages: list[dict], model: str, on_token: Callable[[str], None] | None,
    temperature: float | None, max_tokens: int, tools: list[dict] | None = None,
) -> LLMResponse:
    """One attempt at the LLM call, no error handling."""
    temperature = config.TEMPERATURE if temperature is None else temperature
    if config.LLM_PROVIDER == "ollama":
        return _chat_once_ollama(_messages_for_ollama(messages), model, on_token, temperature, max_tokens, tools)
    return _chat_once_openai(_messages_for_openai(messages), model, on_token, temperature, max_tokens, tools)


# Each provider writes tool-call messages slightly differently; we keep one neutral form and convert.

def _messages_for_ollama(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": [
                {"function": {"name": t["name"], "arguments": t["arguments"]}} for t in m["tool_calls"]]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": m["content"], "tool_name": m.get("name", "")})
        else:
            out.append(m)
    return out


def _messages_for_openai(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or None, "tool_calls": [
                {"id": t["id"], "type": "function", "function": {"name": t["name"], "arguments": json.dumps(t["arguments"])}}
                for t in m["tool_calls"]]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append(m)
    return out


# ---------- Ollama ----------

def _chat_once_ollama(
    messages: list[dict], model: str, on_token: Callable[[str], None] | None, temperature: float, max_tokens: int,
    tools: list[dict] | None = None,
) -> LLMResponse:
    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
        "num_ctx": config.CONTEXT_WINDOW,
    }

    if on_token is None:
        response = _ollama_client.chat(model=model, messages=messages, options=options, tools=tools)
        result = _ollama_to_llm_response(response, response.message.content or "")
        result.tool_calls = [
            ToolCall(id=f"call_{i}", name=t.function.name, arguments=dict(t.function.arguments or {}))
            for i, t in enumerate(response.message.tool_calls or [])
        ]
        return result

    start = time.perf_counter()
    first_token_at = None
    pieces = []
    for chunk in _ollama_client.chat(model=model, messages=messages, options=options, stream=True):
        piece = chunk.message.content
        if piece:
            if first_token_at is None:
                first_token_at = time.perf_counter() - start
            pieces.append(piece)
            on_token(piece)

    # The last chunk (done=True) carries the token counts and timings.
    return _ollama_to_llm_response(chunk, "".join(pieces), time_to_first_token=first_token_at)


def _ollama_to_llm_response(
    response: ollama.ChatResponse, text: str, time_to_first_token: float | None = None
) -> LLMResponse:
    # Ollama reports durations in nanoseconds.
    generation_seconds = response.eval_duration / 1e9
    return LLMResponse(
        text=text,
        input_tokens=response.prompt_eval_count,
        output_tokens=response.eval_count,
        total_seconds=response.total_duration / 1e9,
        tokens_per_second=response.eval_count / generation_seconds if generation_seconds else 0.0,
        time_to_first_token=time_to_first_token,
        truncated=response.done_reason == "length",
    )


# ---------- OpenAI-compatible APIs (Groq, OpenRouter, ...) ----------

def _chat_once_openai(
    messages: list[dict], model: str, on_token: Callable[[str], None] | None, temperature: float, max_tokens: int,
    tools: list[dict] | None = None,
) -> LLMResponse:
    client = _get_openai_client()
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if config.LLM_REASONING_EFFORT:
        params["reasoning_effort"] = config.LLM_REASONING_EFFORT
    if tools:
        params["tools"] = tools
    start = time.perf_counter()

    if on_token is None:
        response = client.chat.completions.create(**params)
        choice = response.choices[0]
        result = _openai_to_llm_response(
            choice.message.content or "", response.usage, time.perf_counter() - start, None, choice.finish_reason
        )
        result.tool_calls = [
            ToolCall(id=t.id, name=t.function.name, arguments=_parse_arguments(t.function.arguments))
            for t in (choice.message.tool_calls or [])
        ]
        return result

    first_token_at = None
    pieces = []
    usage = None
    finish_reason = None
    # include_usage asks the server to send token counts in one final chunk (which has no text).
    stream = client.chat.completions.create(**params, stream=True, stream_options={"include_usage": True})
    for chunk in stream:
        usage = chunk.usage or usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        finish_reason = choice.finish_reason or finish_reason
        piece = choice.delta.content
        if piece:
            if first_token_at is None:
                first_token_at = time.perf_counter() - start
            pieces.append(piece)
            on_token(piece)

    return _openai_to_llm_response(
        "".join(pieces), usage, time.perf_counter() - start, first_token_at, finish_reason
    )


def _parse_arguments(raw: str) -> dict:
    """OpenAI-style APIs send tool arguments as a JSON string; a malformed one becomes {} (the tool reports it)."""
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _openai_to_llm_response(
    text: str, usage, total_seconds: float, time_to_first_token: float | None, finish_reason: str | None
) -> LLMResponse:
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    generation_seconds = total_seconds - (time_to_first_token or 0.0)
    return LLMResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_seconds=total_seconds,
        tokens_per_second=output_tokens / generation_seconds if generation_seconds > 0 else 0.0,
        time_to_first_token=time_to_first_token,
        truncated=finish_reason == "length",
    )
