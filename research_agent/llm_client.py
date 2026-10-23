"""The only module that talks to the LLM. The rest of the project calls chat().

Two providers, chosen by config.LLM_PROVIDER:
- "ollama": local Ollama server
- "openai_compatible": any hosted API using the OpenAI request format (e.g. Groq)
Both return the same LLMResponse and raise the same LLMError, so callers never know which is in use.
"""

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

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
    try:
        if config.LLM_PROVIDER == "ollama":
            return any(m.model == config.MODEL for m in _ollama_client.list().models)
        return any(m.id == config.MODEL for m in _get_openai_client().models.list())
    except (LLMError, ConnectionError, httpx.TransportError, ollama.ResponseError,
            openai.APIStatusError, openai.APIConnectionError):
        return False


def chat(
    messages: list[dict],
    model: str = config.MODEL,
    on_token: Callable[[str], None] | None = None,
) -> LLMResponse:
    """Send a list of {"role", "content"} messages and return the model's reply.

    The model is stateless: it sees only the messages passed in this call.
    If on_token is given, the reply is streamed and on_token is called with
    each piece of text as soon as it is generated.

    Transient failures are retried with exponential backoff. Anything that
    still fails is raised as LLMError.
    """
    for attempt in range(1, config.MAX_ATTEMPTS + 1):
        stream_started = False

        def forward(piece: str) -> None:
            nonlocal stream_started
            stream_started = True
            on_token(piece)

        try:
            return _chat_once(messages, model, forward if on_token else None)
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
    messages: list[dict], model: str, on_token: Callable[[str], None] | None
) -> LLMResponse:
    """One attempt at the LLM call, no error handling."""
    if config.LLM_PROVIDER == "ollama":
        return _chat_once_ollama(messages, model, on_token)
    return _chat_once_openai(messages, model, on_token)


# ---------- Ollama ----------

def _chat_once_ollama(
    messages: list[dict], model: str, on_token: Callable[[str], None] | None
) -> LLMResponse:
    options = {
        "temperature": config.TEMPERATURE,
        "num_predict": config.MAX_OUTPUT_TOKENS,
        "num_ctx": config.CONTEXT_WINDOW,
    }

    if on_token is None:
        response = _ollama_client.chat(model=model, messages=messages, options=options)
        return _ollama_to_llm_response(response, response.message.content)

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
    messages: list[dict], model: str, on_token: Callable[[str], None] | None
) -> LLMResponse:
    client = _get_openai_client()
    params = {
        "model": model,
        "messages": messages,
        "temperature": config.TEMPERATURE,
        "max_tokens": config.MAX_OUTPUT_TOKENS,
    }
    start = time.perf_counter()

    if on_token is None:
        response = client.chat.completions.create(**params)
        choice = response.choices[0]
        return _openai_to_llm_response(
            choice.message.content or "", response.usage, time.perf_counter() - start, None, choice.finish_reason
        )

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
