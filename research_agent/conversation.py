"""Follow-up questions: rewrite "and the big one?" into a question that makes sense on its own.

Retrieval only sees the words of the question, so a follow-up that leans on earlier turns would find
nothing useful. One quick LLM call turns it into a standalone question first ("query condensation").
"""

import re

from research_agent.llm_client import LLMError, chat

MAX_TURNS = 3          # earlier question/answer pairs given as context
ANSWER_PREVIEW = 300   # characters of each earlier answer (enough to resolve "it", "that", ...)

SYSTEM_PROMPT = """You rewrite a user's follow-up question into a standalone question that can be understood
without the conversation. You never answer questions."""

REWRITE_REQUEST = """Rewrite my last question, "{question}", as a standalone question: replace words like "it",
"that", "they" or "the big one" with what they refer to, and keep what my question changes compared to
before (for example, after asking about the capital of France, "and Germany?" becomes
"What is the capital of Germany?"). If it is already standalone, repeat it unchanged.
Reply with only the question."""


def condense_question(question: str, history: list[dict]) -> str:
    """A standalone version of the question, given earlier {"role", "content"} messages (oldest first)."""
    turns = [m for m in history if m["role"] in ("user", "assistant")][-2 * MAX_TURNS:]
    if not turns:
        return question
    # The conversation is given as real chat turns: the format chat models were trained on.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in turns:
        text = " ".join(m["content"].split())
        messages.append({"role": m["role"], "content": text if m["role"] == "user" else text[:ANSWER_PREVIEW]})
    messages.append({"role": "user", "content": REWRITE_REQUEST.format(question=question)})
    try:
        rewritten = chat(messages, temperature=0, max_tokens=100).text.strip().strip('"')
    except LLMError:
        return question
    rewritten = re.sub(r"^(standalone question|question|user)\s*:\s*", "", rewritten, flags=re.IGNORECASE).strip('"')
    # A sane rewrite is one line, not an answer or an essay; otherwise keep the user's own words.
    return rewritten if rewritten and "\n" not in rewritten and len(rewritten) < 4 * len(question) + 200 else question
