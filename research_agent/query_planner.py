"""Turn a user's question into better search queries with one quick LLM call.

- Rewriting: use the words the papers would use ("token order" -> "positional encoding").
- Decomposition: a question needing several documents or facts becomes one query per part,
  so each part gets its own search and one document can't crowd out the others.
Falls back to the original question if anything goes wrong.
"""

import json
import re

from research_agent import config
from research_agent.llm_client import LLMError, chat

MAX_QUERIES = 3

PLANNER_PROMPT = """You turn a user's question into search queries for a search engine over these documents:
{documents}

Rules:
- If the question asks about ONE thing, return ONE query: the question rewritten with the technical
  terms a research paper would use.
- If it needs facts from DIFFERENT documents, or several separate facts, return one query per part
  (at most {max_queries}), each naming its document or topic.
- Keep each query short and specific. Do not answer the question.

Question: {question}

Reply with JSON only: {{"queries": ["...", "..."]}}"""


def plan_queries(question: str, documents: list[str]) -> list[str]:
    """1 to MAX_QUERIES search queries for the question (the question itself if planning fails)."""
    prompt = PLANNER_PROMPT.format(
        documents="\n".join(f"- {d}" for d in documents) or "- (none)", max_queries=MAX_QUERIES, question=question
    )
    try:
        reply = chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=200).text
    except LLMError:
        return [question]
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        queries = json.loads(match.group(0))["queries"] if match else []
    except (json.JSONDecodeError, KeyError, TypeError):
        queries = []
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()][:MAX_QUERIES]
    return queries or [question]
