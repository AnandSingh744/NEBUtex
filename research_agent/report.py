"""Research report workflow, built with LangGraph.

The workflow is a graph: a shared STATE, NODES (plain functions that read the state and return updates)
and EDGES (what runs next, including a conditional edge):

    plan -> research -> (too many unanswered and not yet re-planned? -> plan again) -> write -> END

Usage:
    python -m research_agent.report "How does retrieval-augmented generation work, and what retriever does it use?"
"""

import argparse
import json
import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from research_agent.llm_client import chat
from research_agent.rag import answer
from research_agent.retriever import StoreRetriever
from research_agent.store import DocumentStore, get_store

MAX_SUBQUESTIONS = 4


class ReportState(TypedDict, total=False):
    topic: str
    subquestions: list[str]
    findings: list[dict]      # {question, answer, status, sources: {label: citation}}
    replanned: bool
    report: str


PLAN_PROMPT = """Break this research topic into {n} short, specific sub-questions that can each be answered
from research papers. {avoid}
Topic: {topic}
Reply with JSON only: {{"subquestions": ["...", "..."]}}"""

WRITE_PROMPT = """Write a short research report on: {topic}

Use ONLY the findings below. Keep their citations exactly as written (like [rag_lewis_2020.pdf, p. 4]).
Structure: a one-paragraph overview, then one short section per finding, then "Open questions" listing
anything the findings could not answer.

{findings}"""


def make_graph(store: DocumentStore):
    retriever = StoreRetriever(store)

    def plan(state: ReportState) -> dict:
        avoid = ""
        if state.get("findings"):  # re-planning: steer away from questions the documents couldn't answer
            failed = [f["question"] for f in state["findings"] if f["status"] == "not_found"]
            avoid = "Avoid these, which could not be answered: " + "; ".join(failed)
        reply = chat([{"role": "user", "content": PLAN_PROMPT.format(n=MAX_SUBQUESTIONS, avoid=avoid, topic=state["topic"])}],
                     temperature=0, max_tokens=300, json_mode=True).text
        try:
            subs = [q for q in json.loads(reply).get("subquestions", []) if isinstance(q, str) and q.strip()]
        except json.JSONDecodeError:
            subs = []
        return {"subquestions": subs[:MAX_SUBQUESTIONS] or [state["topic"]], "replanned": bool(state.get("findings"))}

    def research(state: ReportState) -> dict:
        findings = []
        for q in state["subquestions"]:
            result = answer(q, retriever, temperature=0)
            # Replace [S1]-style labels with full citations so the writer can keep them across findings.
            text = result.text
            sources = {}
            for n, src in result.cited_sources():
                text = text.replace(f"[S{n}]", f"[{src.citation}]")
                sources[f"S{n}"] = src.citation
            findings.append({"question": q, "answer": text, "status": result.status, "sources": sources})
        return {"findings": findings}

    def needs_replan(state: ReportState) -> str:  # the conditional edge
        unanswered = sum(f["status"] == "not_found" for f in state["findings"])
        return "plan" if unanswered * 2 > len(state["findings"]) and not state.get("replanned") else "write"

    def write(state: ReportState) -> dict:
        answered = [f for f in state["findings"] if f["status"] != "not_found"]
        if not answered:
            return {"report": "The documents don't contain enough information on this topic."}
        findings = "\n\n".join(f"Q: {f['question']}\nA: {f['answer']}" for f in state["findings"])
        return {"report": chat([{"role": "user", "content": WRITE_PROMPT.format(topic=state["topic"], findings=findings)}],
                               temperature=0, max_tokens=1200).text.strip()}

    graph = StateGraph(ReportState)
    graph.add_node("plan", plan)
    graph.add_node("research", research)
    graph.add_node("write", write)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "research")
    graph.add_conditional_edges("research", needs_replan, {"plan": "plan", "write": "write"})
    graph.add_edge("write", END)
    return graph.compile()


def research_report(topic: str, store: DocumentStore | None = None, on_step=None) -> ReportState:
    """Run the whole graph. on_step(node_name, update) is called after each node (for progress display)."""
    app = make_graph(store or get_store())
    state: ReportState = {"topic": topic}
    for event in app.stream(state):  # one event per finished node: {node_name: update}
        for node, update in event.items():
            state.update(update)
            if on_step:
                on_step(node, update)
    return state


def _cited_count(text: str) -> int:
    return len(set(re.findall(r"\[[^\]]+\.pdf, p\. \d+\]", text)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write a research report with a LangGraph workflow.")
    parser.add_argument("topic")
    args = parser.parse_args()

    def show(node: str, update: dict) -> None:
        if node == "plan":
            print(f"📋 plan: {update['subquestions']}")
        elif node == "research":
            print("🔎 research: " + ", ".join(f["status"] for f in update["findings"]))
        elif node == "write":
            print("✍️  write: done")

    final = research_report(args.topic, on_step=show)
    print(f"\n{final['report']}\n\n[{_cited_count(final['report'])} distinct citations]")
