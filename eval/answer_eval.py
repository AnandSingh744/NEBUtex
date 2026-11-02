"""Answer evaluation on eval/testset.jsonl: is each answer correct, and supported by its sources?

Three kinds of checks:
- behaviour (code, exact): unanswerable questions must get "not_found"; answerable ones shouldn't
- correctness (LLM judge): does the answer match the reference answer?
- faithfulness (LLM judge): is every claim supported by the passages the answerer was given?

Usage (from the project folder):
    python -m eval.answer_eval --check-judge    # first: test the judge on answers with known verdicts
    python -m eval.answer_eval                  # then: evaluate the whole test set
    python -m eval.answer_eval --limit 5        # quick run on the first 5 questions
"""

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from research_agent import config
from research_agent.llm_client import LLMError, chat
from research_agent.rag import RAGAnswer, answer
from research_agent.retriever import StoreRetriever
from research_agent.store import get_store
from eval.retrieval_eval import load_testset

ANSWERABLE = {"factual", "multi_doc", "false_premise"}
# Questions processed at the same time. Keep it equal to Ollama's OLLAMA_NUM_PARALLEL; more parallel
# slots need more memory (about 1 GB each at an 8k context), and swapping makes everything far slower.
WORKERS = int(os.getenv("EVAL_WORKERS", "2"))
JUDGE_MAX_TOKENS = 800  # enough for the JSON we ask for; a cap cuts off a model stuck repeating itself
# Answers are generated at temperature 0 during evaluation: repeatable (no run-to-run noise) and cacheable.
ANSWER_TEMPERATURE = 0.0

CORRECTNESS_PROMPT = """You are grading an answer to a question about research papers.

Question: {question}
Reference answer (correct): {reference}
Answer to grade: {answer}

Grade the answer against the reference answer:
- "correct": contains the key facts of the reference and nothing that contradicts it. Different wording is fine.
- "partially_correct": some key facts are right, but others are missing or mixed up.
- "incorrect": the key facts are wrong or missing, or it contradicts the reference.
Rules:
- If the question asks about several things (for example two models), the answer must give the right
  fact for EACH one. Giving one value for several things, or covering only some of them, is "partially_correct".
- If every key fact is wrong, or attached to the wrong thing (for example two numbers swapped), it is "incorrect".
- Ignore citation markers like [S1]. Extra correct detail is fine.

First explain your reasoning in one or two sentences, then give the verdict.
Reply with JSON only: {{"reasoning": "...", "verdict": "correct" | "partially_correct" | "incorrect"}}"""

FAITHFULNESS_PROMPT = """You are checking whether an answer is supported by its source passages.

Source passages:
{passages}

Answer to check: {answer}

Split the answer into its individual factual claims. For EACH claim decide whether the passages
support it. If supported, copy the exact supporting words from a passage as the quote.
A claim is supported if the passages state it, even in different words. Ignore citation markers like [S1].

Reply with JSON only:
{{"claims": [{{"claim": "...", "supported": true, "quote": "exact words from a passage"}},
            {{"claim": "...", "supported": false, "quote": ""}}]}}"""

QUOTE_MATCH = 0.8  # share of a quote's words that must appear in the passages for us to believe it
CLAIM_MATCH = 0.6  # share of a claim's words that must appear in the ANSWER (else the judge invented it)


def _parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)  # models sometimes add words around the JSON
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def judge_correctness(question: str, reference: str, answer_text: str, model: str = config.JUDGE_MODEL) -> dict:
    """{"verdict": correct | partially_correct | incorrect | judge_error, "reasoning": str}"""
    prompt = CORRECTNESS_PROMPT.format(question=question, reference=reference, answer=answer_text)
    data = _parse_json(chat([{"role": "user", "content": prompt}], model=model, temperature=0,
                            max_tokens=JUDGE_MAX_TOKENS).text)
    if not data or data.get("verdict") not in ("correct", "partially_correct", "incorrect"):
        return {"verdict": "judge_error", "reasoning": "unreadable judge reply"}
    return {"verdict": data["verdict"], "reasoning": str(data.get("reasoning", ""))}


def _quote_found(quote: str, passages: list[str]) -> bool:
    """Does the judge's quote really come from the passages? (Checked by code, not trusted blindly.)"""
    words = re.findall(r"\w+", quote.lower())
    passage_words = set(re.findall(r"\w+", " ".join(passages).lower()))
    return bool(words) and sum(w in passage_words for w in words) / len(words) >= QUOTE_MATCH


def _claim_from_answer(claim: str, answer_text: str) -> bool:
    """Is this claim really in the answer? Small judges sometimes 'check' sentences from the passages instead."""
    words = re.findall(r"\w+", claim.lower())
    answer_words = set(re.findall(r"\w+", answer_text.lower()))
    return bool(words) and sum(w in answer_words for w in words) / len(words) >= CLAIM_MATCH


# One complete claim object, for recovering claims from a reply that was cut off mid-way.
_CLAIM_OBJECT = re.compile(
    r'\{\s*"claim"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"supported"\s*:\s*(true|false)\s*,\s*"quote"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'
)


def _salvage_claims(text: str) -> list[dict]:
    """Every complete claim object in a (possibly cut-off) reply."""
    return [{"claim": c, "supported": s == "true", "quote": q} for c, s, q in _CLAIM_OBJECT.findall(text)]


def judge_faithfulness(answer_text: str, passages: list[str], model: str = config.JUDGE_MODEL) -> dict:
    """{"faithful": bool | None (None = judge error), "unsupported_claims": [...], "reasoning": str}

    The judge checks claim by claim and must quote its evidence; a "supported" claim whose quote
    isn't actually in the passages is counted as unsupported.
    """
    numbered = "\n\n".join(f"[S{i}] {p}" for i, p in enumerate(passages, start=1))
    prompt = FAITHFULNESS_PROMPT.format(passages=numbered, answer=answer_text)
    reply = chat([{"role": "user", "content": prompt}], model=model, temperature=0, max_tokens=JUDGE_MAX_TOKENS)
    data = _parse_json(reply.text)
    salvaged = False
    if not data or not isinstance(data.get("claims"), list):
        claims = _salvage_claims(reply.text)  # e.g. the reply hit the token cap: keep the complete claims
        if not claims:
            return {"faithful": None, "unsupported_claims": [], "reasoning": "unreadable judge reply"}
        data, salvaged = {"claims": claims}, True
    unsupported, fake_quotes, invented = [], 0, 0
    for c in data["claims"]:
        if not isinstance(c, dict) or not str(c.get("claim", "")).strip():
            continue
        if not _claim_from_answer(str(c["claim"]), answer_text):
            invented += 1  # not a claim the answer makes: ignore it
            continue
        if c.get("supported") is True and _quote_found(str(c.get("quote", "")), passages):
            continue
        if c.get("supported") is True:
            fake_quotes += 1  # judge said "supported" but its quote isn't in the passages
        unsupported.append(str(c["claim"]))
    note = (f"{len(data['claims'])} claims checked"
            + (f", {fake_quotes} with quotes not found in the passages" if fake_quotes else "")
            + (f", {invented} ignored (not in the answer)" if invented else "")
            + (" (reply was cut off; used the complete claims)" if salvaged else ""))
    return {"faithful": not unsupported, "unsupported_claims": unsupported, "reasoning": note}


@dataclass
class AnswerResult:
    """One question's answer and verdicts, as plain data so it can be saved to and loaded from JSON."""
    item: dict
    answer_text: str | None = None
    status: str | None = None                          # answered | partial | not_found | unsupported
    sources: list[dict] = field(default_factory=list)  # what the answerer was shown: citation, chunk_id, text
    cited: list[int] = field(default_factory=list)     # source numbers the answer cites ([S1] -> 1)
    seconds: float = 0.0
    error: str | None = None
    correctness: dict = field(default_factory=dict)
    faithfulness: dict = field(default_factory=dict)

    @property
    def outcome(self) -> str:
        """What the system did, judged by rules alone (no LLM)."""
        if self.status is None:
            return "error"
        t, status = self.item["type"], self.status
        if t == "not_answerable":
            return "ok" if status == "not_found" else "hallucinated"   # must decline
        if t == "partial":
            return "ok" if status == "partial" else ("refused" if status == "not_found" else "not_flagged")
        if status == "not_found":
            return "refused"                                           # safe, but a miss
        return "unsupported" if status == "unsupported" else "ok"


def generate_answers(items: list[dict], verbose: bool = True, workers: int = WORKERS) -> list[AnswerResult]:
    """Pass 1: run the real RAG system on every question, several at once."""
    retriever = StoreRetriever(get_store())

    def run(item: dict) -> AnswerResult:
        r = AnswerResult(item)
        try:
            rag: RAGAnswer = answer(item["question"], retriever, temperature=ANSWER_TEMPERATURE)
            r.answer_text, r.status, r.cited = rag.text, rag.status, rag.cited
            r.sources = [{"citation": s.citation, "chunk_id": s.chunk.chunk_id, "text": s.chunk.text} for s in rag.sources]
            r.seconds = round(rag.llm.total_seconds, 2)
        except LLMError as e:
            r.error = str(e)
        if verbose:
            print(f"  answered {item['id']} -> {r.status or 'ERROR'}")
        return r

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, items))  # map keeps the results in question order


def judge_answers(results: list[AnswerResult], judge_model: str = config.JUDGE_MODEL, verbose: bool = True,
                  workers: int = WORKERS) -> None:
    """Pass 2: grade the answers that make sense to grade (fills in correctness and faithfulness)."""

    def judge(r: AnswerResult) -> None:
        r.correctness, r.faithfulness = {}, {}
        if r.status in (None, "not_found") or r.item["type"] == "not_answerable":
            return  # nothing to grade: a refusal, or a question whose only right answer is "not found"
        try:
            if r.item["answer"]:
                r.correctness = judge_correctness(r.item["question"], r.item["answer"], r.answer_text, judge_model)
            r.faithfulness = judge_faithfulness(r.answer_text, [s["text"] for s in r.sources], judge_model)
        except LLMError as e:
            r.error = f"judge: {e}"
        if verbose:
            print(f"  judged {r.item['id']} -> {r.correctness.get('verdict', '-')}, faithful={r.faithfulness.get('faithful', '-')}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(judge, results))


def evaluate_answers(items: list[dict], judge_model: str = config.JUDGE_MODEL, verbose: bool = True) -> list[AnswerResult]:
    """Answer everything, then judge everything. (Two passes: if the judge is a different model,
    the computer switches models once instead of once per question.)"""
    results = generate_answers(items, verbose)
    judge_answers(results, judge_model, verbose)
    return results


CORRECTNESS_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


def summarize_answers(results: list[AnswerResult]) -> dict:
    answerable = [r for r in results if r.item["type"] in ANSWERABLE]
    unanswerable = [r for r in results if r.item["type"] == "not_answerable"]
    graded = [r for r in answerable if r.correctness.get("verdict") in CORRECTNESS_SCORE]
    judged_faith = [r for r in results if r.faithfulness.get("faithful") is not None]
    # Refusals score 0 for correctness: from the user's point of view, the question went unanswered.
    correctness = sum(CORRECTNESS_SCORE[r.correctness["verdict"]] for r in graded) / len(answerable) if answerable else 0
    return {
        "questions": len(results),
        "correctness": correctness,                                     # over all answerable questions
        "fully_correct": sum(r.correctness.get("verdict") == "correct" for r in answerable) / max(len(answerable), 1),
        "faithfulness": sum(r.faithfulness["faithful"] for r in judged_faith) / max(len(judged_faith), 1),
        "refusal_rate": sum(r.outcome == "refused" for r in answerable) / max(len(answerable), 1),
        "hallucination_rate": sum(r.outcome == "hallucinated" for r in unanswerable) / max(len(unanswerable), 1),
        "judge_errors": sum(r.correctness.get("verdict") == "judge_error" or
                            ("faithful" in r.faithfulness and r.faithfulness["faithful"] is None) for r in results),
        "outcomes": dict(Counter(r.outcome for r in results)),
    }


def print_answer_report(results: list[AnswerResult]) -> None:
    s = summarize_answers(results)
    print(f"\n{s['questions']} questions | judge: {config.JUDGE_MODEL} | answerer: {config.MODEL}\n")
    print(f"  correctness         {s['correctness']:.2f}   (correct = 1, partially = 0.5; refusals count 0)")
    print(f"  fully correct       {s['fully_correct']:.2f}")
    print(f"  faithfulness        {s['faithfulness']:.2f}   (answers with no unsupported claims)")
    print(f"  refusal rate        {s['refusal_rate']:.2f}   (answerable questions answered 'not found')")
    print(f"  hallucination rate  {s['hallucination_rate']:.2f}   (unanswerable questions answered anyway)")
    print(f"  judge errors        {s['judge_errors']}")
    print(f"  outcomes            {s['outcomes']}")

    by_type = defaultdict(list)
    for r in results:
        by_type[r.item["type"]].append(r)
    print("\nBy type:")
    for t, group in sorted(by_type.items()):
        verdicts = Counter(r.correctness.get("verdict", "-") for r in group)
        print(f"  {t:15} n={len(group):>2}  outcomes={dict(Counter(r.outcome for r in group))}  verdicts={dict(verdicts)}")

    problems = [r for r in results if r.outcome != "ok" or r.correctness.get("verdict") in ("incorrect", "partially_correct")
                or r.faithfulness.get("faithful") is False]
    print(f"\nProblems to read ({len(problems)}):")
    for r in problems:
        print(f"\n  {r.item['id']} [{r.item['type']}] outcome={r.outcome} verdict={r.correctness.get('verdict', '-')} "
              f"faithful={r.faithfulness.get('faithful', '-')}")
        print(f"    Q: {r.item['question']}")
        print(f"    expected: {r.item['answer']}")
        print(f"    got:      {' '.join((r.answer_text or r.error or '').split())[:300]}")
        if r.correctness.get("reasoning"):
            print(f"    judge (correctness): {r.correctness['reasoning'][:250]}")
        if r.faithfulness.get("unsupported_claims"):
            print(f"    judge (unsupported): {r.faithfulness['unsupported_claims']}")


# ---------- testing the judge itself ----------

def check_judge(model: str = config.JUDGE_MODEL) -> None:
    """Grade answers whose correct verdicts we know. If the judge gets these wrong, don't trust its scores."""
    heads_q = "In the Transformer paper, how many parallel attention heads does the base model use?"
    heads_ref = "h = 8 heads (with d_k = d_v = 64)."
    heads_src = ["In this work we employ h = 8 parallel attention layers, or heads. For each of these we use "
                 "dk = dv = dmodel/h = 64."]
    train_q = "In the Transformer paper, how long did training take for the base model compared with the big model?"
    train_ref = "Base models: about 12 hours (100,000 steps); big models: 3.5 days (300,000 steps)."
    train_src = ["We trained the base models for a total of 100,000 steps or 12 hours. The big models were "
                 "trained for 300,000 steps (3.5 days). We trained our models on one machine with 8 NVIDIA P100 GPUs."]

    cases = [  # (name, question, reference, passages, answer, expected verdict, expected faithful)
        ("correct, exact wording", heads_q, heads_ref, heads_src, "The base model uses h = 8 parallel attention heads [S1].", "correct", True),
        ("correct, paraphrased", heads_q, heads_ref, heads_src, "Eight heads are used, each with 64-dimensional keys and values [S1].", "correct", True),
        ("wrong number", heads_q, heads_ref, heads_src, "The base model uses 16 attention heads [S1].", "incorrect", False),
        ("right, plus a hallucination", heads_q, heads_ref, heads_src, "It uses 8 heads [S1], and the model was trained on Google TPUs [S1].", "correct", False),
        ("the '3.5 days' merge", train_q, train_ref, train_src, "Training took 3.5 days on 8 P100 GPUs [S1].", "partially_correct", True),
        ("half the answer", train_q, train_ref, train_src, "The base model trained for about 12 hours [S1].", "partially_correct", True),
        ("fully correct comparison", train_q, train_ref, train_src, "The base model took about 12 hours and the big model 3.5 days [S1].", "correct", True),
        ("swapped numbers", train_q, train_ref, train_src, "The base model took 3.5 days and the big model about 12 hours [S1].", "incorrect", False),
    ]
    agree_c = agree_f = 0
    print(f"Judge check ({model}):\n")
    for name, q, ref, src, ans, want_v, want_f in cases:
        c = judge_correctness(q, ref, ans, model)
        f = judge_faithfulness(ans, src, model)
        ok_c, ok_f = c["verdict"] == want_v, f["faithful"] == want_f
        agree_c += ok_c
        agree_f += ok_f
        print(f"  {'✓' if ok_c else '✗'} correctness {c['verdict']:18} (want {want_v:18}) "
              f"{'✓' if ok_f else '✗'} faithful={str(f['faithful']):5} (want {want_f!s:5})  {name}")
        if not ok_f and f["unsupported_claims"]:
            print(f"      judge said unsupported: {f['unsupported_claims']}")
    print(f"\nJudge agreed with the known verdicts: correctness {agree_c}/{len(cases)}, faithfulness {agree_f}/{len(cases)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate answers with rules and an LLM judge.")
    parser.add_argument("--check-judge", action="store_true", help="test the judge on answers with known verdicts")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    args = parser.parse_args()

    if args.check_judge:
        check_judge()
    else:
        start = time.perf_counter()
        items = load_testset()[: args.limit] if args.limit else load_testset()
        results = evaluate_answers(items)
        print_answer_report(results)
        print(f"\n(took {time.perf_counter() - start:.0f}s)")
