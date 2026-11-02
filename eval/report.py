"""One-command evaluation: retrieval + answers, saved to eval/runs/, compared with the previous run.

Run it before and after every change to see what got better and what got worse.

Usage (from the project folder):
    python -m eval.report --label baseline
    python -m eval.report --label reranker                 # compares itself with the latest saved run
    python -m eval.report --retrieval-only                 # fast (~1 min, no LLM): for search changes
    python -m eval.report --rejudge eval/runs/<run>.json   # re-grade saved answers (e.g. after changing the judge)
    python -m eval.report --compare-to eval/runs/<run>.json
    python -m eval.report --quick                          # every 3rd question (17): a fast check while working
    python -m eval.report --limit 6                        # quick run on the first N questions

LLM replies are cached (eval/.llm_cache.json): answers and verdicts use temperature 0, so an identical
prompt always gets the identical reply. Re-runs only call the model for questions whose prompt changed.
"""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from research_agent import config, llm_client
from research_agent.chunker import CHUNK_OVERLAP, CHUNK_SIZE
from research_agent.embedder import embed_query
from research_agent.retriever import StoreRetriever
from research_agent.store import get_store
from eval.answer_eval import ANSWER_TEMPERATURE, AnswerResult, evaluate_answers, judge_answers, print_answer_report, summarize_answers
from eval.retrieval_eval import TESTSET_PATH, evaluate_retrieval, load_testset, summarize

RUNS_DIR = config.EVAL_DIR / "runs"
CACHE_PATH = config.EVAL_DIR / ".llm_cache.json"

# (section, metric, True if higher is better) -- the numbers compared between runs
COMPARED_METRICS = [
    ("retrieval", "hit@1", True), ("retrieval", "hit@5", True), ("retrieval", "recall@5", True),
    ("retrieval", "mrr", True), ("retrieval", "precision@5", True),
    ("answers", "correctness", True), ("answers", "fully_correct", True), ("answers", "faithfulness", True),
    ("answers", "refusal_rate", False), ("answers", "hallucination_rate", False),
]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=config.PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def run_metadata(label: str, store) -> dict:
    """Everything needed to know WHAT was tested, so two runs can be compared fairly."""
    return {
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD") or "unknown",
        "uncommitted_changes": bool(_git("status", "--porcelain")),
        "testset_sha256": hashlib.sha256(TESTSET_PATH.read_bytes()).hexdigest()[:16],
        "config": {
            "llm_provider": config.LLM_PROVIDER, "model": config.MODEL, "judge_model": config.JUDGE_MODEL,
            "answer_temperature": ANSWER_TEMPERATURE, "embedding_model": config.EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "top_k": config.RAG_TOP_K,
            "retrieval_mode": config.RETRIEVAL_MODE, "candidates": config.RETRIEVAL_CANDIDATES,
            "rerank_model": config.RERANK_MODEL or "none", "query_planning": config.QUERY_PLANNING,
            "max_context_chars": config.RAG_MAX_CONTEXT_CHARS, "store": type(store).__name__,
        },
    }


def retrieval_records(results) -> list[dict]:
    return [{
        "id": r.item["id"], "first_hit": r.first_hit, "evidence_ranks": r.evidence_ranks,
        "retrieved": [x.citation for x in r.retrieved],
    } for r in results]


def save_run(run: dict) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(run["meta"]["timestamp"]).strftime("%Y%m%d-%H%M%S")
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in run["meta"]["label"])
    path = RUNS_DIR / f"{stamp}-{safe_label}.json"
    path.write_text(json.dumps(run, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def load_run(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def latest_run(exclude: Path | None = None) -> Path | None:
    runs = sorted(p for p in RUNS_DIR.glob("*.json") if p != exclude)
    return runs[-1] if runs else None


def answers_from_run(run: dict, ids: set[str]) -> list[AnswerResult]:
    """Saved answers (for re-judging), restricted to the given question ids."""
    return [AnswerResult(**rec) for rec in run.get("answers", {}).get("questions", []) if rec["item"]["id"] in ids]


def compare(prev: dict, cur: dict, prev_path: Path) -> None:
    pm, cm = prev["meta"], cur["meta"]
    print(f"\n{'=' * 70}\nCOMPARISON: '{pm['label']}' ({pm['timestamp']}, {prev_path.name})  ->  '{cm['label']}'\n{'=' * 70}")

    # 1. Were the two runs testing the same thing? If not, say so before showing any numbers.
    if pm["testset_sha256"] != cm["testset_sha256"]:
        print("⚠ The test set changed between these runs: the averages are NOT directly comparable.")
    changed = {k: (pm["config"].get(k), v) for k, v in cm["config"].items() if pm["config"].get(k) != v}
    print("What changed in the configuration: " + (", ".join(f"{k}: {a} -> {b}" for k, (a, b) in changed.items()) or "nothing"))
    print(f"Code: {pm['git_commit']}{' (+uncommitted)' if pm['uncommitted_changes'] else ''} -> "
          f"{cm['git_commit']}{' (+uncommitted)' if cm['uncommitted_changes'] else ''}")

    prev_ids = {q["id"] for q in prev.get("retrieval", {}).get("questions", [])}
    cur_ids = {q["id"] for q in cur.get("retrieval", {}).get("questions", [])}
    if prev_ids != cur_ids:
        print(f"⚠ Different question sets ({len(prev_ids)} vs {len(cur_ids)} retrieval questions): "
              f"compare the per-question changes, not the averages.")

    # 2. The headline numbers.
    print(f"\n{'metric':28} {'before':>8} {'after':>8} {'change':>8}")
    for section, metric, higher_is_better in COMPARED_METRICS:
        a = prev.get(section, {}).get("summary", {}).get(metric)
        b = cur.get(section, {}).get("summary", {}).get(metric)
        if a is None or b is None:
            continue
        delta = b - a
        verdict = "" if abs(delta) < 0.005 else ("better" if (delta > 0) == higher_is_better else "WORSE")
        print(f"{section + ' ' + metric:28} {a:>8.2f} {b:>8.2f} {delta:>+8.2f}  {verdict}")

    # 3. Which questions changed, and how. (Averages can stay flat while some get better and others worse.)
    print("\nQuestions that changed:")
    shown = 0
    prev_r = {q["id"]: q for q in prev.get("retrieval", {}).get("questions", [])}
    for q in cur.get("retrieval", {}).get("questions", []):
        old = prev_r.get(q["id"])
        if old and old["first_hit"] != q["first_hit"]:
            print(f"  {q['id']} retrieval: first right passage at rank {old['first_hit'] or '>10'} -> {q['first_hit'] or '>10'}")
            shown += 1
    prev_a = {rec["item"]["id"]: rec for rec in prev.get("answers", {}).get("questions", [])}
    for rec in cur.get("answers", {}).get("questions", []):
        old = prev_a.get(rec["item"]["id"])
        if not old:
            continue
        before = (AnswerResult(**old).outcome, old["correctness"].get("verdict", "-"), old["faithfulness"].get("faithful", "-"))
        after = (AnswerResult(**rec).outcome, rec["correctness"].get("verdict", "-"), rec["faithfulness"].get("faithful", "-"))
        if before != after:
            print(f"  {rec['item']['id']} answer: outcome/verdict/faithful {before} -> {after}")
            shown += 1
    if not shown:
        print("  none")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the evaluation, save it, and compare with the previous run.")
    parser.add_argument("--label", default="run", help="a short name for this run, e.g. 'baseline' or 'reranker'")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the (slow) answer evaluation")
    parser.add_argument("--rejudge", metavar="RUN", help="re-grade the answers saved in RUN instead of generating new ones")
    parser.add_argument("--compare-to", metavar="RUN", help="compare with this run (default: the latest saved run)")
    parser.add_argument("--quick", action="store_true", help="every 3rd question only (a fast check)")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--no-cache", action="store_true", help="call the model even for prompts seen before")
    args = parser.parse_args()

    items = load_testset()
    if args.quick:
        items = items[::3]
    if args.limit:
        items = items[: args.limit]
    if not args.no_cache:
        llm_client.enable_cache(CACHE_PATH)
    started = datetime.now()
    store = get_store()
    run = {"meta": run_metadata(args.label, store)}
    embed_query("warm-up")

    print(f"Evaluating {len(items)} questions ({run['meta']['config']['store']}, model {config.MODEL})...\n")
    retrieval = evaluate_retrieval(StoreRetriever(store), items)
    rs = summarize(retrieval)
    run["retrieval"] = {"summary": rs, "questions": retrieval_records(retrieval)}
    print(f"Retrieval: hit@5 {rs['hit@5']:.2f} | recall@5 {rs['recall@5']:.2f} | MRR {rs['mrr']:.2f} "
          f"| precision@5 {rs['precision@5']:.2f}  ({rs['n']} questions)")

    answers = None
    if args.rejudge:
        source = load_run(args.rejudge)
        answers = answers_from_run(source, {i["id"] for i in items})
        run["meta"]["rejudged_from"] = Path(args.rejudge).name
        print(f"\nRe-judging {len(answers)} saved answers from {Path(args.rejudge).name} with {config.JUDGE_MODEL}...")
        judge_answers(answers)
    elif not args.retrieval_only:
        print("\nGenerating and judging answers (slow: every question runs the real system)...")
        answers = evaluate_answers(items)
    if answers is not None:
        run["answers"] = {"summary": summarize_answers(answers), "questions": [asdict(a) for a in answers]}
        print_answer_report(answers)

    llm_client.save_cache()
    path = save_run(run)
    print(f"\nSaved this run to {path.relative_to(config.PROJECT_ROOT)} (took {(datetime.now() - started).seconds}s)")
    prev_path = Path(args.compare_to) if args.compare_to else latest_run(exclude=path)
    if prev_path:
        compare(load_run(prev_path), run, prev_path)
    else:
        print("(No earlier run to compare with: this one is your baseline.)")
