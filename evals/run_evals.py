"""Run the eval set on one contract.

    python evals/run_evals.py --contract data/contracts/bail_exemple.pdf --gold evals/gold/bail_exemple.yaml
    python evals/run_evals.py ... --no-embeddings   # BM25-only baseline, to measure hybrid gains
    python evals/run_evals.py ... --judge           # + LLM-as-judge on answer correctness

Questions go through the same path as the API (router + query rewrite -> retriever ->
answerer), so the numbers describe what users actually get.

Metrics
- extraction_field_accuracy   extracted fields equal to gold
- router_accuracy             evals/router.yaml + every eval question must route to "ask"
- retrieval_recall_at_k       gold page is among the retrieved chunks (retriever alone)
- citation_page_accuracy      at least one citation on the gold page
- quote_verified_rate         cited quotes that really appear in the cited chunk
- abstention_rate             unanswerable questions answered with found=false
- false_abstention_rate       answerable questions wrongly answered with found=false
- answer_accuracy (--judge)   LLM judge compares the answer with the gold answer
- compliance_precision/recall flags raised vs `compliance:` in the gold file (rule ids)
- by_tag                      pass rate per tag (failure taxonomy)

    --fail-under metric=value   exit 1 if a metric is below the threshold (used by CI)
"""
import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.answerer import answer  # noqa: E402
from app.agents.compliance import check  # noqa: E402
from app.agents.extractor import extract  # noqa: E402
from app.agents.graph import plan_request  # noqa: E402
from app.config import settings  # noqa: E402
from app.ingest import ingest  # noqa: E402
from app.llm import chat_json  # noqa: E402
from app.retrieval import Retriever, default_embedder  # noqa: E402

NOT_IN_CONTRACT = "NOT_IN_CONTRACT"
MAX_CONSECUTIVE_ERRORS = 3  # quota exhausted or endpoint down: stop instead of burning the run


class Judgement(BaseModel):
    correct: bool
    reason: str


JUDGE_SYSTEM = (
    "You grade answers about a contract. Given a question, the reference answer and a "
    "candidate answer, decide if the candidate is correct: same facts as the reference "
    "(wording, language and extra correct details do not matter; a missing key fact or a "
    "wrong number does). Be strict on numbers, dates and names."
)


def judge(question: str, expected: str, candidate: str) -> Judgement:
    user = f"Question: {question}\nReference answer: {expected}\nCandidate answer: {candidate}"
    return chat_json(JUDGE_SYSTEM, user, Judgement)


def norm(v):
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    return v


def field_accuracy(pred: dict, gold: dict) -> tuple[float, dict]:
    detail = {k: norm(pred.get(k)) == norm(v) for k, v in gold.items()}
    return sum(detail.values()) / max(len(detail), 1), detail


def rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def eval_router(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for case in yaml.safe_load(path.read_text(encoding="utf-8")):
        try:
            got = plan_request(case["text"], strict=True).route
        except Exception as e:
            rows.append({"text": case["text"], "expected": case["route"], "error": str(e)[:300]})
            if len(rows) >= MAX_CONSECUTIVE_ERRORS and all("error" in r for r in rows[-MAX_CONSECUTIVE_ERRORS:]):
                print(f"Router eval stopped: {MAX_CONSECUTIVE_ERRORS} errors in a row", file=sys.stderr)
                break
            continue
        rows.append({"text": case["text"], "expected": case["route"], "got": got, "ok": got == case["route"]})
    return rows


def eval_question(g: dict, q: dict, retriever: Retriever, use_judge: bool) -> dict:
    question = q["question"]
    unanswerable = g["expected"] == NOT_IN_CONTRACT
    row = {"id": g["id"], "question": question, "tags": q.get("tags", []), "expected": g["expected"]}
    t0 = time.perf_counter()
    try:
        plan = plan_request(question, strict=True)
        retrieved = retriever.search(question, keywords=plan.keywords)
        a = answer(question, retrieved)
    except Exception as e:  # one failing call must not kill a 50-question run
        row.update(ok=False, error=str(e)[:300], latency_s=round(time.perf_counter() - t0, 2))
        return row
    row["latency_s"] = round(time.perf_counter() - t0, 2)
    row.update(
        route=plan.route,
        keywords=plan.keywords,
        answer=a.answer,
        found=a.found,
        citations=[c.model_dump() for c in a.citations],
        retrieved_pages=[c.page for c in retrieved],
    )
    if unanswerable:
        row["ok"] = not a.found
        return row
    page = g.get("page")
    row["retrieval_hit"] = page in row["retrieved_pages"] if page else None
    row["retrieval_hit_at_1"] = row["retrieved_pages"][:1] == [page] if page else None
    row["citation_page_ok"] = any(c.page == page for c in a.citations) if page else None
    if use_judge:
        try:
            j = judge(question, str(g["expected"]), a.answer)
            row["judge_correct"], row["judge_reason"] = j.correct, j.reason
        except Exception as e:
            row["judge_correct"], row["judge_reason"] = None, f"judge failed: {e}"[:300]
        row["ok"] = bool(row["judge_correct"])
    else:
        row["ok"] = bool(a.found and row["citation_page_ok"])
    return row


def eval_compliance(chunks, extraction, expected: set[str]) -> dict:
    t0 = time.perf_counter()
    try:
        rep = check(chunks, extraction)
    except Exception as e:
        return {"compliance_error": str(e)[:300]}
    raised = {f.rule_id for f in rep.flags}
    tp = raised & expected
    return {
        "compliance_latency_s": round(time.perf_counter() - t0, 2),
        "compliance_precision": round(len(tp) / len(raised), 3) if raised else (1.0 if not expected else 0.0),
        "compliance_recall": round(len(tp) / len(expected), 3) if expected else 1.0,
        "compliance_false_positives": sorted(raised - expected),
        "compliance_missed": sorted(expected - raised),
        "compliance_flags": [f.model_dump() for f in rep.flags],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--questions", default="evals/questions.yaml")
    ap.add_argument("--router", default="evals/router.yaml")
    ap.add_argument("--out", default="evals/results.json")
    ap.add_argument("--no-embeddings", action="store_true", help="BM25-only baseline")
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge on answer correctness")
    ap.add_argument("--skip-extraction", action="store_true")
    ap.add_argument("--skip-questions", action="store_true")
    ap.add_argument(
        "--fail-under", action="append", default=[], metavar="METRIC=VALUE",
        help="e.g. --fail-under answer_accuracy=0.9 (repeatable)",
    )
    args = ap.parse_args()

    gold = yaml.safe_load(Path(args.gold).read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in yaml.safe_load(Path(args.questions).read_text(encoding="utf-8"))}
    chunks = ingest(args.contract)
    embedder = None if args.no_embeddings else default_embedder()
    retriever = Retriever(chunks, embedder)
    print(f"{len(chunks)} chunks, retrieval={retriever.mode}, model={settings.llm_model}", file=sys.stderr)

    report: dict = {"contract": args.contract, "retrieval_mode": retriever.mode, "llm_model": settings.llm_model, "top_k": settings.top_k}

    extraction = None
    if not args.skip_extraction:
        t0 = time.perf_counter()
        try:
            extraction = extract(chunks)
        except Exception as e:  # recorded, not fatal: compliance will retry its own extraction
            report["extraction_error"] = str(e)[:300]
        else:
            report["extraction_latency_s"] = round(time.perf_counter() - t0, 2)
            acc, detail = field_accuracy(extraction.model_dump(), gold.get("fields", {}))
            report["extraction_field_accuracy"] = round(acc, 3)
            report["extraction_detail"] = detail

    if "compliance" in gold:
        report.update(eval_compliance(chunks, extraction, set(gold["compliance"] or [])))

    router_rows = [] if args.skip_questions else eval_router(Path(args.router))

    rows, aborted = [], None
    for i, g in enumerate([] if args.skip_questions else gold.get("questions", []), start=1):
        row = eval_question(g, questions[g["id"]], retriever, args.judge)
        rows.append(row)
        print(f"[{i}/{len(gold['questions'])}] {row['id']} {'ok' if row['ok'] else 'FAIL'}", file=sys.stderr)
        if len(rows) >= MAX_CONSECUTIVE_ERRORS and all("error" in r for r in rows[-MAX_CONSECUTIVE_ERRORS:]):
            aborted = f"{MAX_CONSECUTIVE_ERRORS} errors in a row at {row['id']}: {row['error']}"
            print(f"Aborting, partial results kept: {aborted}", file=sys.stderr)
            break

    answerable = [r for r in rows if r["expected"] != NOT_IN_CONTRACT and "error" not in r]
    unanswerable = [r for r in rows if r["expected"] == NOT_IN_CONTRACT and "error" not in r]
    citations = [c for r in rows for c in r.get("citations", [])]
    latencies = sorted(r["latency_s"] for r in rows if "error" not in r)
    route_checks = [r["ok"] for r in router_rows if "error" not in r] + [r["route"] == "ask" for r in rows if "route" in r]

    by_tag = defaultdict(list)
    for r in rows:
        for t in r["tags"]:
            by_tag[t].append(r["ok"])

    report.update(
        aborted=aborted,
        questions_total=len(rows),
        questions_answered=sum("error" not in r for r in rows),
        errors=sum("error" in r for r in rows),
        answer_accuracy=rate([r["judge_correct"] is True for r in answerable]) if args.judge else None,
        router_accuracy=rate(route_checks),
        retrieval_recall_at_k=rate([r["retrieval_hit"] for r in answerable if r.get("retrieval_hit") is not None]),
        retrieval_hit_at_1=rate([r["retrieval_hit_at_1"] for r in answerable if r.get("retrieval_hit_at_1") is not None]),
        citation_page_accuracy=rate([r["citation_page_ok"] for r in answerable if r.get("citation_page_ok") is not None]),
        quote_verified_rate=rate([c["verified"] for c in citations]),
        abstention_rate=rate([r["ok"] for r in unanswerable]),
        false_abstention_rate=rate([not r["found"] for r in answerable]),
        median_answer_latency_s=round(statistics.median(latencies), 2) if latencies else None,
        p95_answer_latency_s=latencies[int(0.95 * (len(latencies) - 1))] if latencies else None,
        by_tag={t: {"n": len(v), "pass": rate(v)} for t, v in sorted(by_tag.items())},
        router=router_rows,
        questions=rows,
    )
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    hidden = ("questions", "router", "extraction_detail", "compliance_flags")
    summary = {k: v for k, v in report.items() if k not in hidden}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failures = [f"{r['id']}: {r.get('answer') or r.get('error')}" for r in rows if not r["ok"]]
    if failures:
        print("\nFailures:\n  " + "\n  ".join(failures), file=sys.stderr)
    sys.exit(2 if aborted else check_thresholds(report, args.fail_under))


def check_thresholds(report: dict, specs: list[str]) -> int:
    failed = []
    for spec in specs:
        name, _, value = spec.partition("=")
        got = report.get(name)
        if got is None or got < float(value):
            failed.append(f"{name}={got} < {value}")
    if failed:
        print("\nThresholds not met: " + ", ".join(failed), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    main()
