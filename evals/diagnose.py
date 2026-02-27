"""
Diagnostic CLI for RAG evaluation results.

Reads analyzed result files and answers the questions that aggregate averages
can't: which queries are failing, why, and whether the problem is retrieval
or generation.

Failure mode classification
----------------------------
Each query is classified into one of five modes based on its metrics:

  OK                 sem_sim >= 0.65 — answer is good
  GENERATION_WEAK    good retrieval, answer is mediocre (sem_sim 0.45–0.65)
  GENERATION_COLLAPSE good retrieval, model produced a bad answer (sem_sim < 0.45,
                      ctx_recall >= 0.45)
  RETRIEVAL_MISS     the relevant content was not retrieved (ctx_recall < 0.45)
  ERROR              the query raised an exception

These map directly to fixes:
  RETRIEVAL_MISS      → increase k, add hybrid search, re-embed with better model
  GENERATION_COLLAPSE → model context window overflow, or model too small
  GENERATION_WEAK     → prompt tuning, larger model, re-ranking
  OK                  → nothing to do

Usage
-----
    # Full report across all analyzed files
    python evals/diagnose.py

    # Show worst N queries (default 10)
    python evals/diagnose.py --worst 15

    # Deep-dive on one query across all experiments
    python evals/diagnose.py --query q016

    # Filter to one category
    python evals/diagnose.py --category performance

    # Filter to one experiment
    python evals/diagnose.py --strategy Recursive_256_Overlap_30

    # Read from a different directory
    python evals/diagnose.py --input evals/results/analyzed/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

ANALYZED_DIR = Path("evals/results/analyzed")

# ---------------------------------------------------------------------------
# Failure mode classification
# ---------------------------------------------------------------------------

# Thresholds — deliberately conservative so borderline cases go to WEAK not OK
SEM_SIM_OK       = 0.65
SEM_SIM_WEAK     = 0.45
CTX_RECALL_FLOOR = 0.45    # below this = retrieval miss regardless of sem_sim


def classify(sem_sim: Optional[float], ctx_recall: Optional[float]) -> str:
    if sem_sim is None:
        return "ERROR"
    if ctx_recall is not None and ctx_recall < CTX_RECALL_FLOOR:
        return "RETRIEVAL_MISS"
    if sem_sim >= SEM_SIM_OK:
        return "OK"
    if sem_sim >= SEM_SIM_WEAK:
        return "GENERATION_WEAK"
    return "GENERATION_COLLAPSE"


FAILURE_MODE_ORDER = ["OK", "GENERATION_WEAK", "GENERATION_COLLAPSE", "RETRIEVAL_MISS", "ERROR"]

FAILURE_MODE_ADVICE = {
    "RETRIEVAL_MISS":     "Increase k, add hybrid/BM25 search, or use a better embedding model.",
    "GENERATION_COLLAPSE":"Model context window overflow or model too small for the chunk size.",
    "GENERATION_WEAK":    "Try prompt tuning, re-ranking retrieved chunks, or a larger model.",
    "OK":                 "No action needed.",
    "ERROR":              "Check logs — LLM call failed.",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(analyzed_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all records from every analyzed JSON file into a flat list.
    Each record gets experiment metadata fields injected for easy grouping.
    """
    files = sorted(analyzed_dir.glob("*_analyzed.json"))
    if not files:
        logger.error(f"No analyzed files found in {analyzed_dir}")
        sys.exit(1)

    records = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        exp_label = f"{meta.get('llm_provider')}/{meta.get('llm_model_id')} × {meta.get('chunking_strategy')}"
        for rec in data.get("results", []):
            m = rec.get("metrics", {})
            sem   = m.get("semantic_similarity")
            prec  = m.get("context_precision")
            recall= m.get("context_recall")
            faith = m.get("faithfulness_score")
            records.append({
                # experiment
                "exp_label":  exp_label,
                "llm":        f"{meta.get('llm_provider')}/{meta.get('llm_model_id')}",
                "strategy":   meta.get("chunking_strategy", ""),
                "chunk_size": meta.get("chunk_size"),
                # query
                "query_id":   rec.get("query_id", ""),
                "query":      rec.get("query", ""),
                "category":   rec.get("category", ""),
                "difficulty": rec.get("difficulty", ""),
                "ground_truth": rec.get("ground_truth", ""),
                "llm_answer": rec.get("llm_answer", ""),
                "error":      rec.get("error"),
                # metrics
                "sem_sim":    sem,
                "ctx_prec":   prec,
                "ctx_rec":    recall,
                "faithfulness": faith,
                "latency_s":  rec.get("llm_response_time_s"),
                # derived
                "failure_mode": classify(sem, recall),
            })
    return records


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], decimals: int = 3) -> str:
    return f"{v:.{decimals}f}" if v is not None else "N/A"


def _avg(values) -> Optional[float]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _mode_bar(mode_counts: Dict[str, int], total: int) -> str:
    """Simple ASCII bar showing failure mode distribution."""
    parts = []
    for mode in FAILURE_MODE_ORDER:
        n = mode_counts.get(mode, 0)
        if n:
            pct = n / total * 100
            parts.append(f"{mode}={n}({pct:.0f}%)")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def report_overview(records: List[Dict]) -> None:
    """High-level failure mode distribution across all experiments."""
    mode_counts: Dict[str, int] = defaultdict(int)
    for r in records:
        mode_counts[r["failure_mode"]] += 1
    total = len(records)

    print(f"\n{'='*70}")
    print(f"  OVERVIEW  ({total} query×experiment records across {len(set(r['exp_label'] for r in records))} experiments)")
    print(f"{'='*70}")

    rows = []
    for mode in FAILURE_MODE_ORDER:
        n = mode_counts.get(mode, 0)
        if n == 0:
            continue
        rows.append([
            mode,
            n,
            f"{n/total*100:.1f}%",
            FAILURE_MODE_ADVICE[mode],
        ])
    print(tabulate(rows, headers=["Failure Mode", "N", "%", "What to do"], tablefmt="simple"))


def report_by_category(records: List[Dict]) -> None:
    """Per-category breakdown with failure mode distribution."""
    cats: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        cats[r["category"]].append(r)

    print(f"\n{'='*70}")
    print("  BY CATEGORY")
    print(f"{'='*70}")

    rows = []
    for cat in sorted(cats):
        subset = cats[cat]
        mode_counts: Dict[str, int] = defaultdict(int)
        for r in subset:
            mode_counts[r["failure_mode"]] += 1
        rows.append([
            cat,
            len(subset),
            _fmt(_avg(r["sem_sim"] for r in subset)),
            _fmt(_avg(r["ctx_rec"] for r in subset)),
            _mode_bar(mode_counts, len(subset)),
        ])
    print(tabulate(
        rows,
        headers=["Category", "N", "Avg Sem.Sim", "Avg Ctx.Rec", "Failure modes"],
        tablefmt="simple",
    ))


def report_by_difficulty(records: List[Dict]) -> None:
    """Per-difficulty breakdown."""
    print(f"\n{'='*70}")
    print("  BY DIFFICULTY")
    print(f"{'='*70}")

    rows = []
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in records if r["difficulty"] == diff]
        if not subset:
            continue
        mode_counts: Dict[str, int] = defaultdict(int)
        for r in subset:
            mode_counts[r["failure_mode"]] += 1
        rows.append([
            diff,
            len(subset),
            _fmt(_avg(r["sem_sim"] for r in subset)),
            _fmt(_avg(r["ctx_rec"] for r in subset)),
            _mode_bar(mode_counts, len(subset)),
        ])
    print(tabulate(
        rows,
        headers=["Difficulty", "N", "Avg Sem.Sim", "Avg Ctx.Rec", "Failure modes"],
        tablefmt="simple",
    ))


def report_worst_queries(records: List[Dict], n: int = 10) -> None:
    """
    Queries that score worst on average semantic similarity across all
    experiments.  Consistently bad queries point to fundamental retrieval
    or documentation gaps.
    """
    by_query: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_query[r["query_id"]].append(r)

    query_avgs: List[Tuple[str, float, str, str, str]] = []
    for qid, recs in by_query.items():
        avg_sem = _avg(r["sem_sim"] for r in recs)
        avg_rec = _avg(r["ctx_rec"] for r in recs)
        mode_counts: Dict[str, int] = defaultdict(int)
        for r in recs:
            mode_counts[r["failure_mode"]] += 1
        dominant_mode = max(mode_counts, key=mode_counts.get)
        query_avgs.append((
            qid,
            avg_sem if avg_sem is not None else 1.0,
            avg_rec,
            recs[0]["difficulty"],
            recs[0]["category"],
            recs[0]["query"][:55],
            dominant_mode,
        ))

    query_avgs.sort(key=lambda x: x[1])

    print(f"\n{'='*70}")
    print(f"  WORST {n} QUERIES  (averaged across all experiments)")
    print(f"{'='*70}")

    rows = []
    for qid, avg_sem, avg_rec, diff, cat, q_short, mode in query_avgs[:n]:
        rows.append([
            qid,
            diff,
            cat,
            _fmt(avg_sem),
            _fmt(avg_rec),
            mode,
            q_short,
        ])
    print(tabulate(
        rows,
        headers=["ID", "Diff", "Category", "Avg Sem.Sim", "Avg Ctx.Rec", "Dominant Mode", "Query"],
        tablefmt="simple",
    ))


def report_consistent_failures(records: List[Dict], threshold: float = 0.5) -> None:
    """
    Queries that fail in EVERY experiment (sem_sim below threshold everywhere).
    These are the hardest problems — not fixable by switching models.
    """
    by_query: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_query[r["query_id"]].append(r)

    n_experiments = len(set(r["exp_label"] for r in records))

    consistent_failures = []
    for qid, recs in by_query.items():
        scored = [r for r in recs if r["sem_sim"] is not None]
        if not scored:
            continue
        n_failing = sum(1 for r in scored if r["sem_sim"] < threshold)
        if n_failing == len(scored):  # fails in every experiment
            avg_sem = _avg(r["sem_sim"] for r in scored)
            avg_rec = _avg(r["ctx_rec"] for r in scored)
            consistent_failures.append((qid, avg_sem, avg_rec, recs[0]["difficulty"],
                                        recs[0]["category"], recs[0]["query"][:55]))

    consistent_failures.sort(key=lambda x: x[1] if x[1] is not None else 1.0)

    print(f"\n{'='*70}")
    print(f"  CONSISTENT FAILURES  (sem_sim < {threshold} in all {n_experiments} experiments)")
    print(f"{'='*70}")

    if not consistent_failures:
        print(f"  None — no query fails in every experiment at threshold {threshold}")
        return

    rows = [[qid, diff, cat, _fmt(avg_sem), _fmt(avg_rec), q]
            for qid, avg_sem, avg_rec, diff, cat, q in consistent_failures]
    print(tabulate(
        rows,
        headers=["ID", "Diff", "Category", "Avg Sem.Sim", "Avg Ctx.Rec", "Query"],
        tablefmt="simple",
    ))
    print(f"\n  → These {len(consistent_failures)} queries need retrieval or documentation fixes,")
    print(  "    not just model/prompt tuning.")


def report_query_deep_dive(records: List[Dict], query_id: str) -> None:
    """Show one query's results across every experiment side by side."""
    matching = [r for r in records if r["query_id"] == query_id]
    if not matching:
        print(f"Query '{query_id}' not found.")
        return

    print(f"\n{'='*70}")
    print(f"  DEEP DIVE: {query_id}")
    print(f"{'='*70}")
    print(f"  Query    : {matching[0]['query']}")
    print(f"  Category : {matching[0]['category']}  Difficulty: {matching[0]['difficulty']}")
    print(f"  Ground truth: {matching[0]['ground_truth'][:120]}")
    print()

    rows = []
    for r in sorted(matching, key=lambda x: x["sem_sim"] or 0, reverse=True):
        rows.append([
            r["llm"],
            r["strategy"],
            _fmt(r["sem_sim"]),
            _fmt(r["ctx_prec"]),
            _fmt(r["ctx_rec"]),
            r["failure_mode"],
            (r["llm_answer"] or "")[:80],
        ])
    print(tabulate(
        rows,
        headers=["LLM", "Strategy", "Sem.Sim", "Ctx.Prec", "Ctx.Rec", "Mode", "Answer"],
        tablefmt="simple",
    ))


def report_experiment_comparison(records: List[Dict]) -> None:
    """Cross-experiment summary sorted by avg semantic similarity."""
    by_exp: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_exp[r["exp_label"]].append(r)

    rows = []
    for exp, recs in by_exp.items():
        mode_counts: Dict[str, int] = defaultdict(int)
        for r in recs:
            mode_counts[r["failure_mode"]] += 1
        avg_sem = _avg(r["sem_sim"] for r in recs)
        rows.append([
            exp,
            _fmt(avg_sem),
            _fmt(_avg(r["ctx_prec"] for r in recs)),
            _fmt(_avg(r["ctx_rec"] for r in recs)),
            _mode_bar(mode_counts, len(recs)),
        ])
    rows.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*70}")
    print("  EXPERIMENT COMPARISON  (best to worst)")
    print(f"{'='*70}")
    print(tabulate(
        rows,
        headers=["Experiment", "Avg Sem.Sim", "Avg Ctx.Prec", "Avg Ctx.Rec", "Failure modes"],
        tablefmt="simple",
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_diagnose(
    analyzed_dir: Path,
    worst_n: int = 10,
    filter_category: Optional[str] = None,
    filter_strategy: Optional[str] = None,
    deep_dive_query: Optional[str] = None,
) -> None:
    records = load_records(analyzed_dir)

    # Apply filters
    if filter_category:
        records = [r for r in records if r["category"].lower() == filter_category.lower()]
        if not records:
            print(f"No records found for category '{filter_category}'")
            return

    if filter_strategy:
        records = [r for r in records if filter_strategy.lower() in r["strategy"].lower()]
        if not records:
            print(f"No records found matching strategy '{filter_strategy}'")
            return

    if deep_dive_query:
        all_records = load_records(analyzed_dir)   # unfiltered for deep dive
        report_query_deep_dive(all_records, deep_dive_query)
        return

    report_overview(records)
    report_by_difficulty(records)
    report_by_category(records)
    report_experiment_comparison(records)
    report_worst_queries(records, n=worst_n)
    report_consistent_failures(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose RAG evaluation results — find failure modes, worst queries, and actionable gaps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        metavar="DIR",
        default=str(ANALYZED_DIR),
        help=f"Directory of analyzed JSON files (default: {ANALYZED_DIR})",
    )
    parser.add_argument(
        "--worst",
        type=int,
        default=10,
        metavar="N",
        help="Number of worst queries to show (default 10).",
    )
    parser.add_argument(
        "--query",
        metavar="ID",
        help="Deep-dive on a single query ID (e.g. q016) across all experiments.",
    )
    parser.add_argument(
        "--category",
        metavar="NAME",
        help="Filter all reports to one category (e.g. performance, DDL, SQL).",
    )
    parser.add_argument(
        "--strategy",
        metavar="NAME",
        help="Filter all reports to experiments matching this strategy name.",
    )
    args = parser.parse_args()

    run_diagnose(
        analyzed_dir=Path(args.input),
        worst_n=args.worst,
        filter_category=args.category,
        filter_strategy=args.strategy,
        deep_dive_query=args.query,
    )
