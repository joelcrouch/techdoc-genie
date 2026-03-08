"""
Phase 2 — Analysis

Reads raw result files produced by generate.py and computes evaluation
metrics.  Because this step is decoupled from generation you can:

  - Run it after the GPU has cooled down
  - Re-run it with different parameters without re-querying the LLM
  - Add new metrics to old result files

Currently implemented metrics
------------------------------
  semantic_similarity  — cosine similarity between the LLM answer and the
                         ground truth string (SentenceTransformer, fast)
  faithfulness         — fraction of answer claims grounded in retrieved
                         context, judged by a local Ollama model (slow;
                         use --skip-faithfulness to skip)

Planned (not yet implemented here)
------------------------------------
  context_precision    — fraction of retrieved chunks that were relevant
  context_recall       — fraction of ground-truth info present in context
  answer_relevance     — does the answer actually address the question?

Usage
-----
    # Analyze all raw files
    python evals/analyze.py

    # Analyze a specific file
    python evals/analyze.py --input evals/results/raw/postgresql__ollama__phi3_mini__Recursive_512_Overlap_50.json

    # Skip the slow faithfulness judge (good first pass)
    python evals/analyze.py --skip-faithfulness

    # Throttle faithfulness calls to avoid redlining the GPU
    python evals/analyze.py --throttle 1.5

    # Use a different judge model
    python evals/analyze.py --judge-model qwen2.5:1.5b

    # Re-analyze even if an analyzed file already exists
    python evals/analyze.py --overwrite

Output
------
    evals/results/analyzed/<same_stem>_analyzed.json

Each file keeps the original result records and adds a "metrics" dict to
every record, plus a "summary" block at the top level.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.metrics.faithfulness import FaithfulnessScorer
from evals.metrics.context_metrics import ContextMetricsScorer, ContextMetricsResult
from evals.metrics.correctness import CorrectnessScorer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

RAW_DIR = Path("evals/results/raw")
ANALYZED_DIR = Path("evals/results/analyzed")
SIMILARITY_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _compute_metrics_for_record(
    rec: Dict[str, Any],
    sim_model: SentenceTransformer,
    context_scorer: ContextMetricsScorer,
    faithfulness_scorer: Optional[FaithfulnessScorer],
    correctness_scorer: Optional[CorrectnessScorer],
    throttle_s: float,
) -> Dict[str, Any]:
    """
    Compute all requested metrics for a single result record.
    Returns a metrics dict (never raises — errors are captured inside).
    """
    metrics: Dict[str, Any] = {}

    query: str = rec.get("query", "")
    answer: str = rec.get("llm_answer", "")
    ground_truth: str = rec.get("ground_truth", "")
    chunks: List[Dict] = rec.get("retrieved_chunks", [])
    chunk_texts: List[str] = [c["text"] for c in chunks if c.get("text")]
    is_error = bool(rec.get("error")) or answer.startswith("ERROR")

    # ---- semantic similarity ------------------------------------------------
    if answer and ground_truth and not is_error:
        try:
            embs = sim_model.encode([answer, ground_truth])
            metrics["semantic_similarity"] = float(
                cosine_similarity([embs[0]], [embs[1]])[0][0]
            )
        except Exception as exc:
            logger.warning(f"  sem_sim failed for {rec.get('query_id')}: {exc}")
            metrics["semantic_similarity"] = None
    else:
        metrics["semantic_similarity"] = None

    # ---- context precision & recall (no LLM needed) ------------------------
    if query and chunk_texts and ground_truth and not is_error:
        ctx_result: ContextMetricsResult = context_scorer.score(
            query=query,
            ground_truth=ground_truth,
            chunk_texts=chunk_texts,
        )
        metrics.update(ctx_result.as_dict())
    else:
        metrics["context_precision"] = None
        metrics["context_recall"] = None
        metrics["n_retrieved"] = len(chunk_texts)
        metrics["n_relevant"] = None
        metrics["chunk_relevance_scores"] = []
        metrics["best_chunk_index"] = None
        metrics["context_metrics_error"] = "skipped"

    # ---- faithfulness (LLM-as-judge, optional) ------------------------------
    if faithfulness_scorer is not None and chunk_texts and answer and not is_error:
        try:
            fr = faithfulness_scorer.score(
                answer=answer,
                context_chunks=chunk_texts,
            )
            metrics.update(fr.as_dict())
        except Exception as exc:
            logger.warning(f"  faithfulness failed for {rec.get('query_id')}: {exc}")
            metrics["faithfulness_score"] = None
            metrics["faithfulness_error"] = str(exc)

        if throttle_s > 0:
            time.sleep(throttle_s)
    else:
        metrics["faithfulness_score"] = None

    # ---- correctness (LLM-as-judge, optional) -------------------------------
    if correctness_scorer is not None and answer and ground_truth and not is_error:
        try:
            cr = correctness_scorer.score(
                llm_answer=answer,
                ground_truth=ground_truth,
            )
            metrics.update(cr.as_dict())
        except Exception as exc:
            logger.warning(f"  correctness failed for {rec.get('query_id')}: {exc}")
            metrics["correctness_score"] = None
            metrics["correctness_error"] = str(exc)

        if throttle_s > 0:
            time.sleep(throttle_s)
    else:
        metrics["correctness_score"] = None

    return metrics


def _summarize(annotated: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate summary statistics across all records."""
    non_error = [r for r in annotated if not r.get("error") and not str(r.get("llm_answer", "")).startswith("ERROR")]

    def _avg(values):
        valid = [v for v in values if v is not None]
        return float(np.mean(valid)) if valid else None

    sem_scores = [r["metrics"].get("semantic_similarity") for r in non_error]
    faith_scores = [r["metrics"].get("faithfulness_score") for r in non_error]
    precision_scores = [r["metrics"].get("context_precision") for r in non_error]
    recall_scores = [r["metrics"].get("context_recall") for r in non_error]
    ap_scores = [r["metrics"].get("average_precision") for r in non_error]
    correctness_scores = [r["metrics"].get("correctness_score") for r in non_error]
    latencies = [r.get("llm_response_time_s") for r in annotated if r.get("llm_response_time_s")]

    # Slice by difficulty
    by_difficulty: Dict[str, Any] = {}
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in non_error if r.get("difficulty") == diff]
        by_difficulty[diff] = {
            "n": len(subset),
            "avg_semantic_similarity": _avg([r["metrics"].get("semantic_similarity") for r in subset]),
            "avg_context_precision": _avg([r["metrics"].get("context_precision") for r in subset]),
            "avg_context_recall": _avg([r["metrics"].get("context_recall") for r in subset]),
            "avg_average_precision": _avg([r["metrics"].get("average_precision") for r in subset]),
            "avg_faithfulness": _avg([r["metrics"].get("faithfulness_score") for r in subset]),
            "avg_correctness": _avg([r["metrics"].get("correctness_score") for r in subset]),
        }

    # Slice by category
    categories = sorted({r.get("category", "") for r in annotated if r.get("category")})
    by_category: Dict[str, Any] = {}
    for cat in categories:
        subset = [r for r in non_error if r.get("category") == cat]
        by_category[cat] = {
            "n": len(subset),
            "avg_semantic_similarity": _avg([r["metrics"].get("semantic_similarity") for r in subset]),
            "avg_context_precision": _avg([r["metrics"].get("context_precision") for r in subset]),
            "avg_context_recall": _avg([r["metrics"].get("context_recall") for r in subset]),
            "avg_average_precision": _avg([r["metrics"].get("average_precision") for r in subset]),
            "avg_faithfulness": _avg([r["metrics"].get("faithfulness_score") for r in subset]),
            "avg_correctness": _avg([r["metrics"].get("correctness_score") for r in subset]),
        }

    return {
        "n_total": len(annotated),
        "n_errors": sum(1 for r in annotated if r.get("error")),
        "n_analyzed": len(non_error),
        "avg_semantic_similarity": _avg(sem_scores),
        "avg_context_precision": _avg(precision_scores),
        "avg_context_recall": _avg(recall_scores),
        "avg_average_precision": _avg(ap_scores),
        "avg_faithfulness": _avg(faith_scores),
        "avg_correctness": _avg(correctness_scores),
        "avg_latency_s": _avg(latencies),
        "answer_rate_pct": round(len(non_error) / len(annotated) * 100, 1) if annotated else 0.0,
        "by_difficulty": by_difficulty,
        "by_category": by_category,
    }


def analyze_file(
    raw_path: Path,
    sim_model: SentenceTransformer,
    context_scorer: ContextMetricsScorer,
    faithfulness_scorer: Optional[FaithfulnessScorer],
    correctness_scorer: Optional[CorrectnessScorer] = None,
    throttle_s: float = 0.0,
    overwrite: bool = False,
) -> Optional[Path]:
    """
    Analyze a single raw results file.  Returns the path to the analyzed
    output file, or None if skipped.
    """
    out_path = ANALYZED_DIR / raw_path.name.replace(".json", "_analyzed.json")
    if out_path.exists() and not overwrite:
        logger.info(f"Already analyzed — skipping {raw_path.name} (use --overwrite)")
        return None

    with open(raw_path) as f:
        data = json.load(f)

    results: List[Dict] = data["results"]
    meta: Dict = data.get("metadata", {})

    logger.info(
        f"Analyzing {raw_path.name}  "
        f"({meta.get('llm_provider')}/{meta.get('llm_model_id')} × "
        f"{meta.get('chunking_strategy')}, {len(results)} queries)"
    )

    annotated = []
    for rec in tqdm(results, desc=f"  {raw_path.stem[:60]}", unit="q", leave=False):
        rec = dict(rec)  # don't mutate original
        rec["metrics"] = _compute_metrics_for_record(
            rec, sim_model, context_scorer, faithfulness_scorer, correctness_scorer, throttle_s
        )
        annotated.append(rec)

    summary = _summarize(annotated)

    # Print a quick summary to stdout
    _print_summary(meta, summary)

    ANALYZED_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "2.0",
        "source_file": str(raw_path),
        "analyzed_at": datetime.now().isoformat(),
        "metadata": meta,
        "summary": summary,
        "results": annotated,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"  Saved → {out_path}")
    return out_path


def _print_summary(meta: Dict, summary: Dict) -> None:
    """Print a formatted summary table to stdout."""
    label = (
        f"{meta.get('llm_provider')}/{meta.get('llm_model_id')} "
        f"× {meta.get('chunking_strategy')}"
    )
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    sem   = summary.get("avg_semantic_similarity")
    prec  = summary.get("avg_context_precision")
    rec   = summary.get("avg_context_recall")
    faith = summary.get("avg_faithfulness")
    lat   = summary.get("avg_latency_s")

    def _fmt(v): return f"{v:.4f}" if v is not None else "N/A"

    overview = [
        ["Queries analyzed", summary["n_analyzed"]],
        ["Errors",           summary["n_errors"]],
        ["Answer rate",      f"{summary['answer_rate_pct']}%"],
        ["Avg sem similarity",   _fmt(sem)],
        ["Avg context precision", _fmt(prec)],
        ["Avg context recall",    _fmt(rec)],
        ["Avg faithfulness",      _fmt(faith)],
        ["Avg latency",      f"{lat:.1f}s" if lat is not None else "N/A"],
    ]
    print(tabulate(overview, tablefmt="simple"))

    # Difficulty breakdown
    diff_rows = []
    for diff, stats in summary.get("by_difficulty", {}).items():
        if stats["n"] == 0:
            continue
        diff_rows.append([
            diff, stats["n"],
            _fmt(stats.get("avg_semantic_similarity")),
            _fmt(stats.get("avg_context_precision")),
            _fmt(stats.get("avg_context_recall")),
            _fmt(stats.get("avg_faithfulness")),
        ])
    if diff_rows:
        print("\nBy difficulty:")
        print(tabulate(
            diff_rows,
            headers=["Difficulty", "N", "Sem.Sim", "Ctx.Prec", "Ctx.Rec", "Faithful"],
            tablefmt="simple",
        ))

    # Category breakdown
    cat_rows = []
    for cat, stats in summary.get("by_category", {}).items():
        if stats["n"] == 0:
            continue
        cat_rows.append([
            cat, stats["n"],
            _fmt(stats.get("avg_semantic_similarity")),
            _fmt(stats.get("avg_context_precision")),
            _fmt(stats.get("avg_context_recall")),
            _fmt(stats.get("avg_faithfulness")),
        ])
    if cat_rows:
        print("\nBy category:")
        print(tabulate(
            cat_rows,
            headers=["Category", "N", "Sem.Sim", "Ctx.Prec", "Ctx.Rec", "Faithful"],
            tablefmt="simple",
        ))


# ---------------------------------------------------------------------------
# Cross-experiment summary
# ---------------------------------------------------------------------------

def _print_cross_experiment_summary(analyzed_paths: List[Path]) -> None:
    """After all files are analyzed, print a side-by-side comparison table."""
    rows = []
    for path in analyzed_paths:
        try:
            with open(path) as f:
                data = json.load(f)
            m = data.get("metadata", {})
            s = data.get("summary", {})
            def _f(v): return f"{v:.4f}" if v is not None else "N/A"
            s_lat = s.get("avg_latency_s")
            rows.append([
                m.get("llm_provider", "") + "/" + m.get("llm_model_id", ""),
                m.get("chunking_strategy", ""),
                _f(s.get("avg_semantic_similarity")),
                _f(s.get("avg_context_precision")),
                _f(s.get("avg_context_recall")),
                _f(s.get("avg_faithfulness")),
                f"{s.get('answer_rate_pct', 0):.1f}%",
                f"{s_lat:.1f}s" if s_lat else "N/A",
            ])
        except Exception:
            pass

    if not rows:
        return

    print(f"\n{'='*90}")
    print("  CROSS-EXPERIMENT SUMMARY")
    print(f"{'='*90}")
    print(tabulate(
        rows,
        headers=["LLM", "Strategy", "Sem.Sim", "Ctx.Prec", "Ctx.Rec", "Faithful", "Ans.Rate", "Latency"],
        tablefmt="grid",
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(
    input_paths: List[Path],
    skip_faithfulness: bool = False,
    skip_correctness: bool = False,
    judge_model: Optional[str] = None,
    judge_timeout: int = 60,
    throttle_s: float = 0.0,
    overwrite: bool = False,
) -> None:
    if not input_paths:
        logger.warning("No raw result files found to analyze.")
        return

    logger.info(f"=== Analysis Phase: {len(input_paths)} file(s) to process ===")

    # Load similarity model once — shared by sem_sim AND context metrics
    logger.info(f"Loading SentenceTransformer: {SIMILARITY_MODEL}")
    sim_model = SentenceTransformer(SIMILARITY_MODEL)
    context_scorer = ContextMetricsScorer(sim_model)

    # Optionally init correctness scorer
    correctness_scorer: Optional[CorrectnessScorer] = None
    if not skip_correctness:
        try:
            kwargs_c: Dict[str, Any] = {"judge_timeout": judge_timeout}
            if judge_model:
                kwargs_c["judge_model"] = judge_model
            correctness_scorer = CorrectnessScorer(**kwargs_c)
            logger.info(
                f"CorrectnessScorer ready "
                f"(model={judge_model or 'default'}, timeout={judge_timeout}s)"
            )
        except Exception as exc:
            logger.warning(
                f"Could not init CorrectnessScorer ({exc}). "
                f"Correctness will be skipped. "
                f"Is Ollama running? Try: ollama serve"
            )
    else:
        logger.info("Correctness scoring disabled (--skip-correctness).")

    # Optionally init faithfulness scorer
    faithfulness_scorer: Optional[FaithfulnessScorer] = None
    if not skip_faithfulness:
        try:
            kwargs: Dict[str, Any] = {"judge_timeout": judge_timeout}
            if judge_model:
                kwargs["judge_model"] = judge_model
            faithfulness_scorer = FaithfulnessScorer(**kwargs)
            logger.info(
                f"FaithfulnessScorer ready "
                f"(model={judge_model or 'default'}, timeout={judge_timeout}s)"
            )
        except Exception as exc:
            logger.warning(
                f"Could not init FaithfulnessScorer ({exc}). "
                f"Faithfulness will be skipped. "
                f"Is Ollama running? Try: ollama serve"
            )
    else:
        logger.info("Faithfulness scoring disabled (--skip-faithfulness).")

    analyzed_paths: List[Path] = []
    for raw_path in input_paths:
        out = analyze_file(
            raw_path=raw_path,
            sim_model=sim_model,
            context_scorer=context_scorer,
            faithfulness_scorer=faithfulness_scorer,
            correctness_scorer=correctness_scorer,
            throttle_s=throttle_s,
            overwrite=overwrite,
        )
        if out:
            analyzed_paths.append(out)

    if len(analyzed_paths) > 1:
        _print_cross_experiment_summary(analyzed_paths)

    logger.info(f"\n=== Analysis Phase: Complete — {len(analyzed_paths)} file(s) written ===")
    logger.info(f"Analyzed results in: {ANALYZED_DIR}/")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: Compute evaluation metrics on stored raw results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        metavar="PATH",
        help=(
            "Path to a single raw result file, or a directory of them.  "
            f"Defaults to {RAW_DIR}/"
        ),
    )
    parser.add_argument(
        "--skip-faithfulness",
        action="store_true",
        help=(
            "Skip the LLM-as-judge faithfulness step.  "
            "Semantic similarity is still computed (fast, CPU-only)."
        ),
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help=(
            "Skip the LLM-as-judge correctness step.  "
            "Use this with --skip-faithfulness to run fast, CPU-only analysis."
        ),
    )
    parser.add_argument(
        "--judge-model",
        metavar="MODEL_ID",
        help="Ollama model to use as the faithfulness judge (default: from settings).",
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=60,
        metavar="SECS",
        help="Per-call timeout for the faithfulness judge (default 60s).",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.0,
        metavar="SECS",
        help=(
            "Seconds to wait between faithfulness judge calls.  "
            "Use 1-2 if the GPU is close to 100%% during analysis."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-analyze and overwrite existing analyzed files.",
    )
    args = parser.parse_args()

    # Resolve input paths
    if args.input:
        p = Path(args.input)
        if p.is_file():
            raw_files = [p]
        elif p.is_dir():
            raw_files = sorted(p.glob("*.json"))
        else:
            logger.error(f"Input path not found: {p}")
            sys.exit(1)
    else:
        if not RAW_DIR.exists():
            logger.error(
                f"Default raw dir not found: {RAW_DIR}\n"
                f"Run generate.py first: python evals/generate.py"
            )
            sys.exit(1)
        raw_files = sorted(RAW_DIR.glob("*.json"))

    if not raw_files:
        logger.error("No .json files found to analyze.")
        sys.exit(1)

    run_analysis(
        input_paths=raw_files,
        skip_faithfulness=args.skip_faithfulness,
        skip_correctness=args.skip_correctness,
        judge_model=args.judge_model,
        judge_timeout=args.judge_timeout,
        throttle_s=args.throttle,
        overwrite=args.overwrite,
    )
