"""
Phase 1 — Generation

Runs every experiment (document × LLM × chunking strategy) defined in
evaluation_configs.json and stores the raw results:

    query + LLM answer + retrieved chunks (full text + metadata)

NO metrics are computed here.  Keeping generation lean means the GPU only
does LLM inference; it doesn't also run a faithfulness judge at the same time.

Run analysis in a separate step once generation is complete:

    python evals/analyze.py

Usage
-----
    # Run all experiments in evaluation_configs.json
    python evals/generate.py

    # Run a single LLM (useful when GPU is limited)
    python evals/generate.py --provider ollama --model phi3:mini

    # Run specific chunking strategies only
    python evals/generate.py --strategies Recursive_512_Overlap_50,Recursive_1024_Overlap_100

    # Reduce concurrency (default 3; use 1 if the GPU is already at 100%)
    python evals/generate.py --workers 1

    # Force re-run even if output file already exists
    python evals/generate.py --overwrite

Output
------
    evals/results/raw/<doc>__<provider>__<model>__<strategy>.json

Each file contains a metadata block and a list of result records.  The
filename is stable (no timestamp) so re-running an experiment overwrites
it — use --overwrite to confirm.  Partial progress is saved to
evals/checkpoints/generate/ so a crash can be resumed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.rag_chain import RAGChain
from src.ingestion.providers.huggingface import HuggingFaceEmbeddingProvider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

EVAL_CONFIGS_PATH = Path("evals/evaluation_configs.json")
CHECKPOINT_DIR = Path("evals/checkpoints/generate")
RAW_RESULTS_DIR = Path("evals/results/raw")
CHECKPOINT_INTERVAL = 5          # save checkpoint every N completed queries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(s: str) -> str:
    """Make a string safe for use in a filename."""
    return s.replace(":", "_").replace("/", "_").replace(" ", "_")


def _experiment_key(doc: str, provider: str, model: str, strategy: str) -> str:
    return f"{_safe(doc)}__{_safe(provider)}__{_safe(model)}__{_safe(strategy)}"


def _raw_output_path(doc: str, provider: str, model: str, strategy: str) -> Path:
    return RAW_RESULTS_DIR / f"{_experiment_key(doc, provider, model, strategy)}.json"


def _checkpoint_path(doc: str, provider: str, model: str, strategy: str) -> Path:
    return CHECKPOINT_DIR / f"{_experiment_key(doc, provider, model, strategy)}.pkl"


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _save_checkpoint(results: List[Dict], completed_ids: Set[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"results": results, "completed_ids": list(completed_ids)}, f)
    logger.debug(f"Checkpoint saved — {len(completed_ids)} queries done")


def _load_checkpoint(path: Path):
    if not path.exists():
        return [], set()
    with open(path, "rb") as f:
        data = pickle.load(f)
    completed = set(data["completed_ids"])
    logger.info(f"Checkpoint loaded — {len(completed)} queries already completed")
    return data["results"], completed


# ---------------------------------------------------------------------------
# Per-query worker
# ---------------------------------------------------------------------------

def _run_single_query(
    query_item: Dict[str, Any],
    rag_chain: RAGChain,
    doc_name: str,
    provider: str,
    model_id: str,
    strategy_name: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_method: str,
) -> Dict[str, Any]:
    """
    Run one query through the RAG chain.  Returns a record dict ready to be
    written to the raw results file.  Never raises — errors are captured in
    the 'error' field so the experiment can continue.
    """
    query = query_item["query"]
    ground_truth = query_item.get("ground_truth", "")

    llm_answer = ""
    retrieved_chunks: List[Dict[str, Any]] = []
    llm_response_time_s: Optional[float] = None
    error: Optional[str] = None

    try:
        t0 = time.perf_counter()
        result = rag_chain.query_with_citations(query)
        llm_response_time_s = round(time.perf_counter() - t0, 3)

        llm_answer = result.get("answer", "")
        citations = result.get("citations", [])

        # Store full chunk text AND metadata so the analysis phase has
        # everything it needs for context precision / recall / faithfulness.
        retrieved_chunks = [
            {
                "rank": i + 1,
                "text": c.get("full_content", c.get("snippet", "")),
                "metadata": c.get("metadata", {}),
            }
            for i, c in enumerate(citations)
            if c.get("full_content") or c.get("snippet")
        ]

    except Exception as exc:
        error = str(exc)
        logger.error(f"  Query '{query_item['id']}' failed: {exc}")

    return {
        # — experiment context —
        "document": doc_name,
        "llm_provider": provider,
        "llm_model_id": model_id,
        "chunking_strategy": strategy_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunking_method": chunk_method,
        # — query —
        "query_id": query_item["id"],
        "category": query_item.get("category", ""),
        "difficulty": query_item.get("difficulty", ""),
        "expected_topics": query_item.get("expected_topics", []),
        "query": query,
        "ground_truth": ground_truth,
        # — raw outputs —
        "llm_answer": llm_answer,
        "retrieved_chunks": retrieved_chunks,
        "llm_response_time_s": llm_response_time_s,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Result file I/O
# ---------------------------------------------------------------------------

def _write_raw_results(
    results: List[Dict],
    out_path: Path,
    doc: str,
    provider: str,
    model_id: str,
    strategy: Dict,
    embedder_cfg: Dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_errors = sum(1 for r in results if r.get("error"))
    payload = {
        "schema_version": "2.0",
        "generated_at": datetime.now().isoformat(),
        "metadata": {
            "document": doc,
            "llm_provider": provider,
            "llm_model_id": model_id,
            "chunking_strategy": strategy["name"],
            "chunk_size": strategy["chunk_size"],
            "chunk_overlap": strategy["chunk_overlap"],
            "chunking_method": strategy["method"],
            "embedder_provider": embedder_cfg.get("provider"),
            "embedder_model": embedder_cfg.get("model"),
            "total_queries": len(results),
            "n_errors": n_errors,
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_generation(
    filter_provider: Optional[str] = None,
    filter_model: Optional[str] = None,
    filter_strategies: Optional[List[str]] = None,
    workers: int = 3,
    overwrite: bool = False,
) -> None:
    logger.info("=== Generation Phase: Starting ===")

    # Load evaluation config
    if not EVAL_CONFIGS_PATH.exists():
        logger.error(f"Config not found: {EVAL_CONFIGS_PATH}")
        sys.exit(1)
    with open(EVAL_CONFIGS_PATH) as f:
        cfg = json.load(f)

    doc_paths: Dict[str, str] = cfg["document_paths"]
    test_queries_path: str = cfg.get("test_queries_path", "evals/test_queries.json")
    llm_configs: List[Dict] = cfg["llm_configs"]
    embedder_cfg: Dict = cfg["embedder_config"]
    chunking_strategies: List[Dict] = cfg["chunking_strategies"]

    # Apply filters
    if filter_provider:
        llm_configs = [c for c in llm_configs if c["provider"] == filter_provider]
        if filter_model:
            llm_configs = [c for c in llm_configs if c["model_id"] == filter_model]
    if filter_strategies:
        chunking_strategies = [
            s for s in chunking_strategies if s["name"] in filter_strategies
        ]

    if not llm_configs:
        logger.error("No LLM configs match the given filters.")
        sys.exit(1)
    if not chunking_strategies:
        logger.error("No chunking strategies match the given filters.")
        sys.exit(1)

    # Load test queries
    with open(test_queries_path) as f:
        test_queries: List[Dict] = json.load(f)["queries"]
    logger.info(f"Loaded {len(test_queries)} test queries")

    # Validate embedder (needed only to confirm the cached vector stores exist;
    # the embedder itself is pre-baked into the pickled VectorStore objects)
    if embedder_cfg["provider"] != "huggingface":
        logger.error(f"Unsupported embedder provider: {embedder_cfg['provider']}")
        sys.exit(1)

    total_experiments = len(doc_paths) * len(llm_configs) * len(chunking_strategies)
    logger.info(
        f"Experiments to run: {len(doc_paths)} docs × {len(llm_configs)} LLMs × "
        f"{len(chunking_strategies)} strategies = {total_experiments}"
    )

    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Outer loops: document → LLM → chunking strategy                     #
    # ------------------------------------------------------------------ #
    for doc_name, _doc_path_str in doc_paths.items():
        logger.info(f"\n--- Document: {doc_name} ---")

        for llm_cfg in llm_configs:
            provider = llm_cfg["provider"]
            model_id = llm_cfg["model_id"]
            logger.info(f"  LLM: {provider}/{model_id}")

            for strategy in chunking_strategies:
                s_name = strategy["name"]
                s_size = strategy["chunk_size"]
                s_overlap = strategy["chunk_overlap"]
                s_method = strategy["method"]

                out_path = _raw_output_path(doc_name, provider, model_id, s_name)
                if out_path.exists() and not overwrite:
                    logger.info(
                        f"  [{s_name}] Already exists — skipping "
                        f"(use --overwrite to force re-run)"
                    )
                    continue

                logger.info(f"  [{s_name}] Starting…")

                # Load cached vector store (must be pre-built by build_vectorstores.py)
                cache_key = f"vector_store_{doc_name}_{s_name}.pkl"
                cache_path = Path("evals/cache") / cache_key
                if not cache_path.exists():
                    logger.error(
                        f"  Vector store cache missing: {cache_path}\n"
                        f"  Run: python evals/build_vectorstores.py"
                    )
                    continue

                with open(cache_path, "rb") as f:
                    vector_store = pickle.load(f)

                rag_chain = RAGChain(
                    vector_store=vector_store,
                    llm_provider_type=provider,
                    model_id=model_id,
                )

                # Resume from checkpoint if available
                ckpt_path = _checkpoint_path(doc_name, provider, model_id, s_name)
                results, completed_ids = _load_checkpoint(ckpt_path)
                remaining = [q for q in test_queries if q["id"] not in completed_ids]

                logger.info(
                    f"  [{s_name}] {len(remaining)} queries remaining "
                    f"({len(completed_ids)} already done)"
                )

                if not remaining:
                    logger.info(f"  [{s_name}] All queries done — writing output file.")
                    _write_raw_results(
                        results, out_path, doc_name, provider, model_id,
                        strategy, embedder_cfg,
                    )
                    ckpt_path.unlink(missing_ok=True)
                    continue

                # -------------------------------------------------------- #
                # Run queries in parallel                                    #
                # NOTE: Ollama serializes inference on one GPU anyway, so   #
                # workers > 1 mainly reduces Python overhead between calls. #
                # If the GPU is at 100%, drop --workers to 1.               #
                # -------------------------------------------------------- #
                lock = threading.Lock()
                counter = [len(completed_ids)]

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            _run_single_query,
                            q, rag_chain,
                            doc_name, provider, model_id,
                            s_name, s_size, s_overlap, s_method,
                        ): q
                        for q in remaining
                    }

                    desc = f"{provider}/{model_id} × {s_name}"
                    with tqdm(total=len(remaining), desc=desc, unit="q") as pbar:
                        for future in as_completed(futures):
                            rec = future.result()
                            with lock:
                                results.append(rec)
                                completed_ids.add(rec["query_id"])
                                counter[0] += 1
                                if counter[0] % CHECKPOINT_INTERVAL == 0:
                                    _save_checkpoint(results, completed_ids, ckpt_path)
                            pbar.update(1)

                # Final checkpoint + write output
                _save_checkpoint(results, completed_ids, ckpt_path)
                _write_raw_results(
                    results, out_path, doc_name, provider, model_id,
                    strategy, embedder_cfg,
                )
                ckpt_path.unlink(missing_ok=True)
                n_errors = sum(1 for r in results if r.get("error"))
                logger.info(
                    f"  [{s_name}] Done — {len(results)} queries, "
                    f"{n_errors} errors → {out_path}"
                )

    logger.info("\n=== Generation Phase: Complete ===")
    logger.info(f"Raw results saved to: {RAW_RESULTS_DIR}/")
    logger.info("Next step: python evals/analyze.py")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1: Generate raw RAG results (no metrics).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--provider",
        metavar="PROVIDER",
        help="Filter to one LLM provider (e.g. ollama, gemini).",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL_ID",
        help="Filter to one model ID. Requires --provider.",
    )
    parser.add_argument(
        "--strategies",
        metavar="NAME,NAME",
        help="Comma-separated list of chunking strategy names to run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Parallel query workers per experiment (default 3). "
            "Use 1 if the GPU is at 100%% and you want zero contention."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run and overwrite experiments that already have output files.",
    )
    args = parser.parse_args()

    strategy_filter = (
        [s.strip() for s in args.strategies.split(",") if s.strip()]
        if args.strategies
        else None
    )

    run_generation(
        filter_provider=args.provider,
        filter_model=args.model,
        filter_strategies=strategy_filter,
        workers=args.workers,
        overwrite=args.overwrite,
    )
