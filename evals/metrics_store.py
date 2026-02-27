"""
SQLite metrics store for RAG evaluation results.

Ingests analyzed result files (produced by analyze.py) into a queryable
SQLite database.  Once data is in the store you can:

  - Track how metrics change across runs over time
  - Query by experiment, category, difficulty, or failure mode
  - Feed the Streamlit dashboard without re-reading JSON files

Schema
------
  runs           — one row per experiment (doc × LLM × strategy × run)
  query_results  — one row per (run × query), with all metric values

Idempotency
-----------
Ingesting the same file twice is safe — the run is identified by
(source_file, analyzed_at) and skipped if it already exists.

Usage
-----
    # Ingest all analyzed files
    python evals/metrics_store.py --ingest

    # Ingest a specific file
    python evals/metrics_store.py --ingest --file evals/results/analyzed/foo_analyzed.json

    # Print a summary of what's in the store
    python evals/metrics_store.py --summary

    # Use as a library
    from evals.metrics_store import MetricsStore
    store = MetricsStore()
    store.ingest_dir(Path("evals/results/analyzed"))
    df = store.runs_dataframe()
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.logger import setup_logger

# Re-use the same failure mode classifier so the store stays consistent
# with what diagnose.py reports.
from evals.diagnose import classify

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH    = Path("evals/metrics.db")
ANALYZED_DIR       = Path("evals/results/analyzed")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    run_at              TEXT NOT NULL,          -- ISO timestamp from analyzed_at
    source_file         TEXT NOT NULL,
    document            TEXT,
    llm_provider        TEXT,
    llm_model_id        TEXT,
    chunking_strategy   TEXT,
    chunk_size          INTEGER,
    chunk_overlap       INTEGER,
    chunking_method     TEXT,
    embedder_provider   TEXT,
    embedder_model      TEXT,
    n_queries           INTEGER,
    n_errors            INTEGER,
    avg_semantic_similarity REAL,
    avg_context_precision   REAL,
    avg_context_recall      REAL,
    avg_faithfulness        REAL,
    avg_latency_s           REAL,
    answer_rate_pct         REAL
);

CREATE TABLE IF NOT EXISTS query_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    query_id            TEXT NOT NULL,
    query               TEXT,
    category            TEXT,
    difficulty          TEXT,
    ground_truth        TEXT,
    llm_answer          TEXT,
    semantic_similarity REAL,
    context_precision   REAL,
    context_recall      REAL,
    faithfulness_score  REAL,
    latency_s           REAL,
    failure_mode        TEXT,
    error               TEXT
);

CREATE INDEX IF NOT EXISTS idx_qr_run     ON query_results(run_id);
CREATE INDEX IF NOT EXISTS idx_qr_query   ON query_results(query_id);
CREATE INDEX IF NOT EXISTS idx_qr_cat     ON query_results(category);
CREATE INDEX IF NOT EXISTS idx_qr_diff    ON query_results(difficulty);
CREATE INDEX IF NOT EXISTS idx_qr_mode    ON query_results(failure_mode);
CREATE INDEX IF NOT EXISTS idx_runs_llm   ON runs(llm_provider, llm_model_id);
CREATE INDEX IF NOT EXISTS idx_runs_strat ON runs(chunking_strategy);
"""


# ---------------------------------------------------------------------------
# MetricsStore
# ---------------------------------------------------------------------------

class MetricsStore:
    """
    Thin wrapper around a SQLite database for RAG evaluation metrics.

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  Created on first use.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_DDL)
        logger.debug(f"MetricsStore ready: {db_path}")

    # ------------------------------------------------------------------ #
    # Connection helper                                                    #
    # ------------------------------------------------------------------ #

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Ingestion                                                            #
    # ------------------------------------------------------------------ #

    def _make_run_id(self, source_file: str, analyzed_at: str) -> str:
        """Stable, human-readable run ID."""
        stem = Path(source_file).stem  # e.g. postgresql__ollama__phi3_mini__..._analyzed
        # Truncate timestamp to minute precision for readability
        ts = analyzed_at[:16].replace(":", "").replace("-", "").replace("T", "_")
        return f"{stem}__{ts}"

    def ingest_file(self, path: Path) -> bool:
        """
        Ingest one analyzed JSON file.  Returns True if ingested, False if
        already present (idempotent).
        """
        with open(path) as f:
            data = json.load(f)

        source_file = str(path)
        analyzed_at = data.get("analyzed_at", datetime.now().isoformat())
        run_id = self._make_run_id(source_file, analyzed_at)

        with self._conn() as conn:
            existing = conn.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if existing:
                logger.debug(f"Already ingested: {path.name}")
                return False

            meta = data.get("metadata", {})
            summary = data.get("summary", {})

            conn.execute(
                """INSERT INTO runs VALUES (
                    :run_id, :run_at, :source_file,
                    :document, :llm_provider, :llm_model_id,
                    :chunking_strategy, :chunk_size, :chunk_overlap, :chunking_method,
                    :embedder_provider, :embedder_model,
                    :n_queries, :n_errors,
                    :avg_semantic_similarity, :avg_context_precision,
                    :avg_context_recall, :avg_faithfulness,
                    :avg_latency_s, :answer_rate_pct
                )""",
                {
                    "run_id":             run_id,
                    "run_at":             analyzed_at,
                    "source_file":        source_file,
                    "document":           meta.get("document"),
                    "llm_provider":       meta.get("llm_provider"),
                    "llm_model_id":       meta.get("llm_model_id"),
                    "chunking_strategy":  meta.get("chunking_strategy"),
                    "chunk_size":         meta.get("chunk_size"),
                    "chunk_overlap":      meta.get("chunk_overlap"),
                    "chunking_method":    meta.get("chunking_method"),
                    "embedder_provider":  meta.get("embedder_provider"),
                    "embedder_model":     meta.get("embedder_model"),
                    "n_queries":          summary.get("n_total"),
                    "n_errors":           summary.get("n_errors"),
                    "avg_semantic_similarity": summary.get("avg_semantic_similarity"),
                    "avg_context_precision":   summary.get("avg_context_precision"),
                    "avg_context_recall":      summary.get("avg_context_recall"),
                    "avg_faithfulness":        summary.get("avg_faithfulness"),
                    "avg_latency_s":           summary.get("avg_latency_s"),
                    "answer_rate_pct":         summary.get("answer_rate_pct"),
                },
            )

            rows = []
            for rec in data.get("results", []):
                m = rec.get("metrics", {})
                sem   = m.get("semantic_similarity")
                recall= m.get("context_recall")
                rows.append({
                    "run_id":              run_id,
                    "query_id":            rec.get("query_id"),
                    "query":               rec.get("query"),
                    "category":            rec.get("category"),
                    "difficulty":          rec.get("difficulty"),
                    "ground_truth":        rec.get("ground_truth"),
                    "llm_answer":          rec.get("llm_answer"),
                    "semantic_similarity": sem,
                    "context_precision":   m.get("context_precision"),
                    "context_recall":      recall,
                    "faithfulness_score":  m.get("faithfulness_score"),
                    "latency_s":           rec.get("llm_response_time_s"),
                    "failure_mode":        classify(sem, recall),
                    "error":               rec.get("error"),
                })

            conn.executemany(
                """INSERT INTO query_results (
                    run_id, query_id, query, category, difficulty,
                    ground_truth, llm_answer,
                    semantic_similarity, context_precision, context_recall,
                    faithfulness_score, latency_s, failure_mode, error
                ) VALUES (
                    :run_id, :query_id, :query, :category, :difficulty,
                    :ground_truth, :llm_answer,
                    :semantic_similarity, :context_precision, :context_recall,
                    :faithfulness_score, :latency_s, :failure_mode, :error
                )""",
                rows,
            )

        logger.info(f"Ingested {len(rows)} records from {path.name} → run_id={run_id}")
        return True

    def ingest_dir(self, directory: Path = ANALYZED_DIR) -> int:
        """Ingest all *_analyzed.json files in a directory. Returns count of new files ingested."""
        files = sorted(directory.glob("*_analyzed.json"))
        if not files:
            logger.warning(f"No analyzed files found in {directory}")
            return 0
        ingested = sum(1 for f in files if self.ingest_file(f))
        logger.info(f"Ingested {ingested}/{len(files)} files (rest already present)")
        return ingested

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def runs(self) -> List[sqlite3.Row]:
        """All runs, newest first."""
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM runs ORDER BY run_at DESC"
            ).fetchall()

    def query_results(
        self,
        run_id: Optional[str] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        failure_mode: Optional[str] = None,
    ) -> List[sqlite3.Row]:
        """Filtered query results."""
        clauses = []
        params: List[Any] = []
        if run_id:
            clauses.append("run_id = ?");   params.append(run_id)
        if category:
            clauses.append("category = ?"); params.append(category)
        if difficulty:
            clauses.append("difficulty = ?"); params.append(difficulty)
        if failure_mode:
            clauses.append("failure_mode = ?"); params.append(failure_mode)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._conn() as conn:
            return conn.execute(
                f"SELECT * FROM query_results {where} ORDER BY run_id, query_id",
                params,
            ).fetchall()

    def metric_history(self, metric: str = "avg_semantic_similarity") -> List[sqlite3.Row]:
        """
        Time-series of a summary metric across all runs, ordered by run date.
        Useful for dashboard line charts.
        """
        allowed = {
            "avg_semantic_similarity", "avg_context_precision",
            "avg_context_recall", "avg_faithfulness", "avg_latency_s",
        }
        if metric not in allowed:
            raise ValueError(f"metric must be one of {allowed}")
        with self._conn() as conn:
            return conn.execute(
                f"""SELECT run_at, llm_provider, llm_model_id, chunking_strategy,
                           {metric}
                    FROM runs
                    ORDER BY run_at ASC""",
            ).fetchall()

    def worst_queries(self, n: int = 10) -> List[sqlite3.Row]:
        """Queries with the lowest average semantic similarity across all runs."""
        with self._conn() as conn:
            return conn.execute(
                """SELECT query_id, category, difficulty, query,
                          AVG(semantic_similarity) AS avg_sem_sim,
                          AVG(context_recall)      AS avg_ctx_rec,
                          COUNT(*)                 AS n_runs
                   FROM query_results
                   WHERE semantic_similarity IS NOT NULL
                   GROUP BY query_id
                   ORDER BY avg_sem_sim ASC
                   LIMIT ?""",
                (n,),
            ).fetchall()

    def failure_mode_summary(self) -> List[sqlite3.Row]:
        """Count of each failure mode across all ingested data."""
        with self._conn() as conn:
            return conn.execute(
                """SELECT failure_mode, COUNT(*) AS n
                   FROM query_results
                   GROUP BY failure_mode
                   ORDER BY n DESC""",
            ).fetchall()

    # ------------------------------------------------------------------ #
    # Summary print                                                        #
    # ------------------------------------------------------------------ #

    def print_summary(self) -> None:
        from tabulate import tabulate

        run_rows = self.runs()
        print(f"\n{'='*70}")
        print(f"  METRICS STORE: {self.db_path}  ({len(run_rows)} runs)")
        print(f"{'='*70}")

        if not run_rows:
            print("  Empty — run: python evals/metrics_store.py --ingest")
            return

        def _f(v): return f"{v:.4f}" if v is not None else "N/A"

        table = []
        for r in run_rows:
            table.append([
                r["run_at"][:16],
                f"{r['llm_provider']}/{r['llm_model_id']}",
                r["chunking_strategy"],
                _f(r["avg_semantic_similarity"]),
                _f(r["avg_context_precision"]),
                _f(r["avg_context_recall"]),
                _f(r["avg_faithfulness"]),
                r["n_queries"],
            ])
        print(tabulate(
            table,
            headers=["Run at", "LLM", "Strategy", "Sem.Sim", "Ctx.Prec", "Ctx.Rec", "Faithful", "N"],
            tablefmt="simple",
        ))

        print("\nFailure mode distribution across all ingested data:")
        fm_rows = self.failure_mode_summary()
        total = sum(r["n"] for r in fm_rows)
        fm_table = [[r["failure_mode"], r["n"], f"{r['n']/total*100:.1f}%"] for r in fm_rows]
        print(tabulate(fm_table, headers=["Mode", "N", "%"], tablefmt="simple"))

        print("\nWorst 5 queries (avg sem_sim across all runs):")
        wq = self.worst_queries(5)
        wq_table = [
            [r["query_id"], r["category"], r["difficulty"],
             f"{r['avg_sem_sim']:.3f}", f"{r['avg_ctx_rec']:.3f}",
             r["query"][:55]]
            for r in wq
        ]
        print(tabulate(
            wq_table,
            headers=["ID", "Category", "Diff", "Avg Sem.Sim", "Avg Ctx.Rec", "Query"],
            tablefmt="simple",
        ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SQLite metrics store for RAG evaluation results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest analyzed files into the store.",
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        help="Ingest a single analyzed file (use with --ingest).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a summary of what's currently in the store.",
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=str(DEFAULT_DB_PATH),
        help=f"Path to the SQLite database (default: {DEFAULT_DB_PATH}).",
    )
    args = parser.parse_args()

    store = MetricsStore(db_path=Path(args.db))

    if args.ingest:
        if args.file:
            store.ingest_file(Path(args.file))
        else:
            store.ingest_dir(ANALYZED_DIR)

    if args.summary or not (args.ingest or args.summary):
        store.print_summary()
