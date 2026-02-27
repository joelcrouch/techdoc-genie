"""
Context Precision and Context Recall metrics for RAG evaluation.

Both metrics are computed using embedding similarity only — no LLM calls
required, so they run fast even without GPU.

Context Precision
-----------------
"Of the chunks we retrieved, how many were actually relevant to the query?"

    precision = |relevant retrieved| / |retrieved|

A chunk is considered relevant if its embedding cosine-similarity to the
query embedding exceeds a threshold (default 0.35).  This threshold is
deliberately low — we want to catch chunks that are topically related, not
just exact matches.

High precision + low semantic_similarity → model is hallucinating despite
good retrieval.

Low precision → retriever is pulling in off-topic chunks (chunking too
coarse, or embedding model mismatch).

Context Recall
--------------
"Did the retrieved context actually contain the information needed to
answer the query?"

    recall = max similarity between any retrieved chunk and the ground truth

We encode the ground truth and take the maximum cosine similarity across all
retrieved chunks.  A score near 1.0 means at least one chunk closely matches
what the correct answer says.  A score near 0 means the answer wasn't in the
retrieved context at all.

Low recall → retriever missed the relevant document section entirely.  Fix:
larger k, different chunking strategy, or hybrid search.

Typical usage
-------------
    from sentence_transformers import SentenceTransformer
    from evals.metrics.context_metrics import ContextMetricsScorer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    scorer = ContextMetricsScorer(model)

    result = scorer.score(
        query="How do I create an index in PostgreSQL?",
        ground_truth="Use CREATE INDEX ...",
        chunk_texts=["PostgreSQL supports B-tree indexes ...", ...],
    )
    print(result.context_precision)   # 0.0 – 1.0
    print(result.context_recall)      # 0.0 – 1.0

Standalone test
---------------
    python evals/metrics/context_metrics.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class ContextMetricsResult:
    """Scores for a single query."""

    # non-default fields first
    context_precision: Optional[float]
    """Fraction of retrieved chunks judged relevant to the query (0–1)."""

    n_retrieved: int
    """Total number of retrieved chunks."""

    n_relevant: int
    """Number of chunks judged relevant."""

    context_recall: Optional[float]
    """Max cosine similarity between any retrieved chunk and the ground truth."""

    # default fields after
    chunk_relevance_scores: List[float] = field(default_factory=list)
    """Per-chunk cosine similarity to the query embedding."""

    best_chunk_index: Optional[int] = None
    """Index (0-based) of the chunk most similar to the ground truth."""

    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "context_precision": self.context_precision,
            "n_retrieved": self.n_retrieved,
            "n_relevant": self.n_relevant,
            "chunk_relevance_scores": self.chunk_relevance_scores,
            "context_recall": self.context_recall,
            "best_chunk_index": self.best_chunk_index,
            "context_metrics_error": self.error,
        }


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class ContextMetricsScorer:
    """
    Compute context precision and recall using a SentenceTransformer model.

    Parameters
    ----------
    model:
        A loaded SentenceTransformer instance.  Reuse the one already loaded
        in analyze.py to avoid loading it twice.
    precision_threshold:
        Minimum cosine similarity between a chunk and the query for the chunk
        to be counted as "relevant".  Default 0.35.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        precision_threshold: float = 0.35,
    ):
        self.model = model
        self.precision_threshold = precision_threshold

    def score(
        self,
        query: str,
        ground_truth: str,
        chunk_texts: List[str],
    ) -> ContextMetricsResult:
        """
        Compute context precision and recall for one query.

        Parameters
        ----------
        query:
            The original user question.
        ground_truth:
            The reference answer used for recall scoring.
        chunk_texts:
            The text content of every retrieved chunk (in retrieval rank order).
        """
        if not chunk_texts:
            return ContextMetricsResult(
                context_precision=None,
                n_retrieved=0,
                n_relevant=0,
                context_recall=None,
                error="no chunks provided",
            )

        try:
            # Encode everything in one batch for efficiency
            texts_to_encode = [query, ground_truth] + chunk_texts
            embeddings = self.model.encode(texts_to_encode, show_progress_bar=False)

            query_emb = embeddings[0:1]           # shape (1, D)
            gt_emb = embeddings[1:2]              # shape (1, D)
            chunk_embs = embeddings[2:]           # shape (N, D)

            # ---- precision: query vs each chunk ----------------------------
            q_chunk_sims = cosine_similarity(query_emb, chunk_embs)[0]  # (N,)
            chunk_relevance_scores = [float(s) for s in q_chunk_sims]
            n_relevant = int(np.sum(q_chunk_sims >= self.precision_threshold))
            context_precision = n_relevant / len(chunk_texts)

            # ---- recall: ground truth vs each chunk ------------------------
            gt_chunk_sims = cosine_similarity(gt_emb, chunk_embs)[0]   # (N,)
            context_recall = float(np.max(gt_chunk_sims))
            best_chunk_index = int(np.argmax(gt_chunk_sims))

            return ContextMetricsResult(
                context_precision=context_precision,
                n_retrieved=len(chunk_texts),
                n_relevant=n_relevant,
                chunk_relevance_scores=chunk_relevance_scores,
                context_recall=context_recall,
                best_chunk_index=best_chunk_index,
            )

        except Exception as exc:
            logger.error(f"ContextMetricsScorer failed: {exc}")
            return ContextMetricsResult(
                context_precision=None,
                n_retrieved=len(chunk_texts),
                n_relevant=0,
                context_recall=None,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading model…")
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    scorer = ContextMetricsScorer(_model)

    query = "How do I create an index in PostgreSQL?"
    ground_truth = (
        "Use CREATE INDEX statement. Supports B-tree (default), Hash, GiST, GIN."
    )
    chunks = [
        # relevant
        "PostgreSQL supports several index types: B-tree, Hash, GiST, GIN, and BRIN. "
        "Use CREATE INDEX idx_name ON table(column) to create a B-tree index.",
        # relevant
        "Indexes speed up SELECT queries but slow down writes. Use EXPLAIN ANALYZE "
        "to verify an index is being used.",
        # not relevant
        "The pg_hba.conf file controls client authentication. Each record specifies "
        "a connection type, database, user, address, and authentication method.",
        # not relevant
        "WAL (Write-Ahead Logging) ensures data integrity by writing changes to a "
        "log before applying them to the data files.",
        # somewhat relevant
        "Partial indexes index only a subset of rows: "
        "CREATE INDEX idx ON orders(status) WHERE status = 'pending'.",
    ]

    result = scorer.score(query=query, ground_truth=ground_truth, chunk_texts=chunks)

    print(f"\nContext Precision : {result.context_precision:.3f}  "
          f"({result.n_relevant}/{result.n_retrieved} chunks relevant)")
    print(f"Context Recall    : {result.context_recall:.3f}  "
          f"(best match: chunk {result.best_chunk_index})")
    print("\nPer-chunk relevance scores (vs query):")
    for i, score in enumerate(result.chunk_relevance_scores):
        marker = "✓" if score >= scorer.precision_threshold else "✗"
        print(f"  [{marker}] chunk {i}: {score:.3f}")
