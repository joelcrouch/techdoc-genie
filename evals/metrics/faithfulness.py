"""
Faithfulness metric for RAG evaluation.

Measures the fraction of claims in an LLM-generated answer that are
supported by the retrieved context chunks.

    score = supported_claims / total_claims

This is an LLM-as-judge approach that runs entirely locally via Ollama,
so no external API keys are required.  Works with small models such as
phi3:mini and qwen2.5:1.5b, using a structured prompt that produces
simple YES/NO output per claim.

Typical usage
-------------
    from evals.metrics.faithfulness import FaithfulnessScorer

    scorer = FaithfulnessScorer(judge_model="phi3:mini")
    result = scorer.score(
        answer="Indexes speed up queries by...",
        context_chunks=["PostgreSQL supports B-tree indexes...", ...],
    )
    print(result.score)           # 0.0 – 1.0, or None if undecidable
    print(result.verdicts)        # per-claim breakdown

Standalone test
---------------
    python evals/metrics/faithfulness.py
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Allow running this file directly from the repo root or evals/ directory.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.providers.ollama_provider import OllamaProvider
from src.utils.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClaimVerdict:
    """Judgment for a single claim extracted from the answer."""
    claim: str
    supported: Optional[bool]   # True=YES, False=NO, None=could not decide
    raw_response: str = ""      # Judge's raw token for this claim


@dataclass
class FaithfulnessResult:
    """Aggregate result of a faithfulness evaluation."""
    score: Optional[float]      # None when no claims could be decided
    num_claims: int
    num_supported: int
    num_unsupported: int
    num_undecided: int
    verdicts: List[ClaimVerdict] = field(default_factory=list)
    error: Optional[str] = None

    def as_dict(self) -> dict:
        """Serialise to a plain dict for JSON storage."""
        return {
            "faithfulness_score": self.score,
            "num_claims": self.num_claims,
            "num_supported": self.num_supported,
            "num_unsupported": self.num_unsupported,
            "num_undecided": self.num_undecided,
            "faithfulness_verdicts": [
                {
                    "claim": v.claim,
                    "supported": v.supported,
                    "raw_response": v.raw_response,
                }
                for v in self.verdicts
            ],
            "faithfulness_error": self.error,
        }


# ---------------------------------------------------------------------------
# Claim / sentence splitting
# ---------------------------------------------------------------------------

# Sentence boundary: end-of-sentence punctuation followed by whitespace,
# but not preceded by a single uppercase letter + period (abbreviations).
_SENT_BOUNDARY = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+')


def _split_into_claims(text: str) -> List[str]:
    """
    Split an answer into individual claims (sentences).

    Handles:
    - Numbered / bulleted lists  (strips the marker)
    - Markdown bold/italic       (strips the markers)
    - Very short fragments       (filtered out, < 10 chars)
    """
    # Strip markdown emphasis markers
    text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)
    # Normalise newlines so bullet lists split cleanly
    text = re.sub(r'\n+', ' ', text.strip())

    raw_sentences = _SENT_BOUNDARY.split(text)

    claims = []
    for s in raw_sentences:
        # Remove list markers: "1. ", "- ", "• ", etc.
        s = re.sub(r'^\s*[-*•\d]+[.)]\s*', '', s).strip()
        if len(s) >= 10:
            claims.append(s)

    return claims


# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

# Batch prompt: all claims in one call.  Faster, preferred.
_BATCH_PROMPT = """\
You are a strict fact-checker evaluating an AI assistant's answer.

Your task: for each numbered claim below, decide whether it is directly \
supported by the CONTEXT passages provided.

Rules:
- YES  → the context explicitly states or clearly implies the claim.
- NO   → the context does not mention it, contradicts it, or you are unsure.
- Reply with ONLY one word per line (YES or NO), in the same order as the claims.
- Do NOT add explanations, numbers, or punctuation.

CONTEXT:
{context}

CLAIMS:
{numbered_claims}

Answer (one YES or NO per line, nothing else):"""

# Single-claim prompt: fallback when batch response is malformed.
_SINGLE_PROMPT = """\
Does the following CONTEXT support the CLAIM?

CONTEXT:
{context}

CLAIM: {claim}

Reply with only YES or NO."""


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class FaithfulnessScorer:
    """
    LLM-as-judge faithfulness scorer backed by a local Ollama model.

    Parameters
    ----------
    judge_model:
        Ollama model name used as judge.  Defaults to the application's
        configured default (``ollama_model_id`` in settings).
    max_context_chars:
        Truncate concatenated context to this many characters before
        sending to the judge.  Keeps the prompt within the model's
        context window.  3 000 chars ≈ 750 tokens, safe for phi3:mini.
    base_url:
        Ollama server URL.  Defaults to settings value.
    fallback_threshold:
        Fraction of undecided verdicts in the batch response that triggers
        the slower per-claim fallback.  Default 0.5.
    judge_timeout:
        Per-call timeout in seconds for the judge LLM.  Shorter than the
        main RAG timeout so a slow judge fails fast rather than stalling
        the whole eval run.  Default 60 s (GPU) is plenty; on CPU you may
        need to increase this or use ``--skip-faithfulness``.
    """

    def __init__(
        self,
        judge_model: Optional[str] = None,
        max_context_chars: int = 3000,
        base_url: Optional[str] = None,
        fallback_threshold: float = 0.5,
        judge_timeout: int = 60,
    ):
        settings = get_settings()
        resolved_model = judge_model or settings.ollama_model_id
        resolved_url = base_url or settings.ollama_base_url

        self.max_context_chars = max_context_chars
        self.fallback_threshold = fallback_threshold
        self.judge_timeout = judge_timeout

        self.provider = OllamaProvider(
            model_name=resolved_model,
            base_url=resolved_url,
        )
        logger.info(
            f"FaithfulnessScorer ready — judge: {resolved_model} "
            f"@ {resolved_url} (timeout={judge_timeout}s)"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        answer: str,
        context_chunks: List[str],
    ) -> FaithfulnessResult:
        """
        Score the faithfulness of *answer* against *context_chunks*.

        Parameters
        ----------
        answer:
            The LLM-generated answer to evaluate.
        context_chunks:
            Raw text strings from retrieved document chunks.  Typically
            ``[c['full_content'] for c in result['citations']]`` from the
            RAGChain output.

        Returns
        -------
        FaithfulnessResult
            Contains the aggregate score (0.0–1.0) and per-claim verdicts.
            ``score`` is ``None`` when no claims could be evaluated.
        """
        if not answer or not answer.strip():
            return FaithfulnessResult(
                score=None, num_claims=0,
                num_supported=0, num_unsupported=0, num_undecided=0,
                error="Empty answer — nothing to evaluate.",
            )

        if not context_chunks:
            return FaithfulnessResult(
                score=None, num_claims=0,
                num_supported=0, num_unsupported=0, num_undecided=0,
                error="No context chunks provided.",
            )

        claims = _split_into_claims(answer)
        if not claims:
            return FaithfulnessResult(
                score=None, num_claims=0,
                num_supported=0, num_unsupported=0, num_undecided=0,
                error="Could not extract any claims from the answer.",
            )

        context_text = self._prepare_context(context_chunks)
        verdicts = self._judge_claims(claims, context_text)

        supported  = sum(1 for v in verdicts if v.supported is True)
        unsupported = sum(1 for v in verdicts if v.supported is False)
        undecided  = sum(1 for v in verdicts if v.supported is None)

        decidable = supported + unsupported
        score = (supported / decidable) if decidable > 0 else None

        return FaithfulnessResult(
            score=score,
            num_claims=len(claims),
            num_supported=supported,
            num_unsupported=unsupported,
            num_undecided=undecided,
            verdicts=verdicts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_context(self, chunks: List[str]) -> str:
        joined = "\n\n---\n\n".join(chunks)
        if len(joined) > self.max_context_chars:
            joined = joined[: self.max_context_chars] + "\n[context truncated]"
        return joined

    def _judge_claims(
        self, claims: List[str], context_text: str
    ) -> List[ClaimVerdict]:
        """Try batch first; fall back to per-claim if the response is garbled."""
        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
        prompt = _BATCH_PROMPT.format(
            context=context_text,
            numbered_claims=numbered,
        )

        try:
            raw = self.provider.generate_text(
                prompt, temperature=0.0, timeout=self.judge_timeout
            )
            verdicts = self._parse_batch_response(claims, raw)

            undecided_frac = sum(
                1 for v in verdicts if v.supported is None
            ) / len(verdicts)

            if undecided_frac > self.fallback_threshold:
                logger.warning(
                    f"Batch parse: {undecided_frac:.0%} undecided "
                    f"(>{self.fallback_threshold:.0%} threshold) — "
                    "falling back to per-claim calls."
                )
                return self._judge_per_claim(claims, context_text)

            return verdicts

        except Exception as exc:
            logger.error(
                f"Batch judge call failed ({exc}); falling back to per-claim."
            )
            return self._judge_per_claim(claims, context_text)

    def _parse_batch_response(
        self, claims: List[str], raw: str
    ) -> List[ClaimVerdict]:
        """
        Extract ordered YES/NO tokens from the judge's free-text response.
        Ignores any surrounding prose or numbering.
        """
        tokens = re.findall(r'\b(YES|NO)\b', raw.upper())
        verdicts = []
        for i, claim in enumerate(claims):
            if i < len(tokens):
                verdicts.append(ClaimVerdict(
                    claim=claim,
                    supported=(tokens[i] == "YES"),
                    raw_response=tokens[i],
                ))
            else:
                verdicts.append(ClaimVerdict(
                    claim=claim,
                    supported=None,
                    raw_response="[missing — batch response too short]",
                ))
        return verdicts

    def _judge_per_claim(
        self, claims: List[str], context_text: str
    ) -> List[ClaimVerdict]:
        """Evaluate one claim at a time (slower but more reliable)."""
        verdicts = []
        for claim in claims:
            prompt = _SINGLE_PROMPT.format(
                context=context_text, claim=claim
            )
            try:
                raw = self.provider.generate_text(
                    prompt, temperature=0.0, timeout=self.judge_timeout
                )
                tokens = re.findall(r'\b(YES|NO)\b', raw.upper())
                supported = (tokens[0] == "YES") if tokens else None
                verdicts.append(ClaimVerdict(
                    claim=claim,
                    supported=supported,
                    raw_response=raw.strip()[:80],
                ))
            except Exception as exc:
                logger.error(
                    f"Per-claim judge failed for '{claim[:60]}': {exc}"
                )
                verdicts.append(ClaimVerdict(
                    claim=claim,
                    supported=None,
                    raw_response=f"ERROR: {exc}",
                ))
        return verdicts


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("FaithfulnessScorer — standalone smoke test")
    print("=" * 60)

    CONTEXT = [
        (
            "PostgreSQL supports several index types. The default index type "
            "is B-tree, which can handle equality and range queries. Other "
            "types include Hash (equality only), GiST, GIN, SP-GiST, and BRIN."
        ),
        (
            "To create an index: CREATE INDEX index_name ON table_name (column). "
            "A partial index adds a WHERE clause to index only a subset of rows."
        ),
    ]

    ANSWER_GOOD = (
        "PostgreSQL supports multiple index types including B-tree, Hash, GiST, "
        "and GIN. B-tree is the default and handles range queries. "
        "You can create an index with CREATE INDEX."
    )

    ANSWER_HALLUCINATED = (
        "PostgreSQL supports B-tree indexes. "
        "Indexes are stored in a separate Redis cache for speed. "
        "You must restart the server after creating any index."
    )

    scorer = FaithfulnessScorer()

    for label, answer in [("GOOD", ANSWER_GOOD), ("HALLUCINATED", ANSWER_HALLUCINATED)]:
        print(f"\n--- {label} ANSWER ---")
        print(f"Answer: {answer}\n")
        result = scorer.score(answer=answer, context_chunks=CONTEXT)
        print(f"Faithfulness score : {result.score:.2f}" if result.score is not None else "Score: N/A")
        print(f"Claims: {result.num_claims} total | "
              f"{result.num_supported} supported | "
              f"{result.num_unsupported} unsupported | "
              f"{result.num_undecided} undecided")
        print("Verdicts:")
        for v in result.verdicts:
            symbol = "✓" if v.supported else ("✗" if v.supported is False else "?")
            print(f"  [{symbol}] {v.claim[:80]}")
