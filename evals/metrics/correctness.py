"""
Correctness metric for RAG evaluation.

Measures whether an LLM-generated answer is factually correct and complete
when compared to a ground-truth reference answer.

Two sub-scores are computed using LLM-as-judge (same Ollama-backed approach
as FaithfulnessScorer — no external API keys required):

Factual Accuracy
----------------
"Does the answer avoid contradicting the ground truth?"
Claims extracted from the LLM answer are checked against the ground truth.
A claim is "correct" if the ground truth supports or is consistent with it.

    factual_accuracy = correct_claims / total_answer_claims

Completeness
------------
"Does the answer cover the key points in the ground truth?"
Key points extracted from the ground truth are checked against the answer.
A point is "covered" if the answer addresses it.

    completeness = covered_points / total_ground_truth_points

Final Score
-----------
    correctness = 0.6 * factual_accuracy + 0.4 * completeness

Factual accuracy is weighted higher because a confidently wrong answer is
worse than an incomplete answer for technical documentation use cases.

Unlike semantic similarity (which rewards word overlap) this score can catch
cases where the LLM states the correct topic but gets the details backwards.

Typical usage
-------------
    from evals.metrics.correctness import CorrectnessScorer

    scorer = CorrectnessScorer(judge_model="phi3:mini")
    result = scorer.score(
        llm_answer="Indexes speed up queries ...",
        ground_truth="Use CREATE INDEX ... B-tree is default ...",
    )
    print(result.score)              # 0.0 – 1.0
    print(result.factual_accuracy)   # 0.0 – 1.0
    print(result.completeness)       # 0.0 – 1.0

Standalone test
---------------
    python evals/metrics/correctness.py
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.metrics.faithfulness import _split_into_claims
from src.agent.providers.ollama_provider import OllamaProvider
from src.utils.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CorrectnessResult:
    """Aggregate result of a correctness evaluation for one query."""

    score: Optional[float]
    """
    Weighted correctness score (0–1).
    0.6 * factual_accuracy + 0.4 * completeness.
    None when neither sub-score could be computed.
    """

    factual_accuracy: Optional[float]
    """
    Fraction of answer claims consistent with the ground truth (0–1).
    Low value: the answer is stating things the ground truth contradicts.
    """

    completeness: Optional[float]
    """
    Fraction of ground-truth key points covered by the answer (0–1).
    Low value: the answer is missing important information.
    """

    num_answer_claims: int
    num_correct_claims: int

    num_gt_points: int
    num_covered_points: int

    answer_verdicts: List[dict] = field(default_factory=list)
    """Per-claim breakdown for factual accuracy (claim, correct: bool|None)."""

    completeness_verdicts: List[dict] = field(default_factory=list)
    """Per-point breakdown for completeness (point, covered: bool|None)."""

    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "correctness_score": self.score,
            "factual_accuracy": self.factual_accuracy,
            "completeness": self.completeness,
            "num_answer_claims": self.num_answer_claims,
            "num_correct_claims": self.num_correct_claims,
            "num_gt_points": self.num_gt_points,
            "num_covered_points": self.num_covered_points,
            "answer_verdicts": self.answer_verdicts,
            "completeness_verdicts": self.completeness_verdicts,
            "correctness_error": self.error,
        }


# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

_FACTUAL_ACCURACY_PROMPT = """\
You are a strict fact-checker for technical documentation.

GROUND TRUTH is the definitive correct answer. For each numbered CLAIM from \
an AI-generated answer, decide whether the ground truth supports or is \
consistent with that claim.

Rules:
- YES → the ground truth confirms or is consistent with the claim.
- NO  → the ground truth contradicts the claim, or the claim says something \
factually wrong relative to the ground truth.
- Reply with ONLY one word per line (YES or NO), in the same order as the claims.
- Do NOT add explanations, numbers, or punctuation.

GROUND TRUTH:
{ground_truth}

CLAIMS FROM AI ANSWER:
{numbered_claims}

Answer (one YES or NO per line, nothing else):"""


_COMPLETENESS_PROMPT = """\
You are evaluating whether an AI-generated answer covers the key information \
from a ground-truth reference.

For each numbered KEY POINT from the ground truth, decide whether the AI \
ANSWER addresses it.

Rules:
- YES → the answer explicitly or clearly covers this point.
- NO  → the answer omits or contradicts this point.
- Reply with ONLY one word per line (YES or NO), in the same order as the points.
- Do NOT add explanations, numbers, or punctuation.

AI ANSWER:
{answer}

KEY POINTS FROM GROUND TRUTH:
{numbered_points}

Answer (one YES or NO per line, nothing else):"""


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class CorrectnessScorer:
    """
    LLM-as-judge correctness scorer backed by a local Ollama model.

    Parameters
    ----------
    judge_model:
        Ollama model name.  Defaults to the application's configured default.
    max_text_chars:
        Truncate ground truth / answer text to this length before sending to
        the judge.  Keeps prompts within small-model context windows.
    base_url:
        Ollama server URL.  Defaults to settings value.
    judge_timeout:
        Per-call timeout in seconds.
    factual_weight:
        Weight given to factual accuracy in the final score.  Completeness
        receives (1 - factual_weight).  Default 0.6.
    """

    FACTUAL_WEIGHT = 0.6

    def __init__(
        self,
        judge_model: Optional[str] = None,
        max_text_chars: int = 2000,
        base_url: Optional[str] = None,
        judge_timeout: int = 60,
        factual_weight: float = 0.6,
    ):
        settings = get_settings()
        resolved_model = judge_model or settings.ollama_model_id
        resolved_url = base_url or settings.ollama_base_url

        self.max_text_chars = max_text_chars
        self.judge_timeout = judge_timeout
        self.factual_weight = factual_weight

        self.provider = OllamaProvider(
            model_name=resolved_model,
            base_url=resolved_url,
        )
        logger.info(
            f"CorrectnessScorer ready — judge: {resolved_model} "
            f"@ {resolved_url} (timeout={judge_timeout}s)"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        llm_answer: str,
        ground_truth: str,
    ) -> CorrectnessResult:
        """
        Score the correctness of *llm_answer* against *ground_truth*.

        Parameters
        ----------
        llm_answer:
            The LLM-generated answer to evaluate.
        ground_truth:
            The reference answer from the test query set.

        Returns
        -------
        CorrectnessResult
            Contains aggregate score, factual_accuracy, completeness, and
            per-claim breakdowns.  score is None when no claims could be judged.
        """
        if not llm_answer or not llm_answer.strip():
            return self._empty_result(error="Empty answer — nothing to evaluate.")

        if not ground_truth or not ground_truth.strip():
            return self._empty_result(error="Empty ground truth — cannot evaluate.")

        answer_text = self._truncate(llm_answer)
        gt_text = self._truncate(ground_truth)

        # ---- Factual Accuracy: answer claims vs ground truth ---------------
        answer_claims = _split_into_claims(answer_text)
        if not answer_claims:
            factual_accuracy = None
            num_correct = 0
            answer_verdicts = []
        else:
            correct_flags, answer_verdicts = self._judge_claims_vs_reference(
                claims=answer_claims,
                reference=gt_text,
                prompt_template=_FACTUAL_ACCURACY_PROMPT,
                reference_key="ground_truth",
                verdict_key="correct",
            )
            num_correct = sum(1 for f in correct_flags if f is True)
            decidable = sum(1 for f in correct_flags if f is not None)
            factual_accuracy = (num_correct / decidable) if decidable > 0 else None

        # ---- Completeness: ground truth points vs answer -------------------
        gt_points = _split_into_claims(gt_text)
        if not gt_points:
            completeness = None
            num_covered = 0
            completeness_verdicts = []
        else:
            covered_flags, completeness_verdicts = self._judge_claims_vs_reference(
                claims=gt_points,
                reference=answer_text,
                prompt_template=_COMPLETENESS_PROMPT,
                reference_key="answer",
                verdict_key="covered",
            )
            num_covered = sum(1 for f in covered_flags if f is True)
            decidable = sum(1 for f in covered_flags if f is not None)
            completeness = (num_covered / decidable) if decidable > 0 else None

        # ---- Combined score ------------------------------------------------
        if factual_accuracy is not None and completeness is not None:
            combined = self.factual_weight * factual_accuracy + (1 - self.factual_weight) * completeness
        elif factual_accuracy is not None:
            combined = factual_accuracy
        elif completeness is not None:
            combined = completeness
        else:
            combined = None

        return CorrectnessResult(
            score=combined,
            factual_accuracy=factual_accuracy,
            completeness=completeness,
            num_answer_claims=len(answer_claims) if answer_claims else 0,
            num_correct_claims=num_correct,
            num_gt_points=len(gt_points) if gt_points else 0,
            num_covered_points=num_covered,
            answer_verdicts=answer_verdicts,
            completeness_verdicts=completeness_verdicts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _truncate(self, text: str) -> str:
        if len(text) > self.max_text_chars:
            return text[: self.max_text_chars] + "\n[truncated]"
        return text

    def _empty_result(self, error: str) -> CorrectnessResult:
        return CorrectnessResult(
            score=None,
            factual_accuracy=None,
            completeness=None,
            num_answer_claims=0,
            num_correct_claims=0,
            num_gt_points=0,
            num_covered_points=0,
            error=error,
        )

    def _judge_claims_vs_reference(
        self,
        claims: List[str],
        reference: str,
        prompt_template: str,
        reference_key: str,
        verdict_key: str,
    ):
        """
        Call the judge LLM with a numbered claim list and parse YES/NO responses.

        Returns (flags, verdicts) where flags is List[Optional[bool]] and
        verdicts is a list of dicts suitable for JSON storage.
        """
        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))

        if reference_key == "ground_truth":
            prompt = prompt_template.format(
                ground_truth=reference,
                numbered_claims=numbered,
            )
        else:  # answer
            prompt = prompt_template.format(
                answer=reference,
                numbered_points=numbered,
            )

        try:
            raw = self.provider.generate_text(
                prompt, temperature=0.0, timeout=self.judge_timeout
            )
            tokens = re.findall(r'\b(YES|NO)\b', raw.upper())
        except Exception as exc:
            logger.error(f"Judge call failed: {exc}")
            tokens = []

        flags: List[Optional[bool]] = []
        verdicts = []
        for i, claim in enumerate(claims):
            if i < len(tokens):
                flag = tokens[i] == "YES"
                raw_token = tokens[i]
            else:
                flag = None
                raw_token = "[missing]"
            flags.append(flag)
            verdicts.append({
                "text": claim,
                verdict_key: flag,
                "raw_response": raw_token,
            })

        return flags, verdicts


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("CorrectnessScorer — standalone smoke test")
    print("=" * 60)

    GROUND_TRUTH = (
        "PostgreSQL supports B-tree (default), Hash, GiST, GIN, SP-GiST, and BRIN "
        "index types. Use CREATE INDEX idx_name ON table(column) to create one. "
        "B-tree handles equality and range queries. Partial indexes add a WHERE clause."
    )

    ANSWER_CORRECT = (
        "To create an index in PostgreSQL use CREATE INDEX. "
        "The default index type is B-tree which supports equality and range queries. "
        "Other types include Hash, GiST, and GIN."
    )

    ANSWER_WRONG = (
        "PostgreSQL only supports one index type: Hash. "
        "You create indexes using the MAKE INDEX command. "
        "Indexes are stored in RAM and lost on restart."
    )

    ANSWER_INCOMPLETE = (
        "PostgreSQL supports B-tree indexes."
    )

    scorer = CorrectnessScorer()

    cases = [
        ("CORRECT", ANSWER_CORRECT),
        ("WRONG",   ANSWER_WRONG),
        ("INCOMPLETE", ANSWER_INCOMPLETE),
    ]

    for label, answer in cases:
        print(f"\n--- {label} ---")
        result = scorer.score(llm_answer=answer, ground_truth=GROUND_TRUTH)
        score_str = f"{result.score:.3f}" if result.score is not None else "N/A"
        fa_str = f"{result.factual_accuracy:.3f}" if result.factual_accuracy is not None else "N/A"
        comp_str = f"{result.completeness:.3f}" if result.completeness is not None else "N/A"
        print(f"  Correctness      : {score_str}")
        print(f"  Factual Accuracy : {fa_str}  ({result.num_correct_claims}/{result.num_answer_claims} claims correct)")
        print(f"  Completeness     : {comp_str}  ({result.num_covered_points}/{result.num_gt_points} points covered)")
        if result.error:
            print(f"  Error: {result.error}")
