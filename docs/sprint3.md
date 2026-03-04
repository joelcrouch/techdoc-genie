# Sprint 3: Advanced Metrics & Practitioner Insights

**Duration**: 3–5 Days
**Goal**: Transform "vibes-based" evaluation into actionable engineering data by implementing industry-standard ranking and correctness metrics.

---

## User Stories & Deliverables

### 1. Ranking-Aware Retrieval (Average Precision)
*   **User Story**: As a RAG Engineer, I want to know if my relevant chunks are appearing at the top of the results, so I can determine if I need a Re-ranker or a better Embedding model.
*   **Deliverable**: Update `evals/metrics/context_metrics.py` to calculate **Mean Average Precision (MAP)**.
*   **Technical Task**: Replace simple boolean precision with a weighted formula that penalizes relevant information found at lower ranks (e.g., position #5 vs #1).

### 2. Logical Answer Correctness (LLM-as-judge)
*   **User Story**: As a Product Owner, I want a correctness score that understands logic rather than just word similarity, so I can identify when the LLM is confidently stating the opposite of the truth.
*   **Deliverable**: Create `evals/metrics/correctness.py`.
*   **Technical Task**: Implement a judge prompt that compares `llm_answer` to `ground_truth` based on **Factual Accuracy** and **Completeness**, returning a 1–5 scale or a weighted % score.

### 3. Advanced Failure Diagnosis
*   **User Story**: As a Developer, I want the system to tell me exactly *what* to fix (Docs vs. Prompt vs. Retrieval), so I don't waste time optimizing the wrong component.
*   **Deliverable**: Update `evals/diagnose.py` and `evals/dashboard.py`.
*   **Technical Task**: Map the intersection of Faithfulness and Correctness to specific "Actionable Advice" (e.g., "Knowledge Gap: Answer is wrong but LLM followed the docs — check your source data").

---

## Success Metrics
*   **Diagnostic Accuracy**: Manual review confirms `diagnose.py` correctly identifies the root cause for 90% of failed queries.
*   **Metric Correlation**: "Correctness" score correlates more closely with human judgment than "Semantic Similarity" in 80% of test cases.

---

## Technical Details

### Average Precision Formula
Instead of `n_relevant / n_retrieved`, we will implement:
`AP = Σ (Precision@k * relevance_k) / total_relevant_retrieved`
where `relevance_k` is 1 if chunk `k` is relevant, 0 otherwise.

### Correctness Judge Prompt
The prompt will ask the LLM to analyze:
1.  **Factual consistency**: Does the answer contradict the ground truth?
2.  **Completeness**: Does it address all parts of the question mentioned in the ground truth?
3.  **No Hallucinations**: Does it include extra info not in the ground truth that contradicts common knowledge?
