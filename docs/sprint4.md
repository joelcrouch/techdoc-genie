# Sprint 4: Quality Engineering & Automated Testing

**Duration**: 3–5 Days
**Goal**: Implement a "Safety Net" of automated tests to ensure that improvements in one area don't cause silent regressions in another.

---

## User Stories & Deliverables

### 1. Metric Unit Testing (The Scorer Test)
*   **User Story**: As a Developer, I want to verify my metric math is correct using "known" edge cases, so I can trust the scores I'm seeing on the dashboard.
*   **Deliverable**: Create `tests/evals/test_metrics.py`.
*   **Technical Task**: Write unit tests for `ContextMetricsScorer` and `FaithfulnessScorer` using dummy data where the result is mathematically pre-determined.

### 2. RAG Regression Testing (The Gold Set)
*   **User Story**: As a Developer, I want to run a "fast-eval" on 5 critical queries every time I change a prompt, so I don't accidentally break core functionality while fixing a niche bug.
*   **Deliverable**: Create `tests/integration/test_regression_eval.py`.
*   **Technical Task**: Create a "Gold Set" of 5–10 high-priority queries that must maintain a minimum Correctness score of 0.8 to pass CI.

### 3. Provider & Pipeline Mocking
*   **User Story**: As a Developer, I want to test my RAG orchestration logic without spending API credits or waiting for local LLM inference, so I can iterate on code structure in seconds.
*   **Deliverable**: Create `src/agent/providers/mock_provider.py` and corresponding tests.
*   **Technical Task**: Implement a Mock LLM that returns pre-defined strings, allowing the full `rag_chain.py` to be tested for logic, citation formatting, and error handling without a "real" backend.

---

## Success Metrics
*   **Test Coverage**: Core metric and retrieval logic achieve >80% code coverage.
*   **Iteration Speed**: Regression tests for the "Gold Set" run in under 30 seconds (using mocks where appropriate).
*   **Reliability**: Zero "silent regressions" (bugs reaching the dashboard) during the final 2 days of development.

---

## Implementation Plan

### Regression Set (`Gold Set`)
The following queries from `test_queries.json` will be used as the regression baseline:
- `q001` (Basic DDL)
- `q002` (Basic SQL/Joins)
- `q004` (Complex Architecture - MVCC)
- `q011` (Performance - EXPLAIN)
- `q016` (Architecture - WAL)

### Mocking Strategy
The `MockProvider` will implement the `BaseLLMProvider` interface but allow the test suite to "inject" expected responses:
```python
provider.set_mock_response("How do I create an index?", "Use CREATE INDEX...")
```
