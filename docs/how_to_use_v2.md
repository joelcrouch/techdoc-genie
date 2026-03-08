# TechDoc Genie — How To Use (v2)

A RAG (Retrieval-Augmented Generation) system for querying technical documentation.
Supports PDF and HTML source documents, multiple LLM backends, and a full evaluation
pipeline for measuring and improving retrieval and answer quality.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Adding a New Document](#2-adding-a-new-document)
3. [Building a Vector Store](#3-building-a-vector-store)
4. [Querying Interactively](#4-querying-interactively)
5. [Running Evaluation](#5-running-evaluation)
   - 5.1 [Pre-build evaluation vector stores](#51-pre-build-evaluation-vector-stores)
   - 5.2 [Generate raw results](#52-generate-raw-results)
   - 5.3 [Analyze results](#53-analyze-results)
   - 5.4 [Diagnose failure modes](#54-diagnose-failure-modes)
   - 5.5 [Store metrics and launch dashboard](#55-store-metrics-and-launch-dashboard)
6. [Metrics Reference](#6-metrics-reference)
7. [LLM Provider Reference](#7-llm-provider-reference)
8. [Configuration Reference](#8-configuration-reference)
9. [End-to-End Example: Python Docs](#9-end-to-end-example-python-docs)
10. [Typical Iteration Workflow](#10-typical-iteration-workflow)

---

## 1. Prerequisites

### Python environment
```bash
cd ~/Projects/techdoc-genie
source techdoc-genie-venv/bin/activate
pip install -r requirements.txt
```

### Local LLM (Ollama)
```bash
# Check if running
systemctl status ollama

# Start if stopped
sudo systemctl start ollama

# Pull models you want to use
ollama pull phi3:mini
ollama pull qwen2.5:1.5b
```

### Cloud LLM keys (optional)
Copy `.env.example` to `.env` and fill in the keys you need:
```
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

---

## 2. Adding a New Document

### Option A — PDF
Drop the PDF into a directory under `data/raw/`:
```bash
mkdir -p data/raw/python_docs
cp ~/Downloads/python-3.12-docs.pdf data/raw/python_docs/
```

### Option B — HTML (downloaded docs site)
```bash
mkdir -p data/raw/python_docs

# Download the HTML zip from https://docs.python.org/3/download.html then unzip
unzip python-3.12-docs-html.zip -d /tmp/pydocs
cp /tmp/pydocs/*.html data/raw/python_docs/

# Or mirror with wget
wget -r -l 1 -A "*.html" -nd -P data/raw/python_docs/ \
     https://docs.python.org/3/library/functions.html
```

The HTML loader strips navigation, scripts, and footers automatically — it
only keeps headings and body text.

### What the loader supports

| Format | What it does |
|--------|-------------|
| `pdf`  | Splits into logical sections by detecting headings and chapter markers |
| `html` | Strips boilerplate (nav/footer/scripts), converts headings to Markdown |
| `auto` | Loads both PDF and HTML from the same directory |

---

## 3. Building a Vector Store

Run `build_generic_vectorstore.py` to ingest documents and write a FAISS
vector store to disk.

```bash
python scripts/build_generic_vectorstore.py \
    -d data/raw/python_docs/ \
    -f html \
    -n python_docs
```

| Flag | Description |
|------|-------------|
| `-d` | Directory containing your documents |
| `-f` | Format: `html`, `pdf`, or `auto` |
| `-n` | Name for the output vector store (saved to `data/vector_store/<name>/`) |
| `--no-split-sections` | For PDFs: treat each page as a chunk instead of splitting by heading |

The script prints a summary when done:
```
✅ Vector store build completed!
Documents loaded : 142
Chunks created   : 1 847
Vector store location: data/vector_store/python_docs
```

---

## 4. Querying Interactively

```bash
# Default: phi3:mini + default vector store
python src/agent/query_interface.py

# Specify vector store and model
python src/agent/query_interface.py \
    --vector-store python_docs \
    --provider ollama \
    --model qwen2.5:1.5b

# Use Gemini (cloud)
python src/agent/query_interface.py \
    --vector-store python_docs \
    --provider gemini \
    --model gemini-2.5-flash-lite

# Use a different prompt style
python src/agent/query_interface.py \
    --vector-store python_docs \
    --prompt detailed        # options: base | detailed | troubleshooting | code
```

Type `exit` or `quit` to end the session.

---

## 5. Running Evaluation

The evaluation pipeline is split into phases so GPU-heavy generation and
CPU-only analysis don't compete:

```
generate.py  →  raw results (query + LLM answer + retrieved chunks)
analyze.py   →  metrics added to each record + summary stats
diagnose.py  →  failure mode report, worst queries, actionable gaps
metrics_store.py + dashboard.py  →  SQLite store + Streamlit UI
```

### 5.1 Pre-build evaluation vector stores

Build cached vector stores once per document/strategy combination:

```bash
python evals/build_vectorstores.py
```

This reads `evals/evaluation_configs.json` and builds one `.pkl` vector store
per chunking strategy. Takes 5–15 minutes; only needs to run again if you add
a new strategy.

### 5.2 Generate raw results

Phase 1 — runs LLM inference and stores raw outputs. No metrics are computed here.

```bash
# All experiments in evaluation_configs.json
python evals/generate.py --workers 1

# Filter to one LLM
python evals/generate.py --provider ollama --model qwen2.5:1.5b --workers 1

# Filter to specific chunking strategies
python evals/generate.py \
    --provider ollama --model phi3:mini \
    --strategies Recursive_256_Overlap_30,Recursive_512_Overlap_50 \
    --workers 1

# Cloud providers can use more workers
python evals/generate.py --provider gemini --model gemini-2.5-flash-lite --workers 5
```

**Workers note:** Ollama serialises GPU inference regardless of thread count —
`--workers 1` is safest when the GPU is at 100%. For cloud providers higher
values help.

Output goes to `evals/results/raw/`. Re-running skips experiments that already
have an output file. Use `--overwrite` to force a re-run.

### 5.3 Analyze results

Phase 2 — reads raw files and computes all metrics. No LLM or vector store
needed for the fast pass.

```bash
# Fast pass: all metrics except faithfulness (CPU only, ~1-2 min for 20 files)
python evals/analyze.py --skip-faithfulness

# Full pass: adds faithfulness scoring via LLM-as-judge (slower)
python evals/analyze.py --throttle 1.5

# Re-analyze already-processed files (e.g. after adding a new metric)
python evals/analyze.py --skip-faithfulness --overwrite

# Analyze a single file
python evals/analyze.py \
    --input evals/results/raw/postgresql__ollama__qwen2.5_1.5b__Recursive_512_Overlap_50.json \
    --skip-faithfulness

# Use a different judge model for faithfulness
python evals/analyze.py --judge-model qwen2.5:1.5b
```

Output goes to `evals/results/analyzed/`. Each file adds a `metrics` dict to
every record and a top-level `summary` block.

The analysis prints a per-experiment table and a cross-experiment summary:

```
======================================================================
  ollama/qwen2.5:1.5b × Recursive_512_Overlap_50
======================================================================
Queries analyzed         44
Answer rate              97.8%
Avg sem similarity       0.6891
Avg context precision    0.9864
Avg context recall       0.6509
Avg average precision    0.9821   ← MAP (new in Sprint 3)
Avg faithfulness         N/A
```

### 5.4 Diagnose failure modes

```bash
# Full report: failure modes, category/difficulty breakdown, worst queries
python evals/diagnose.py

# Focus on one category
python evals/diagnose.py --category performance

# Show more worst queries
python evals/diagnose.py --worst 20

# Deep dive on a specific query across all experiments
python evals/diagnose.py --query q028

# Filter to one chunking strategy
python evals/diagnose.py --strategy Recursive_256_Overlap_30
```

The Experiment Comparison table now includes a MAP column:

```
Experiment                                    Avg Sem.Sim  Avg Ctx.Prec  Avg Ctx.Rec  MAP    Failure modes
--------------------------------------------  -----------  ------------  -----------  -----  -------------------------
ollama/qwen2.5:1.5b × Recursive_256_Overlap   0.693        1.000         0.670        0.982  OK=31  GENERATION_WEAK=10 ...
```

#### Failure modes

| Mode | Meaning | Action |
|------|---------|--------|
| `OK` | Answer is good | Nothing |
| `GENERATION_WEAK` | Good retrieval, mediocre answer | Prompt tuning, larger model |
| `GENERATION_COLLAPSE` | Good retrieval, very bad answer | Context window overflow — reduce chunk size |
| `RETRIEVAL_MISS` | Right content not retrieved | Increase k, add hybrid search |
| `ERROR` | LLM call failed | Check logs |

### 5.5 Store metrics and launch dashboard

```bash
# Ingest all analyzed files into the SQLite store
python evals/metrics_store.py --ingest

# Print a summary of what's in the store
python evals/metrics_store.py --summary

# Launch the interactive dashboard
streamlit run evals/dashboard.py
```

The dashboard opens at `http://localhost:8501` and shows:

- **Overview** — failure mode distribution donut chart + top-line metrics
- **By Experiment** — metric comparison bar charts (select: Sem.Sim, Ctx.Prec, Ctx.Rec, **MAP**, Faithfulness)
- **By Category** — which query categories score best/worst (includes MAP as purple bar)
- **By Difficulty** — easy / medium / hard breakdown
- **Worst Queries** — table with per-query deep dive and bar chart
- **Time Series** — metric trends across runs over time

**Re-ingesting after re-analysis:** Delete the DB first to avoid stale data:
```bash
rm evals/metrics.db
python evals/metrics_store.py --ingest
```

---

## 6. Metrics Reference

### Retrieval metrics (computed by `analyze.py`, no LLM needed)

| Metric | What it measures | Fix when low |
|--------|-----------------|--------------|
| **Semantic similarity** | Cosine distance between LLM answer and ground truth | General quality signal — look at failure mode for root cause |
| **Context precision** | Fraction of retrieved chunks relevant to the query | Chunking too coarse — try smaller chunks |
| **Context recall** | Did retrieved context contain the answer? | Increase k; add BM25 hybrid search |
| **MAP** (Average Precision) | Rank-aware precision — penalises relevant chunks found at rank 5 vs rank 1 | Low MAP with high precision → relevant chunks ranking too low; consider a re-ranker |

**MAP vs Context Precision:**
Context precision tells you *how many* relevant chunks were retrieved.
MAP tells you *where* they ranked. A system that finds 3 relevant chunks but
puts them at positions 3, 4, 5 (not 1, 2, 3) scores lower on MAP — that
matters when the LLM uses the top-k chunks and ignores the rest.

### Answer quality metrics (require LLM-as-judge)

| Metric | What it measures | Fix when low |
|--------|-----------------|--------------|
| **Faithfulness** | Fraction of answer claims grounded in retrieved context | Model hallucinating — use stronger model or reduce chunk size |
| **Correctness** | Factual accuracy (0.6 weight) + completeness (0.4 weight) vs ground truth | Low factual: model contradicts docs; low completeness: answer is too brief |

#### Correctness sub-scores

`correctness = 0.6 × factual_accuracy + 0.4 × completeness`

- **Factual accuracy** — are the claims in the answer consistent with the ground truth?
- **Completeness** — does the answer cover the key points from the ground truth?

Use the `CorrectnessScorer` directly:
```python
from evals.metrics.correctness import CorrectnessScorer

scorer = CorrectnessScorer(judge_model="qwen2.5:1.5b")
result = scorer.score(
    llm_answer="PostgreSQL supports B-tree and Hash indexes...",
    ground_truth="Use CREATE INDEX. B-tree is default. Supports Hash, GiST, GIN.",
)
print(result.score)             # 0.0 – 1.0
print(result.factual_accuracy)  # 0.0 – 1.0
print(result.completeness)      # 0.0 – 1.0
```

Run the standalone smoke test to verify your Ollama connection:
```bash
python evals/metrics/correctness.py
```

---

## 7. LLM Provider Reference

| Provider | `--provider` | Example `--model` | Requires |
|----------|-------------|-------------------|---------|
| Ollama (local) | `ollama` | `phi3:mini`, `qwen2.5:1.5b` | Ollama running |
| Google Gemini | `gemini` | `gemini-2.5-flash-lite` | `GEMINI_API_KEY` in `.env` |
| OpenAI | `openai` | `gpt-4-turbo-preview` | `OPENAI_API_KEY` in `.env` |
| Anthropic | `claude` | `claude-3-haiku-20240307` | `ANTHROPIC_API_KEY` in `.env` |

To add a new LLM to `evaluation_configs.json`:
```json
{
  "provider": "gemini",
  "model_id": "gemini-2.5-flash-lite"
}
```

---

## 8. Configuration Reference

### `evals/evaluation_configs.json`

Controls which experiments the evaluation pipeline runs.

```json
{
  "document_paths": {
    "postgresql": "data/raw/postgresql/postgresql-16-A4.pdf"
  },
  "test_queries_path": "evals/test_queries.json",
  "llm_configs": [
    { "provider": "ollama", "model_id": "phi3:mini" },
    { "provider": "ollama", "model_id": "qwen2.5:1.5b" },
    { "provider": "gemini", "model_id": "gemini-2.5-flash-lite" }
  ],
  "embedder_config": {
    "provider": "huggingface",
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  },
  "chunking_strategies": [
    { "name": "Recursive_256_Overlap_30",   "chunk_size": 256,  "chunk_overlap": 30,  "method": "chunk_recursive" },
    { "name": "Recursive_512_Overlap_50",   "chunk_size": 512,  "chunk_overlap": 50,  "method": "chunk_recursive" },
    { "name": "Recursive_1024_Overlap_100", "chunk_size": 1024, "chunk_overlap": 100, "method": "chunk_recursive" },
    { "name": "Recursive_2048_Overlap_200", "chunk_size": 2048, "chunk_overlap": 200, "method": "chunk_recursive" },
    { "name": "Semantic_512_Overlap_50",    "chunk_size": 512,  "chunk_overlap": 50,  "method": "chunk_semantic"  }
  ]
}
```

To add a new chunking strategy: add an entry and re-run `build_vectorstores.py`,
then `generate.py`. Existing results are untouched.

### Key paths

| Path | Purpose |
|------|---------|
| `data/raw/<name>/` | Source documents (PDF or HTML) |
| `data/vector_store/<name>/` | Query-time vector store |
| `evals/cache/` | Evaluation vector stores (built by `build_vectorstores.py`) |
| `evals/results/raw/` | Raw generation outputs |
| `evals/results/analyzed/` | Results with metrics added |
| `evals/metrics.db` | SQLite metrics store |
| `evals/checkpoints/` | Crash-recovery checkpoints |
| `evals/metrics/context_metrics.py` | Context precision, recall, MAP scorer |
| `evals/metrics/faithfulness.py` | Faithfulness LLM-as-judge scorer |
| `evals/metrics/correctness.py` | Correctness LLM-as-judge scorer (Sprint 3) |

---

## 9. End-to-End Example: Python Docs

### Step 1 — Download the HTML docs
```bash
mkdir -p data/raw/python_docs
wget https://docs.python.org/3/archives/python-3.12-docs-html.zip -P /tmp/
unzip /tmp/python-3.12-docs-html.zip -d /tmp/pydocs/
cp /tmp/pydocs/python-3.12-docs-html/*.html data/raw/python_docs/
```

### Step 2 — Build the query-time vector store
```bash
python scripts/build_generic_vectorstore.py \
    -d data/raw/python_docs/ \
    -f html \
    -n python_docs
```

### Step 3 — Query it
```bash
python src/agent/query_interface.py \
    --vector-store python_docs \
    --provider ollama \
    --model qwen2.5:1.5b
```

### Step 4 — Add to evaluation config
Edit `evals/evaluation_configs.json`:
```json
"document_paths": {
    "postgresql": "data/raw/postgresql/postgresql-16-A4.pdf",
    "python_docs": "data/raw/python_docs"
}
```

### Step 5 — Build evaluation vector stores
```bash
python evals/build_vectorstores.py
```

### Step 6 — Generate results
```bash
python evals/generate.py --provider ollama --model qwen2.5:1.5b --workers 1
```

### Step 7 — Analyze and diagnose
```bash
python evals/analyze.py --skip-faithfulness
python evals/diagnose.py
```

### Step 8 — Ingest and view dashboard
```bash
rm evals/metrics.db   # clear stale data if re-ingesting
python evals/metrics_store.py --ingest
streamlit run evals/dashboard.py
```

---

## 10. Typical Iteration Workflow

```
1. Edit evaluation_configs.json        ← add a new strategy or LLM
2. python evals/build_vectorstores.py  ← only if new strategies added
3. python evals/generate.py --workers 1
4. python evals/analyze.py --skip-faithfulness
5. python evals/diagnose.py
6. rm evals/metrics.db && python evals/metrics_store.py --ingest
7. streamlit run evals/dashboard.py
8. Read results → decide what to change → go to step 1
```

Because `generate.py` skips experiments that already have output files and
`analyze.py` skips files that are already analyzed, re-running after adding
a single new strategy only processes that new experiment.

### When to use `--overwrite`
- Added a new metric to `analyze.py` and want it in old result files
- Changed the precision threshold in `ContextMetricsScorer`
- Fixed a bug in the scoring logic

```bash
python evals/analyze.py --skip-faithfulness --overwrite
rm evals/metrics.db
python evals/metrics_store.py --ingest
```
