# TechDoc Genie — How To Use

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
6. [LLM Provider Reference](#6-llm-provider-reference)
7. [Configuration Reference](#7-configuration-reference)
8. [End-to-End Example: Python Docs](#8-end-to-end-example-python-docs)

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
```bash
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
Download the HTML documentation and place the `.html` files in a directory:
```bash
mkdir -p data/raw/python_docs

# Python docs: download the HTML zip from https://docs.python.org/3/download.html
# then unzip the html files into the directory
unzip python-3.12-docs-html.zip -d /tmp/pydocs
cp /tmp/pydocs/*.html data/raw/python_docs/

# Or use wget to mirror a docs site (example: Python docs)
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
3 and 4 are essentially smoke tests to make sure it works.  The real analyiss begins to happen at 5.
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

Example session:
```
================================================================================
Technical Documentation Assistant
Provider: ollama, Model: qwen2.5:1.5b, Prompt: base
Vector Store: python_docs
Type 'exit' or 'quit' to end.
================================================================================

Your question: how do I use list comprehensions?

🔍 Searching documentation and generating answer...

================================================================================
ANSWER:
────────────────────────────────────────────────────────────────────────────────
List comprehensions provide a concise way to create lists...

SOURCES (5):
────────────────────────────────────────────────────────────────────────────────
[1] functions.html, Page N/A
    ...
```

Type `exit` or `quit` to end the session.

---

## 5. Running Evaluation

The evaluation pipeline is split into two phases so generation (GPU-heavy)
and analysis (CPU-only) don't fight each other.

```
generate.py  →  raw results (query + LLM answer + retrieved chunks)
analyze.py   →  metrics added to raw results
diagnose.py  →  failure mode report, worst queries, actionable gaps
metrics_store.py + dashboard.py  →  SQLite store + Streamlit UI
```

### 5.1 Pre-build evaluation vector stores

The evaluation runner loads cached vector stores.  Build them once:

```bash
python evals/build_vectorstores.py
```

This reads `evals/evaluation_configs.json` and builds one cached `.pkl` vector
store per chunking strategy.  Takes 5-15 minutes, only needs to run once per
document.

### 5.2 Generate raw results

Phase 1 — runs every configured experiment and stores raw outputs.
**No metrics are computed here** — the GPU only does LLM inference.

```bash
# Run all experiments defined in evaluation_configs.json
python evals/generate.py --workers 1

# Filter to one LLM (fastest for a first pass)
python evals/generate.py --provider ollama --model qwen2.5:1.5b --workers 1

# Filter to specific chunking strategies
python evals/generate.py \
    --provider ollama --model phi3:mini \
    --strategies Recursive_256_Overlap_30,Recursive_512_Overlap_50 \
    --workers 1

# Use Gemini (cloud — can use more workers)
python evals/generate.py --provider gemini --model gemini-2.5-flash-lite --workers 5
```

**Workers note:** Ollama serialises GPU inference regardless of thread count.
`--workers 1` is the safest choice when the GPU is at 100%.  For cloud
providers (Gemini, OpenAI) higher values help.

Output goes to `evals/results/raw/`.  Files are named:
```
postgresql__ollama__qwen2.5_1.5b__Recursive_512_Overlap_50.json
```

Re-running skips experiments that already have an output file.  Use
`--overwrite` to force a re-run.

### 5.3 Analyze results

Phase 2 — reads raw files and computes metrics.  No LLM or vector store needed.

```bash
# Fast pass: semantic similarity + context precision/recall only (CPU, ~30s)
python evals/analyze.py --skip-faithfulness

# Full pass: adds faithfulness (LLM-as-judge via Ollama, slower)
python evals/analyze.py --throttle 1.5

# Re-analyze files already processed (e.g. after adding a new metric)
python evals/analyze.py --skip-faithfulness --overwrite

# Analyze a single file
python evals/analyze.py \
    --input evals/results/raw/postgresql__ollama__qwen2.5_1.5b__Recursive_512_Overlap_50.json \
    --skip-faithfulness
```

The analysis prints a table per experiment and a cross-experiment summary:

```
======================================================================
  ollama/qwen2.5:1.5b × Recursive_512_Overlap_50
======================================================================
Queries analyzed         44
Answer rate              97.8%
Avg sem similarity       0.6891
Avg context precision    0.9864
Avg context recall       0.6509
Avg faithfulness         N/A
```

Output goes to `evals/results/analyzed/`.

#### Metrics explained

| Metric | What it measures | Fix when low |
|--------|-----------------|--------------|
| **Semantic similarity** | Cosine distance between answer and ground truth | General quality signal |
| **Context precision** | Fraction of retrieved chunks that are relevant | Chunking too coarse; try smaller chunks |
| **Context recall** | Did retrieved context contain the answer? | Increase k; add BM25 hybrid search |
| **Faithfulness** | Fraction of answer claims grounded in context | Model hallucinating; use stronger model |

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

#### Failure modes

| Mode | Meaning | Action |
|------|---------|--------|
| `OK` | Answer is good | Nothing |
| `GENERATION_WEAK` | Good retrieval, mediocre answer | Prompt tuning, larger model |
| `GENERATION_COLLAPSE` | Good retrieval, very bad answer | Context window overflow — reduce chunk size |
| `RETRIEVAL_MISS` | Right content not retrieved | Increase k, add hybrid search |
| `ERROR` | LLM call failed | Check logs |

Example output:
```
CONSISTENT FAILURES (sem_sim < 0.5 in all experiments)
q028  hard  performance  0.342  How can I optimize a slow query...
  → These queries need retrieval fixes, not just model/prompt tuning.
```

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
- **Overview** — failure mode distribution donut chart
- **By Experiment** — metric comparison across all LLM × strategy combinations
- **By Category** — which query categories score best and worst
- **By Difficulty** — easy / medium / hard breakdown
- **Worst Queries** — table with per-query deep dive and bar chart
- **Time Series** — metric trends across runs over time

---

## 6. LLM Provider Reference

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

## 7. Configuration Reference

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
    { "name": "Semantic_512_Overlap_50",    "chunk_size": 512,  "chunk_overlap": 50,  "method": "chunk_semantic"   }
  ]
}
```

To add a new chunking strategy, add an entry and re-run `build_vectorstores.py`
then `generate.py`.  Existing results are untouched.

### Key paths

| Path | Purpose |
|------|---------|
| `data/raw/<name>/` | Source documents (PDF or HTML) |
| `data/vector_store/<name>/` | Query-time vector store (built by `build_generic_vectorstore.py`) |
| `evals/cache/` | Evaluation vector stores (built by `build_vectorstores.py`) |
| `evals/results/raw/` | Raw generation outputs |
| `evals/results/analyzed/` | Results with metrics added |
| `evals/metrics.db` | SQLite metrics store |
| `evals/checkpoints/` | Crash-recovery checkpoints |

---

## 8. End-to-End Example: Python Docs

This walks through adding a completely new document — the Python 3.12 standard
library docs — from scratch through to evaluation.

### Step 1 — Download the HTML docs

```bash
mkdir -p data/raw/python_docs

# Download the Python 3.12 HTML docs (single-page HTML zip from python.org)
wget https://docs.python.org/3/archives/python-3.12-docs-html.zip -P /tmp/
unzip /tmp/python-3.12-docs-html.zip -d /tmp/pydocs/

# Copy HTML files into the project
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

Try asking:
- *How do I use list comprehensions?*
- *What is the difference between a list and a tuple?*
- *How does the `with` statement work?*

### Step 4 — Add to evaluation config

Edit `evals/evaluation_configs.json` to add the new document:

```json
"document_paths": {
    "postgresql": "data/raw/postgresql/postgresql-16-A4.pdf",
    "python_docs": "data/raw/python_docs"
}
```

Create a test queries file for Python (or reuse the pattern from
`evals/test_queries.json`) with category/difficulty/ground_truth fields.

Update `test_queries_path` in the config if you create a separate file.

### Step 5 — Build evaluation vector stores

```bash
python evals/build_vectorstores.py
```

### Step 6 — Generate results

```bash
python evals/generate.py --provider ollama --model qwen2.5:1.5b --workers 1
```

Go get coffee. Come back when it's done.

### Step 7 — Analyze and diagnose

```bash
python evals/analyze.py --skip-faithfulness
python evals/diagnose.py
```

### Step 8 — Ingest and view dashboard

```bash
python evals/metrics_store.py --ingest
streamlit run evals/dashboard.py
```

---

## Typical iteration workflow

```
1. Edit evaluation_configs.json  ← add new strategy or LLM
2. python evals/build_vectorstores.py   ← only if new strategies added
3. python evals/generate.py --workers 1  ← go to lunch
4. python evals/analyze.py --skip-faithfulness
5. python evals/diagnose.py
6. python evals/metrics_store.py --ingest
7. streamlit run evals/dashboard.py
8. Read results → decide what to change → go to step 1
```

Because `generate.py` skips experiments that already have output files and
`analyze.py` skips files that are already analyzed, re-running after adding
a single new strategy only processes that new experiment.
