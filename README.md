# TechDoc Genie

A Retrieval-Augmented Generation (RAG) system for querying technical documentation using natural language. Supports PDF and HTML source documents, multiple LLM backends (local and cloud), and a full evaluation pipeline for measuring and improving retrieval and answer quality.

---

## Features

- **Semantic search** — vector-based retrieval that understands intent beyond keywords
- **Multi-provider LLM support** — Ollama (local), Google Gemini, OpenAI, and Anthropic Claude
- **Grounded answers with citations** — responses reference source documents
- **Configurable chunking strategies** — compare recursive and semantic chunking at various sizes
- **Full evaluation pipeline** — generate, analyze, diagnose, and visualize RAG quality metrics
- **Interactive Streamlit dashboard** — track metrics across experiments over time
- **MCP server integration** — Model Context Protocol for extensible document access

---

## Project Structure

```
techdoc-genie/
├── src/
│   ├── ingestion/        # Document loading, chunking, embedding
│   ├── retrieval/        # FAISS vector store
│   ├── agent/            # RAG chain, LLM providers, query interface
│   │   └── providers/    # ollama, openai, claude, gemini
│   ├── mcp_server/       # MCP server implementation
│   ├── cli/              # CLI entry points
│   └── utils/            # Config, logging
├── evals/
│   ├── build_vectorstores.py   # Build cached eval vector stores
│   ├── generate.py             # Phase 1: run experiments, save raw results
│   ├── analyze.py              # Phase 2: compute metrics on raw results
│   ├── diagnose.py             # Failure mode analysis
│   ├── metrics_store.py        # Ingest metrics into SQLite
│   ├── dashboard.py            # Streamlit metrics dashboard
│   └── evaluation_configs.json # Experiment configuration
├── scripts/
│   └── build_generic_vectorstore.py  # Build a query-time vector store
├── data/
│   ├── raw/              # Source documents (PDF, HTML)
│   └── vector_store/     # Built vector stores
└── docs/                 # Design docs, sprint notes, guides
```

---

## Prerequisites

- Python 3.10+
- (Optional) [Ollama](https://ollama.com/) for local LLM inference

### Setup

```bash
git clone <repo-url>
cd techdoc-genie
python -m venv techdoc-genie-venv
source techdoc-genie-venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in any cloud API keys you need:

```bash
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

### Ollama (local LLM)

```bash
# Check status
systemctl status ollama

# Start if stopped
sudo systemctl start ollama

# Pull models
ollama pull phi3:mini
ollama pull qwen2.5:1.5b
```

---

## Quick Start

### 1. Add documents

Place PDF or HTML files into a directory under `data/raw/`:

```bash
mkdir -p data/raw/python_docs
cp ~/Downloads/python-3.12-docs.pdf data/raw/python_docs/
```

### 2. Build a vector store

```bash
python scripts/build_generic_vectorstore.py \
    -d data/raw/python_docs/ \
    -f pdf \
    -n python_docs
```

| Flag | Description |
|------|-------------|
| `-d` | Directory containing documents |
| `-f` | Format: `pdf`, `html`, or `auto` |
| `-n` | Name for the output vector store |

### 3. Query interactively

```bash
# Default (Ollama phi3:mini)
python src/agent/query_interface.py

# Specify vector store and model
python src/agent/query_interface.py \
    --vector-store python_docs \
    --provider ollama \
    --model qwen2.5:1.5b

# Use Gemini
python src/agent/query_interface.py \
    --vector-store python_docs \
    --provider gemini \
    --model gemini-2.5-flash-lite

# Change prompt style (base | detailed | troubleshooting | code)
python src/agent/query_interface.py \
    --vector-store python_docs \
    --prompt detailed
```

Type `exit` or `quit` to end the session.

---

## LLM Providers

| Provider | `--provider` | Example `--model` | Requires |
|----------|-------------|-------------------|---------|
| Ollama (local) | `ollama` | `phi3:mini`, `qwen2.5:1.5b` | Ollama running |
| Google Gemini | `gemini` | `gemini-2.5-flash-lite` | `GEMINI_API_KEY` |
| OpenAI | `openai` | `gpt-4-turbo-preview` | `OPENAI_API_KEY` |
| Anthropic | `claude` | `claude-3-haiku-20240307` | `ANTHROPIC_API_KEY` |

---

## Evaluation Pipeline

The evaluation pipeline is split into phases so GPU inference and CPU-based analysis don't compete.

```
generate.py  →  raw results (query + answer + chunks)
analyze.py   →  metrics added to raw results
diagnose.py  →  failure mode report, worst queries, actionable gaps
metrics_store.py + dashboard.py  →  SQLite store + Streamlit UI
```

### Step 1 — Build evaluation vector stores (once per document)

```bash
python evals/build_vectorstores.py
```

### Step 2 — Generate raw results

```bash
# All experiments from evaluation_configs.json
python evals/generate.py --workers 1

# Filter to one LLM
python evals/generate.py --provider ollama --model qwen2.5:1.5b --workers 1

# Cloud providers can use more workers
python evals/generate.py --provider gemini --model gemini-2.5-flash-lite --workers 5
```

Re-running skips experiments that already have output files. Use `--overwrite` to force a re-run.

### Step 3 — Analyze results

```bash
# Fast (semantic similarity + context metrics, CPU only)
python evals/analyze.py --skip-faithfulness

# Full (adds LLM-as-judge faithfulness scoring)
python evals/analyze.py --throttle 1.5
```

#### Metrics

| Metric | What it measures | Fix when low |
|--------|-----------------|--------------|
| **Semantic similarity** | Cosine distance between answer and ground truth | General quality signal |
| **Context precision** | Fraction of retrieved chunks that are relevant | Try smaller chunks |
| **Context recall** | Did retrieved context contain the answer? | Increase k; add BM25 |
| **Faithfulness** | Fraction of answer claims grounded in context | Use a stronger model |

### Step 4 — Diagnose failure modes

```bash
python evals/diagnose.py

# Focus on one category or query
python evals/diagnose.py --category performance
python evals/diagnose.py --query q028
```

| Mode | Meaning | Action |
|------|---------|--------|
| `OK` | Answer is good | — |
| `GENERATION_WEAK` | Good retrieval, mediocre answer | Prompt tuning or larger model |
| `GENERATION_COLLAPSE` | Good retrieval, very bad answer | Reduce chunk size (context overflow) |
| `RETRIEVAL_MISS` | Right content not retrieved | Increase k, add hybrid search |
| `ERROR` | LLM call failed | Check logs |

### Step 5 — Store and visualize

```bash
python evals/metrics_store.py --ingest
streamlit run evals/dashboard.py
```

Dashboard opens at `http://localhost:8501` and shows failure mode distribution, metric comparisons by experiment/category/difficulty, worst queries, and time-series trends.

---

## Configuration

### `evals/evaluation_configs.json`

Controls which experiments the pipeline runs:

```json
{
  "document_paths": {
    "postgresql": "data/raw/postgresql/postgresql-16-A4.pdf"
  },
  "test_queries_path": "evals/test_queries.json",
  "llm_configs": [
    { "provider": "ollama", "model_id": "phi3:mini" },
    { "provider": "gemini", "model_id": "gemini-2.5-flash-lite" }
  ],
  "embedder_config": {
    "provider": "huggingface",
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  },
  "chunking_strategies": [
    { "name": "Recursive_512_Overlap_50", "chunk_size": 512, "chunk_overlap": 50, "method": "chunk_recursive" }
  ]
}
```

### Key paths

| Path | Purpose |
|------|---------|
| `data/raw/<name>/` | Source documents |
| `data/vector_store/<name>/` | Query-time vector store |
| `evals/cache/` | Evaluation vector stores |
| `evals/results/raw/` | Raw generation outputs |
| `evals/results/analyzed/` | Results with metrics |
| `evals/metrics.db` | SQLite metrics store |

---

## Running Tests

```bash
pytest
```

---

## Typical Iteration Workflow

```
1. Edit evaluation_configs.json     ← add a strategy or LLM
2. python evals/build_vectorstores.py  ← only if new strategies added
3. python evals/generate.py --workers 1
4. python evals/analyze.py --skip-faithfulness
5. python evals/diagnose.py
6. python evals/metrics_store.py --ingest
7. streamlit run evals/dashboard.py
8. Read results → decide what to change → repeat
```

---

## License

MIT
