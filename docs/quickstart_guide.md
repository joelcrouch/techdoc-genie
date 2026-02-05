# Quickstart Guide: TechDoc Genie

This guide provides the essential commands to build a vector store and run the RAG (Retrieval-Augmented Generation) query assistant.

---

## 1. Running the Local LLM (Ollama)

The local LLM is managed by the Ollama server running as a system service.

### Check Server Status
To see if the Ollama server is already running:
```bash
systemctl status ollama
```

### Start/Stop Server
If the server is not active, you can start it with:
```bash
sudo systemctl start ollama
```

To stop it:
```bash
sudo systemctl stop ollama
```

---

## 2. Building a New Vector Store

You can ingest new documentation from a directory of files (PDFs, HTML, etc.) into a new vector store.

### Generic Build Command
Use this command to build a new vector store. Replace the placeholders (`<...>` a/k/a chevrons) with your specific paths and names.

```bash
python scripts/build_generic_vectorstore.py -d <path_to_your_docs_directory> -f <doc_format> -n <new_vector_store_name>
```

### Example: Building the Ubuntu Docs Vector Store
This command ingests PDF files from the `data/raw/ubuntu_docs/` directory and creates a vector store named `ubuntu_docs_pdf`.

```bash
python scripts/build_generic_vectorstore.py -d /home/dell-linux-dev3/Projects/techdoc-genie/data/raw/ubuntu_docs/ -f pdf -n ubuntu_docs_pdf
```

---

## 3. Querying with the RAG Assistant

Once you have a vector store, you can use the interactive assistant to ask questions.

### Basic Query (Local LLM, Default Vector Store)
This command starts the assistant using the local `phi3:mini` model and the default vector store (derived from your config, e.g., `vectorstore_chunk512_overlap50`).

```bash
python src/agent/query_interface.py
```

### Query a Specific Vector Store
Use the `--vector-store` flag to point to a different vector store.

```bash
# Example for Ubuntu docs
python src/agent/query_interface.py --vector-store ubuntu_docs_pdf
```

### Query with a Different Prompt Style
Use the `--prompt` flag to change the LLM's behavior.

```bash
python src/agent/query_interface.py --vector-store ubuntu_docs_pdf --prompt detailed
```

### Query Using the OpenAI Provider
You can switch the provider and model to use a remote LLM like GPT-4. (Ensure your `OPENAI_API_KEY` is set in your `.env` file).

```bash
python src/agent/query_interface.py --provider openai --model gpt-4-turbo-preview --vector-store ubuntu_docs_pdf
```

---

## 4. Running Evaluation Scripts

There are two primary evaluation scripts to help you tune your RAG pipeline:

### 4.1. Prompt Comparison Evaluation (`prompt_comparison.py`)

This script runs a set of test queries against different prompt templates (e.g., 'base', 'detailed') using the **default PostgreSQL vector store** and your **local `phi3:mini` LLM**. It allows you to see how different prompts affect the LLM's generated answers.

```bash
python evals/prompt_comparison.py
```

*Note: This script makes multiple LLM calls and may take some time to complete, especially for the more complex prompts.*

### 4.2. Retrieval Evaluation (`retrieval_evaluation.py`)

This script helps you tune the "deterministic" portion of your RAG system by comparing different **chunking strategies** (size, overlap, method) based purely on their ability to retrieve relevant documents. It uses the `postgresql-16-A4.pdf` file and a set of PostgreSQL-specific queries.

It outputs objective metrics like **Hit Rate** and **Mean Reciprocal Rank (MRR)**.

```bash
python evals/retrieval_evaluation.py
```

*Note: This script performs many similarity searches for each chunking strategy and query, and while it doesn't involve the LLM, it can still take a significant amount of time.*
