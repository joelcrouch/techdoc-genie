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
python src/agent/query_interface.py  <vector_db> <model>
```

examples:
```
 python src/agent/query_interface.py --vector-store ubuntu_docs_pdf --model qwen2.5:1.5b 
2026-02-17 11:30:52,099 - __main__ - INFO - Loading vector store...
2026-02-17 11:30:54,774 - src.ingestion.embedder - INFO - Using HuggingFaceEmbedder with model all-MiniLM-L6-v2
2026-02-17 11:30:54,798 - src.retrieval.vector_store - INFO - Vector store loaded from data/vector_store/ubuntu_docs_pdf/ubuntu_docs_pdf
2026-02-17 11:30:54,798 - __main__ - INFO - Vector store 'ubuntu_docs_pdf' loaded successfully from data/vector_store/ubuntu_docs_pdf.
2026-02-17 11:30:54,798 - __main__ - INFO - Initializing RAG chain with provider 'ollama' and model 'qwen2.5:1.5b'...
2026-02-17 11:30:54,803 - src.agent.providers.ollama_provider - INFO - Successfully connected to Ollama server at http://localhost:11434
2026-02-17 11:30:54,804 - src.agent.rag_chain - INFO - Initialized RAG chain with Ollama model: qwen2.5:1.5b
2026-02-17 11:30:54,804 - src.agent.rag_chain - INFO - RAG chain fully initialized with provider 'ollama' and retriever.

================================================================================
Technical Documentation Assistant
Provider: ollama, Model: qwen2.5:1.5b, Prompt: base
Vector Store: ubuntu_docs_pdf
Type 'exit' or 'quit' to end.
================================================================================


Your question: how do i contribute to ubuntu

🔍 Searching documentation and generating answer...
2026-02-17 11:31:06,595 - src.agent.rag_chain - INFO - Processing query: how do i contribute to ubuntu
2026-02-17 11:31:12,007 - src.agent.rag_chain - INFO - Generated response with 5 sources.

================================================================================
ANSWER:
────────────────────────────────────────────────────────────────────────────────
To contribute to Ubuntu, you can start by following these steps:

1. **Pick up an existing GitHub Issue**: Find a project or issue on the Ubuntu GitHub repository that interests you and assign it to yourself. This could be anything from improving documentation to fixing bugs.

2. **Update and remove old documentation**: If you notice outdated or obsolete information in the project, submit a pull request (PR) with updates or deletions of those sections.

3. **Migrate content from Ubuntu Wiki**: Look for articles on the Ubuntu Wiki that are either outdated or no longer relevant. Migrate this content to the Read the Docs instance by following the guidelines provided there.

4. **Prepare your application template**: Use the provided template (https://wiki.ubuntu.com/Kernel/Dev/PPUApplicationTemplate) and submit it along with a brief explanation of why you want to contribute and how you plan to do so.

5. **Seek confirmation from existing members**: At least three Ubuntu kernel uploaders must confirm that they have worked with you sufficiently, assessed your skills, and verified that you meet the criteria for contributing to the project.

Remember, contributions can be as simple as fixing a typo or as complex as improving documentation or migrating content from one platform to another. The key is to find something that aligns with your interests and skills while also benefiting the community.

================================================================================
SOURCES (5):
────────────────────────────────────────────────────────────────────────────────

[1] [1] ubuntu.pdf, Page N/A
    We believe that everyone has something valuable to contribute, whether you’re a coder, a
writer, or a tester. Here’s how and why you can get involved:
•Why join us? Work with like-minded people, devel...

[2] [2] ubuntu.pdf, Page N/A
    autolinked-references-and-urls
19https://www.conventionalcommits.org/
19 of 55
Startcontributing
If you are ready to contribute but unsure where to start, here are some suggested starting
points.
1.Pi...

[3] [3] ubuntu.pdf, Page N/A
    2.Update and remove old documentation.
If you browse through the project and find information or whole pages that are either
outdated or obsolete, submit a PR with changes to update or delete them.
3....

[4] [4] ubuntu.pdf, Page N/A
    Applicationtemplate
If you are interested in joining, start by preparing your application using the following tem-
plate:
https://wiki.ubuntu.com/Kernel/Dev/PPUApplicationTemplate
An example applicati...

[5] [5] ubuntu.pdf, Page N/A
    # Project and community
Section: Project and community

Kernel documentation is a member of the Ubuntu family. It’s an open source documentation
project that warmly welcomes community contributions, s...

==========================================================
```

deFAULT
```
python src/agent/query_interface.py
2026-02-17 11:22:16,023 - __main__ - INFO - Loading vector store...
2026-02-17 11:22:18,557 - src.ingestion.embedder - INFO - Using HuggingFaceEmbedder with model all-MiniLM-L6-v2
2026-02-17 11:22:18,851 - src.retrieval.vector_store - INFO - Vector store loaded from data/vector_store/vectorstore_chunk512_overlap50/vectorstore_chunk512_overlap50
2026-02-17 11:22:18,851 - __main__ - INFO - Vector store 'vectorstore_chunk512_overlap50' loaded successfully from data/vector_store/vectorstore_chunk512_overlap50.
2026-02-17 11:22:18,851 - __main__ - INFO - Initializing RAG chain with provider 'ollama' and model 'phi3:mini'...
2026-02-17 11:22:18,857 - src.agent.providers.ollama_provider - INFO - Successfully connected to Ollama server at http://localhost:11434
2026-02-17 11:22:18,858 - src.agent.rag_chain - INFO - Initialized RAG chain with Ollama model: phi3:mini
2026-02-17 11:22:18,858 - src.agent.rag_chain - INFO - RAG chain fully initialized with provider 'ollama' and retriever.

================================================================================
Technical Documentation Assistant
Provider: ollama, Model: phi3:mini, Prompt: base
Vector Store: vectorstore_chunk512_overlap50
Type 'exit' or 'quit' to end.
================================================================================


Your question: how do i make a table

🔍 Searching documentation and generating answer...
2026-02-17 11:22:41,817 - src.agent.rag_chain - INFO - Processing query: how do i make a table
2026-02-17 11:22:49,330 - src.agent.rag_chain - INFO - Generated response with 5 sources.

================================================================================
ANSWER:
────────────────────────────────────────────────────────────────────────────────
To create a table in PostgreSQL, you can use the `CREATE TABLE` statement. Here is an example of syntax for creating a simple table named 'employees':

```sql
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100) UNIQUE NOT NULL,
    hire_date DATE NOT NULL
);
```
In this example:
- `id` is an auto-incrementing integer that serves as the primary key.
- `first_name` and `last_name` are variable character fields with a maximum length of 50 characters each, representing employee names.
- `email` is also a variable character field but has a larger limit (100) to accommodate different email formats; it's unique for every record in the table and cannot be null.
- `hire_date` represents when an employee was hired as a date, which must not be empty or null.

================================================================================
SOURCES (5):
────────────────────────────────────────────────────────────────────────────────

[1] [1] postgresql-16-A4.pdf, Page N/A
    # CREATE TABLE
Section: CREATE TABLE

[2] [2] postgresql-16-A4.pdf, Page N/A
    # CREATE TABLE
Section: CREATE TABLE

[3] [3] postgresql-16-A4.pdf, Page N/A
    # CREATE TABLE
Section: CREATE TABLE

[4] [4] postgresql-16-A4.pdf, Page N/A
    # CREATE TABLE
Section: CREATE TABLE

[5] [5] postgresql-16-A4.pdf, Page N/A
    # CREATE TABLE
Section: CREATE TABLE

================================================================================


```

Openai example
```
 python src/agent/query_interface.py --provider gemini --model gemini-2.5-flash-lite
2026-02-17 11:34:28,162 - __main__ - INFO - Loading vector store...
2026-02-17 11:34:30,807 - src.ingestion.embedder - INFO - Using HuggingFaceEmbedder with model all-MiniLM-L6-v2
2026-02-17 11:34:31,111 - src.retrieval.vector_store - INFO - Vector store loaded from data/vector_store/vectorstore_chunk512_overlap50/vectorstore_chunk512_overlap50
2026-02-17 11:34:31,112 - __main__ - INFO - Vector store 'vectorstore_chunk512_overlap50' loaded successfully from data/vector_store/vectorstore_chunk512_overlap50.
2026-02-17 11:34:31,112 - __main__ - INFO - Initializing RAG chain with provider 'gemini' and model 'gemini-2.5-flash-lite'...
2026-02-17 11:34:31,112 - src.agent.rag_chain - INFO - Initialized RAG chain with Gemini model: gemini-2.5-flash-lite
2026-02-17 11:34:31,112 - src.agent.rag_chain - INFO - RAG chain fully initialized with provider 'gemini' and retriever.

================================================================================
Technical Documentation Assistant
Provider: gemini, Model: gemini-2.5-flash-lite, Prompt: base
Vector Store: vectorstore_chunk512_overlap50
Type 'exit' or 'quit' to end.
================================================================================


Your question: difference between a left join and middle join

🔍 Searching documentation and generating answer...
2026-02-17 11:34:51,229 - src.agent.rag_chain - INFO - Processing query: difference between a left join and middle join
2026-02-17 11:34:53,440 - src.agent.rag_chain - INFO - Generated response with 5 sources.

================================================================================
ANSWER:
────────────────────────────────────────────────────────────────────────────────
I don't have enough information

================================================================================
SOURCES (5):
────────────────────────────────────────────────────────────────────────────────

[1] [1] postgresql-16-A4.pdf, Page N/A
    of the joined table by inserting null values for the right-hand columns. Note that only the JOIN
clause's own condition is considered while deciding which rows have matches. Outer conditions
are appli...

[2] [2] postgresql-16-A4.pdf, Page N/A
    binds more tightly than the commas separating FROM-list items. All the JOIN options are just a
notational convenience, since they do nothing you couldn't do with plain FROM and WHERE.
LEFT OUTER JOIN ...

[3] [3] postgresql-16-A4.pdf, Page N/A
    # RIGHT OUTER JOIN
Section: RIGHT OUTER JOIN

First, an inner join is performed. Then, for each row in T2 that does not satisfy the join
condition with any row in T1, a joined row is added with null v...

[4] [4] postgresql-16-A4.pdf, Page N/A
    JOIN binds more tightly than comma. For example FROM T1 CROSS JOIN T2
INNER JOIN T3 ON condition is not the same as FROM T1, T2 INNER JOIN
115
Queries
T3 ON condition because the condition can referen...

[5] [5] postgresql-16-A4.pdf, Page N/A
    FULL OUTER JOIN returns all the joined rows, plus one row for each unmatched left-hand row
(extended with nulls on the right), plus one row for each unmatched right-hand row (extended
with nulls on th...

================================================================================
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
