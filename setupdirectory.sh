#!/bin/bash

# Create directories
mkdir -p .github/workflows
mkdir -p docs
mkdir -p data/raw data/processed data/sample_queries
mkdir -p src/ingestion src/retrieval src/agent src/mcp_server src/api src/utils
mkdir -p tests
mkdir -p evals/results
mkdir -p notebooks
mkdir -p scripts
mkdir -p ui

# Create files
touch docs/architecture.md docs/evaluation-results.md docs/mcp-protocol.md docs/design-decisions.md
touch src/__init__.py
touch src/ingestion/__init__.py src/ingestion/document_loader.py src/ingestion/chunker.py src/ingestion/embedder.py
touch src/retrieval/__init__.py src/retrieval/vector_store.py
touch src/agent/__init__.py
touch src/mcp_server/__init__.py
touch src/api/__init__.py
touch src/utils/__init__.py src/utils/config.py src/utils/logger.py
touch tests/test_ingestion.py tests/test_retrieval.py tests/test_utils.py
touch evals/test_queries.json evals/eval_runner.py
touch notebooks/exploration.ipynb
touch scripts/download_docs.sh scripts/ingest_docs.py scripts/test_retrieval.py
touch ui/app.py
touch .env.example .gitignore docker-compose.yml Dockerfile pyproject.toml README.md Makefile

echo "Project structure created successfully!"