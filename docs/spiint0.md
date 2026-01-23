# Sprint 0: Foundation & Setup - Detailed Plan

**Duration**: Week 1 (7 days)
**Goal**: Establish solid foundation with working development environment and complete data ingestion pipeline

---

## Sprint Goals

### Primary Objectives
1. Set up professional project structure and development environment
2. Implement document ingestion pipeline with multiple chunking strategies
3. Create and populate vector database with embeddings
4. Validate baseline retrieval capability
5. Establish development workflows (Git, Docker, testing)

### Success Metrics
- ✅ Complete project scaffolding with all directories and config files
- ✅ Docker environment runs successfully on first try
- ✅ Successfully ingest and chunk 100+ pages (target: 500-1000 chunks)
- ✅ Vector search returns semantically relevant results for 5 hand-crafted test queries
- ✅ All dependencies pinned and documented

---

## Day-by-Day Breakdown

### Day 1: Project Structure & Environment Setup

**Time Estimate**: 4-6 hours

#### Tasks

**1.1 Initialize Git Repository**   
```bash
# Create repo on GitHub: technical-doc-assistant
# Clone locally and set up initial structure
```

- Create repository with MIT license
- Add comprehensive .gitignore for Python
- Set up branch protection (optional for solo project)
- Initialize with README stub

**1.2 Create Project Structure**
```
technical-doc-assistant/
├── .github/
│   └── workflows/          # CI/CD (future)
├── docs/
│   ├── architecture.md
│   ├── evaluation-results.md
│   ├── mcp-protocol.md
│   └── design-decisions.md
├── data/
│   ├── raw/               # Original PDFs/documents
│   ├── processed/         # Chunked documents
│   └── sample_queries/    # Test queries
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_loader.py
│   │   ├── chunker.py
│   │   └── embedder.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── vector_store.py
│   ├── agent/
│   │   └── __init__.py
│   ├── mcp_server/
│   │   └── __init__.py
│   ├── api/
│   │   └── __init__.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logger.py
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_utils.py
├── evals/
│   ├── test_queries.json
│   ├── eval_runner.py
│   └── results/
├── notebooks/              # Jupyter for experiments
│   └── exploration.ipynb
├── scripts/
│   ├── download_docs.sh
│   ├── ingest_docs.py
│   └── test_retrieval.py
├── ui/
│   └── app.py
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml          # or requirements.txt
├── README.md
└── Makefile               # Common commands
```

**1.3 Set Up Python Environment**
**1.3 See conda_adaptation_guide.md**

- Create virtual environment: `python3.10 -m venv venv`
- Set up pyproject.toml or requirements.txt:

```toml
# pyproject.toml
[tool.poetry]
name = "technical-doc-assistant"
version = "0.1.0"
description = "RAG-based technical documentation assistant"

[tool.poetry.dependencies]
python = "^3.10"
langchain = "^0.1.0"
langchain-openai = "^0.0.5"
langchain-community = "^0.0.20"
faiss-cpu = "^1.7.4"
sentence-transformers = "^2.2.2"
pypdf = "^3.17.0"
python-dotenv = "^1.0.0"
pydantic = "^2.5.0"
pydantic-settings = "^2.1.0"
fastapi = "^0.109.0"
uvicorn = "^0.27.0"
tiktoken = "^0.5.2"
openai = "^1.10.0"

[tool.poetry.dev-dependencies]
pytest = "^7.4.0"
black = "^24.0.0"
ruff = "^0.1.0"
mypy = "^1.8.0"
jupyter = "^1.0.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

**1.4 Configure Environment Variables**

Create `.env.example`:
```bash
# LLM API Keys
OPENAI_API_KEY=sk-your-key-here

# Embedding Model
EMBEDDING_MODEL=text-embedding-3-small

# Vector Store
VECTOR_STORE_PATH=./data/vector_store
COLLECTION_NAME=tech_docs

# Application
LOG_LEVEL=INFO
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

**1.5 Set Up Configuration Management**

Create `src/utils/config.py`:
```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    vector_store_path: str = "./data/vector_store"
    collection_name: str = "tech_docs"
    log_level: str = "INFO"
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

**1.6 Set Up Logging**

Create `src/utils/logger.py`:
```python
import logging
from .config import get_settings

def setup_logger(name: str) -> logging.Logger:
    settings = get_settings()
    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger
```

**Deliverables**:
- ✅ Project structure created
- ✅ Virtual environment set up  (pip works, conda/mamba have some issues)
- ✅ Dependencies installed and locked  (req.txt )
- ✅ Configuration and logging utilities ready => tested and ready to go.

---

### Day 2: Docker Setup & Document Collection

**Time Estimate**: 4-5 hours

#### Tasks

**2.1 Create Dockerfile**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Copy application code
COPY . .

# Expose port for API
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**2.2 Create docker-compose.yml**
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./src:/app/src
    env_file:
      - .env
    depends_on:
      - vectordb
    command: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

  vectordb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma-data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE

  notebook:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/app/notebooks
      - ./data:/app/data
      - ./src:/app/src
    env_file:
      - .env
    command: jupyter notebook --ip=0.0.0.0 --allow-root --NotebookApp.token=''

volumes:
  chroma-data:
```

**2.3 Download PostgreSQL Documentation**

Create `scripts/download_docs.sh`:
```bash
#!/bin/bash

# Create data directories
mkdir -p data/raw/postgresql

# Download PostgreSQL 16 docs (HTML version for easier parsing)
wget -r -l 1 -np -nd -P data/raw/postgresql \
  https://www.postgresql.org/docs/16/

# Alternative: Download PDF version
# wget -O data/raw/postgresql/postgresql-16-A4.pdf \
#   https://www.postgresql.org/files/documentation/pdf/16/postgresql-16-A4.pdf

echo "Documentation downloaded to data/raw/postgresql/"
```

**Alternative Manual Download**:
- PostgreSQL 16 Documentation PDF: https://www.postgresql.org/files/documentation/pdf/16/postgresql-16-A4.pdf
- Save to `data/raw/postgresql/postgresql-16-A4.pdf`

**2.4 Create Makefile for Common Commands**
```makefile
.PHONY: setup install test format lint docker-build docker-up

setup:
	python -m venv venv
	. venv/bin/activate && pip install poetry
	. venv/bin/activate && poetry install

install:
	poetry install

test:
	pytest tests/ -v

format:
	black src/ tests/
	ruff check src/ tests/ --fix

lint:
	ruff check src/ tests/
	mypy src/

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

ingest:
	python scripts/ingest_docs.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf data/processed/* data/vector_store/*
```

**Deliverables**:
- ✅ Dockerfile and docker-compose or docker compose (depends upon which version) configuration
- ✅ Makefile with common commands
- ✅ PostgreSQL documentation downloaded (~3MB PDF or HTML files)
- ✅ Docker environment tested and running
```
docker compose build
[+] Building 707.2s (16/16) FINISHED                                                                                                                                    
 => [internal] load local bake definitions                                                                                                                         0.0s
 => => reading from stdin 1.01kB                                                                                                                                   0.0s
 => [app internal] load build definition from Dockerfile                                                                                                           0.0s
 => => transferring dockerfile: 1.40kB                                                                                                                             0.0s
 => [notebook internal] load metadata for docker.io/library/python:3.10-slim                                                                                       0.7s
 => [auth] library/python:pull token for registry-1.docker.io                                                                                                      0.0s
 => [notebook internal] load .dockerignore                                                                                                                         0.0s
 => => transferring context: 2B                                                                                                                                    0.0s
 => [notebook internal] load build context                                                                                                                         1.6s
 => => transferring context: 6.85MB                                                                                                                                1.5s
 => [notebook 1/6] FROM docker.io/library/python:3.10-slim@sha256:f5d029fe39146b08200bcc73595795ac19b85997ad0e5001a02c7c32e8769efa                                 0.0s
 => CACHED [notebook 2/6] WORKDIR /app                                                                                                                             0.0s
 => CACHED [notebook 3/6] RUN apt-get update && apt-get install -y     build-essential     curl     && rm -rf /var/lib/apt/lists/*                                 0.0s
 => [notebook 4/6] COPY requirements.txt .                                                                                                                         0.1s
 => [notebook 5/6] RUN pip install --no-cache-dir -r requirements.txt                                                                                            670.5s
 => [app 6/6] COPY . .                                                                                                                                            13.9s
 => [notebook] exporting to image                                                                                                                                 20.1s 
 => => exporting layers                                                                                                                                           20.0s
 => => writing image sha256:1571eac0755066c60e6a285b9dcd400133b36995980ae28ffc5eabe4a100958f                                                                       0.0s
 => => naming to docker.io/library/techdoc-genie-notebook                                                                                                          0.0s
 => [app] exporting to image                                                                                                                                      20.1s
 => => exporting layers                                                                                                                                           20.0s
 => => writing image sha256:cd2838038e1492622875b98250f6c3a3afe5101f364157a9e6ee78c7309d88c7                                                                       0.0s
 => => naming to docker.io/library/techdoc-genie-app                                                                                                               0.0s
 => [notebook] resolving provenance for metadata file                                                                                                              0.0s
 => [app] resolving provenance for metadata file                                                                                                                   0.0s
[+] Building 2/2
 ✔ techdoc-genie-app       Built                                                                                                                                   0.0s 
 ✔ techdoc-genie-notebook  Built                                                                                                                                   0.0s 
(techdoc-genie-venv) (base) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/techdoc-genie$ docker ps
CONTAINER ID   IMAGE         COMMAND                  CREATED      STATUS      PORTS                                         NAMES
f0da256e8dc7   postgres:16   "docker-entrypoint.s…"   5 days ago   Up 5 days   0.0.0.0:6432->5432/tcp, [::]:6432->5432/tcp   pythoncrud-db-1
(techdoc-genie-venv) (base) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/techdoc-genie$ docker compose up -d
[+] Running 4/4
 ✔ Network techdoc-genie_default       Created                                                                                                                     0.0s 
 ✔ Container techdoc-genie-vectordb-1  Started                                                                                                                     0.4s 
 ✔ Container techdoc-genie-notebook-1  Started                                                                                                                     0.4s 
 ✔ Container techdoc-genie-app-1       Started                                                                                                                     0.5s 
(techdoc-genie-venv) (base) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/techdoc-genie$ docker ps
CONTAINER ID   IMAGE                    COMMAND                  CREATED         STATUS         PORTS                                                   NAMES
f61db8349f5a   techdoc-genie-app        "uvicorn src.api.mai…"   5 seconds ago   Up 4 seconds   0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp             techdoc-genie-app-1
1bb876ffd092   techdoc-genie-notebook   "jupyter notebook --…"   5 seconds ago   Up 4 seconds   8000/tcp, 0.0.0.0:8888->8888/tcp, [::]:8888->8888/tcp   techdoc-genie-notebook-1
3f1ebdb0610d   chromadb/chroma:latest   "dumb-init -- chroma…"   5 seconds ago   Up 4 seconds   0.0.0.0:8001->8000/tcp, [::]:8001->8000/tcp             techdoc-genie-vectordb-1
f0da256e8dc7   postgres:16              "docker-entrypoint.s…"   5 days ago      Up 5 days      0.0.0.0:6432->5432/tcp, [::]:6432->5432/tcp             pythoncrud-db-1
(techdoc-genie-venv) (base) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/techdoc-genie$ cat data/raw/postgresql
cat: data/raw/postgresql: Is a directory
(techdoc-genie-venv) (base) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/techdoc-genie$ cat data/raw/postgresql/
cat: data/raw/postgresql/: Is a directory
(techdoc-genie-venv) (base) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/techdoc-genie$ ls data/raw/postgresql/
acronyms.html                contrib-prog.html       gist.html                logicaldecoding.html        reference-client.html       storage.html
admin.html                   custom-rmgr.html        glossary.html            logical-replication.html    reference.html              tableam.html
appendixes.html              custom-scan.html        hash-index.html          maintenance.html            reference-server.html       tablesample-method.html
appendix-obsolete.html       datatype.html           high-availability.html   managing-databases.html     regress.html                textsearch.html
archive-modules.html         datetime-appendix.html  history.html             monitoring.html             release.html                transactions.html
backup.html                  ddl.html                indexam.html             mvcc.html                   replication-origins.html    triggers.html
backup-manifest-format.html  diskusage.html          indexes.html             nls.html                    resources.html              tutorial-advanced.html
bgworker.html                dml.html                index.html               notation.html               robots.txt                  tutorial.html
biblio.html                  docguide.html           index.html.1             overview.html               rules.html                  tutorial-sql.html
bki.html                     ecpg.html               information-schema.html  parallel-query.html         runtime-config.html         tutorial-start.html
bookindex.html               errcodes-appendix.html  installation.html        performance-tips.html       runtime.html                typeconv.html
brin.html                    event-triggers.html     install-binaries.html    planner-stats-details.html  server-programming.html     user-manag.html
btree.html                   extend.html             install-windows.html     plhandler.html              source.html                 views.html
bug-reporting.html           external-projects.html  internals.html           plperl.html                 sourcerepo.html             wal.html
catalogs.html                fdwhandler.html         intro-whatis.html        plpgsql.html                spgist.html                 xplang.html
charset.html                 features.html           jit.html                 plpython.html               spi.html
client-authentication.html   functions.html          largeobjects.html        pltcl.html                  sql-commands.html
client-interfaces.html       generic-wal.html        legalnotice.html         preface.html                sql.html
color.html                   geqo.html               libpq.html               protocol.html               sql-keywords-appendix.html
contrib.html                 gin.html                limits.html              queries.html                sql-syntax.html
(techdoc-genie-venv) (base) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/techdoc-genie$ 

```
---

### Day 3: Document Loading & Parsing

**Time Estimate**: 5-6 hours

#### Tasks

**3.1 Implement Document Loader**

Create `src/ingestion/document_loader.py`:
```python
from pathlib import Path
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.schema import Document
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class DocumentLoader:
    """Load documents from various sources."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
    
    def load_pdfs(self, pattern: str = "**/*.pdf") -> List[Document]:
        """Load all PDF files matching pattern."""
        logger.info(f"Loading PDFs from {self.data_dir}")
        
        loader = DirectoryLoader(
            str(self.data_dir),
            glob=pattern,
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages")
        
        return documents
    
    def add_metadata(self, documents: List[Document]) -> List[Document]:
        """Enrich documents with metadata."""
        for doc in documents:
            # Add source file name
            if 'source' in doc.metadata:
                doc.metadata['filename'] = Path(doc.metadata['source']).name
            
            # Add document type (can be extended)
            doc.metadata['doc_type'] = 'technical_manual'
            
        return documents
    
    def load_and_prepare(self) -> List[Document]:
        """Load documents and prepare with metadata."""
        documents = self.load_pdfs()
        documents = self.add_metadata(documents)
        
        logger.info(f"Prepared {len(documents)} documents")
        return documents
```

**3.2 Create Simple Test**

Create `tests/test_ingestion.py`:
```python
import pytest
from pathlib import Path
from src.ingestion.document_loader import DocumentLoader

def test_document_loader():
    """Test basic document loading."""
    loader = DocumentLoader(data_dir="data/raw")
    
    # This test will only pass if documents exist
    documents = loader.load_and_prepare()
    
    assert len(documents) > 0, "Should load at least one document"
    assert all(hasattr(doc, 'page_content') for doc in documents)
    assert all(hasattr(doc, 'metadata') for doc in documents)

def test_metadata_enrichment():
    """Test metadata is properly added."""
    loader = DocumentLoader(data_dir="data/raw")
    documents = loader.load_and_prepare()
    
    if documents:
        assert 'filename' in documents[0].metadata
        assert 'doc_type' in documents[0].metadata
```

**Deliverables**:
- ✅ Document loader implementation
- ✅ Basic tests passing
- ✅ Successfully load PostgreSQL docs (verify with test)

```
pytest -v
========================================================================= test session starts ==========================================================================
platform linux -- Python 3.11.8, pytest-7.4.0, pluggy-1.6.0 -- /home/dell-linux-dev3/Projects/techdoc-genie/techdoc-genie-venv/bin/python3
cachedir: .pytest_cache
rootdir: /home/dell-linux-dev3/Projects/techdoc-genie
plugins: anyio-4.12.1
collected 9 items                                                                                                                                                      

tests/test_ingestion.py::test_document_loader PASSED                                                                                                             [ 11%]
tests/test_ingestion.py::test_metadata_enrichment PASSED                                                                                                         [ 22%]
tests/test_utils.py::test_settings_reads_required_openapi_key PASSED                                                                                             [ 33%]
tests/test_utils.py::test_settings_uses_default_values_when_not_overridden PASSED                                                                                [ 44%]
tests/test_utils.py::test_settings_overrides_defaults_from_env PASSED                                                                                            [ 55%]
tests/test_utils.py::test_setup_logger_returns_logger PASSED                                                                                                     [ 66%]
tests/test_utils.py::test_logger_level_respects_settings PASSED                                                                                                  [ 77%]
tests/test_utils.py::test_logger_outputs_message PASSED                                                                                                          [ 88%]
tests/test_utils.py::test_setup_logger_does_not_duplicate_handlers PASSED                                                                                        [100%]

========================================================================== 9 passed in 8.46s ===========================================================================
```
---

### Day 4: Chunking Strategies

**Time Estimate**: 6-7 hours

#### Tasks

**4.1 Implement Multiple Chunking Strategies**

Create `src/ingestion/chunker.py`:
```python
from typing import List, Dict, Literal
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain.schema import Document
from ..utils.logger import setup_logger
from ..utils.config import get_settings

logger = setup_logger(__name__)

class DocumentChunker:
    """Split documents into chunks using various strategies."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    def chunk_recursive(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """Recursive character-based chunking (default strategy)."""
        logger.info(f"Chunking with recursive strategy: size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def chunk_by_tokens(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """Token-based chunking (for precise token limits)."""
        logger.info(f"Chunking with token strategy")
        
        splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        chunks = splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def chunk_semantic(
        self,
        documents: List[Document],
        breakpoint_threshold: float = 0.5
    ) -> List[Document]:
        """Semantic chunking (placeholder for future enhancement)."""
        # For Sprint 0, we'll use recursive as fallback
        logger.warning("Semantic chunking not yet implemented, using recursive")
        return self.chunk_recursive(documents)
    
    def add_chunk_metadata(
        self,
        chunks: List[Document]
    ) -> List[Document]:
        """Add chunk-specific metadata."""
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        return chunks
    
    def chunk_documents(
        self,
        documents: List[Document],
        strategy: Literal["recursive", "token", "semantic"] = "recursive"
    ) -> List[Document]:
        """Chunk documents using specified strategy."""
        
        if strategy == "recursive":
            chunks = self.chunk_recursive(documents)
        elif strategy == "token":
            chunks = self.chunk_by_tokens(documents)
        elif strategy == "semantic":
            chunks = self.chunk_semantic(documents)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        chunks = self.add_chunk_metadata(chunks)
        
        return chunks
```

**4.2 Experiment with Chunking Parameters**

Create notebook `notebooks/chunking_experiments.ipynb`:
```python
# Cell 1: Setup
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker

loader = DocumentLoader()
documents = loader.load_and_prepare()

print(f"Loaded {len(documents)} documents")
print(f"Total characters: {sum(len(doc.page_content) for doc in documents):,}")

# Cell 2: Test different chunk sizes
chunk_sizes = [256, 512, 1024]

for size in chunk_sizes:
    chunker = DocumentChunker(chunk_size=size, chunk_overlap=50)
    chunks = chunker.chunk_documents(documents, strategy="recursive")
    
    print(f"\nChunk size: {size}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Avg chunk length: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f}")
    print(f"  Sample chunk:\n  {chunks[0].page_content[:200]}...")

# Cell 3: Analyze chunk distribution
import matplotlib.pyplot as plt

chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk_documents(documents)

chunk_lengths = [len(chunk.page_content) for chunk in chunks]

plt.figure(figsize=(10, 5))
plt.hist(chunk_lengths, bins=50)
plt.xlabel('Chunk Length (characters)')
plt.ylabel('Frequency')
plt.title('Distribution of Chunk Lengths')
plt.show()

print(f"Min: {min(chunk_lengths)}, Max: {max(chunk_lengths)}, Median: {sorted(chunk_lengths)[len(chunk_lengths)//2]}")
```

**Deliverables**:
- ✅ Chunking implementation with multiple strategies
- ✅ Notebook comparing chunking approaches (not notebook run with a local script in scripts/test_chunking or somethin like taht)
- Decision on optimal chunk size for PostgreSQL docs => pushed back to when we have llm installed and running


---

### Day 5: Embedding & Vector Store Setup

**Time Estimate**: 6-7 hours

#### Tasks

**5.1 Implement Embedder**

Create `src/ingestion/embedder.py`:
```python
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from ..utils.logger import setup_logger
from ..utils.config import get_settings

logger = setup_logger(__name__)

class DocumentEmbedder:
    """Generate embeddings for document chunks."""
    
    def __init__(self, model: str = None):
        settings = get_settings()
        self.model = model or settings.embedding_model
        
        self.embeddings = OpenAIEmbeddings(
            model=self.model,
            openai_api_key=settings.openai_api_key
        )
        
        logger.info(f"Initialized embedder with model: {self.model}")
    
    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Process in batches to avoid rate limits
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch)}/{len(texts)} texts")
        
        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embeddings.embed_query(query)
```

**5.2 Implement Vector Store**

Create `src/retrieval/vector_store.py`:
```python✅
from typing import List, Tuple, Optional
from pathlib import Path
import faiss
import pickle
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from ..utils.logger import setup_logger
from ..utils.config import get_settings

logger = setup_logger(__name__)

class VectorStore:
    """Manage vector storage and retrieval."""
    
    def __init__(self, persist_path: str = None):
        settings = get_settings()
        self.persist_path = Path(persist_path or settings.vector_store_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )
        
        self.vectorstore = None
    
    def create_from_documents(
        self,
        documents: List[Document]
    ) -> None:
        """Create vector store from documents."""
        logger.info(f"Creating vector store from {len(documents)} documents")
        
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        logger.info("Vector store created successfully")
    
    def save(self, name: str = "index") -> None:
        """Save vector store to disk."""
        if not self.vectorstore:
            raise ValueError("No vector store to save")
        
        save_path = self.persist_path / name
        self.vectorstore.save_local(str(save_path))
        
        logger.info(f"Vector store saved to {save_path}")
    
    def load(self, name: str = "index") -> None:
        """Load vector store from disk."""
        load_path = self.persist_path / name
        
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store not found at {load_path}")
        
        self.vectorstore = FAISS.load_local(
            str(load_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        logger.info(f"Vector store loaded from {load_path}")
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """Retrieve top-k similar documents."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents with similarity scores."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k
        )
        
        return results
```

**5.3 Create Ingestion Pipeline Script**

Create `scripts/ingest_docs.py`:
```python
"""
Complete ingestion pipeline: Load → Chunk → Embed → Store
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker
from src.retrieval.vector_store import VectorStore
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    """Run full ingestion pipeline."""
    logger.info("Starting document ingestion pipeline")
    
    # Step 1: Load documents
    logger.info("Step 1: Loading documents...")
    loader = DocumentLoader(data_dir="data/raw")
    documents = loader.load_and_prepare()
    logger.info(f"Loaded {len(documents)} documents")
    
    # Step 2: Chunk documents
    logger.info("Step 2: Chunking documents...")
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_documents(documents, strategy="recursive")
    logger.info(f"Created {len(chunks)} chunks")
    
    # Step 3: Create vector store
    logger.info("Step 3: Creating vector store...")
    vector_store = VectorStore()
    vector_store.create_from_documents(chunks)
    
    # Step 4: Save vector store
    logger.info("Step 4: Saving vector store...")
    vector_store.save(name="postgresql_docs")
    
    logger.info("✅ Ingestion pipeline completed successfully!")
    logger.info(f"   - Total chunks: {len(chunks)}")
    logger.info(f"   - Vector store saved to: data/vector_store/postgresql_docs")

if __name__ == "__main__":
    main()
```

**Deliverables**:
- ✅ Embedder implementation
- ✅ Vector store with FAISS
- ✅ Complete ingestion pipeline script
- ✅ Vector database populated with 500+ chunks

---

### Day 6: Basic Retrieval & Testing

**Time Estimate**: 5-6 hours

#### Tasks

**6.1 Create Retrieval Test Script**

Create `scripts/test_retrieval.py`:
```python
"""
Test vector search with sample queries
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.vector_store import VectorStore
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Test queries for PostgreSQL documentation
TEST_QUERIES = [
    "How do I create an index in PostgreSQL?",
    "What are the differences between INNER JOIN and LEFT JOIN?",
    "How do I configure connection pooling?",
    "What is MVCC and how does it work?",
    "How do I back up a PostgreSQL database?",
]

def test_retrieval():
    """Test retrieval with sample queries."""
    logger.info("Loading vector store...")
    vector_store = VectorStore()
    vector_store.load(name="postgresql_docs")
    
    logger.info(f"\nTesting with {len(TEST_QUERIES)} queries\n")
    logger.info("=" * 80)
    
    for i, query in enumerate(TEST_QUERIES, 1):
        logger.info(f"\nQuery {i}: {query}")
        logger.info("-" * 80)
        
        # Get top 3 results with scores
        results = vector_store.similarity_search_with_score(query, k=3)
        
        for j, (doc, score) in enumerate(results, 1):
            logger.info(f"\nResult {j} (Score: {score:.4f}):")
            logger.info(f"Source: {doc.metadata.get('filename', 'unknown')}")
            logger.info(f"Page: {doc.metadata.get('page', 'N/A')}")
            logger.info(f"Content preview: {doc.page_content[:200]}...")
        
        logger.info("\n" + "=" * 80)

if __name__ == "__main__":
    test_retrieval()
```

**6.2 Create Test Query Dataset**

Create `data/sample_queries/test_queries.json`:
```json
{
  "queries": [
    {
      "id": 1,
      "query": "How do I create an index in PostgreSQL?",
      "category": "DDL",
      "expected_topics": ["CREATE INDEX", "index types", "performance"]
    },
    {
      "id": 2,
      "query": "What are the differences between INNER JOIN and LEFT JOIN?",
      "category": "SQL",
      "expected_topics": ["JOIN types", "query examples", "NULL handling"]
    },
    {
      "id": 3,
      "query": "How do I configure connection pooling?",
      "category": "configuration",
      "expected_topics": ["max_connections", "pooling", "pgbouncer"]
    },
    {
      "id": 4,
      "query": "What is MVCC and how does it work?",
      "category": "architecture",
      "expected_topics": ["concurrency", "transactions", "versioning"]
    },
    {
      "id": 5,
      "query": "How do I back up a PostgreSQL database?",
      "category": "administration",
      "expected_topics": ["pg_dump", "backup strategies", "restore"]
    }
  ]
}
```

**6.3 Write Unit Tests**

Update `tests/test_retrieval.py`:
```python
import pytest
from src.retrieval.vector_store import VectorStore
from langchain.schema import Document

def test_vector_store_creation():
    """Test creating a vector store."""
    docs = [
        Document(page_content="PostgreSQL is a powerful database", metadata={"source": "test"}),
        Document(page_content="Indexes improve query performance", metadata={"source": "test"}),
    ]
    
    vector_store = VectorStore(persist_path="data/test_vector_store")
    vector_store.create_from_documents(docs)
    
    assert vector_store.vectorstore is not None

def test_similarity_search():
    """Test similarity search returns results."""
    vector_store = VectorStore()
    
    # This assumes the main vector store exists
    try:
        vector_store.load(name="postgresql_docs")
        results = vector_store.similarity_search("database backup", k=3)
        
        assert len(results) == 3
        assert all(isinstance(doc, Document) for doc in results)
    except FileNotFoundError:
        pytest.skip("Vector store not yet created")
```

**Deliverables**:
- ✅ Retrieval test script