# TechDoc Genie - Project Vision & Roadmap

## 🎯 Core Vision

Build a flexible, MLOps-friendly platform that makes it easy to create and optimize RAG-powered chatbots from any technical documentation source.

**Key Philosophy:** If the backend is solid, frontend integration becomes straightforward.

---

## Current Architecture

### What We Have Today

- **Document Ingestion Pipeline**
  - Multi-format support (HTML, PDF)
  - Configurable chunking strategies (recursive, semantic)
  - Multiple embedding providers (HuggingFace, Gemini, OpenAI)

- **Vector Store & Retrieval**
  - FAISS-based vector store
  - Efficient similarity search

- **LLM Integration**
  - Abstract interface for multiple providers
  - Support for Claude, Gemini, OpenAI
  - Easy to add new LLMs

- **Application Layer**
  - FastAPI backend
  - Web UI for interactive access

- **DevOps Foundation**
  - Testing infrastructure with pytest
  - Code quality tools (black, ruff, mypy)
  - Docker containerization
  - GitHub Actions CI/CD

---

##  Immediate Next Steps

### 1. CLI Refactoring & Enhancement
- [ ] Consolidate operational scripts into unified `techdoc-genie` CLI
- [ ] Implement core commands:
  - `techdoc-genie data download --source-url <url>`
  - `techdoc-genie ingest process --docs-dir <path>`
  - `techdoc-genie vectorstore build --config <strategy>`
  - `techdoc-genie eval run --test-set <path>`
- [ ] Make commands idempotent for automation
- [ ] Add structured output options (JSON) for programmatic use
- [ ] Standardize exit codes for pipeline integration

### 2. Experiment & Evaluation Framework
- [ ] Create declarative experiment configuration (YAML)  <---this has been peculating for a while in my brain
- [ ] Integrate experiment tracking (MLflow, Weights & Biases,<ithinkg mlfow/w&b do stuff like this but ill probalby jsut use my own homebrew> or custom)
- [ ] Expand evaluation metrics beyond basic retrieval
- [ ] Add RAG-specific metrics (RAGAS framework)
- [ ] Implement experiment comparison tooling
- [ ] Consider DVC for data versioning

### 3. Configuration Management
- [ ] Centralize all configurations in `src/utils/config.py`
- [ ] Implement layered configuration (defaults → files → env vars)
- [ ] Create `.env.example` for documentation
- [ ] Plan secrets management for production (Vault, K8s Secrets)

---

## Medium-Term Goals

### Documentation Processing
- [ ] Add support for more formats:
  - [ ] Markdown files
  - [ ] Confluence pages
  - [ ] Notion exports
  - [ ] GitHub repositories
  - [ ] Generic web scraping

### Experimentation Tools
- [ ] Build experiment definition system
- [ ] Create metrics visualization
- [ ] Develop automated A/B testing for configurations
- [ ] Add benchmark datasets for different doc types

### Platform Robustness
- [ ] Optimize Docker configurations
- [ ] Create docker-compose for full stack orchestration
- [ ] Add telemetry and metrics collection (Prometheus compatible)
- [ ] Implement structured logging throughout
- [ ] Consider distributed tracing (OpenTelemetry)

---

## Senior-Level Engineering Priorities

### Architectural Excellence
- [ ] **Elevate CLI to Platform Orchestration API**
  - Enable automation with external schedulers (Airflow, Kubeflow)
  - Support CI/CD pipeline integration
  - Design for unattended operation

- [ ] **Reproducible R&D Framework**
  - Version everything: data, code, models, configs
  - Track all experiment parameters
  - Enable reliable rollbacks
  - Implement scientific method for RAG development

- [ ] **Environment-Agnostic Design**
  - Work seamlessly across dev/staging/production
  - Prevent configuration drift
  - Support multiple client deployments

- [ ] **Extensible Modularity**
  - Strengthen provider interfaces
  - Maintain loose coupling between components
  - Easy to add new chunking algorithms
  - Easy to swap vector stores
  - Easy to integrate new LLMs

- [ ] **Production-Grade Observability**
  - Monitor retrieval latency
  - Track embedding generation time
  - Measure LLM call duration and token usage
  - Enable troubleshooting with request tracing

---

## Frontend Integration (Future)

### API Exposure
- [ ] Design clean, well-documented REST API
- [ ] Provide clear endpoint specifications
- [ ] Enable easy widget integration
- [ ] Support standard chat interface patterns

**Note:** Frontend implementation is out of scope for current focus, but a solid backend will make this straightforward for frontend teams.

---

## Documentation Needs

- [ ] User-facing platform documentation (MkDocs/Sphinx)
- [ ] API documentation (auto-generated from code)
- [ ] Configuration guide for MLOps engineers
- [ ] Experiment setup tutorials
- [ ] Integration examples for new LLMs/embedders
- [ ] Troubleshooting guide

---

##  Validation Strategy

### Testing Different Document Sets
- [ ] Upload Ubuntu kernel documentation
- [ ] Test with Ubuntu contribution guides
- [ ] Try nonprofit organization documents
- [ ] Identify pain points in each domain
- [ ] Document lessons learned

### Performance Benchmarking
- [ ] Establish baseline metrics
- [ ] Compare chunking strategies empirically
- [ ] Evaluate different embedding models
- [ ] Measure retrieval quality across doc types

---

## 💡 Key Principles

1. **MLOps-First Design** - Build for the engineers who will operate and optimize the system
2. **Experimentation-Friendly** - Make it easy to test new approaches and configurations
3. **Reproducibility** - Every experiment should be fully reproducible
4. **Modularity** - Components should be swappable without major refactoring
5. **Observability** - Always know what's happening in production
6. **Automation** - Minimize manual intervention in workflows

---

##  Notes

- Primary user: **MLOps engineers** building and optimizing RAG systems
- Secondary outcome: Easy frontend integration through clean API design
- Current focus: Backend solidity over feature breadth
- Approach: Build with real documentation sets to discover pain points