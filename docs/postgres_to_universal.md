# Refactoring Guide: PostgreSQL → Universal Platform

## Strategy: Crawl, Walk, Run

This guide shows how to evolve your project from a working PostgreSQL assistant into a domain-agnostic platform **without rewriting everything**.

---

## Timeline

- **Sprints 0-3**: Build PostgreSQL version (focus on making it work)
- **Sprint 4**: Identify abstraction opportunities while building UI
- **Sprint 5**: Refactor for multi-domain support
- **Post-project**: Add 2nd domain as proof of concept

---

## Phase 1: PostgreSQL-Specific (Sprints 0-3)

### What to Build First

Focus on making it work **without worrying about abstraction**:

```python
# src/agent/rag_chain.py - INITIAL VERSION (PostgreSQL-specific)

class PostgreSQLRAGChain:
    """RAG chain specifically for PostgreSQL documentation."""
    
    def __init__(self):
        # Hardcoded PostgreSQL-specific values
        self.chunk_size = 512
        self.vector_store_name = "postgresql_docs"
        self.prompt = self._get_postgres_prompt()
        
    def _get_postgres_prompt(self):
        """PostgreSQL-specific prompt."""
        return """You are a PostgreSQL documentation assistant.
        Help users understand PostgreSQL features, syntax, and best practices.
        
        Context: {context}
        Question: {question}
        Answer:"""
    
    def query(self, question: str):
        # Implementation...
        pass
```

**Why this is fine:**
- ✅ Fast to build
- ✅ Easy to test
- ✅ You learn what actually matters
- ✅ Get working demo quickly

### Signs You're Ready to Refactor

During Sprints 1-3, **take notes** when you notice:
- 📝 Hardcoded values that might change per domain
- 📝 Logic that feels "PostgreSQL-specific"
- 📝 Places where you think "this should be configurable"
- 📝 Prompts that reference PostgreSQL explicitly

**Create a file: `docs/refactoring-notes.md`**

```markdown
# Refactoring Notes

## Hardcoded Values Found
- [ ] Vector store name: "postgresql_docs"
- [ ] Chunk size: 512 (works well for technical docs)
- [ ] Prompt template mentions "PostgreSQL" explicitly
- [ ] Evaluation queries are all PostgreSQL-specific
- [ ] Metadata includes PostgreSQL version numbers

## Patterns to Extract
- [ ] Prompt template structure (domain name is only variable part)
- [ ] Chunking strategy (might differ for narrative vs technical)
- [ ] Metadata schema (version, section, page are universal)

## Keep As-Is
- [ ] Vector store implementation (FAISS works for all)
- [ ] Citation system (universal)
- [ ] Evaluation framework structure (universal)
```

---

## Phase 2: Identify Abstraction Points (Sprint 4)

### What Changes Per Domain?

**Domain-Specific:**
```python
# Things that MUST change per domain
domain_name = "PostgreSQL"
data_directory = "data/raw/postgresql"
vector_collection = "postgresql_docs"
prompt_intro = "You are a PostgreSQL documentation assistant"
evaluation_queries = [specific to PostgreSQL]
```

**Domain-Agnostic:**
```python
# Things that DON'T change
vector_db_type = "FAISS"
embedding_model = "text-embedding-3-small"
llm_model = "gpt-4-turbo-preview"
citation_format = "[{id}] {filename}, Page {page}"
response_structure = {"answer": ..., "citations": ...}
```

### Create Abstraction Plan

**Document in `docs/platform-refactoring-plan.md`:**

```markdown
# Platform Refactoring Plan

## Goals
1. Support multiple domains without code changes
2. Make domain addition take <1 hour
3. Keep existing PostgreSQL functionality working

## Changes Required

### 1. Configuration Layer (2-3 hours)
- Create `config/domains.yaml`
- Add domain loader utility
- Update components to read from config

### 2. Prompt Templates (1-2 hours)
- Extract domain name from prompts
- Create template registry
- Add domain parameter to RAG chain

### 3. Directory Structure (1 hour)
- Support `data/raw/{domain}/`
- Support `data/vector_store/{domain}/`
- Update paths in code

### 4. CLI Updates (1 hour)
- Add `--domain` flag to scripts
- Add domain selection to interactive mode

### 5. Documentation (1-2 hours)
- Write domain adaptation guide
- Create setup wizard script
- Update README with multi-domain examples

Total: ~8-10 hours
```

---

## Phase 3: Refactoring Implementation (Sprint 5)

### Step 1: Create Configuration System (Start Small)

**Create `config/domains.yaml`:**

```yaml
# Start with just PostgreSQL
active_domain: postgresql

domains:
  postgresql:
    name: "PostgreSQL Documentation"
    display_name: "PostgreSQL Assistant"
    data_dir: "data/raw/postgresql"
    collection_name: "postgresql_docs"
    
    # Chunking
    chunk_size: 512
    chunk_overlap: 50
    
    # Prompting
    prompt_template: "technical_db"
    system_context: "PostgreSQL database system"
    
    # Metadata
    metadata_fields:
      - version
      - section
      - page
```

**Create `src/utils/domain_config.py`:**

```python
"""Domain configuration management."""
import yaml
from pathlib import Path
from typing import Dict, Optional
from functools import lru_cache

class DomainConfig:
    """Load and manage domain configurations."""
    
    def __init__(self, config_path: str = "config/domains.yaml"):
        self.config_path = Path(config_path)
        self._config = None
    
    @property
    def config(self) -> Dict:
        """Lazy load configuration."""
        if self._config is None:
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f)
        return self._config
    
    def get_domain(self, domain_id: Optional[str] = None) -> Dict:
        """Get configuration for a domain."""
        if domain_id is None:
            domain_id = self.config.get('active_domain', 'postgresql')
        
        domain_config = self.config['domains'].get(domain_id)
        if not domain_config:
            raise ValueError(f"Domain '{domain_id}' not found in config")
        
        # Add domain_id to the config
        domain_config['id'] = domain_id
        return domain_config
    
    def list_domains(self) -> list:
        """List all configured domains."""
        return list(self.config['domains'].keys())

@lru_cache()
def get_domain_config(domain_id: Optional[str] = None) -> Dict:
    """Get domain configuration (cached)."""
    return DomainConfig().get_domain(domain_id)
```

### Step 2: Refactor RAG Chain (Gradual Migration)

**Create new `src/agent/rag_chain_v2.py` (don't break existing):**

```python
"""Domain-agnostic RAG chain."""
from typing import Optional, Dict
from .rag_chain import RAGChain as RAGChainV1  # Original
from ..utils.domain_config import get_domain_config
from .prompts import get_prompt_template

class RAGChain(RAGChainV1):
    """
    Domain-agnostic RAG chain.
    Backwards compatible with PostgreSQL-specific version.
    """
    
    def __init__(
        self,
        vector_store,
        domain: Optional[str] = None,  # NEW: defaults to PostgreSQL
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.0,
        retrieval_k: Optional[int] = None,  # NEW: from config
        prompt_type: Optional[str] = None,   # NEW: from config
    ):
        # Load domain configuration
        self.domain_config = get_domain_config(domain)
        
        # Use config values if not explicitly provided
        if retrieval_k is None:
            retrieval_k = self.domain_config.get('retrieval_k', 5)
        
        if prompt_type is None:
            prompt_type = self.domain_config.get('prompt_template', 'base')
        
        # Call parent constructor
        super().__init__(
            vector_store=vector_store,
            model_name=model_name,
            temperature=temperature,
            retrieval_k=retrieval_k,
            prompt_type=prompt_type
        )
        
        # Override prompt with domain-specific version
        self.prompt_template = self._create_domain_prompt(prompt_type)
    
    def _create_domain_prompt(self, prompt_type: str):
        """Create prompt with domain-specific context."""
        # Get base template
        base_template = get_prompt_template(prompt_type)
        
        # Inject domain name
        template_str = base_template.template.replace(
            "technical documentation assistant",
            f"{self.domain_config['display_name']}"
        )
        
        from langchain.prompts import PromptTemplate
        return PromptTemplate(
            template=template_str,
            input_variables=base_template.input_variables
        )
```

**Why this approach works:**
- ✅ Original code still works (backwards compatible)
- ✅ New code uses config by default
- ✅ Can test new version alongside old
- ✅ Gradual migration path

### Step 3: Update Scripts (Add Domain Support)

**Update `scripts/ingest_docs.py`:**

```python
"""
Ingest documents with domain support.
BACKWARDS COMPATIBLE: Works with or without --domain flag.
"""
import argparse
from src.utils.domain_config import get_domain_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--domain',
        default=None,  # None = use active_domain from config
        help='Domain to ingest (default: active domain from config)'
    )
    args = parser.parse_args()
    
    # Load domain config
    config = get_domain_config(args.domain)
    
    logger.info(f"Ingesting documents for domain: {config['name']}")
    
    # Use config values
    loader = DocumentLoader(data_dir=config['data_dir'])
    chunker = DocumentChunker(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap']
    )
    
    # ... rest of ingestion
    
    vector_store.save(name=config['collection_name'])
    logger.info(f"✅ Ingestion complete for {config['name']}")

if __name__ == "__main__":
    main()
```

**Result:**
```bash
# Still works (uses active_domain from config)
python scripts/ingest_docs.py

# Also works (explicitly specify domain)
python scripts/ingest_docs.py --domain postgresql
```

### Step 4: Update Interactive Query Interface

**Update `src/agent/query_interface.py`:**

```python
import argparse
from src.utils.domain_config import get_domain_config, DomainConfig

def interactive_query():
    """Interactive query with domain selection."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', help='Domain to query')
    args = parser.parse_args()
    
    # Show domain selection if not specified
    if args.domain is None:
        domain_config_mgr = DomainConfig()
        domains = domain_config_mgr.list_domains()
        
        if len(domains) == 1:
            args.domain = domains[0]
        else:
            print("\nAvailable domains:")
            for i, domain in enumerate(domains, 1):
                config = get_domain_config(domain)
                print(f"  {i}. {config['name']} ({domain})")
            
            choice = input("\nSelect domain (1-{}): ".format(len(domains)))
            args.domain = domains[int(choice) - 1]
    
    # Load domain config
    config = get_domain_config(args.domain)
    
    print(f"\n{'='*80}")
    print(f"{config['display_name']}")
    print(f"{'='*80}\n")
    
    # Load vector store for this domain
    vector_store = VectorStore()
    vector_store.load(name=config['collection_name'])
    
    # Create RAG chain with domain
    from src.agent.rag_chain_v2 import RAGChain
    rag_chain = RAGChain(vector_store, domain=args.domain)
    
    # ... rest of interactive loop
```

### Step 5: Test Migration

**Create `tests/test_domain_config.py`:**

```python
"""Test domain configuration system."""
import pytest
from src.utils.domain_config import get_domain_config, DomainConfig

def test_load_postgresql_config():
    """Test loading PostgreSQL config."""
    config = get_domain_config('postgresql')
    
    assert config['id'] == 'postgresql'
    assert config['name'] == 'PostgreSQL Documentation'
    assert config['chunk_size'] == 512
    assert 'postgresql' in config['collection_name']

def test_default_domain():
    """Test default domain loads."""
    config = get_domain_config()  # No domain specified
    assert config is not None
    assert 'name' in config

def test_list_domains():
    """Test listing all domains."""
    mgr = DomainConfig()
    domains = mgr.list_domains()
    
    assert 'postgresql' in domains
    assert len(domains) >= 1
```

**Run tests:**
```bash
pytest tests/test_domain_config.py -v
```

---

## Phase 4: Add Second Domain (Validation)

### Quick Test: Add Linux Kernel Docs

**Update `config/domains.yaml`:**

```yaml
active_domain: postgresql

domains:
  postgresql:
    # ... existing config ...
  
  kernel_dev:  # NEW
    name: "Linux Kernel Development Documentation"
    display_name: "Kernel Dev Assistant"
    data_dir: "data/raw/kernel"
    collection_name: "kernel_docs"
    chunk_size: 1024  # Longer chunks for code
    chunk_overlap: 50
    prompt_template: "code_dev"
    system_context: "Linux kernel development"
    metadata_fields:
      - subsystem
      - maintainer
      - file_path
```

**Add kernel-specific prompt to `src/agent/prompts.py`:**

```python
CODE_DEV_TEMPLATE = """You are a {domain_name} assistant.
Help developers understand code, contribution processes, and best practices.

INSTRUCTIONS:
- Reference specific files or subsystems when relevant
- Include code examples from the context
- Mention maintainers or mailing lists if documented
- Use domain-specific terminology correctly

Context:
{context}

Question: {question}

Answer:"""

def get_prompt_template(template_type: str) -> PromptTemplate:
    templates = {
        "base": BASE_TEMPLATE,
        "technical_db": DETAILED_TEMPLATE,
        "code_dev": CODE_DEV_TEMPLATE,  # NEW
        # ...
    }
    # ...
```

**Test with minimal kernel docs:**

```bash
# 1. Download a few kernel doc files
mkdir -p data/raw/kernel
cd data/raw/kernel
wget https://www.kernel.org/doc/html/latest/process/submitting-patches.html

# 2. Ingest
python scripts/ingest_docs.py --domain kernel_dev

# 3. Query
python src/agent/query_interface.py --domain kernel_dev
```

**If this works, your abstraction is successful!** ✅

---

## Migration Checklist

**Before refactoring:**
- [ ] PostgreSQL version works end-to-end
- [ ] Tests pass
- [ ] Evaluation baseline established
- [ ] Code is committed to git

**During refactoring:**
- [ ] Create `config/domains.yaml`
- [ ] Add domain config loader
- [ ] Update RAG chain (keep v1 working)
- [ ] Update scripts with `--domain` flag
- [ ] Add tests for config system
- [ ] Keep PostgreSQL as active_domain

**Validation:**
- [ ] PostgreSQL still works without changes
- [ ] Scripts work with and without `--domain`
- [ ] Add second domain (even minimal)
- [ ] Second domain works with same code
- [ ] Update documentation

**Completion:**
- [ ] Rename rag_chain_v2.py → rag_chain.py (after v1 removed)
- [ ] Delete old hardcoded versions
- [ ] Update all imports
- [ ] Final test run

---

## Presentation Strategy

### In Your Application

**Show the evolution:**

> "I built this as a PostgreSQL documentation assistant first (Sprints 0-3), ensuring the core RAG pipeline worked reliably. Once I had a solid foundation with 90%+ faithfulness scores, I identified abstraction opportunities and refactored to support multiple domains (Sprint 5). 
>
> I validated the architecture by adding Linux kernel documentation as a second domain - same codebase, different configuration. This demonstrates the system can adapt to different documentation types (database reference vs. code/process docs) without code changes.
>
> For Onto Innovation, this means the same platform could handle equipment manuals, internal knowledge bases, and customer support docs across different product lines."

### In Your README

```markdown
## Architecture Evolution

**v1.0 (Sprints 0-3)**: PostgreSQL-specific implementation
- Focus: Make it work reliably
- Result: 92% faithfulness, <2s latency, complete citation system

**v2.0 (Sprint 5)**: Domain-agnostic platform
- Refactored to support multiple documentation domains
- Configuration-driven (no code changes needed)
- Validated with Linux kernel docs as second domain

**Time to add new domain**: ~30-60 minutes
```

---

## Key Benefits of This Approach

✅ **De-risks the project**: Get working version first
✅ **Learn before abstracting**: Understand actual requirements
✅ **Easier to test**: Concrete before abstract
✅ **Better abstractions**: Based on real patterns, not guesses
✅ **Impressive narrative**: Shows iterative improvement
✅ **Professional**: This is how real software evolves

---

## Timeline Summary

| Sprint | Focus | Abstraction Level |
|--------|-------|------------------|
| 0-1 | Build PostgreSQL version | None (hardcoded is fine) |
| 2-3 | Optimize and evaluate | Take notes on patterns |
| 4 | UI + identify abstractions | Plan refactoring |
| 5 | Refactor + add 2nd domain | Multi-domain support |
| Post | Document + demo | Polish presentation |

**Total time investment in abstraction: ~8-12 hours**
**Value added: Platform vs. single-purpose tool**

This is the engineering approach that will impress Onto Innovation.