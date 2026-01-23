# Domain Adaptation Guide: Universal Documentation Assistant

## Overview

This RAG system is designed as a **domain-agnostic framework**. The core architecture remains the same whether you're working with PostgreSQL docs, Linux kernel development guides, non-profit policies, medical protocols, or corporate knowledge bases. This document explains how to adapt the system to any documentation source.

---

## Core Architecture (Universal)

These components remain unchanged across domains:

```
Document Source → Ingestion → Chunking → Embedding → Vector Store → Retrieval → LLM → Response
```

**What stays the same:**
- Document loading pipeline
- Vector database infrastructure
- RAG chain mechanics
- Citation system
- Evaluation framework structure
- MCP server protocol
- API endpoints
- UI/UX patterns

**What gets customized:**
- Source documents
- Chunking strategy parameters
- Prompt templates (domain vocabulary)
- Evaluation queries
- Metadata schema
- Domain-specific post-processing

---

## Adaptation Checklist

When switching to a new domain, follow this checklist:

### ✅ Phase 1: Data Preparation (2-3 hours)

**1.1 Identify Documentation Sources**
- [ ] Locate primary documentation (PDFs, HTML, Markdown, Word docs)
- [ ] Identify supplementary materials (wikis, FAQs, policies)
- [ ] List any structured data (spreadsheets, databases)
- [ ] Note access requirements (public vs. internal)

**1.2 Download & Organize**
- [ ] Download all documentation to `data/raw/<domain_name>/`
- [ ] Organize by document type if needed
- [ ] Document source URLs and versions
- [ ] Create `data/raw/<domain_name>/README.md` with source info

**1.3 Assess Document Characteristics**
- [ ] Document types: PDF, HTML, Markdown, Word, etc.
- [ ] Average document length
- [ ] Structure: highly structured vs. narrative
- [ ] Technical level: beginner, intermediate, expert
- [ ] Special content: code, diagrams, tables, forms

### ✅ Phase 2: Chunking Strategy (1-2 hours)

Different documentation types need different chunking approaches:

**Technical Reference (PostgreSQL, API Docs)**
```python
chunk_size = 512
chunk_overlap = 50
strategy = "recursive"  # Good for structured technical content
separators = ["\n\n", "\n", ". ", " ", ""]
```

**Narrative Documentation (Policies, Guidelines)**
```python
chunk_size = 768
chunk_overlap = 100
strategy = "recursive"  # Preserve context in longer passages
separators = ["\n\n\n", "\n\n", "\n", ". ", " "]
```

**Code-Heavy Documentation (Kernel Docs, SDK Guides)**
```python
chunk_size = 1024
chunk_overlap = 50
strategy = "recursive"
separators = ["\n\n", "\nclass ", "\ndef ", "\n\n", "\n", " "]
```

**Legal/Compliance (Non-profit Policies, Regulations)**
```python
chunk_size = 512
chunk_overlap = 100
strategy = "recursive"  # Maintain numbered sections
preserve_section_numbers = True
```

**Run experiments in notebook:**
```python
# notebooks/chunking_experiments.ipynb
from src.ingestion.chunker import DocumentChunker

# Test different parameters
configs = [
    {"size": 512, "overlap": 50},
    {"size": 768, "overlap": 100},
    {"size": 1024, "overlap": 50},
]

for config in configs:
    chunker = DocumentChunker(chunk_size=config["size"], chunk_overlap=config["overlap"])
    chunks = chunker.chunk_documents(documents)
    # Analyze distribution, sample chunks
```

### ✅ Phase 3: Prompt Engineering (2-4 hours)

Create domain-specific prompt templates:

**Example: Linux Kernel Development**
```python
KERNEL_DEV_TEMPLATE = """You are an expert Linux kernel development assistant. Help developers understand kernel subsystems, coding standards, and contribution processes.

INSTRUCTIONS:
- Reference specific kernel subsystems when relevant
- Cite Documentation/* paths when available
- Mention relevant maintainers or mailing lists if documented
- Use kernel-specific terminology correctly (e.g., "commit", "patch series", "tree")
- Highlight coding style requirements when discussing code

Context:
{context}

Question: {question}

Answer:"""
```

**Example: Non-Profit Policies**
```python
NONPROFIT_POLICY_TEMPLATE = """You are a helpful assistant for [Organization Name] staff and volunteers. Provide clear, accurate information from our official policies and procedures.

INSTRUCTIONS:
- Be clear and accessible (avoid jargon when possible)
- Always cite the specific policy document and section
- If a question involves compliance or legal matters, emphasize consulting leadership
- Distinguish between required policies and recommended guidelines
- Mention effective dates if policies have changed recently

Context:
{context}

Question: {question}

Answer:"""
```

**Example: Medical Protocols**
```python
MEDICAL_PROTOCOL_TEMPLATE = """You are a clinical protocol assistant. Provide accurate information from approved medical protocols and guidelines.

CRITICAL INSTRUCTIONS:
- Only reference information explicitly stated in the protocols
- Always cite protocol name, version, and page number
- Never extrapolate or provide medical advice beyond the documented protocols
- If information is not in the protocols, clearly state this
- Highlight any warnings, contraindications, or critical steps

Context:
{context}

Question: {question}

Protocol Reference:"""
```

**Template Selection Guide:**

| Domain | Tone | Citation Style | Special Considerations |
|--------|------|----------------|------------------------|
| Technical Docs | Precise, technical | Section/page numbers | Include code examples |
| Corporate Policies | Professional, clear | Policy name + section | Effective dates important |
| Medical/Healthcare | Cautious, precise | Protocol + version | Never extrapolate |
| Legal/Compliance | Formal, exact | Document + clause | Verbatim quotes often needed |
| Educational | Supportive, clear | Chapter/lesson | Multiple explanation styles |
| Customer Support | Friendly, helpful | Article/KB number | Action-oriented |

### ✅ Phase 4: Metadata Schema (1 hour)

Design domain-specific metadata:

**Technical Documentation:**
```python
metadata = {
    'filename': 'postgresql-16-manual.pdf',
    'page': 147,
    'section': '9.5 String Functions',
    'version': '16.0',
    'doc_type': 'reference',
    'last_updated': '2023-09-14'
}
```

**Corporate/Non-Profit:**
```python
metadata = {
    'filename': 'volunteer-handbook.pdf',
    'page': 12,
    'policy_number': 'POL-2024-003',
    'department': 'Human Resources',
    'effective_date': '2024-01-01',
    'revision': '2.1',
    'approver': 'Board of Directors'
}
```

**Code/Development:**
```python
metadata = {
    'filename': 'kernel-dev-process.rst',
    'section': 'Submitting Patches',
    'subsystem': 'networking',
    'maintainer': 'netdev@vger.kernel.org',
    'kernel_version': '6.5',
    'last_commit': 'a1b2c3d'
}
```

**Update document loader:**
```python
# src/ingestion/document_loader.py

def add_domain_metadata(self, documents: List[Document], domain: str) -> List[Document]:
    """Add domain-specific metadata."""
    
    if domain == "kernel_dev":
        return self._add_kernel_metadata(documents)
    elif domain == "nonprofit":
        return self._add_nonprofit_metadata(documents)
    # ... etc
    
    return documents
```

### ✅ Phase 5: Evaluation Dataset (3-4 hours)

Create domain-specific test queries:

**Structure per domain:**
```json
{
  "domain": "linux_kernel_development",
  "queries": [
    {
      "id": "q001",
      "query": "How do I submit a patch to the networking subsystem?",
      "category": "contribution_process",
      "difficulty": "medium",
      "expected_topics": ["patch format", "netdev mailing list", "maintainer tree"],
      "ground_truth": "Submit patches to netdev@vger.kernel.org following format in Documentation/process/submitting-patches.rst",
      "requires_code": false,
      "requires_procedure": true
    }
  ]
}
```

**Query Categories by Domain:**

**Technical Documentation:**
- How-to questions
- Troubleshooting scenarios
- Configuration questions
- Performance optimization
- Feature comparisons

**Policy/Compliance:**
- Eligibility questions
- Procedure lookups
- Requirement clarifications
- Timeline questions
- Exception processes

**Development/Code:**
- Best practices
- API usage
- Integration patterns
- Debugging approaches
- Tool usage

**Medical/Healthcare:**
- Protocol steps
- Dosage information
- Contraindications
- Emergency procedures
- Documentation requirements

### ✅ Phase 6: Domain-Specific Features (Optional, 2-6 hours)

Add specialized functionality:

**For Code Documentation:**
```python
# src/agent/code_extractor.py

class CodeExtractor:
    """Extract and format code examples from responses."""
    
    def extract_code_blocks(self, response: str) -> List[Dict]:
        """Find code blocks in markdown."""
        # Return formatted code with syntax highlighting
        pass
    
    def validate_code_syntax(self, code: str, language: str) -> bool:
        """Basic syntax validation."""
        pass
```

**For Compliance/Policy:**
```python
# src/agent/version_tracker.py

class PolicyVersionTracker:
    """Track policy versions and effective dates."""
    
    def get_current_policy(self, policy_name: str) -> Dict:
        """Return current version of a policy."""
        pass
    
    def check_superseded(self, policy_id: str) -> bool:
        """Check if policy has been superseded."""
        pass
```

**For Medical/Healthcare:**
```python
# src/agent/safety_checker.py

class ClinicalSafetyChecker:
    """Add extra safety checks for medical content."""
    
    def verify_protocol_version(self, protocol: str) -> bool:
        """Ensure using current protocol version."""
        pass
    
    def flag_critical_info(self, response: str) -> List[str]:
        """Identify warnings, contraindications."""
        pass
```

---

## Domain-Specific Configuration File

Create `config/domains.yaml`:

```yaml
domains:
  postgresql:
    name: "PostgreSQL Documentation Assistant"
    data_dir: "data/raw/postgresql"
    chunk_size: 512
    chunk_overlap: 50
    prompt_template: "technical"
    collection_name: "postgresql_docs"
    metadata_fields:
      - version
      - section
      - page
    
  kernel_dev:
    name: "Linux Kernel Development Assistant"
    data_dir: "data/raw/kernel"
    chunk_size: 1024
    chunk_overlap: 50
    prompt_template: "kernel_dev"
    collection_name: "kernel_docs"
    metadata_fields:
      - subsystem
      - maintainer
      - kernel_version
  
  nonprofit_policies:
    name: "Organization Policy Assistant"
    data_dir: "data/raw/policies"
    chunk_size: 512
    chunk_overlap: 100
    prompt_template: "nonprofit"
    collection_name: "policy_docs"
    metadata_fields:
      - policy_number
      - effective_date
      - department
      - approver
```

**Load configuration:**
```python
# src/utils/domain_config.py

import yaml
from pathlib import Path

class DomainConfig:
    """Load domain-specific configuration."""
    
    def __init__(self, config_path: str = "config/domains.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
    
    def get_domain_config(self, domain: str) -> dict:
        """Get configuration for specific domain."""
        return self.config['domains'].get(domain, {})
    
    def list_domains(self) -> List[str]:
        """List all configured domains."""
        return list(self.config['domains'].keys())
```

---

## Multi-Domain Setup Script

Create `scripts/setup_new_domain.py`:

```python
#!/usr/bin/env python3
"""
Setup script for adding a new domain to the RAG system.
"""

import sys
from pathlib import Path
import questionary

def setup_new_domain():
    """Interactive setup for new domain."""
    
    print("\n" + "="*80)
    print("New Domain Setup Wizard")
    print("="*80 + "\n")
    
    # Gather information
    domain_id = questionary.text(
        "Domain ID (e.g., 'kernel_dev', 'nonprofit_policies'):"
    ).ask()
    
    domain_name = questionary.text(
        "Full name (e.g., 'Linux Kernel Development Docs'):"
    ).ask()
    
    doc_type = questionary.select(
        "Primary document type:",
        choices=["PDF", "HTML", "Markdown", "Word", "Mixed"]
    ).ask()
    
    content_type = questionary.select(
        "Content type:",
        choices=[
            "Technical Reference",
            "Code Documentation",
            "Policy/Compliance",
            "Medical/Healthcare",
            "Educational",
            "Other"
        ]
    ).ask()
    
    # Create directory structure
    data_dir = Path(f"data/raw/{domain_id}")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create README
    readme_content = f"""# {domain_name}

## Source Information
- Domain ID: {domain_id}
- Document Type: {doc_type}
- Content Type: {content_type}

## Documents
Place source documents in this directory.

## Notes
- Date added: [DATE]
- Primary source: [URL or description]
- Update frequency: [How often docs change]
"""
    
    (data_dir / "README.md").write_text(readme_content)
    
    # Create evaluation template
    eval_template = {
        "domain": domain_id,
        "description": f"Evaluation queries for {domain_name}",
        "queries": [
            {
                "id": "q001",
                "query": "[Sample query for this domain]",
                "category": "[category]",
                "difficulty": "easy",
                "expected_topics": [],
                "ground_truth": ""
            }
        ]
    }
    
    import json
    eval_path = Path(f"evals/{domain_id}_queries.json")
    eval_path.write_text(json.dumps(eval_template, indent=2))
    
    print(f"\n✅ Domain setup complete!")
    print(f"\nNext steps:")
    print(f"1. Add documents to: {data_dir}")
    print(f"2. Update config/domains.yaml with domain settings")
    print(f"3. Create domain-specific prompt in src/agent/prompts.py")
    print(f"4. Run: python scripts/ingest_docs.py --domain {domain_id}")
    print(f"5. Test with: python src/agent/query_interface.py --domain {domain_id}")

if __name__ == "__main__":
    setup_new_domain()
```

---

## Domain Switching in Code

Update main components to support multiple domains:

```python
# src/agent/rag_chain.py

class RAGChain:
    def __init__(
        self,
        vector_store: VectorStore,
        domain: str = "postgresql",  # NEW parameter
        **kwargs
    ):
        self.domain = domain
        
        # Load domain configuration
        from ..utils.domain_config import DomainConfig
        domain_config = DomainConfig()
        self.config = domain_config.get_domain_config(domain)
        
        # Use domain-specific prompt
        prompt_type = self.config.get('prompt_template', 'base')
        self.prompt_template = get_prompt_template(prompt_type)
        
        # ... rest of initialization
```

```python
# src/agent/query_interface.py

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--domain',
        default='postgresql',
        help='Domain to query (e.g., postgresql, kernel_dev, nonprofit)'
    )
    args = parser.parse_args()
    
    # Load domain-specific vector store
    vector_store = VectorStore()
    collection_name = f"{args.domain}_docs"
    vector_store.load(name=collection_name)
    
    # Create RAG chain with domain
    rag_chain = RAGChain(vector_store, domain=args.domain)
    
    # ... rest of interface
```

---

## Real-World Domain Examples

### Example 1: Ubuntu Community Documentation

**Characteristics:**
- Mix of technical guides and community policies
- Multiple documentation formats (wiki, markdown, man pages)
- Needs to handle both user and developer queries

**Adaptations:**
```python
# Chunking
chunk_size = 768  # Longer for tutorial-style content
chunk_overlap = 100

# Prompt
UBUNTU_TEMPLATE = """You are the Ubuntu Documentation Assistant...
- Distinguish between LTS and regular releases
- Link to relevant Ubuntu wiki pages
- Mention package names and PPAs when relevant
"""

# Metadata
metadata_fields = ['release_version', 'wiki_page', 'last_updated', 'category']
```

### Example 2: Non-Profit Grant Guidelines

**Characteristics:**
- Policy documents with specific eligibility criteria
- Application procedures and deadlines
- Budget templates and requirements

**Adaptations:**
```python
# Chunking
chunk_size = 512  # Preserve numbered requirements
preserve_formatting = True

# Prompt
GRANT_TEMPLATE = """You assist with [Org Name] grant application process...
- Always cite policy effective dates
- Highlight eligibility requirements
- Mention deadlines prominently
- Clarify between 'required' and 'recommended'
"""

# Special features
- Add date validation (check if grant cycle is open)
- Extract dollar amounts and highlight them
- Link to application forms
```

### Example 3: Medical Clinical Protocols

**Characteristics:**
- Step-by-step procedures
- Dosage calculations
- Emergency protocols

**Adaptations:**
```python
# Chunking
chunk_size = 1024  # Keep full procedures together
preserve_steps = True

# Prompt
CLINICAL_TEMPLATE = """You are a clinical protocol reference assistant...
CRITICAL: Only provide information explicitly in protocols
- Always cite protocol name and version
- Highlight warnings and contraindications
- Include all steps in order
- Never extrapolate or infer
"""

# Safety features
- Version checking (ensure current protocol)
- Highlight critical warnings
- Require explicit confirmation for high-risk info
- Log all queries for audit
```

---

## Quick Reference: 30-Minute Domain Adaptation

**For a new domain, minimum steps:**

1. **Create directory** (2 min)
   ```bash
   mkdir -p data/raw/my_domain
   ```

2. **Add documents** (5 min)
   - Copy PDFs/docs to directory

3. **Create prompt** (10 min)
   - Add to `src/agent/prompts.py`
   - Use existing template as base

4. **Ingest** (5 min)
   ```bash
   python scripts/ingest_docs.py --domain my_domain
   ```

5. **Test** (8 min)
   ```python
   python src/agent/query_interface.py --domain my_domain
   ```

**You now have a working system!** 

Optimization and evaluation can happen iteratively.

---

## Benefits of This Architecture

✅ **Reusable**: Same codebase for any documentation domain
✅ **Scalable**: Add new domains without touching core logic
✅ **Maintainable**: Domain configs separate from code
✅ **Testable**: Same evaluation framework across domains
✅ **Flexible**: Easy to customize per domain needs
✅ **Professional**: Shows software engineering thinking

This makes your portfolio project even more impressive - it's not just a PostgreSQL doc assistant, it's a **universal documentation intelligence platform**.