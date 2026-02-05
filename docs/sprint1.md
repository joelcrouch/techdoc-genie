# Sprint 1: Core RAG Pipeline - Detailed Plan

**Duration**: Week 2 (7 days)
**Goal**: Build complete RAG pipeline with LLM integration, citations, and initial evaluation framework

---

## Sprint Goals

### Primary Objectives
1. Integrate LangChain with OpenAI GPT-4 /local llm for answer generation
2. Implement citation system linking answers to source documents
3. Create prompt templates for grounded, accurate responses
4. Build initial evaluation dataset with ground truth answers
5. Optimize retrieval and prompt engineering through iteration
6. Establish baseline quality metrics

### Success Metrics
- ✅ Generate coherent answers with citations for 20 test queries
- ✅ Achieve >80% manual evaluation score for answer relevance
- ✅ Response latency <5 seconds for cold queries
- ✅ Zero hallucinations in responses (all claims cited)
- ✅ Complete prompt engineering documentation

---

## Day-by-Day Breakdown

### Day 1: LLM Integration & Basic RAG

**Time Estimate**: 6-7 hours

#### Tasks

**1.1 Create RAG Chain Implementation**

Create `src/agent/rag_chain.py`:
```python
from typing import List, Dict, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from ..retrieval.vector_store import VectorStore
from ..utils.logger import setup_logger
from ..utils.config import get_settings

logger = setup_logger(__name__)

class RAGChain:
    """Retrieval-Augmented Generation chain for answering queries."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.0,
        retrieval_k: int = 5
    ):
        self.vector_store = vector_store
        self.retrieval_k = retrieval_k
        
        # Initialize LLM
        settings = get_settings()
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=settings.openai_api_key
        )
        
        # Create retriever
        self.retriever = self.vector_store.vectorstore.as_retriever(
            search_kwargs={"k": retrieval_k}
        )
        
        logger.info(f"Initialized RAG chain with {model_name}")
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create prompt template for grounded responses."""
        template = """You are a helpful technical documentation assistant. Use the following pieces of context to answer the question at the end.

IMPORTANT INSTRUCTIONS:
- Only use information from the provided context
- If the context doesn't contain enough information to answer the question, say so
- Cite your sources by mentioning the relevant section or page
- Be precise and technical in your explanations
- Do not make up or infer information not present in the context

Context:
{context}

Question: {question}

Helpful Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def query(
        self,
        question: str,
        return_source_documents: bool = True
    ) -> Dict:
        """
        Query the RAG system.
        
        Returns:
            Dict with 'result' and optionally 'source_documents'
        """
        logger.info(f"Processing query: {question}")
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=return_source_documents,
            chain_type_kwargs={
                "prompt": self._create_prompt_template()
            }
        )
        
        # Run query
        response = qa_chain.invoke({"query": question})
        
        logger.info(f"Generated response with {len(response.get('source_documents', []))} sources")
        
        return response
    
    def query_with_citations(self, question: str) -> Dict:
        """Query and format response with explicit citations."""
        response = self.query(question, return_source_documents=True)
        
        # Format citations
        citations = []
        for i, doc in enumerate(response['source_documents'], 1):
            citations.append({
                'id': i,
                'content': doc.page_content[:200] + "...",
                'metadata': doc.metadata,
                'full_content': doc.page_content
            })
        
        return {
            'answer': response['result'],
            'citations': citations,
            'num_sources': len(citations)
        }
```

**1.2 Create Simple Query Interface**

Create `src/agent/query_interface.py`:
```python
"""Simple interface for querying the RAG system."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.vector_store import VectorStore
from src.agent.rag_chain import RAGChain
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def interactive_query():
    """Interactive query session."""
    logger.info("Loading vector store...")
    vector_store = VectorStore()
    vector_store.load(name="postgresql_docs")
    
    logger.info("Initializing RAG chain...")
    rag_chain = RAGChain(vector_store, retrieval_k=5)
    
    print("\n" + "="*80)
    print("Technical Documentation Assistant")
    print("Type 'exit' to quit")
    print("="*80 + "\n")
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        print("\n🔍 Searching documentation...")
        result = rag_chain.query_with_citations(question)
        
        print("\n" + "="*80)
        print("ANSWER:")
        print("-"*80)
        print(result['answer'])
        print("\n" + "="*80)
        print(f"SOURCES ({result['num_sources']}):")
        print("-"*80)
        
        for citation in result['citations']:
            print(f"\n[{citation['id']}] {citation['metadata'].get('filename', 'Unknown')}")
            print(f"    Page: {citation['metadata'].get('page', 'N/A')}")
            print(f"    {citation['content']}")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    interactive_query()
```

**1.3 Test Basic RAG**

Create `scripts/test_rag_basic.py`:
```python
"""Test basic RAG functionality."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.vector_store import VectorStore
from src.agent.rag_chain import RAGChain
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def test_basic_query():
    """Test a simple query."""
    # Load vector store
    vector_store = VectorStore()
    vector_store.load(name="postgresql_docs")
    
    # Create RAG chain
    rag_chain = RAGChain(vector_store)
    
    # Test query
    question = "How do I create an index in PostgreSQL?"
    result = rag_chain.query_with_citations(question)
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nNumber of sources: {result['num_sources']}")
    
    for citation in result['citations']:
        print(f"\nSource {citation['id']}:")
        print(f"  File: {citation['metadata'].get('filename')}")
        print(f"  Content: {citation['content']}")

if __name__ == "__main__":
    test_basic_query()
```

**Deliverables**:
- ✅ RAG chain implementation
- ✅ Interactive query interface
- ✅ Basic query test passing
- ✅ First successful answer with citations

---

### Day 2: Prompt Engineering & Response Quality

**Time Estimate**: 6-7 hours

#### Tasks

**2.1 Create Multiple Prompt Templates**

Create `src/agent/prompts.py`:
```python
"""Prompt templates for different use cases."""
from langchain.prompts import PromptTemplate

# Base template - conservative, citation-focused
BASE_TEMPLATE = """You are a technical documentation assistant for PostgreSQL. Use only the provided context to answer questions.

RULES:
- Answer ONLY based on the context provided
- If unsure or if context is insufficient, say "I don't have enough information"
- Cite specific sections when making claims
- Be concise but complete
- Use technical terminology correctly

Context:
{context}

Question: {question}

Answer:"""

# Detailed template - more explanation
DETAILED_TEMPLATE = """You are an expert PostgreSQL documentation assistant. Your role is to help users understand PostgreSQL concepts and features.

INSTRUCTIONS:
1. Carefully read the provided context
2. Answer the question using ONLY information from the context
3. Provide detailed explanations when appropriate
4. Include examples if they appear in the context
5. Cite the relevant sections or pages
6. If the context doesn't contain sufficient information, clearly state this

Context:
{context}

Question: {question}

Detailed Answer:"""

# Troubleshooting template - action-oriented
TROUBLESHOOTING_TEMPLATE = """You are a PostgreSQL troubleshooting assistant. Help users solve problems using the documentation.

APPROACH:
- Identify the core issue from the question
- Use the context to find relevant solutions
- Provide step-by-step guidance if available
- Mention relevant configuration parameters or commands
- Cite where in the documentation this information comes from
- If the context doesn't address the issue, recommend what to search for

Context:
{context}

Question: {question}

Troubleshooting Response:"""

# Code example template - focused on practical usage
CODE_EXAMPLE_TEMPLATE = """You are a PostgreSQL code assistant. Help users with SQL syntax and examples.

GUIDELINES:
- Look for code examples in the context
- Explain the syntax and parameters
- Highlight important considerations or warnings
- Only provide examples that appear in or can be directly derived from the context
- Format SQL code clearly
- Reference the documentation sections

Context:
{context}

Question: {question}

Response with Examples:"""

def get_prompt_template(template_type: str = "base") -> PromptTemplate:
    """Get a prompt template by type."""
    templates = {
        "base": BASE_TEMPLATE,
        "detailed": DETAILED_TEMPLATE,
        "troubleshooting": TROUBLESHOOTING_TEMPLATE,
        "code": CODE_EXAMPLE_TEMPLATE
    }
    
    template = templates.get(template_type, BASE_TEMPLATE)
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
```

**2.2 Update RAG Chain to Support Multiple Prompts**

Update `src/agent/rag_chain.py`:
```python
# Add import
from .prompts import get_prompt_template

# Update RAGChain __init__
def __init__(
    self,
    vector_store: VectorStore,
    model_name: str = "gpt-4-turbo-preview",
    temperature: float = 0.0,
    retrieval_k: int = 5,
    prompt_type: str = "base"  # NEW
):
    # ... existing code ...
    self.prompt_type = prompt_type
    self.prompt_template = get_prompt_template(prompt_type)

# Update query method to use self.prompt_template
def query(self, question: str, return_source_documents: bool = True) -> Dict:
    # ... existing code ...
    qa_chain = RetrievalQA.from_chain_type(
        llm=self.llm,
        chain_type="stuff",
        retriever=self.retriever,
        return_source_documents=return_source_documents,
        chain_type_kwargs={"prompt": self.prompt_template}  # Use instance variable
    )
    # ... rest of method ...
```

**2.3 Create Prompt Comparison Notebook**

Create `notebooks/prompt_engineering.ipynb`:
```python
# Cell 1: Setup
from src.retrieval.vector_store import VectorStore
from src.agent.rag_chain import RAGChain
import json

# Load vector store
vector_store = VectorStore()
vector_store.load(name="postgresql_docs")

# Test queries
test_queries = [
    "How do I create an index?",
    "What's the difference between INNER JOIN and LEFT JOIN?",
    "My query is slow, how can I optimize it?"
]

# Cell 2: Compare prompt templates
prompt_types = ["base", "detailed", "troubleshooting", "code"]

for query in test_queries[:1]:  # Start with one query
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}\n")
    
    for prompt_type in prompt_types:
        print(f"\n--- {prompt_type.upper()} TEMPLATE ---")
        
        rag_chain = RAGChain(
            vector_store, 
            prompt_type=prompt_type,
            retrieval_k=3
        )
        
        result = rag_chain.query(query)
        print(result['result'])
        print()

# Cell 3: Evaluate and choose best
# Manually review outputs and document decision
```

**2.4 Create Answer Quality Checklist**

Create `docs/answer-quality-criteria.md`:
```markdown
# Answer Quality Evaluation Criteria

## Grounding & Accuracy
- [ ] All claims are supported by the retrieved context
- [ ] No hallucinated information
- [ ] Technical terms used correctly
- [ ] No contradictions with source material

## Completeness
- [ ] Answers the question fully
- [ ] Addresses implicit sub-questions
- [ ] Provides necessary context
- [ ] Mentions limitations if context is insufficient

## Citation Quality
- [ ] Relevant sources are cited
- [ ] Citations are accurate (match content)
- [ ] Sufficient sources to support claims
- [ ] Source metadata is complete

## Clarity & Usability
- [ ] Answer is easy to understand
- [ ] Appropriate level of detail
- [ ] Well-structured (paragraphs, lists where appropriate)
- [ ] Technical but accessible

## Performance
- [ ] Response time acceptable (<5s cold, <2s warm)
- [ ] Retrieved chunks are relevant
- [ ] No redundant information
```

**Deliverables**:
- ✅ Multiple prompt templates
- ✅ Prompt comparison notebook
- ✅ Quality criteria documentation
- ✅ Selected optimal prompt template

---

### Day 3: Citation System Enhancement

**Time Estimate**: 5-6 hours

#### Tasks

**3.1 Implement Advanced Citation Tracking**

Create `src/agent/citation_manager.py`:
```python
"""Manage citations and source attribution."""
from typing import List, Dict, Tuple
from langchain.schema import Document
import re

class CitationManager:
    """Track and format citations for responses."""
    
    def __init__(self):
        self.citations = []
    
    def add_sources(self, documents: List[Document]) -> None:
        """Add source documents as potential citations."""
        self.citations = []
        
        for i, doc in enumerate(documents, 1):
            self.citations.append({
                'id': i,
                'content': doc.page_content,
                'metadata': doc.metadata,
                'cited': False  # Track if citation was used
            })
    
    def format_citation(self, citation_id: int) -> str:
        """Format a single citation for display."""
        if citation_id > len(self.citations):
            return f"[Citation {citation_id} not found]"
        
        citation = self.citations[citation_id - 1]
        
        # Extract key metadata
        filename = citation['metadata'].get('filename', 'Unknown')
        page = citation['metadata'].get('page', 'N/A')
        
        return f"[{citation_id}] {filename}, Page {page}"
    
    def get_citation_snippet(
        self,
        citation_id: int,
        max_length: int = 200
    ) -> str:
        """Get a snippet of the cited content."""
        if citation_id > len(self.citations):
            return ""
        
        content = self.citations[citation_id - 1]['content']
        
        if len(content) <= max_length:
            return content
        
        return content[:max_length] + "..."
    
    def mark_as_cited(self, citation_ids: List[int]) -> None:
        """Mark citations as used in the response."""
        for cid in citation_ids:
            if 0 < cid <= len(self.citations):
                self.citations[cid - 1]['cited'] = True
    
    def get_all_citations(self) -> List[Dict]:
        """Get all citations with formatting."""
        formatted = []
        
        for citation in self.citations:
            formatted.append({
                'id': citation['id'],
                'reference': self.format_citation(citation['id']),
                'snippet': self.get_citation_snippet(citation['id']),
                'full_content': citation['content'],
                'metadata': citation['metadata'],
                'used': citation['cited']
            })
        
        return formatted
    
    def inject_inline_citations(self, answer: str) -> str:
        """
        Add inline citation markers to answer.
        This is a simple version - can be enhanced with LLM-based attribution.
        """
        # For now, just append all citations at the end
        # In a production system, you'd use LLM to insert citations inline
        
        citation_text = "\n\nSources:\n"
        for citation in self.citations:
            citation_text += f"[{citation['id']}] {self.format_citation(citation['id'])}\n"
        
        return answer + citation_text


class ResponseFormatter:
    """Format RAG responses with citations."""
    
    @staticmethod
    def format_answer_with_citations(
        answer: str,
        citations: List[Dict]
    ) -> Dict:
        """Format a complete response with structured citations."""
        
        # Separate answer and citations
        return {
            'answer': answer,
            'citations': citations,
            'formatted_text': ResponseFormatter._create_formatted_text(
                answer, citations
            )
        }
    
    @staticmethod
    def _create_formatted_text(answer: str, citations: List[Dict]) -> str:
        """Create a formatted text version of the response."""
        text = f"{answer}\n\n"
        text += "─" * 80 + "\n"
        text += "SOURCES:\n"
        text += "─" * 80 + "\n"
        
        for citation in citations:
            text += f"\n[{citation['id']}] {citation['reference']}\n"
            text += f"    {citation['snippet']}\n"
        
        return text
    
    @staticmethod
    def format_for_cli(result: Dict) -> str:
        """Format response for command-line display."""
        output = "\n" + "=" * 80 + "\n"
        output += "ANSWER:\n"
        output += "─" * 80 + "\n"
        output += result['answer'] + "\n"
        output += "\n" + "=" * 80 + "\n"
        output += f"SOURCES ({len(result['citations'])}):\n"
        output += "─" * 80 + "\n"
        
        for citation in result['citations']:
            output += f"\n[{citation['id']}] {citation['reference']}\n"
            output += f"    {citation['snippet']}\n"
        
        output += "\n" + "=" * 80 + "\n"
        
        return output
```

**3.2 Integrate Citation Manager with RAG Chain**

Update `src/agent/rag_chain.py`:
```python
# Add import
from .citation_manager import CitationManager, ResponseFormatter

# Update query_with_citations method
def query_with_citations(self, question: str) -> Dict:
    """Query and format response with detailed citations."""
    response = self.query(question, return_source_documents=True)
    
    # Use citation manager
    citation_mgr = CitationManager()
    citation_mgr.add_sources(response['source_documents'])
    
    # Mark all as cited (in production, use LLM to determine which are actually used)
    citation_mgr.mark_as_cited(list(range(1, len(response['source_documents']) + 1)))
    
    # Get formatted citations
    citations = citation_mgr.get_all_citations()
    
    # Format response
    return ResponseFormatter.format_answer_with_citations(
        response['result'],
        citations
    )
```

**Deliverables**:
- ✅ Citation manager implementation
- ✅ Response formatter
- ✅ Enhanced citation display
- ✅ Updated RAG chain with citation integration

---

### Day 4: Evaluation Dataset Creation

**Time Estimate**: 6-7 hours

#### Tasks

**4.1 Create Comprehensive Test Query Dataset**

Create `evals/test_queries.json`:
```json
{
  "version": "1.0",
  "created": "2024-01-15",
  "description": "Evaluation dataset for PostgreSQL documentation RAG system",
  "queries": [
    {
      "id": "q001",
      "query": "How do I create an index in PostgreSQL?",
      "category": "DDL",
      "difficulty": "easy",
      "expected_topics": ["CREATE INDEX", "index types", "B-tree", "performance"],
      "ground_truth": "Use CREATE INDEX statement. Basic syntax: CREATE INDEX index_name ON table_name (column_name). Supports various index types including B-tree (default), Hash, GiST, and GIN.",
      "context_keywords": ["index", "CREATE INDEX", "btree", "performance"]
    },
    {
      "id": "q002",
      "query": "What's the difference between INNER JOIN and LEFT JOIN?",
      "category": "SQL",
      "difficulty": "easy",
      "expected_topics": ["JOIN types", "NULL handling", "query examples"],
      "ground_truth": "INNER JOIN returns only rows with matching values in both tables. LEFT JOIN returns all rows from left table and matched rows from right table, with NULLs for non-matching right table columns.",
      "context_keywords": ["JOIN", "INNER JOIN", "LEFT JOIN", "NULL"]
    },
    {
      "id": "q003",
      "query": "How do I configure connection pooling in PostgreSQL?",
      "category": "administration",
      "difficulty": "medium",
      "expected_topics": ["max_connections", "connection pooling", "pgbouncer", "configuration"],
      "ground_truth": "PostgreSQL itself doesn't have built-in connection pooling. Configure max_connections parameter in postgresql.conf. For connection pooling, use external tools like PgBouncer or Pgpool-II.",
      "context_keywords": ["max_connections", "postgresql.conf", "connection"]
    },
    {
      "id": "q004",
      "query": "What is MVCC and how does it work?",
      "category": "architecture",
      "difficulty": "hard",
      "expected_topics": ["MVCC", "concurrency", "transactions", "tuple versioning"],
      "ground_truth": "Multi-Version Concurrency Control (MVCC) allows multiple transaction versions of data. Each transaction sees a snapshot of the database. Readers don't block writers and vice versa. Uses tuple versioning and transaction IDs.",
      "context_keywords": ["MVCC", "concurrency", "transaction", "snapshot"]
    },
    {
      "id": "q005",
      "query": "How do I back up a PostgreSQL database?",
      "category": "administration",
      "difficulty": "easy",
      "expected_topics": ["pg_dump", "backup", "restore", "pg_dumpall"],
      "ground_truth": "Use pg_dump for single database backup: pg_dump dbname > backup.sql. Use pg_dumpall for all databases. For point-in-time recovery, configure WAL archiving and use pg_basebackup.",
      "context_keywords": ["pg_dump", "backup", "restore", "WAL"]
    }
    // ... add 15 more queries to reach 20 total
  ]
}
```

**4.2 Create Evaluation Runner**

Create `evals/eval_runner.py`:
```python
"""
Run evaluations on the RAG system.
"""
import json
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.vector_store import VectorStore
from src.agent.rag_chain import RAGChain
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class RAGEvaluator:
    """Evaluate RAG system responses."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.rag_chain = RAGChain(vector_store, retrieval_k=5)
        self.results = []
    
    def load_test_queries(self, filepath: str = "evals/test_queries.json") -> List[Dict]:
        """Load test queries from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data['queries']
    
    def evaluate_query(self, query_data: Dict) -> Dict:
        """Evaluate a single query."""
        logger.info(f"Evaluating query: {query_data['id']}")
        
        # Get response
        result = self.rag_chain.query_with_citations(query_data['query'])
        
        # Manual evaluation metrics (to be filled in)
        evaluation = {
            'query_id': query_data['id'],
            'query': query_data['query'],
            'answer': result['answer'],
            'num_sources': result['num_sources'],
            'sources': [c['reference'] for c in result['citations']],
            'timestamp': datetime.now().isoformat(),
            
            # Manual scores (1-5 scale, to be filled after review)
            'relevance_score': None,  # Does it answer the question?
            'groundedness_score': None,  # Is it based on context?
            'completeness_score': None,  # Is the answer complete?
            'citation_quality_score': None,  # Are citations appropriate?
            
            # Automated checks
            'has_answer': len(result['answer']) > 0,
            'has_citations': result['num_sources'] > 0,
            'answer_length': len(result['answer']),
        }
        
        return evaluation
    
    def run_evaluation(self, output_file: str = None) -> List[Dict]:
        """Run evaluation on all test queries."""
        queries = self.load_test_queries()
        
        logger.info(f"Running evaluation on {len(queries)} queries")
        
        self.results = []
        for query_data in queries:
            eval_result = self.evaluate_query(query_data)
            self.results.append(eval_result)
        
        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evals/results/eval_results_{timestamp}.json"
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'num_queries': len(queries),
                'results': self.results
            }, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_file}")
        
        return self.results
    
    def print_summary(self):
        """Print evaluation summary."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        total = len(self.results)
        with_answers = sum(1 for r in self.results if r['has_answer'])
        with_citations = sum(1 for r in self.results if r['has_citations'])
        
        print(f"\nTotal queries: {total}")
        print(f"Queries with answers: {with_answers} ({with_answers/total*100:.1f}%)")
        print(f"Queries with citations: {with_citations} ({with_citations/total*100:.1f}%)")
        
        avg_length = sum(r['answer_length'] for r in self.results) / total
        avg_sources = sum(r['num_sources'] for r in self.results) / total
        
        print(f"\nAverage answer length: {avg_length:.0f} characters")
        print(f"Average sources per answer: {avg_sources:.1f}")
        
        print("\n" + "="*80)

def main():
    """Run evaluation."""
    # Load vector store
    logger.info("Loading vector store...")
    vector_store = VectorStore()
    vector_store.load(name="postgresql_docs")
    
    # Create evaluator
    evaluator = RAGEvaluator(vector_store)
    
    # Run evaluation
    evaluator.run_evaluation()
    
    # Print summary
    evaluator.print_summary()
    
    print("\n✅ Evaluation complete!")
    print("Review results in evals/results/ and manually score responses.")

if __name__ == "__main__":
    main()
```

**Deliverables**:
- ✅ 20-query evaluation dataset
- ✅ Evaluation runner script
- ✅ Results directory structure
- ✅ Initial evaluation run completed

---

### Days 5-6: Iteration & Optimization

**Time Estimate**: 10-12 hours total

#### Tasks

**5.1 Analyze Initial Results**
- Review all 20 responses manually
- Score each on 4 dimensions (relevance, groundedness, completeness, citations)
- Identify patterns in failures or weaknesses

**5.2 Optimize Retrieval**
- Experiment with different k values (3, 5, 7, 10)
- Try different chunk sizes if needed
- Test metadata filtering

**5.3 Optimize Prompts**
- Refine based on failure modes
- Add specific instructions for common issues
- Test variations

**5.4 Re-evaluate**
- Run evaluation again with optimizations
- Compare before/after metrics
- Document improvements

**5.5 Create Comparison Notebook**

Create `notebooks/sprint1_optimization.ipynb`:
```python
# Document all experiments, show before/after comparisons
# Include visualizations of improvements
# Record decisions and rationale
```

**Deliverables**:
- ✅ Manual evaluation scores for all queries
- ✅ Optimization experiments documented
- ✅ Measurable improvement in quality scores
- ✅ Updated prompt templates if needed

---

### Day 7: Documentation & Sprint Review

**Time Estimate**: 4-5 hours

#### Tasks

**7.1 Document Sprint 1 Results**

Create `docs/sprint1-results.md`:
```markdown
# Sprint 1 Results: Core RAG Pipeline

## Overview
Successfully built end-to-end RAG pipeline with LLM integration and citation system.

## Achievements
- ✅ RAG chain with GPT-4 integration
- ✅ Multiple prompt templates tested
- ✅ Citation system with source tracking
- ✅ 20-query evaluation dataset
- ✅ Initial baseline metrics established

## Metrics

### Response Quality (Manual Evaluation)
- Relevance Score: X.X/5.0
- Groundedness Score: X.X/5.0
- Completeness Score: X.X/5.0
- Citation Quality: X.X/5.0

### Performance
- Average response time: X.Xs
- P95 response time: X.Xs
- Average sources per response: X

### Coverage
- Queries with complete answers: XX/20 (XX%)
- Queries with citations: XX/20 (XX%)
- Hallucination rate: X%

## Key Learnings

### What Worked Well
- [Document findings]

### Challenges
- [Document challenges]

### Optimizations Applied
- [Document what was changed and why]

## Next Steps for Sprint 2
- Implement MCP server
- Add streaming responses
- Improve retrieval precision
```

**7.2 Update README**

Update main `README.md` with:
- Current project status
- Demo GIF or screenshot
- Quick start instructions
- Example queries and responses

**7.3 Code Cleanup**
- Add missing docstrings
- Run formatters (Black, Ruff)
- Update type hints
- Remove debug print statements

**7.4 Create Demo Script**

Create `scripts/demo.py`:
```python
"""
Demo script showing off Sprint 1 capabilities.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.vector_store import VectorStore
from src.agent.rag_chain import RAGChain
from src.agent.citation_manager import ResponseFormatter

def run_demo():
    """Run demo queries."""
    print("\n" + "="*80)
    print("Technical Documentation Assistant - Sprint 1 Demo")
    print("="*80)
    
    # Load system
    print("\n🔄 Loading system...")
    vector_store = VectorStore()
    vector_store.load(name="postgresql_docs")
    rag_chain = RAGChain(vector_store, prompt_type="detailed")
    
    # Demo queries
    demo_queries = [
        "How do I create an index in PostgreSQL?",
        "What's the difference between INNER JOIN and LEFT JOIN?",
        "How do I back up my database?"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*80}")
        print(f"Demo Query {i}/{len(demo_queries)}")
        print(f"{'='*80}")
        print(f"\nQ: {query}")
        print("\n🔍 Searching documentation...")
        
        result = rag_chain.query_with_citations(query)
        formatted = ResponseFormatter.format_for_cli(result)
        print(formatted)
        
        input("\nPress Enter for next query...")
    
    print("\n✅ Demo complete!")
    print("Try the interactive mode: python src/agent/query_interface.py")

if __name__ == "__main__":
    run_demo()
```

**7.5 Sprint Retrospective**

Create `docs/retrospectives/sprint1.md`:
```markdown
# Sprint 1 Retrospective

## What Went Well
- 

## What Could Be Improved
- 

## Action Items for Next Sprint
- 

## Blockers/Risks
- 
```

**Deliverables**:
- ✅ Sprint 1 results documented
- ✅ README updated with demo
- ✅ Code cleaned and formatted
- ✅ Demo script ready
- ✅ Retrospective completed

---

## Sprint 1 Success Criteria Checklist

### Functional Requirements
- [ ] RAG chain successfully answers all 20 test queries
- [ ] Each answer includes citations to source documents
- [ ] No hallucinated information in responses
- [ ] Responses are technically accurate
- [ ] Average response time <5 seconds

### Code Quality
- [ ] All modules have docstrings
- [ ] Code is formatted with Black
- [ ] Type hints on public methods
- [ ] No linting errors
- [ ] Basic tests pass

### Documentation
- [ ] Prompt engineering decisions documented
- [ ] Evaluation methodology explained
- [ ] Sprint results summarized
- [ ] README has working examples
- [ ] Demo script works end-to-end

### Evaluation
- [ ] 20-query dataset complete with ground truth
- [ ] Manual evaluation scores recorded
- [ ] Baseline metrics established
- [ ] Comparison of prompt templates documented
- [ ] Optimization experiments logged

---

## Key Files Created in Sprint 1

```
src/agent/
├── rag_chain.py              # Main RAG implementation
├── prompts.py                # Prompt templates
├── query_interface.py        # Interactive CLI
├── citation_manager.py       # Citation tracking
└── __init__.py

evals/
├── test_queries.json         # Evaluation dataset
├── eval_runner.py            # Evaluation script
└── results/                  # Evaluation outputs

docs/
├── sprint1-results.md        # Sprint summary
├── answer-quality-criteria.md # Evaluation rubric
└── retrospectives/
    └── sprint1.md            # Retrospective

notebooks/
├── prompt_engineering.ipynb  # Prompt experiments
└── sprint1_optimization.ipynb # Optimization work

scripts/
├── demo.py                   # Demo script
└── test_rag_basic.py         # Basic RAG test
```

---

## Risk Mitigation

### Risk: Answer Quality Below Target
- **Mitigation**: Budget extra time Days 5-6 for iteration
- **Fallback**: Document challenges and plan for Sprint 2 improvements

### Risk: Response Latency Too High
- **Mitigation**: Profile code, reduce retrieval k if needed
- **Fallback**: Note as technical debt, optimize in Sprint 2

### Risk: Evaluation Takes Longer Than Expected
- **Mitigation**: Start with subset (10 queries) if time constrained
- **Fallback**: Complete evaluation asynchronously while starting Sprint 2

---

## Transition to Sprint 2

### Handoff Items
- Working RAG pipeline codebase
- Baseline quality metrics
- 20-query evaluation dataset
- Documented optimization experiments

### Sprint 2 Prerequisites
- Vector store with indexed documents (from Sprint 0)
- RAG chain implementation (from Sprint 1)
- Understanding of current quality limitations

### Sprint 2 Focus Areas
- MCP server implementation
- Advanced retrieval strategies
- Production-grade error handling
- API development