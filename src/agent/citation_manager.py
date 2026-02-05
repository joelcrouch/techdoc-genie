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
