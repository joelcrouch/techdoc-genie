from typing import List, Dict, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from ..retrieval.vector_store import VectorStore
from ..utils.logger import setup_logger
from ..utils.config import get_settings

# New imports for our modular architecture
from .llm_wrapper import CustomLLM
from .providers.ollama_provider import OllamaProvider
from .prompts import get_prompt_template
from .citation_manager import CitationManager, ResponseFormatter

logger = setup_logger(__name__)

class RAGChain:
    """Retrieval-Augmented Generation chain for answering queries."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider_type: str = "openai",
        model_id: str = "gpt-4-turbo-preview",
        temperature: float = 0.0,
        retrieval_k: int = 5,
        prompt_type: str = "base"
    ):
        self.vector_store = vector_store
        self.retrieval_k = retrieval_k
        self.prompt_type = prompt_type
        self.settings = get_settings()
        
        # Initialize LLM based on the provider type
        if llm_provider_type.lower() == "openai":
            self.llm = ChatOpenAI(
                model=model_id,
                temperature=temperature,
                openai_api_key=self.settings.openai_api_key
            )
            logger.info(f"Initialized RAG chain with OpenAI model: {model_id}")
        elif llm_provider_type.lower() == "ollama":
            ollama_provider = OllamaProvider(
                model_name=model_id, 
                base_url=self.settings.ollama_base_url
            )
            self.llm = CustomLLM(llm_provider=ollama_provider, model_name=model_id)
            logger.info(f"Initialized RAG chain with Ollama model: {model_id}")
        else:
            raise ValueError(f"Unsupported LLM provider type: {llm_provider_type}")
        
        # Create retriever from the vector store
        self.retriever = self.vector_store.vectorstore.as_retriever(
            search_kwargs={"k": retrieval_k}
        )
        
        # Get the appropriate prompt template
        self.prompt_template = get_prompt_template(self.prompt_type)
        
        logger.info(f"RAG chain fully initialized with provider '{llm_provider_type}' and retriever.")
    
    def query(
        self,
        question: str,
        return_source_documents: bool = True
    ) -> Dict:
        """
        Query the RAG system.
        
        Returns:
            A dictionary with the 'result' and optionally 'source_documents'.
        """
        logger.info(f"Processing query: {question}")
        
        # Create the QA chain with the selected LLM and prompt template
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=return_source_documents,
            chain_type_kwargs={
                "prompt": self.prompt_template
            }
        )
        
        # Run the query
        response = qa_chain.invoke({"query": question})
        
        logger.info(f"Generated response with {len(response.get('source_documents', []))} sources.")
        
        return response
    
    def query_with_citations(self, question: str) -> Dict:
        """
        Query the RAG system and format the response with explicit citations.
        """
        response = self.query(question, return_source_documents=True)
        
        # Use the citation manager to process source documents
        citation_mgr = CitationManager()
        citation_mgr.add_sources(response['source_documents'])
        
        # In a real system, an LLM call would determine which sources were actually used.
        # For now, we'll mark all retrieved sources as cited.
        all_source_ids = list(range(1, len(response['source_documents']) + 1))
        citation_mgr.mark_as_cited(all_source_ids)
        
        # Get the formatted citation data
        citations = citation_mgr.get_all_citations()
        
        # Format the final response structure
        return ResponseFormatter.format_answer_with_citations(
            response['result'],
            citations
        )
