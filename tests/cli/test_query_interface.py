import pytest
from unittest.mock import patch, MagicMock
import io
import sys
from pathlib import Path

# Adjust path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.query_interface import interactive_query, cli_main, __name__ as query_interface_name
import src.agent.query_interface as query_interface
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# --- Fixtures for Mocking Dependencies ---

@pytest.fixture
def mock_get_settings():
    """Mocks get_settings to control config values."""
    with patch('src.agent.query_interface.get_settings') as mock_settings_func:
        mock_settings = MagicMock()
        mock_settings.llm_default_provider = "mock_provider"
        mock_settings.llm_default_model_id = "mock_model"
        mock_settings.embedding_provider = "mock_embedder_provider"
        mock_settings.embedding_model = "mock_embedder_model"
        mock_settings.chunk_size = 512
        mock_settings.chunk_overlap = 50
        mock_settings_func.return_value = mock_settings
        yield mock_settings_func

@pytest.fixture
def mock_document_embedder():
    """Mocks DocumentEmbedder."""
    with patch('src.agent.query_interface.DocumentEmbedder') as mock_embedder_class:
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance
        yield mock_embedder_class

@pytest.fixture
def mock_vector_store():
    """Mocks VectorStore."""
    with patch('src.agent.query_interface.VectorStore') as mock_vector_store_class:
        mock_vector_store_instance = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store_instance
        yield mock_vector_store_class

@pytest.fixture
def mock_rag_chain():
    """Mocks RAGChain."""
    with patch('src.agent.query_interface.RAGChain') as mock_rag_chain_class:
        mock_rag_chain_instance = MagicMock()
        mock_rag_chain_class.return_value = mock_rag_chain_instance
        yield mock_rag_chain_class

@pytest.fixture
def mock_response_formatter():
    """Mocks ResponseFormatter."""
    with patch('src.agent.query_interface.ResponseFormatter') as mock_formatter_class:
        mock_formatter_class.format_for_cli.return_value = "Formatted CLI Response"
        yield mock_formatter_class

@pytest.fixture
def mock_input(monkeypatch):
    """Mocks user input."""
    inputs = []
    def mock_stdin_readline():
        if inputs:
            return inputs.pop(0) + ''
        return 'exit' # Default to exit to prevent infinite loops
    
    # Use io.StringIO to mock stdin for input()
    monkeypatch.setattr('sys.stdin', io.StringIO(''))
    original_input = __builtins__['input']

    def mock_input_func(prompt=""):
        if inputs:
            return inputs.pop(0)
        return 'exit' # Default to exit to prevent infinite loops

    __builtins__['input'] = mock_input_func
    yield inputs
    __builtins__['input'] = original_input


@pytest.fixture
def capsys_output(capsys):
    """Captures stdout/stderr."""
    return capsys

# --- Test Cases for interactive_query function ---

def test_interactive_query_exit_command(
    mock_get_settings, mock_document_embedder, mock_vector_store, mock_rag_chain, mock_input, capsys_output
):
    """Test that the interactive loop exits on 'exit', 'quit', or 'q'."""
    mock_input.extend(['test query', 'exit']) # Query then exit
    interactive_query("mock_provider", "mock_model", "base", "mock_vector_store")
    
    captured = capsys_output.readouterr()
    assert "Goodbye!" in captured.out
    mock_rag_chain.return_value.query_with_citations.assert_called_once_with('test query')

def test_interactive_query_empty_input(
    mock_get_settings, mock_document_embedder, mock_vector_store, mock_rag_chain, mock_input, capsys_output
):
    """Test that empty user input is skipped."""
    mock_input.extend(['', 'test query', 'exit']) # Empty, then query, then exit
    interactive_query("mock_provider", "mock_model", "base", "mock_vector_store")
    
    captured = capsys_output.readouterr()
    assert "Goodbye!" in captured.out
    mock_rag_chain.return_value.query_with_citations.assert_called_once_with('test query')

def test_interactive_query_successful_response(
    mock_get_settings, mock_document_embedder, mock_vector_store, mock_rag_chain, mock_response_formatter, mock_input, capsys_output
):
    """Test a successful RAG query and formatted output."""
    mock_input.extend(['actual query', 'exit'])
    mock_rag_chain.return_value.query_with_citations.return_value = {"answer": "RAG Answer", "citations": []}
    
    interactive_query("mock_provider", "mock_model", "base", "mock_vector_store")
    
    captured = capsys_output.readouterr()
    mock_rag_chain.return_value.query_with_citations.assert_called_once_with('actual query')
    mock_response_formatter.format_for_cli.assert_called_once_with({"answer": "RAG Answer", "citations": []})
    assert "Formatted CLI Response" in captured.out

def test_interactive_query_rag_chain_error_handling(
    mock_get_settings, mock_document_embedder, mock_vector_store, mock_rag_chain, mock_input, capsys_output, caplog
):
    """Test error handling when RAG chain query fails."""
    mock_input.extend(['failing query', 'exit'])
    mock_rag_chain.return_value.query_with_citations.side_effect = Exception("RAG chain failed")
    
    interactive_query("mock_provider", "mock_model", "base", "mock_vector_store")
    
    captured = capsys_output.readouterr()
    assert "Sorry, an error occurred. Please check the logs for more details." in captured.out
    assert "An error occurred during query processing: RAG chain failed" in caplog.text
    mock_rag_chain.return_value.query_with_citations.assert_called_once_with('failing query')

def test_interactive_query_vector_store_load_failure(
    mock_get_settings, mock_document_embedder, mock_vector_store, mock_rag_chain, mock_input, capsys_output, caplog
):
    """Test error handling when vector store loading fails."""
    mock_vector_store.return_value.load.side_effect = Exception("Vector store load failed")
    
    interactive_query("mock_provider", "mock_model", "base", "mock_vector_store")
    
    captured = capsys_output.readouterr()
    assert "Failed to load vector store: Vector store load failed" in caplog.text # Check log output
    assert "Please ensure the vector store exists and its name is correct." in caplog.text
    mock_vector_store.return_value.load.assert_called_once()
    mock_rag_chain.assert_not_called() # RAG chain should not be initialized

# --- Test Cases for argument parsing in __main__ block ---

@patch('src.agent.query_interface.interactive_query')
@patch('argparse.ArgumentParser')
def test_main_argument_parsing_defaults(
    mock_arg_parser, mock_interactive_query, mock_get_settings
):
    """Test argument parsing uses defaults from settings if not provided."""
    mock_arg_parser_instance = MagicMock()
    mock_arg_parser.return_value = mock_arg_parser_instance
    mock_arg_parser_instance.parse_args.return_value = MagicMock(
        provider=mock_get_settings.return_value.llm_default_provider,
        model=mock_get_settings.return_value.llm_default_model_id,
        prompt="base",
        vector_store=None
    )
    
    # Call the cli_main function directly
    query_interface.cli_main()

    mock_interactive_query.assert_called_once_with(
        mock_get_settings.return_value.llm_default_provider,
        mock_get_settings.return_value.llm_default_model_id,
        "base",
        None
    )

@patch('src.agent.query_interface.interactive_query')
@patch('argparse.ArgumentParser')
def test_main_argument_parsing_custom_args(
    mock_arg_parser, mock_interactive_query, mock_get_settings
):
    """Test argument parsing uses custom arguments if provided."""
    mock_arg_parser_instance = MagicMock()
    mock_arg_parser.return_value = mock_arg_parser_instance
    mock_arg_parser_instance.parse_args.return_value = MagicMock(
        provider="custom_provider",
        model="custom_model",
        prompt="custom_prompt",
        vector_store="custom_vector_store"
    )
    
    # Call the cli_main function directly
    query_interface.cli_main()

    mock_interactive_query.assert_called_once_with(
        "custom_provider",
        "custom_model",
        "custom_prompt",
        "custom_vector_store"
    )

