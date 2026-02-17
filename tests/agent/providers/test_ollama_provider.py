import pytest
from unittest.mock import MagicMock, patch
from requests.exceptions import RequestException, ConnectionError, HTTPError
from src.agent.providers.ollama_provider import OllamaProvider
from src.utils.config import get_settings # Assuming get_settings is imported

# Mock get_settings to control timeout and base_url during tests
@pytest.fixture
def mock_get_settings():
    with patch('src.agent.providers.ollama_provider.get_settings') as mock_settings:
        mock_settings.return_value = MagicMock(
            ollama_base_url="http://mock-ollama.com",
            ollama_timeout=10
        )
        yield mock_settings

@pytest.fixture
def ollama_provider(mock_get_settings):
    # This fixture mocks requests.get for init to ensure success
    with patch('requests.get') as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"models": [{"name": "phi3:mini"}, {"name": "llama2"}]},
            raise_for_status=MagicMock()
        )
        provider = OllamaProvider(model_name="phi3:mini", base_url="http://mock-ollama.com")
        return provider

class MockResponse:
    def __init__(self, status_code, json_data=None, raise_for_status_exception=None):
        self.status_code = status_code
        self._json_data = json_data
        self._raise_for_status_exception = raise_for_status_exception

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self._raise_for_status_exception:
            error = self._raise_for_status_exception(response=self)
            error.response = self
            raise error
        if 400 <= self.status_code < 600:
            error = HTTPError(response=self)
            error.response = self
            raise error

class TestOllamaProvider:

    # --- __init__ tests ---
    @patch('requests.get')
    def test_init_success(self, mock_get, mock_get_settings):
        # Setup mock_get for successful server connection and model availability
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"models": [{"name": "phi3:mini"}, {"name": "llama2"}]},
            raise_for_status=MagicMock()
        )
        provider = OllamaProvider(model_name="phi3:mini", base_url="http://mock-ollama.com")
        assert provider.model_name == "phi3:mini"
        assert provider.base_url == "http://mock-ollama.com"
        mock_get.assert_called_once_with("http://mock-ollama.com/api/tags", timeout=10)

    @patch('requests.get')
    def test_init_success_model_not_found_warning(self, mock_get, mock_get_settings, caplog):
        # Setup mock_get for successful server connection but model not in tags
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"models": [{"name": "llama2"}]}, # phi3:mini is missing
            raise_for_status=MagicMock()
        )
        with caplog.at_level('WARNING'):
            provider = OllamaProvider(model_name="phi3:mini", base_url="http://mock-ollama.com")
            assert "Model 'phi3:mini' not found on Ollama server. " in caplog.text
        assert provider.model_name == "phi3:mini"

    @patch('requests.get')
    def test_init_connection_error_raises_exception(self, mock_get, mock_get_settings):
        mock_get.side_effect = ConnectionError("Ollama server not running")
        with pytest.raises(ConnectionError, match="Ollama server not running"):
            OllamaProvider(model_name="phi3:mini", base_url="http://mock-ollama.com")

    @patch('requests.get')
    def test_init_server_http_error_raises_exception(self, mock_get, mock_get_settings):
        mock_get.return_value = MockResponse(500, raise_for_status_exception=HTTPError)
        with pytest.raises(HTTPError):
            OllamaProvider(model_name="phi3:mini", base_url="http://mock-ollama.com")

    # --- generate_text happy path tests ---
    @patch('requests.post')
    @patch('requests.get') # Need to patch get as well for init
    def test_generate_text_success(self, mock_get, mock_post, ollama_provider):
        mock_response_obj = MagicMock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = {
            "message": {"content": "Generated Ollama response."}
        }
        mock_response_obj.raise_for_status.return_value = None # No HTTP errors
        mock_post.return_value = mock_response_obj
        
        response = ollama_provider.generate_text("test prompt")
        assert response == "Generated Ollama response."
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs['json']['model'] == "phi3:mini"
        assert kwargs['json']['messages'][0]['content'] == "test prompt"

    @patch('requests.post')
    @patch('requests.get') # Need to patch get as well for init
    def test_generate_text_empty_content(self, mock_get, mock_post, ollama_provider):
        mock_response_obj = MagicMock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = {
            "message": {"content": ""}
        }
        mock_response_obj.raise_for_status.return_value = None
        mock_post.return_value = mock_response_obj
        
        response = ollama_provider.generate_text("test prompt")
        assert response == ""

    @patch('requests.post')
    @patch('requests.get') # Need to patch get as well for init
    def test_generate_text_malformed_response(self, mock_get, mock_post, ollama_provider):
        mock_response_obj = MagicMock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = {} # Missing 'message'
        mock_response_obj.raise_for_status.return_value = None
        mock_post.return_value = mock_response_obj
        
        with pytest.raises(KeyError): # Expecting KeyError if 'message' is missing
            ollama_provider.generate_text("test prompt")

    # --- generate_text not-happy path tests ---
    @patch('requests.post')
    @patch('requests.get') # Need to patch get as well for init
    def test_generate_text_network_error(self, mock_get, mock_post, ollama_provider):
        mock_post.side_effect = RequestException("Ollama network error")
        with pytest.raises(RequestException, match="Ollama network error"):
            ollama_provider.generate_text("test prompt")

    @patch('requests.post')
    @patch('requests.get') # Need to patch get as well for init
    def test_generate_text_http_error(self, mock_get, mock_post, ollama_provider):
        mock_response_obj = MagicMock()
        mock_response_obj.status_code = 500
        mock_response_obj.raise_for_status.side_effect = HTTPError("HTTP Error", response=mock_response_obj)
        mock_post.return_value = mock_response_obj

        with pytest.raises(HTTPError):
            ollama_provider.generate_text("test prompt")

    @patch('requests.post')
    @patch('requests.get') # Need to patch get as well for init
    def test_generate_text_unexpected_exception(self, mock_get, mock_post, ollama_provider):
        mock_post.side_effect = Exception("Unexpected Ollama problem")
        with pytest.raises(Exception, match="Unexpected Ollama problem"):
            ollama_provider.generate_text("test prompt")

    # --- embed_texts test ---
    @patch('requests.get') # Need to patch get as well for init
    def test_embed_texts_raises_not_implemented_error(self, mock_get, ollama_provider):
        with pytest.raises(NotImplementedError, match="Embedding generation is not implemented for OllamaProvider yet."):
            ollama_provider.embed_texts(["text1"])