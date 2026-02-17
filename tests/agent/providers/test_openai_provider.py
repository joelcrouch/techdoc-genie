import pytest
from unittest.mock import MagicMock, patch
from requests.exceptions import RequestException, HTTPError
from src.agent.providers.openai_provider import OpenAIProvider
from src.utils.config import get_settings # Assuming get_settings is imported by OpenAIProvider

# Mock get_settings to control API key and timeout during tests
@pytest.fixture
def mock_get_settings():
    with patch('src.agent.providers.openai_provider.get_settings') as mock_settings:
        mock_settings.return_value = MagicMock(
            openai_base_url="http://mock-openai.com",
            openai_timeout=10,
            OPENAI_API_KEY="test_openai_api_key" # Ensure API key is present for init tests
        )
        yield mock_settings

@pytest.fixture
def openai_provider(mock_get_settings):
    # Ensure API key is provided for instantiation
    return OpenAIProvider(model_name="gpt-test-model", api_key="test_openai_api_key")

class MockResponse:
    def __init__(self, status_code, json_data=None, raise_for_status_exception=None):
        self.status_code = status_code
        self._json_data = json_data
        self._raise_for_status_exception = raise_for_status_exception

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self._raise_for_status_exception:
            # Simulate requests.exceptions.HTTPError being raised
            error = self._raise_for_status_exception(response=self)
            error.response = self # Attach the response object to the exception
            raise error
        if 400 <= self.status_code < 600:
            error = HTTPError(response=self)
            error.response = self
            raise error

class TestOpenAIProvider:

    # --- __init__ tests ---
    def test_init_success(self, mock_get_settings):
        provider = OpenAIProvider(model_name="gpt-test-model", api_key="test_openai_api_key")
        assert provider.model_name == "gpt-test-model"
        assert provider.api_key == "test_openai_api_key"

    def test_init_no_api_key_raises_error(self, mock_get_settings):
        mock_get_settings.return_value.OPENAI_API_KEY = None # Ensure API_KEY is None from settings
        with pytest.raises(ValueError, match="OpenAI API key is required."):
            OpenAIProvider(model_name="gpt-test-model", api_key=None)

    # --- generate_text happy path tests ---
    @patch('requests.post')
    def test_generate_text_success(self, mock_post, openai_provider):
        mock_post.return_value = MockResponse(
            200,
            json_data={
                "choices": [{"message": {"content": "Generated OpenAI response."}}]
            }
        )
        response = openai_provider.generate_text("test prompt")
        assert response == "Generated OpenAI response."
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs['headers']['Authorization'] == "Bearer test_openai_api_key"
        assert kwargs['json']['model'] == "gpt-test-model"
        assert kwargs['json']['messages'][0]['content'] == "test prompt"

    @patch('requests.post')
    def test_generate_text_empty_content(self, mock_post, openai_provider):
        mock_post.return_value = MockResponse(
            200,
            json_data={
                "choices": [{"message": {"content": ""}}]
            }
        )
        response = openai_provider.generate_text("test prompt")
        assert response == ""

    @patch('requests.post')
    def test_generate_text_malformed_response_no_choices(self, mock_post, openai_provider):
        mock_post.return_value = MockResponse(200, json_data={}) # Missing 'choices'
        with pytest.raises(KeyError): # Expecting KeyError if 'choices' is missing
            openai_provider.generate_text("test prompt")

    # --- generate_text not-happy path tests ---
    @patch('requests.post')
    def test_generate_text_network_error(self, mock_post, openai_provider):
        mock_post.side_effect = RequestException("Network error")
        with pytest.raises(RequestException, match="Network error"):
            openai_provider.generate_text("test prompt")

    @patch('requests.post')
    def test_generate_text_http_error_non_429(self, mock_post, openai_provider):
        mock_response = MockResponse(400, raise_for_status_exception=HTTPError)
        mock_post.return_value = mock_response
        with pytest.raises(HTTPError):
            openai_provider.generate_text("test prompt")

    @patch('requests.post')
    def test_generate_text_rate_limit_error_429(self, mock_post, openai_provider):
        mock_response = MockResponse(429, raise_for_status_exception=HTTPError)
        mock_post.return_value = mock_response
        with pytest.raises(HTTPError): # OpenAIProvider does not retry 429 errors
            openai_provider.generate_text("test prompt")
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_generate_text_unexpected_exception(self, mock_post, openai_provider):
        mock_post.side_effect = Exception("Unexpected problem")
        with pytest.raises(Exception, match="Unexpected problem"):
            openai_provider.generate_text("test prompt")

    # --- embed_texts test ---
    def test_embed_texts_raises_not_implemented_error(self, openai_provider):
        with pytest.raises(NotImplementedError, match="Embedding generation is not implemented for OpenAIProvider yet."):
            openai_provider.embed_texts(["text1"])

