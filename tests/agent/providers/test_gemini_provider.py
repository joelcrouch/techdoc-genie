import pytest
from unittest.mock import MagicMock, patch
from requests.exceptions import RequestException, HTTPError
from src.agent.providers.gemini_provider import GeminiProvider
from src.utils.config import get_settings # Assuming get_settings is imported by GeminiProvider

# Mock get_settings to control API key and timeout during tests
@pytest.fixture
def mock_get_settings():
    with patch('src.agent.providers.gemini_provider.get_settings') as mock_settings:
        mock_settings.return_value = MagicMock(
            gemini_base_url="http://mock-gemini.com",
            gemini_timeout=10,
            GEMINI_API_KEY="test_api_key" # Ensure API key is present for init tests
        )
        yield mock_settings

@pytest.fixture
def gemini_provider(mock_get_settings):
    # Ensure API key is provided for instantiation
    return GeminiProvider(model_name="gemini-test-model", api_key="test_api_key")

class MockResponse:
    def __init__(self, status_code, json_data=None, raise_for_status_exception=None):
        self.status_code = status_code
        self._json_data = json_data
        self._raise_for_status_exception = raise_for_status_exception

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self._raise_for_status_exception:
            raise self._raise_for_status_exception(response=self)
        if 400 <= self.status_code < 600:
            raise HTTPError(response=self)

@patch('time.sleep', MagicMock()) # Mock time.sleep globally for tests

class TestGeminiProvider:

    # --- __init__ tests ---
    def test_init_success(self, mock_get_settings):
        # API key is already set by mock_get_settings fixture
        provider = GeminiProvider(model_name="gemini-test-model", api_key="test_api_key")
        assert provider.model_name == "gemini-test-model"
        assert provider.api_key == "test_api_key"

    def test_init_no_api_key_raises_error(self, mock_get_settings):
        mock_get_settings.return_value.GEMINI_API_KEY = None # Ensure API_KEY is None from settings
        with pytest.raises(ValueError, match="Gemini API key is required."):
            GeminiProvider(model_name="gemini-test-model", api_key=None) # Pass None directly

    # --- generate_text happy path tests ---
    @patch('requests.post')
    def test_generate_text_success(self, mock_post, gemini_provider):
        mock_post.return_value = MockResponse(
            200,
            json_data={
                "candidates": [{"content": {"parts": [{"text": "Generated response."}]}}]
            }
        )
        response = gemini_provider.generate_text("test prompt")
        assert response == "Generated response."
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_generate_text_empty_response_content(self, mock_post, gemini_provider):
        mock_post.return_value = MockResponse(
            200,
            json_data={
                "candidates": [{"content": {"parts": [{"text": ""}]}}]
            }
        )
        response = gemini_provider.generate_text("test prompt")
        assert response == ""

    @patch('requests.post')
    def test_generate_text_no_candidates_in_response(self, mock_post, gemini_provider):
        mock_post.return_value = MockResponse(200, json_data={"candidates": []})
        response = gemini_provider.generate_text("test prompt")
        assert response == ""

    @patch('requests.post')
    def test_generate_text_malformed_response(self, mock_post, gemini_provider):
        mock_post.return_value = MockResponse(200, json_data={}) # Missing 'candidates'
        response = gemini_provider.generate_text("test prompt")
        assert response == ""

    # --- generate_text not-happy path tests ---
    @patch('requests.post')
    def test_generate_text_request_exception(self, mock_post, gemini_provider):
        mock_post.side_effect = RequestException("Network error")
        with pytest.raises(RequestException, match="Network error"):
            gemini_provider.generate_text("test prompt")

    @patch('requests.post')
    def test_generate_text_http_error_non_429(self, mock_post, gemini_provider):
        mock_response = MockResponse(400, raise_for_status_exception=HTTPError)
        mock_post.return_value = mock_response
        with pytest.raises(HTTPError):
            gemini_provider.generate_text("test prompt")

    @patch('requests.post')
    def test_generate_text_rate_limit_retry_success(self, mock_post, gemini_provider):
        # First call fails with 429, second call succeeds
        mock_post.side_effect = [
            MockResponse(429, raise_for_status_exception=HTTPError),
            MockResponse(200, json_data={"candidates": [{"content": {"parts": [{"text": "Retry success."}]}}]})
        ]
        response = gemini_provider.generate_text("test prompt")
        assert response == "Retry success."
        assert mock_post.call_count == 2 # Called once, then retried once

    @patch('requests.post')
    def test_generate_text_rate_limit_retry_fail(self, mock_post, gemini_provider):
        # Both calls fail with 429
        mock_post.side_effect = [
            MockResponse(429, raise_for_status_exception=HTTPError),
            MockResponse(429, raise_for_status_exception=HTTPError)
        ]
        with pytest.raises(HTTPError): # The re-raised HTTPError from the second attempt
            gemini_provider.generate_text("test prompt")
        assert mock_post.call_count == 2

    @patch('requests.post')
    def test_generate_text_unexpected_exception(self, mock_post, gemini_provider):
        mock_post.side_effect = Exception("Unexpected problem")
        with pytest.raises(Exception, match="Unexpected problem"):
            gemini_provider.generate_text("test prompt")

    def test_embed_texts_pass(self, gemini_provider):
        # This method just passes, so calling it should not raise an error
        try:
            gemini_provider.embed_texts(["text1", "text2"])
        except Exception as e:
            pytest.fail(f"embed_texts should not raise an error, but it raised {e}")

