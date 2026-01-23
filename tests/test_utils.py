# import pytest
# from unittest.mock import patch, MagicMock
# from pathlib import Path
# import os
# import logging
# from src.utils.config import Settings, get_settings
# from src.utils.logger import setup_logger

# # --- Test for config.py ---

# @pytest.fixture(scope="function")
# def clear_settings_cache():
#     """Fixture to clear the lru_cache for get_settings."""
#     get_settings.cache_clear()
#     yield
#     get_settings.cache_clear()

# @pytest.fixture(scope="function")
# def mock_env_file(tmp_path):
#     """Fixture to create a temporary .env file for testing."""
#     env_content = """
# OPENAPI_KEY=test_api_key_123
# EMBEDDING_MODEL=test-embedding-model
# LOG_LEVEL=DEBUG
# """
#     env_file = tmp_path / ".env"
#     env_file.write_text(env_content)
#     # Patch the env_file attribute of Settings.Config to point to our temporary file
#     with patch("src.utils.config.Settings.Config.env_file", str(env_file)):
#         yield env_file

# def test_settings_load_from_env_file(clear_settings_cache, mock_env_file):
#     """Test that settings load correctly from a .env file."""
#     settings = get_settings()
#     assert settings.openapi_key == "test_api_key_123"
#     assert settings.embedding_model == "test-embedding-model"
#     assert settings.log_level == "DEBUG"
#     # Test a default value not in .env
#     assert settings.vector_store_path == "./data/vector_store"

# def test_settings_default_values(clear_settings_cache):
#     """Test that default values are used when not specified in .env."""
#     # Ensure no .env file interferes
#     with patch("src.utils.config.Settings.Config.env_file", None):
#         # Store original OPENAPI_KEY if it exists, initialize to None if not
#         original_openapi_key = os.environ.get("OPENAPI_KEY", None)
        
#         # Set a dummy value for the required openapi_key for this test
#         os.environ["OPENAPI_KEY"] = "dummy_key_for_default_test"
        
#         try:
#             settings = get_settings()
#             assert settings.embedding_model == "text-embedding-3-small"
#             assert settings.log_level == "INFO" # Default value
#             assert settings.chunk_size == 512
#             assert settings.chunk_overlap == 50
#             # Ensure the dummy required key is also there
#             assert settings.openapi_key == "dummy_key_for_default_test"
#         finally:
#             # Clean up the environment variable after the test
#             if original_openapi_key is not None:
#                 os.environ["OPENAPI_KEY"] = original_openapi_key
#             else:
#                 del os.environ["OPENAPI_KEY"]


# # --- Test for logger.py ---

# def test_setup_logger_returns_logger_instance(clear_settings_cache, mock_env_file):
#     """Test that setup_logger returns a logging.Logger instance."""
#     logger_name = "test_logger_instance"
#     logger = setup_logger(logger_name)
#     assert isinstance(logger, logging.Logger)
#     assert logger.name == logger_name

# def test_setup_logger_level_from_settings(clear_settings_cache, mock_env_file):
#     """Test that the logger level is set according to settings."""
#     logger_name = "test_logger_level"
#     logger = setup_logger(logger_name)
#     # The mock_env_file sets LOG_LEVEL=DEBUG
#     assert logger.level == logging.DEBUG

# def test_setup_logger_message_format(clear_settings_cache, mock_env_file, caplog):
#     """Test that log messages are formatted correctly."""
#     logger_name = "test_logger_format"
#     logger = setup_logger(logger_name)
#     logger.info("This is an info message.")

#     assert len(caplog.records) == 1
#     record = caplog.records[0]
#     assert record.levelname == "INFO"
#     assert record.message == "This is an info message."
#     # The formatter includes asctime, name, levelname, message
#     # We can't directly check the full string from caplog.text because asctime is dynamic.
#     # But checking components of the record is sufficient.
#     assert logger_name in caplog.text
#     assert "INFO" in caplog.text
#     assert "This is an info message." in caplog.text

# def test_setup_logger_no_duplicate_handlers(clear_settings_cache, mock_env_file):
#     """Test that calling setup_logger multiple times doesn't add duplicate handlers."""
#     logger_name = "test_no_duplicate_handlers"
#     logger1 = setup_logger(logger_name)
#     initial_handlers_count = len(logger1.handlers)

#     logger2 = setup_logger(logger_name) # Call again with the same name
#     assert len(logger2.handlers) == initial_handlers_count
import logging
import pytest

from src.utils.config import get_settings
from src.utils.logger import setup_logger


# -------------------------
# shared fixtures
# -------------------------

@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Ensure get_settings cache is cleared between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# -------------------------
# config.py tests
# -------------------------

def test_settings_reads_required_openapi_key(monkeypatch):
    """Required OPENAPI_KEY is read from environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_key_123")

    settings = get_settings()

    assert settings.OPENAI_API_KEY == "test_key_123"


def test_settings_uses_default_values_when_not_overridden(monkeypatch):
    """Defaults are used when env vars are not provided."""
    monkeypatch.setenv("OPENAPI_KEY", "dummy_key")

    settings = get_settings()

    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.vector_store_path == "./data/vector_store"
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 50
    assert settings.log_level == "INFO"


def test_settings_overrides_defaults_from_env(monkeypatch):
    """Environment variables override defaults."""
    monkeypatch.setenv("OPENAPI_KEY", "dummy_key")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = get_settings()

    assert settings.embedding_model == "test-embedding"
    assert settings.log_level == "DEBUG"


# -------------------------
# logger.py tests
# -------------------------

def test_setup_logger_returns_logger(monkeypatch):
    """setup_logger returns a logging.Logger instance."""
    monkeypatch.setenv("OPENAPI_KEY", "dummy_key")

    logger = setup_logger("test_logger")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"


def test_logger_level_respects_settings(monkeypatch):
    """Logger level matches LOG_LEVEL setting."""
    monkeypatch.setenv("OPENAPI_KEY", "dummy_key")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    logger = setup_logger("level_test")

    assert logger.level == logging.DEBUG


def test_logger_outputs_message(caplog, monkeypatch):
    """Logger emits messages with correct content."""
    monkeypatch.setenv("OPENAPI_KEY", "dummy_key")

    logger = setup_logger("format_test")

    with caplog.at_level(logging.INFO):
        logger.info("hello world")

    assert len(caplog.records) == 1
    record = caplog.records[0]

    assert record.levelname == "INFO"
    assert record.message == "hello world"
    assert "format_test" in caplog.text


def test_setup_logger_does_not_duplicate_handlers(monkeypatch):
    """Calling setup_logger twice does not add extra handlers."""
    monkeypatch.setenv("OPENAPI_KEY", "dummy_key")

    logger1 = setup_logger("no_dupes")
    handler_count = len(logger1.handlers)

    logger2 = setup_logger("no_dupes")

    assert len(logger2.handlers) == handler_count