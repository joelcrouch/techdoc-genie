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

    assert settings.embedding_model == "all-MiniLM-L6-v2"
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