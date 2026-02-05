import logging
from .config import get_settings

def setup_logger(name: str) -> logging.Logger:
    settings = get_settings()

    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level)

    if not logger.handlers:  # important: avoid duplicate handlers in tests  running inot weird test
        #faliurs that took a long time to figure out
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
