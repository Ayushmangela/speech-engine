import logging
import sys
from src.config import settings


def setup_logger(name: str = "pipeline") -> logging.Logger:
    """Configures and returns a logger that logs to stdout and logs/pipeline.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Formatter with time, level, module, function name, and line number
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = settings.LOGS_DIR / "pipeline.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        console_handler.flush()
        print(f"Warning: Could not set up file logging: {e}", file=sys.stderr)

    return logger


logger = setup_logger()
