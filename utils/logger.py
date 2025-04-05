import sys
from loguru import logger

# Remove default logger
logger.remove()

# Configure logger with colorful format and emojis
logger.configure(
    handlers=[
        {
            "sink": sys.stdout,
            "format": "<green>{time:HH:mm:ss}</green> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            "colorize": True,
            "diagnose": True,
        }
    ]
)


def info(message, *args, **kwargs):
    logger.opt(depth=1, record=True).info(f"ℹ️ {message}", *args, **kwargs)


def debug(message, *args, **kwargs):
    logger.opt(depth=1, record=True).debug(f"🔍 {message}", *args, **kwargs)


def warning(message, *args, **kwargs):
    logger.opt(depth=1, record=True).warning(f"⚠️ {message}", *args, **kwargs)


def error(message, *args, **kwargs):
    logger.opt(depth=1, record=True).error(f"❌ {message}", *args, **kwargs)


def critical(message, *args, **kwargs):
    logger.opt(depth=1, record=True).critical(f"🔥 {message}", *args, **kwargs)


def success(message, *args, **kwargs):
    logger.opt(depth=1, record=True).success(f"✅ {message}", *args, **kwargs)
