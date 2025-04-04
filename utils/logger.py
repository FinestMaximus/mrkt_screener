import sys
from loguru import logger

# Remove default logger
logger.remove()

# Configure logger with colorful format and emojis
logger.configure(
    handlers=[
        {
            "sink": sys.stdout,
            "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            "colorize": True,
            "diagnose": True,
        }
    ]
)


# Add emoji decorators for different log levels
def info(message, *args, **kwargs):
    logger.info(f"ℹ️ {message}", *args, **kwargs)


def debug(message, *args, **kwargs):
    logger.debug(f"🔍 {message}", *args, **kwargs)


def warning(message, *args, **kwargs):
    logger.warning(f"⚠️ {message}", *args, **kwargs)


def error(message, *args, **kwargs):
    logger.error(f"❌ {message}", *args, **kwargs)


def critical(message, *args, **kwargs):
    logger.critical(f"🔥 {message}", *args, **kwargs)


def success(message, *args, **kwargs):
    logger.success(f"✅ {message}", *args, **kwargs)


# Example usage:
# from utils.logger import info, debug, warning, error, critical, success
#
# info("Application started")
# debug("Processing data")
# warning("Resource usage high")
# error("Failed to connect to database")
# critical("System shutdown")
# success("Task completed successfully")
