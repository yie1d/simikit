import os
import sys
from enum import Enum

import loguru._logger
from dotenv import load_dotenv
from loguru import logger as _loguru_logger

load_dotenv()


class LogLevel(Enum):
    TRACE: int = 5
    DEBUG: int = 10
    INFO: int = 20
    SUCCESS: int = 25
    WARNING: int = 30
    ERROR: int = 40
    CRITICAL: int = 50


def set_logger(
    print_level: str | LogLevel = LogLevel.INFO,
) -> loguru._logger.Logger:
    """Sets up the logger and returns it.

    Args:
        print_level (str | LogLevel): Log level to use. Defaults to
         LogLevel.INFO.

    Returns:
        loguru._logger.Logger: Configured logger instance.

    """
    _format = '{time:YYYY-MM-DD HH:mm:ss.SSS} |<lvl>{level:8}</>| {name:8} : {module}:{line:4} | - <lvl>{message}</>'

    if isinstance(print_level, str):
        try:
            print_level = LogLevel[print_level]
        except KeyError:
            raise KeyError(f'Invalid log level {print_level}')

    _loguru_logger.configure(
        handlers=[
            {
                'sink': sys.stdout,
                'format': _format,
                'colorize': True,
                'level': print_level.value,
            }
        ]
    )
    return _loguru_logger


logger = set_logger(os.environ.get('LOG_LEVEL', 'INFO').upper())
