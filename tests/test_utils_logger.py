import pytest

from simikit.utils.logger import LogLevel, set_logger


def test_set_logger():
    log_level = 'DEBUG'

    logger = set_logger(log_level)
    assert next(iter(logger._core.handlers.values()))._levelno == LogLevel[log_level].value

    with pytest.raises(KeyError):
        set_logger('INVALID')
