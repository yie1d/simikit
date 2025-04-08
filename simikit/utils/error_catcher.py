from contextlib import contextmanager
from functools import wraps

from pydantic import ValidationError

__all__ = [
    'context_error_catcher',
    'wraps_error_catcher'
]


@contextmanager
def context_error_catcher():
    """
    A context manager for handling Pydantic ValidationError.

    This context manager allows you to run a block of code and catches any
    ValidationError that occurs within it.
    """
    try:
        yield
    except ValidationError as _e:
        raise _e


def wraps_error_catcher(func):
    """
    A function decorator that uses the context_error_catcher to handle ValidationError.
    This decorator can be applied to any function. It wraps the function in the
    context_error_catcher context manager.

    Args:
        func (callable): The function to be decorated.
    Returns:
        callable: A wrapped function that handles ValidationError.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with context_error_catcher():
            return func(*args, **kwargs)

    return wrapper
