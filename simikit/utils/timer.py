import time
from functools import wraps

from simikit.utils.logger import logger

__all__ = [
    'timer'
]


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res_dict = func(*args, **kwargs)
        res_dict['time'] = round(time.time() - start_time, 5)
        logger.info(f'{func.__name__} cost {res_dict["time"]:.4f} seconds')
        return res_dict

    return wrapper
