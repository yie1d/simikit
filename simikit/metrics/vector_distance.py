import numpy as np

__all__ = [
    'hamming_distance',
    'euclidean_distance',
    'manhattan_distance',
]


def hamming_distance(vector1: np.ndarray, vector2: np.ndarray) -> int:
    return int(np.sum(vector1 != vector2))


def euclidean_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    return float(np.linalg.norm(vector1 - vector2))


def manhattan_distance(vector1: np.ndarray, vector2: np.ndarray) -> int:
    return int(np.sum(np.abs(vector1 - vector2)))
