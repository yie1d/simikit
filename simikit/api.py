from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, field_validator

from simikit.features.base import BaseExtractor
from simikit.utils.error_catcher import context_error_catcher, wraps_error_catcher

__all__ = [
    'Comparator'
]

CompareAlgoType = tuple[BaseExtractor, Callable]


class CompareAlgo(BaseModel):
    model_config = {
        'arbitrary_types_allowed': True
    }
    features_func: BaseExtractor
    metrics_func: Callable

    @field_validator('features_func', mode='before')
    @classmethod
    def validate_features_func(cls, v):
        if isinstance(v, BaseExtractor) is False:
            raise TypeError('features_func must be an instance of BaseExtractor')
        return v


class InputImage(BaseModel):
    image: str | Path


class Comparator:
    def __init__(self, by: tuple[CompareAlgoType, ...] | list[CompareAlgoType]) -> None:
        with context_error_catcher():
            self._algos = self._load_algos(by)

    @staticmethod
    def _load_algos(by: tuple[CompareAlgoType, ...] | list[CompareAlgoType]) -> list[CompareAlgo]:
        if not isinstance(by, (tuple, list)):
            raise TypeError('By must be tuple or list')

        _algos = []
        for _algo in by:
            if not isinstance(_algo, tuple):
                raise TypeError('CompareAlgo must be tuple')
            _algos.append(CompareAlgo(**{'features_func': _algo[0], 'metrics_func': _algo[1]}))

        return _algos

    @wraps_error_catcher
    def compare_image(self, image1: str | Path, image2: str | Path):
        image1 = InputImage(image=image1).image
        image2 = InputImage(image=image2).image

        compare_results = []
        for _algo in self._algos:
            compare_results.append({
                'features_func': _algo.features_func.__class__.__name__,
                'metrics_func': _algo.metrics_func.__name__,
                'result': _algo.metrics_func(
                    _algo.features_func.encode(image1).value,
                    _algo.features_func.encode(image2).value
                )
            })
        return compare_results
