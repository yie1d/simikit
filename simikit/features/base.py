from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image

from simikit.utils.images import load_image


class BaseFeature:
    """
    A base class representing a feature.
    This class encapsulates feature data in a NumPy array.
    """

    TYPE: str = ''

    def __init__(self, data: np.ndarray):
        """
        Initialize a BaseFeature object.

        Args:
            data (np.ndarray): The feature data stored as a NumPy array.
        """
        self._data = data
        self._judge()

    def _judge(self):
        if not self.TYPE:
            raise ValueError('Feature type is not defined.')

    @property
    def type(self) -> str:
        """
        Get the type of the feature.

        Returns:
            str: The type of the feature.
        """
        return self.TYPE


class BaseExtractor(ABC):
    """
    An abstract base class for feature extractors.
    This class provides a template for extracting features from images.
    Subclasses should implement the _extract_algo method.
    """

    def __init__(self):
        """
        Initialize the BaseExtractor object.
        As an abstract base class, it serves as a foundation for subclasses.
        """
        ...

    def encode(self, image: str | Path | Image.Image) -> BaseFeature:
        """
        Encode an image into a feature object.
        If the input is a string or a Path object, the image is loaded using the load_image function.
        Then the feature is extracted using the _extract_algo method.

        Args:
            image (str | Path | Image.Image): The input image, which can be a file path or an Image object.

        Returns:
            BaseFeature: A feature object containing the extracted feature data.
        """
        if isinstance(image, (str, Path)):
            image = load_image(image)

        return self._extract_algo(image)

    @abstractmethod
    def _extract_algo(self, image: Image.Image) -> BaseFeature:
        """
        Extract features from an image.
        This is an abstract method that must be implemented by subclasses.
        It should take an Image object and return a BaseFeature object.

        Args:
            image (Image.Image): The input image.

        Returns:
            BaseFeature: The extracted feature object.
        """
        ...
