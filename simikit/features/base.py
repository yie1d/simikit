from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, field_validator

from simikit.utils.images import load_image

__all__ = [
    'BaseFeature',
    'BaseExtractor'
]


class BaseFeature(BaseModel):
    """
    A Pydantic model representing a base feature.

    This class is used to encapsulate feature data along with its type.
    It uses Pydantic for data validation, ensuring that the feature data is in the correct format.
    """
    model_config = {
        'arbitrary_types_allowed': True
    }  # allow arbitrary types
    type: str = Field(..., description='The type of the feature.')
    data: np.ndarray = Field(..., description='The feature data stored as a NumPy array.')

    @field_validator('data', mode='before')
    @classmethod
    def validate_data(cls, v: np.ndarray) -> np.ndarray:
        """
        Validate that the provided data is a NumPy array.

        This class method is used as a field validator for the 'data' field.
        It ensures that the input value for the 'data' field is of type numpy.ndarray.

        Args:
            v (np.ndarray): The input value for the 'data' field.

        Raises:
            TypeError: If the input value is not a NumPy array.

        Returns:
            np.ndarray: The input value if it is a NumPy array.
        """
        if isinstance(v, np.ndarray) is False:
            raise TypeError('The data must be a NumPy array.')
        return v

    @property
    def value(self) -> np.ndarray:
        """
        Get the value of the feature data.

        This property provides a convenient way to access the 'data' field of the BaseFeature instance.

        Returns:
            The feature data as a NumPy array.
        """
        return self.data


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
