from abc import abstractmethod
from typing import Literal

import numpy as np
import pywt
from PIL import Image
from pydantic import Field
from scipy.fftpack import dct

from simikit.features.base import BaseExtractor, BaseFeature
from simikit.utils.images import resize_image

__all__ = [
    'AHash',
    'DHash',
    'PHash',
    'WHash',
]
WhashWavelet = Literal['haar', 'db4']


class HashFeature(BaseFeature):
    """
    Base class for hash features.

    This class serves as a base for all hash - based feature classes. It inherits from BaseFeature
    and provides a common structure for hash features, including a specific type and a method to
    convert the binary hash array to a hexadecimal string.
    """
    _hex: str = None
    type: str = Field('hash', description='The type of the feature.')

    @property
    def hex(self) -> str:
        if self._hex is None:
            self._hex = self._binary_array_to_hex(self.data)
        return self._hex

    @staticmethod
    def _binary_array_to_hex(hash_array: np.ndarray) -> str:
        """
        Convert a binary array to a hexadecimal string.

        Args:
            hash_array (np.ndarray): The binary array to be converted.

        Returns:
            str: The hexadecimal string representation of the binary array.
        """
        formatter = np.vectorize(lambda x: f'{x:02x}')
        packed = np.packbits(hash_array)
        hex_chars = formatter(packed)
        return ''.join(hex_chars)


class BaseImageHash(BaseExtractor):
    """
    An abstract base class for image hashing algorithms.
    This class provides a common structure for different image hashing algorithms.
    Subclasses should implement the _hash_algo method.
    """

    def __init__(self, hash_size: int = 8):
        """
        Initialize a BaseImageHash object.

        Args:
            hash_size (int, optional): The size of the hash. Must be greater than or equal to 2. Defaults to 8.
        """
        super().__init__()
        if hash_size < 2:
            raise ValueError('Hash size must be greater than or equal to 2')
        self._image_size = (hash_size, hash_size)
        self._origin_image: None | Image.Image = None

    def _extract_algo(self, image: Image.Image) -> HashFeature:
        """
        Extract the hash feature from an image.
        Preprocess the image, apply the hashing algorithm, and return a HashFeature object.

        Args:
            image (Image.Image): The input image.

        Returns:
            HashFeature: The extracted hash feature.
        """
        image_array = self._preprocess_image(image)
        hash_array = self._hash_algo(image_array).astype(int)
        return HashFeature(data=hash_array)

    def _special_preprocess(self, image: Image.Image) -> Image.Image:
        """
        Perform special pre-processing on the image.
        By default, it returns the original image. Subclasses can override this method.

        Args:
            image (Image.Image): The input image.

        Returns:
            Image.Image: The pre-processed image.
        """
        return image

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess the image for hashing.
        Perform special pre-processing, convert to grayscale, resize the image, and return a NumPy array.

        Args:
            image (Image.Image): The input image.

        Returns:
            np.ndarray: The pre-processed image as a NumPy array.
        """
        image = self._special_preprocess(image)

        if image.mode != 'L':
            image = image.convert('L')

        resized_image = resize_image(image, self._image_size)
        return np.asarray(resized_image)

    @abstractmethod
    def _hash_algo(self, image_array: np.ndarray) -> np.ndarray:
        """
        Apply the hashing algorithm to the pre-processed image array.
        This is an abstract method that must be implemented by subclasses.

        Args:
            image_array (np.ndarray): The pre-processed image array.

        Returns:
            np.ndarray: The binary array representing the hash.
        """
        ...


class AHash(BaseImageHash):
    """Average Hash"""

    def _hash_algo(self, image_array: np.ndarray) -> np.ndarray:
        """Computes the average hash of an image.

        The average hash algorithm calculates the mean intensity of the image's pixels.
        Each pixel's intensity is then compared to this mean. If the pixel's intensity
        is greater than the mean, it is represented by '1', otherwise by '0'. This produces
        a binary hash that uniquely represents the image.

        Args:
            image_array (np.ndarray): The image array to be hashed.

        Returns:
            np.ndarray: A binary array representing the average hash of the input image.

        """
        avg = np.mean(image_array)
        return image_array > avg


class DHash(BaseImageHash):
    """Difference Hash."""

    def __init__(self, hash_size: int = 8, vertical: bool = False):
        """Initializes the DHash object with specified hash size and orientation.

        Args:
            hash_size (int): The size of the hash. Default is 8.
            vertical (bool): If True, computes the hash vertically; otherwise, horizontally. Default is False.

        """
        super().__init__(hash_size)
        self._vertical = vertical
        if self._vertical:
            self._image_size = (hash_size, hash_size + 1)
        else:
            self._image_size = (hash_size + 1, hash_size)

    def _hash_algo(self, image_array: np.ndarray) -> np.ndarray:
        """Computes the difference hash of the given image array.

        Args:
            image_array (np.ndarray): A numpy array representing the image.

        Returns:
            np.ndarray: A numpy array containing the computed hash values.

        """
        if self._vertical:
            return image_array[1:, :] > image_array[:-1, :]
        else:
            return image_array[:, 1:] > image_array[:, :-1]


class PHash(BaseImageHash):
    """Perceptual Hash (pHash)."""

    def __init__(
        self,
        hash_size: int = 8,
        simple: bool = False,
        highfreq_factor: int = 4,
    ):
        """Initializes the PHash object with specified parameters.

        Args:
            hash_size (int): The size of the hash. Default is 8.
            simple (bool): If True, uses a simpler version of the algorithm. Default is False.
            highfreq_factor (int): A factor to adjust the size of the DCT result. Default is 4.

        """
        super().__init__(hash_size * highfreq_factor)
        self._simple = simple
        self._hash_size = hash_size

    def _hash_algo(self, image_array: np.ndarray) -> np.ndarray:
        """Computes the perceptual hash of the given image array.

        Args:
            image_array (np.ndarray): A numpy array representing the image.

        Returns:
            np.ndarray: A numpy array containing the computed hash values.

        """
        if self._simple:
            dct_result = dct(image_array)
            # Extract the low-frequency components of the DCT result.
            # In the simple mode, we skip the first column to focus on non-DC components.
            low_freq = dct_result[: self._hash_size, 1: self._hash_size + 1]
            threshold = low_freq.mean()
        else:
            dct_result = dct(dct(image_array, axis=0), axis=1)
            # Extract the low-frequency components from the top-left corner of the 2D DCT result
            low_freq = dct_result[: self._hash_size, : self._hash_size]
            # Flatten the low-frequency components and exclude the DC component (first element)
            flattened_low_freq = np.ndarray.flatten(low_freq)[1:]
            threshold = np.median(flattened_low_freq)

        return np.asarray(low_freq > threshold)


class WHash(BaseImageHash):
    """Wavelet Hash"""

    def __init__(
        self,
        hash_size: int = 8,
        image_scale: int | None = None,
        wavelet_func: WhashWavelet = 'haar',
    ):
        """Initialize the WHash object.

        Args:
            hash_size (int): The size of the hash. It must be a power of 2. Default is 8.
            image_scale (int | None): The scale of the image. If None, it will be calculated automatically.
                                      It must be a power of 2 if provided. Default is None.
            wavelet_func (WhashWavelet): The wavelet function to use for wavelet decomposition. Default is 'haar'.

        """
        if image_scale is not None:
            assert image_scale & (image_scale - 1) == 0, 'image_scale is not power of 2'
        assert hash_size & (hash_size - 1) == 0, 'hash_size is not power of 2'

        super().__init__(hash_size)
        self._wavelet_func = wavelet_func
        self._hash_size = hash_size

        self._image_scale = image_scale
        self._decomposition_level: None | int = int(np.log2(self._hash_size))
        self._dwt_level: None | int = None

    def _special_preprocess(self, image: Image.Image) -> Image.Image:
        """Perform special pre-processing on the input image.

        Args:
            image (Image.Image): The input image to be pre-processed.

        Returns:
            Image.Image: The pre-processed image.

        """
        image_scale = self._image_scale
        if image_scale is None:
            # Calculate the natural scale of the image
            image_natural_scale = 2 ** int(np.log2(min(image.size)))
            image_scale = max(image_natural_scale, self._hash_size)

        max_ll_level = int(np.log2(image_scale))

        assert self._decomposition_level <= max_ll_level, 'hash_size in a wrong range'
        self._dwt_level = max_ll_level - self._decomposition_level
        self._image_size = (image_scale, image_scale)

        return image

    def _hash_algo(self, image_array: np.ndarray) -> np.ndarray:
        """Compute the wavelet hash for the input image array.

        Args:
            image_array (np.ndarray): The input image array.

        Returns:
            np.ndarray: The computed wavelet hash as a binary array.

        """
        image_array = image_array / 255
        # Perform wavelet decomposition
        wavelet_coefficients = pywt.wavedec2(data=image_array, wavelet=self._wavelet_func, level=self._dwt_level)

        low_frequency_coefficients = wavelet_coefficients[0]

        # Calculate the median of the low-frequency coefficients
        median_coefficient_value = np.median(np.ndarray.flatten(low_frequency_coefficients))

        return np.asarray(low_frequency_coefficients > median_coefficient_value)
