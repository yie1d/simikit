from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from simikit.features.hash import AHash, BaseImageHash, DHash, HashFeature, PHash, WHash


class TestHashFeature:
    def test_init(self):
        data = np.array([1, 0, 1])
        feature = HashFeature(data)
        assert isinstance(feature._data, np.ndarray)
        assert np.array_equal(feature._data, data)

    def test_binary_array_to_hex(self):
        data = np.array([1, 0, 1])
        feature = HashFeature(data)
        hex_str = feature._binary_array_to_hex(data)
        assert isinstance(hex_str, str)

    def test_str_and_repr(self):
        data = np.array([1, 0, 1])
        feature = HashFeature(data)
        assert str(feature) == feature._hex
        assert repr(feature) == feature._hex


class ConcreteBaseImageHash(BaseImageHash):
    def _hash_algo(self, image_array: np.ndarray) -> np.ndarray:
        return image_array


class TestBaseImageHash:
    def test_init(self):
        with pytest.raises(ValueError):
            ConcreteBaseImageHash(hash_size=1)
        hash_obj = ConcreteBaseImageHash()
        assert hash_obj._image_size == (8, 8)

    @patch('simikit.features.hash.resize_image')
    def test_preprocess_image(self, mock_resize):
        mock_image = Image.new('RGB', (100, 100))
        hash_obj = ConcreteBaseImageHash()
        result = hash_obj._preprocess_image(mock_image)
        mock_resize.assert_called_once_with(mock_image.convert('L'), (8, 8))
        assert isinstance(result, np.ndarray)

    def test_extract_algo(self):
        mock_image = Image.new('RGB', (100, 100))

        hash_obj = ConcreteBaseImageHash()
        with patch.object(hash_obj, '_preprocess_image', return_value=np.array([[1, 2], [3, 4]])) as mock_preprocess:
            with patch.object(hash_obj, '_hash_algo', return_value=np.array([1, 0, 1])) as mock_hash_algo:
                result = hash_obj._extract_algo(mock_image)
                mock_preprocess.assert_called_once_with(mock_image)
                mock_hash_algo.assert_called_once_with(mock_preprocess.return_value)
                assert isinstance(result, HashFeature)


class TestAHash:
    def test_hash_algo(self):
        image_array = np.array([[1, 2], [3, 4]])
        a_hash = AHash()
        hash_result = a_hash._hash_algo(image_array)
        assert isinstance(hash_result, np.ndarray)


class TestDHash:
    def test_init(self):
        d_hash = DHash()
        assert d_hash._image_size == (9, 8)
        d_hash_vertical = DHash(vertical=True)
        assert d_hash_vertical._image_size == (8, 9)

    def test_hash_algo(self):
        image_array = np.array([[1, 2], [3, 4]])
        d_hash = DHash()
        hash_result = d_hash._hash_algo(image_array)
        assert isinstance(hash_result, np.ndarray)


class TestPHash:
    def test_init(self):
        p_hash = PHash()
        assert p_hash._hash_size == 8

    def test_simple(self):
        p_hash = PHash(simple=True)
        image_array = np.array([[1, 2], [3, 4]])
        hash_result = p_hash._hash_algo(image_array)
        assert isinstance(hash_result, np.ndarray)

    @patch('scipy.fftpack.dct')
    def test_hash_algo(self, mock_dct):
        image_array = np.array([[1, 2], [3, 4]])
        p_hash = PHash()
        hash_result = p_hash._hash_algo(image_array)
        assert isinstance(hash_result, np.ndarray)


class TestWHash:
    def test_init(self):
        with pytest.raises(AssertionError):
            WHash(hash_size=3)
        w_hash = WHash()
        assert w_hash._hash_size == 8

    def test_image_scale(self):
        with pytest.raises(AssertionError):
            WHash(image_scale=3)

    def test_special_preprocess(self):
        mock_image = Image.new('RGB', (100, 100))
        w_hash = WHash()
        result = w_hash._special_preprocess(mock_image)
        assert isinstance(result, Image.Image)

    @patch('pywt.wavedec2')
    def test_hash_algo(self, mock_wavedec2):
        image_array = np.array([[1, 2], [3, 4]])
        mock_wavedec2.return_value = [image_array]
        w_hash = WHash()
        w_hash._dwt_level = 1
        hash_result = w_hash._hash_algo(image_array)
        assert isinstance(hash_result, np.ndarray)
