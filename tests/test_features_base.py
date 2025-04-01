from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from simikit.features.base import BaseExtractor, BaseFeature


class ConcreteFeature(BaseFeature):
    TYPE = 'test'


class TestBaseFeature:
    def test_base_feature_initialization(self):
        data = np.random.rand(4, 8)

        feature = ConcreteFeature(data)
        assert isinstance(feature, BaseFeature)
        assert np.array_equal(feature._data, data)

    def test_base_feature_type_property(self):
        data = np.random.rand(4, 8)

        feature = ConcreteFeature(data)
        assert feature.type == 'test'

    def test_base_feature_type_not_defined(self):
        data = np.random.rand(4, 8)

        with pytest.raises(ValueError) as excinfo:
            BaseFeature(data)

        assert 'Feature type is not defined.' in str(excinfo.value)


class ConcreteExtractor(BaseExtractor):
    def _extract_algo(self, image: Image.Image) -> BaseFeature:
        mock_data = np.random.rand(4, 8)
        return ConcreteFeature(mock_data)


class TestBaseExtractor:
    def test_base_extractor_encode_with_image_object(self):
        extractor = ConcreteExtractor()
        mock_image = Image.new('RGB', (100, 100))

        result = extractor.encode(mock_image)
        assert isinstance(result, ConcreteFeature)

    def test_base_extractor_encode_with_file_path(self):
        extractor = ConcreteExtractor()
        mock_file_path = '/fake/image.png'
        mock_image = Image.new('RGB', (100, 100))

        with patch('simikit.features.base.load_image', return_value=mock_image) as mock_load:
            result = extractor.encode(mock_file_path)
            mock_load.assert_called_once_with(mock_file_path)
            assert isinstance(result, BaseFeature)

    def test_abstract_method_not_implemented(self):
        class IncompleteExtractor(BaseExtractor):
            ...

        with pytest.raises(TypeError):
            IncompleteExtractor()
