from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image
from transformers import BaseImageProcessor, BatchFeature, PreTrainedModel

from simikit.features.transformer import CACHE_DIR, BaseTransformer, DinoV2, TransformerFeature, Vit


class TestTransformerFeature:
    def test_str_and_repr(self):
        data = np.array([1, 2, 3])
        feature = TransformerFeature(data)
        assert str(feature) == str(data)
        assert repr(feature) == str(data)


class TestBaseTransformer:
    @patch.object(BaseTransformer, '_judge_model')
    def test_singleton_pattern(self, mock_judge):
        class MockTransformer(BaseTransformer):
            def _init_model(self):
                pass

        instance1 = MockTransformer()
        instance2 = MockTransformer()
        assert instance1 is instance2

    def test_new_instance(self):
        class MockTransformer(BaseTransformer):
            def _init_model(self):
                pass

        with patch.object(MockTransformer, '_judge_model') as mock_judge:
            with patch.object(MockTransformer, '_init_model') as mock_init:
                instance = MockTransformer()
                mock_init.assert_called_once()
                mock_judge.assert_called_once()

    def test_judge_model(self):
        class MockTransformer(BaseTransformer):
            def _init_model(self):
                self.model = MagicMock(spec=PreTrainedModel)
                self.image_processor = MagicMock(spec=BaseImageProcessor)
                self.pretrained_model_name_or_path = 'test_model'

        transformer = MockTransformer()
        transformer._judge_model()

    @patch.object(BaseTransformer, 'image_processor')
    @patch.object(BaseTransformer, '_get_embedding')
    def test_extract_algo(self, mock_get_embedding, mock_image_processor):
        class MockTransformer(BaseTransformer):
            def _init_model(self):
                pass

        mock_image = Image.new('RGB', (100, 100))
        mock_image_array = MagicMock(spec=BatchFeature)
        mock_image_processor.return_value = mock_image_array

        with patch.object(MockTransformer, '_judge_model'):
            with patch.object(MockTransformer, '_init_model'):
                transformer = MockTransformer()
                result = transformer._extract_algo(mock_image)
                mock_image_processor.assert_called_once_with(mock_image, return_tensors='pt')
                mock_get_embedding.assert_called_once_with(mock_image_array)
                assert result == mock_get_embedding.return_value

    @patch('torch.no_grad')
    def test_get_embedding(self, mock_no_grad):
        class MockTransformer(BaseTransformer):
            def _init_model(self):
                self.model = MagicMock(spec=PreTrainedModel)

        mock_image_array = MagicMock(spec=BatchFeature)
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.tensor([[[1, 2, 3]]])
        mock_no_grad.return_value.__enter__.return_value = None

        with patch.object(MockTransformer, '_judge_model'):
            MockTransformer().model.return_value = mock_outputs
            transformer = MockTransformer()
            result = transformer._get_embedding(mock_image_array)
            assert isinstance(result, np.ndarray)


class TestVit:
    @patch('transformers.AutoImageProcessor.from_pretrained')
    @patch('transformers.ViTModel.from_pretrained')
    def test_init_model(self, mock_vit_model, mock_image_processor):
        with patch.object(Vit, '_judge_model'):
            vit = Vit()
            mock_image_processor.assert_called_once_with(
                vit.pretrained_model_name_or_path, cache_dir=CACHE_DIR, use_fast=True
            )
            mock_vit_model.assert_called_once_with(vit.pretrained_model_name_or_path, cache_dir=CACHE_DIR)


class TestDinoV2:
    @patch('transformers.AutoImageProcessor.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_init_model(self, mock_auto_model, mock_image_processor):
        with patch.object(DinoV2, '_judge_model'):
            dino_v2 = DinoV2()
            mock_image_processor.assert_called_once_with(
                dino_v2.pretrained_model_name_or_path, cache_dir=CACHE_DIR, use_fast=False
            )
            mock_auto_model.assert_called_once_with(dino_v2.pretrained_model_name_or_path, cache_dir=CACHE_DIR)
