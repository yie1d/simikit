from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, BaseImageProcessor, BatchFeature, PreTrainedModel, ViTModel

from simikit.config import config
from simikit.features.base import BaseExtractor, BaseFeature

CACHE_DIR = Path(config.transformers.cache_dir)

if CACHE_DIR:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


class TransformerFeature(BaseFeature):
    """
    A class representing features extracted by a transformer model.
    It inherits from the BaseFeature class and provides specific string and representation methods.
    """

    TYPE = 'transformer'

    def __str__(self):
        """
        Return the string representation of the feature data.

        Returns:
            str: The feature data.
        """
        return str(self._data)

    def __repr__(self):
        """
        Return the string representation of the feature data for debugging purposes.

        Returns:
            str: The feature data.
        """
        return str(self._data)


class BaseTransformer(BaseExtractor):
    """
    An abstract base class for transformer - based feature extractors.
    It implements the singleton pattern and provides methods for initializing and validating models,
    as well as extracting features from images.
    """

    _INSTANCE = None
    _initialized = False
    image_processor: None | BaseImageProcessor = None
    model: None | PreTrainedModel = None
    pretrained_model_name_or_path: str = ''

    def __new__(cls, *args, **kwargs):
        """
        Implement the singleton pattern. Create a new _INSTANCE if it doesn't exist.

        Returns:
            BaseTransformer: The singleton _INSTANCE of the class.
        """
        if cls._INSTANCE is None:
            cls._INSTANCE = super().__new__(cls, *args, **kwargs)
        return cls._INSTANCE

    def __init__(self):
        """
        Initialize the transformer feature extractor.
        This method should be called only once per class, and it will initialize the model and image processor.
        """
        if not self._initialized:
            super().__init__()
            self._init_model()
            self._judge_model()
            self._initialized = True

    @abstractmethod
    def _init_model(self):
        """
        Abstract method to initialize the image processor and the model.
        Subclasses should implement this method to set up the specific model and image processor.
        """
        ...

    def _judge_model(self):
        """
        Validate the model and image processor.
        Ensure that the model is an instance of PreTrainedModel and the image processor
        is an instance of BaseImageProcessor.
        """
        assert isinstance(self.model, PreTrainedModel), 'model must be a transformers model'
        assert isinstance(self.image_processor, BaseImageProcessor), (
            'image_processor must be a transformers image processor'
        )
        assert self.pretrained_model_name_or_path, 'pretrained_model_name_or_path must be specified'

    def _extract_algo(self, image: Image.Image):
        """
        Extract features from an image.
        Process the image using the image processor and then get the embedding from the model.

        Args:
            image (Image.Image): The input image.

        Returns:
            np.ndarray: The extracted feature embedding.
        """
        image_array = self.image_processor(image, return_tensors='pt')
        return self._get_embedding(image_array)

    def _get_embedding(self, image_array: BatchFeature) -> np.ndarray:
        """
        Get the feature embedding from the model's output.
        Use torch.no_grad() to disable gradient calculation and extract the last hidden state.

        Args:
            image_array (BatchFeature): The processed image batch.

        Returns:
            np.ndarray: The feature embedding as a NumPy array.
        """
        with torch.no_grad():
            outputs = self.model(**image_array)
        embedding = outputs.last_hidden_state
        embedding = embedding[:, 0, :].squeeze(1)
        return embedding.numpy()


class Vit(BaseTransformer):
    """
    A class representing the Vision Transformer (ViT) model for feature extraction.
    It inherits from BaseTransformer and implements the model initialization method.
    """

    pretrained_model_name_or_path = 'google/vit-large-patch16-224-in21k'

    def _init_model(self):
        """
        Initialize the image processor and the ViT model.
        Use pre-trained models from the specified checkpoints and cache directory.
        """
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.pretrained_model_name_or_path, cache_dir=CACHE_DIR, use_fast=True
        )
        self.model = ViTModel.from_pretrained(self.pretrained_model_name_or_path, cache_dir=CACHE_DIR)


class DinoV2(BaseTransformer):
    """
    A class representing the DINO-v2 model for feature extraction.
    It inherits from BaseTransformer and implements the model initialization method.
    """

    pretrained_model_name_or_path = 'facebook/dinov2-base'

    def _init_model(self):
        """
        Initialize the image processor and the DINO-v2 model.
        Use pre-trained models from the specified checkpoints and cache directory.
        """
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.pretrained_model_name_or_path, cache_dir=CACHE_DIR, use_fast=False
        )
        self.model = AutoModel.from_pretrained(self.pretrained_model_name_or_path, cache_dir=CACHE_DIR)
