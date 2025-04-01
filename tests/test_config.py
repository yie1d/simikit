import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from simikit.config import get_project_root, LogConfig, TransformerConfig, Config


def test_get_project_root():
    root = get_project_root()
    assert isinstance(root, Path)


def test_log_config_default():
    log_config = LogConfig()
    assert log_config.level == 'DEBUG'


def test_transformer_config_no_cache_dir():
    transformer_config = TransformerConfig()
    assert transformer_config.cache_dir is None


def test_transformer_config_with_cache_dir():
    mock_cache_dir = 'test_cache'
    with patch('pathlib.Path.absolute', return_value=Path('/absolute/test_cache')):
        transformer_config = TransformerConfig(cache_dir=mock_cache_dir)
        assert transformer_config.cache_dir == str(Path('/absolute/test_cache'))


class TestConfig:
    @staticmethod
    def _release_config():
        Config._INSTANCE = None
        Config._initialized = False

    @patch.object(Config, '_get_config_path')
    @patch.object(Config, '_load_config')
    def test_config_singleton(self, mock_load_config, mock_get_config_path):
        self._release_config()
        mock_config_path = MagicMock()
        mock_get_config_path.return_value = mock_config_path
        mock_load_config.return_value = {'log': {'level': 'INFO'}, 'transformers': {'cache_dir': 'test'}}
        config1 = Config()
        config2 = Config()
        assert config1 is config2

    @patch.object(Config, '_get_config_path')
    @patch.object(Config, '_load_config')
    def test_config_log_property(self, mock_load_config, mock_get_config_path):
        self._release_config()
        mock_config_path = MagicMock()
        mock_get_config_path.return_value = mock_config_path
        mock_load_config.return_value = {'log': {'level': 'INFO'}, 'transformers': {'cache_dir': 'test'}}

        config = Config()
        log_config = config.log
        assert isinstance(log_config, LogConfig)
        assert log_config.level == 'INFO'

    @patch.object(Config, '_get_config_path')
    @patch.object(Config, '_load_config')
    def test_config_transformers_property(self, mock_load_config, mock_get_config_path):
        self._release_config()
        mock_config_path = MagicMock()
        mock_get_config_path.return_value = mock_config_path
        mock_load_config.return_value = {'log': {'level': 'INFO'}, 'transformers': {'cache_dir': 'test'}}
        config = Config()
        transformer_config = config.transformers
        assert isinstance(transformer_config, TransformerConfig)
        assert transformer_config.cache_dir is not None

    @patch('pathlib.Path.exists', return_value=False)
    def test_config_file_not_found(self, mock_get_config_path):
        self._release_config()
        with pytest.raises(FileNotFoundError):
            Config()
