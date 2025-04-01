import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from simikit.config import tomllib, get_project_root, LogConfig, TransformerConfig, Config


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

    @patch.object(Config, '_load_config')
    def test_config_singleton(self, mock_load_config):
        self._release_config()
        mock_load_config.return_value = {'log': {'level': 'INFO'}, 'transformers': {'cache_dir': 'test'}}
        config1 = Config()
        config2 = Config()
        assert config1 is config2

    @patch.object(Config, '_load_config')
    def test_config_log_property(self, mock_load_config):
        self._release_config()
        mock_load_config.return_value = {'log': {'level': 'INFO'}, 'transformers': {'cache_dir': 'test'}}
        config = Config()
        log_config = config.log
        assert isinstance(log_config, LogConfig)
        assert log_config.level == 'INFO'

    @patch.object(Config, '_load_config')
    def test_config_transformers_property(self, mock_load_config):
        self._release_config()
        mock_load_config.return_value = {'log': {'level': 'INFO'}, 'transformers': {'cache_dir': 'test'}}
        config = Config()
        transformer_config = config.transformers
        assert isinstance(transformer_config, TransformerConfig)
        assert transformer_config.cache_dir == str(Path('test').absolute())

    @patch.object(tomllib, 'load')
    def test_load_config_from_dict(self, mock_toml_load):
        self._release_config()
        mock_config_dict = {'log': {'level': 'DEBUG'}, 'transformers': {'cache_dir': 'test'}}
        result = Config._load_config(mock_config_dict)
        assert result == mock_config_dict
        mock_toml_load.assert_not_called()

    def test_load_config_from_str(self):
        self._release_config()
        fake_path = 'fake.toml'
        mock_config_dict = {}
        mock_file = MagicMock(spec=str)
        mock_file.return_value = 'fake.toml'
        with patch('pathlib.Path.exists', return_value=False):
            result = Config._load_config(fake_path)
        assert result == mock_config_dict

    def test_load_config_from_none(self):
        self._release_config()
        result = Config._load_config(None)
        assert result == {}

    @patch.object(tomllib, 'load')
    def test_load_config_from_file(self, mock_toml_load):
        self._release_config()
        mock_file_path = MagicMock(spec=Path)
        mock_file_path.exists.return_value = True
        mock_file_path.suffix = '.toml'
        mock_file = MagicMock()
        mock_file_path.open.return_value.__enter__.return_value = mock_file
        mock_toml_load.return_value = {'log': {'level': 'DEBUG'}, 'transformers': {'cache_dir': 'test'}}
        result = Config._load_config(mock_file_path)
        assert result == mock_toml_load.return_value
        mock_toml_load.assert_called_once_with(mock_file)

    def test_load_config_invalid_file_extension(self):
        self._release_config()
        mock_file_path = MagicMock(spec=Path)
        mock_file_path.suffix = '.txt'
        with pytest.raises(ValueError):
            Config._load_config(mock_file_path)

    def test_load_config_nonexistent_file(self):
        self._release_config()
        mock_file_path = MagicMock(spec=Path)
        mock_file_path.exists.return_value = False
        mock_file_path.suffix = '.toml'
        result = Config._load_config(mock_file_path)
        assert result == {}

