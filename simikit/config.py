from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    # >= python3.11
    import tomllib
except ImportError:
    import tomli as tomllib


def get_project_root() -> Path:
    """
    Get the project root directory

    Returns:
        Path: The project root directory as a Path object.
    """
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()


@dataclass
class LogConfig:
    """
    A dataclass representing the logging configuration.

    Attributes:
        level (Literal['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL']):
            The logging level. Defaults to 'DEBUG'.
    """

    level: Literal['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'] = 'DEBUG'


@dataclass
class TransformerConfig:
    """
    A dataclass representing the transformer configuration.

    Attributes:
        cache_dir (None | str): The cache directory for transformers.
            If provided, it will be converted to an absolute path.
    """

    cache_dir: None | str = None

    def __post_init__(self):
        """
        Post-initialization method for TransformerConfig.

        If the cache directory is provided, it converts the path to an absolute path.
        Otherwise, it sets the cache directory to None.
        """
        if self.cache_dir:
            self.cache_dir = str(Path(self.cache_dir).absolute())
        else:
            self.cache_dir = None


class Config:
    """
    A singleton class for managing application configurations.

    This class loads configuration from a TOML file and provides access to logging and transformer configurations.
    """

    _INSTANCE = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """
        Implement the singleton pattern.

        If the singleton instance does not exist, create a new one. Otherwise, return the existing instance.

        Returns:
            Config: The singleton instance of the Config class.
        """
        if cls._INSTANCE is None:
            cls._INSTANCE = super().__new__(cls, *args, **kwargs)
        return cls._INSTANCE

    def __init__(self):
        """
        Initialize the configuration object.

        If the object has not been initialized, load the configuration from the TOML file,
        create LogConfig and TransformerConfig instances, and mark the object as initialized.
        """
        if not self._initialized:
            self.config = self._load_config()
            self._log_config = LogConfig(**self.config.get('log', {}))
            self._transformer_config = TransformerConfig(**self.config.get('transformers', {}))
            self._initialized = True

    @staticmethod
    def _get_config_path() -> Path:
        """
        Get the path to the configuration file.

        This method constructs the path to the configuration file based on the project root.
        If the file exists, it returns the path; otherwise, it raises a FileNotFoundError.

        Returns:
            Path: The path to the configuration file.

        Raises:
            FileNotFoundError: If the configuration file is not found in the config directory.
        """
        config_path = PROJECT_ROOT / 'config' / 'config.toml'
        if config_path.exists():
            return config_path
        raise FileNotFoundError('No configuration file found in config directory')

    def _load_config(self) -> dict[str, dict[str, Any]]:
        """
        Load the configuration from the TOML file.

        This method reads the TOML file using tomllib and returns the parsed configuration as a dictionary.

        Returns:
            Dict[str, Dict[str, Any]]: The parsed configuration dictionary.
        """
        config_path = self._get_config_path()
        with config_path.open('rb') as f:
            return tomllib.load(f)

    @property
    def log(self):
        """
        Get the logging configuration.

        Returns:
            LogConfig: An instance of the LogConfig class representing the logging configuration.
        """
        return self._log_config

    @property
    def transformers(self):
        """
        Get the transformer configuration.

        Returns:
            TransformerConfig: An instance of the TransformerConfig class representing the transformer configuration.
        """
        return self._transformer_config


config = Config()
