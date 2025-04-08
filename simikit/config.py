import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from simikit.utils.error_catcher import context_error_catcher


def get_project_root() -> Path:
    """
    Get the project root directory

    Returns:
        Path: The project root directory as a Path object.
    """
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()


class LogConfig(BaseModel):
    level: Literal['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        'SUCCESS',
        description='The logging level.'
    )


class TransformerConfig(BaseModel):
    cache_dir: None | str = Field(None, description="The cache directory for transformers.")

    @field_validator('cache_dir', mode='after')
    @classmethod
    def validate_cache_dir(cls, cache_dir: str | None) -> str | None:
        if cache_dir:
            cache_dir = str(Path(cache_dir).absolute())
        else:
            cache_dir = None
        return cache_dir


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
            cls._INSTANCE = super().__new__(cls)
        return cls._INSTANCE

    def __init__(self, config_or_file_path: str | None | dict | Path = None):
        """
        Initialize the Config instance.

        If the instance has not been initialized, it attempts to load the configuration using the provided input.
        The input can be a dictionary containing the configuration data, a string or Path object representing
        the path to a TOML file. Based on the loaded configuration, it initializes the logging and transformer
        configuration instances.

        Args:
            config_or_file_path (str | None | dict | Path, optional):
                Can be a dictionary with configuration data, a string or Path object representing the path to a
                configuration file. If None, an empty configuration will be used. Defaults to None.
        """
        if not self._initialized:
            with context_error_catcher():
                self.config = self._load_config(config_or_file_path)

                self._log_config = LogConfig(**self.config.get('log', {}))
                self._transformer_config = TransformerConfig(**self.config.get('transformers', {}))
                self._initialized = True

    @staticmethod
    def _load_config(config_or_file_path: str | None | dict | Path = None) -> dict[str, dict[str, Any]]:
        """
        Load the configuration data.

        This method attempts to load the configuration based on the provided input.
        If the input is a dictionary, it is directly returned as the configuration.
        If the input is a string or Path object and the corresponding file exists, it reads the TOML file and
        returns the parsed data. Otherwise, it returns an empty dictionary.

        Args:
            config_or_file_path (str | None | dict | Path, optional):
                Can be a dictionary with configuration data, a string or Path object representing the path to
                a configuration file. If None, an empty dictionary will be returned. Defaults to None.

        Returns:
            dict[str, dict[str, Any]]: The loaded configuration data as a dictionary.
        """
        if isinstance(config_or_file_path, dict):
            return config_or_file_path
        elif isinstance(config_or_file_path, (str, Path)):
            if isinstance(config_or_file_path, str):
                config_or_file_path = Path(config_or_file_path)
            if config_or_file_path.suffix != '.toml':
                raise ValueError(f'Config file must have a .toml extension: {config_or_file_path}')
            if config_or_file_path.exists():
                with config_or_file_path.open('rb') as f:
                    return tomllib.load(f)
            else:
                return {}
        else:
            return {}

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


config = Config(PROJECT_ROOT / 'config' / 'config.toml')
