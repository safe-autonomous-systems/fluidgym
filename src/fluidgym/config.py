"""Global configuration for FluidGym."""

import logging
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from platformdirs import user_data_dir

logger = logging.getLogger("fluidgym.config")

DEFAULT_PALETTE = [
    "#003a7d",
    "#008dff",
    "#ff73b6",
    "#ff9d3a",
    "#4ecb8d",
    "#f9e858",
    "#d83034",
    "#c701ff",
]


FP32 = torch.float32
FP64 = torch.float64


ALLOWED_DTYPES = {
    "FP32": FP32,
    "FP64": FP64,
}


class ConfigKey(Enum):
    """Configuration keys for FluidGym."""

    HF_LOCAL_DOMAINS_REPO_ID = "hf_intial_domains_repo_id"
    LOCAL_DATA_PATH = "local_data_path"
    DTYPE = "dtype"


class Config:
    """Global configuration for FluidGym."""

    def __init__(self):
        self.settings = {
            ConfigKey.HF_LOCAL_DOMAINS_REPO_ID: (
                "safe-autonomous-systems/fluidgym-data"
            ),
            ConfigKey.LOCAL_DATA_PATH: Path(user_data_dir("FluidGym", "fluidgym")),
            ConfigKey.DTYPE: FP32,
        }

    def __parse_key(self, key: str) -> ConfigKey:
        try:
            _key = ConfigKey(key)
        except ValueError as err:
            raise ValueError(
                f"Key '{key}' is not a valid configuration key. Allowed keys are: "
                f"{list(ConfigKey)}"
            ) from err

        return _key

    def __parse_dtype(self, value: str) -> Any:
        if value not in ALLOWED_DTYPES:
            raise ValueError(
                f"Value '{value}' is not a valid data type. Allowed values are:"
                f"{list(ALLOWED_DTYPES.keys())}"
            )

        return ALLOWED_DTYPES[value]

    def update(self, key: str, value: str) -> None:
        """Update the configuration with the given key and value.

        Parameters
        ----------
        key: str
            The configuration key.

        value: str
            The value to set for the configuration key.

        Raises
        ------
        ValueError
            If the key is not valid or the value is not valid for the key.
        """
        _key = self.__parse_key(key)

        if _key == ConfigKey.DTYPE:
            self.settings[_key] = self.__parse_dtype(value)
        elif _key == ConfigKey.LOCAL_DATA_PATH:
            self.settings[_key] = Path(value).resolve()
        else:
            raise ValueError(f"Unhandled configuration key: {_key}")

    def get(self, key: str) -> Any | None:
        """Get the value for the given configuration key.

        Parameters
        ----------
        key: str
            The configuration key.

        Returns
        -------
        Any | None
            The value associated with the key, or None if the key does not exist.
        """
        _key = self.__parse_key(key)

        return self.settings.get(_key)

    def __getitem__(self, key: str) -> Any:
        _key = self.__parse_key(key)

        value = self.settings.get(_key)

        # Since we check the key in the get method,
        # we can assume value is not None here.
        assert value is not None

        return value

    def __setitem__(self, key: str, value: str) -> None:
        self.update(key, value)

    @property
    def hf_intial_domains_repo_id(self) -> str:
        """The Hugging Face repository ID for initial domain data."""
        return self.settings[ConfigKey.HF_LOCAL_DOMAINS_REPO_ID]

    @property
    def local_data_path(self) -> Path:
        """Path to the local data directory."""
        return self.settings[ConfigKey.LOCAL_DATA_PATH]

    @property
    def initial_domains_path(self) -> Path:
        """Path to the initial domains data directory."""
        return self.local_data_path / "initial_domains"

    @property
    def dtype(self) -> torch.dtype:
        """The default data type for tensors in FluidGym."""
        return self.settings[ConfigKey.DTYPE]

    @property
    def palette(self) -> list[str]:
        """The default color palette for visualizations in FluidGym."""
        return DEFAULT_PALETTE


config = Config()
