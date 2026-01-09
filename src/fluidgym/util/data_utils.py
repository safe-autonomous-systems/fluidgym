"""Utilitiy functions for data saving and loading."""

import json
import logging
import os
from pathlib import Path

import pandas as pd

from fluidgym.config import config as fluidgym_config

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "True"

from huggingface_hub import snapshot_download

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

STATISTICS_FILENAME = "domain_statistics.json"
UNCONTROLLED_EPISODE_FILENAME = "uncontrolled_episode.csv"


def prepare_initial_domains(initial_domain_id: str) -> None:
    """Check if the initial domain data exists and download it  from HuggingFace if not.

    Parameters
    ----------
    initial_domain_id: str
        The initial domain identifier.
    """
    domain_dir = fluidgym_config.initial_domains_path / initial_domain_id
    exists = domain_dir.exists() and any(domain_dir.iterdir())

    logger = logging.getLogger("fluidgym")
    if not exists:
        logger.info("Initial domain data not found. Downloading it from huggingface...")
        download_initial_domains(initial_domain_id)
        logger.info("Initial domain data downloaded.")

    # Check again
    exists = domain_dir.exists() and any(domain_dir.iterdir())
    if not exists:
        raise RuntimeError(
            f"Failed to download initial domain data for {initial_domain_id}."
        )


def download_initial_domains(initial_domain_id: str) -> None:
    """Download initial domain data to the user data directory.

    Parameters
    ----------
    initial_domain_id: str
        The initial domain identifier.
    """
    fluidgym_config.local_data_path.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=fluidgym_config.hf_intial_domains_repo_id,
        repo_type="dataset",
        allow_patterns=f"initial_domains/{initial_domain_id}/**",
        local_dir=fluidgym_config.local_data_path,
    )


def save_statistics(path: Path, data: dict[str, dict[str, float]]) -> None:
    """Save a domain statistics dictionary to a JSON file.

    Parameters
    ----------
    path: Path
        The file path where the JSON will be saved.

    data: dict[str, float]
        The statistics dictionary to save.
    """
    with open(path / STATISTICS_FILENAME, "w") as f:
        json.dump(data, f, indent=4)


def load_statistics(path: Path) -> dict:
    """Load a domain statistics dictionary from a JSON file.

    Parameters
    ----------
    path: Path
        The file path from which to load the JSON.

    Returns
    -------
    dict
        The loaded statistics dictionary.
    """
    with open(path / STATISTICS_FILENAME) as f:
        data = json.load(f)
    return data


def save_uncontrolled_episode(
    path: Path, mode_name: str, episode_df: pd.DataFrame
) -> None:
    """Save uncontrolled episode data to a CSV file.

    Parameters
    ----------
    path: Path
        The directory path where the CSV will be saved.

    mode_name: str
        The environment mode name ('train', 'val', 'test').

    episode_df: pd.DataFrame
        The DataFrame containing episode metrics.
    """
    episode_df.to_csv(
        path / (mode_name + "_" + UNCONTROLLED_EPISODE_FILENAME), index=False
    )


def load_uncontrolled_episode(path: Path, mode_name: str) -> pd.DataFrame:
    """Load uncontrolled episode data from a CSV file.

    Parameters
    ----------
    path: Path
        The directory path from which to load the CSV.

    mode_name: str
        The environment mode name ('train', 'val', 'test').

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame containing episode metrics.
    """
    return pd.read_csv(path / (mode_name + "_" + UNCONTROLLED_EPISODE_FILENAME))
