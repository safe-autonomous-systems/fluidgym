import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

import fluidgym
from fluidgym.envs.cylinder.cylinder_env_base import CylinderEnvBase
from fluidgym.envs.fluid_env import N_INITIAL_DOMAINS, EnvMode

logger = logging.getLogger("fluidgym.preparation")
logging.basicConfig(level=logging.WARNING)


fluidgym.config.update("local_data_path", "./local_data")


@hydra.main(version_base="1.3", config_path="../configs", config_name="preparation")
def main(cfg: DictConfig):
    try:
        aggregate_statistics(cfg)
    except Exception as e:
        logger.exception("Job crashed with an exception")
        logger.error(e)
        raise


def aggregate_statistics(cfg: DictConfig):
    env = fluidgym.make(
        cfg.env_id,
        **cfg.env_kwargs,
        load_initial_domain=False,
        load_domain_statistics=False,
    )
    logger.info("Initialized environment.")
    env.reset(seed=42)

    all_stats = []
    for i in range(N_INITIAL_DOMAINS):
        for mode in [EnvMode.TRAIN, EnvMode.VAL, EnvMode.TEST]:
            try:
                all_stats += [env._load_uncontrolled_episode(idx=i, mode=mode)]
            except FileNotFoundError:
                logger.warning(
                    f"Uncontrolled episode for mode {mode} and domain {i} not found. "
                    f"Skipping."
                )
                continue

    all_stats = pd.concat(all_stats, ignore_index=True)
    logger.info("Collected all uncontrolled episode statistics.")

    metrics = env.metrics
    if isinstance(env, CylinderEnvBase):
        metrics += ["abs_lift"]

    domain_stats: dict[str, dict[str, float]] = defaultdict(dict)
    for metric in env.metrics:
        domain_stats[metric]["mean"] = float(all_stats[metric].mean())
        domain_stats[metric]["min"] = all_stats[metric].min()
        domain_stats[metric]["max"] = all_stats[metric].max()

        for p in [5, 25, 50, 75, 95]:
            domain_stats[metric][f"p{p}"] = float(np.percentile(all_stats[metric], p))

    for name in ["velocity_magnitude", "pressure", "vorticity_magnitude"]:
        for stat in ["mean", "min", "max", "p5", "p25", "p50", "p75", "p95"]:
            col = f"{name}_{stat}"
            if col not in all_stats.columns:
                continue

            if stat == "min":
                domain_stats[name][stat] = all_stats[col].min()
            elif stat == "max":
                domain_stats[name][stat] = all_stats[col].max()
            else:
                domain_stats[name][stat] = all_stats[col].mean()

    env._save_domain_statistics(domain_stats)

    logger.info("Saved domain statistics.")


if __name__ == "__main__":
    main()
