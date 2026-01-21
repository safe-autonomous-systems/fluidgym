import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

import fluidgym
from fluidgym.envs.airfoil.airfoil_env_base import AirfoilEnvBase
from fluidgym.envs.cylinder.cylinder_env_base import CylinderEnvBase
from fluidgym.envs.fluid_env import N_INITIAL_DOMAINS, EnvMode
from fluidgym.envs.tcf.tcf_env import TCF3DBottomEnv

logger = logging.getLogger("fluidgym.preparation")
logging.basicConfig(level=logging.WARNING)


fluidgym.config.update("local_data_path", "./local_data")


@hydra.main(version_base="1.3", config_path="../configs", config_name="preparation")
def main(cfg: DictConfig):
    try:
        collect_uncontrolled_episode_metrics(cfg)
    except Exception as e:
        logger.exception("Job crashed with an exception")
        logger.error(e)
        raise


def add_metrics(
    all_metrics: dict[str, list[float]], key: str, data: torch.Tensor
) -> None:
    data_arr = data.cpu().numpy()

    p5, p25, p50, p75, p95 = np.percentile(data_arr, [5, 25, 50, 75, 95])

    all_metrics[f"{key}_mean"] += [float(np.mean(data_arr))]
    all_metrics[f"{key}_min"] += [float(np.min(data_arr))]
    all_metrics[f"{key}_max"] += [float(np.max(data_arr))]
    all_metrics[f"{key}_p5"] += [float(p5)]
    all_metrics[f"{key}_p25"] += [float(p25)]
    all_metrics[f"{key}_p50"] += [float(p50)]
    all_metrics[f"{key}_p75"] += [float(p75)]
    all_metrics[f"{key}_p95"] += [float(p95)]


def collect_uncontrolled_episode_metrics(cfg: DictConfig):
    env_kwargs = dict(cfg.get("env_kwargs", {}))
    if "TCF" in cfg.env_id:
        # We train for 1000 but test for longer episodes
        env_kwargs["episode_length"] = 2500

    env = fluidgym.make(cfg.env_id, **env_kwargs, load_domain_statistics=False)
    logger.info("Initialized environment.")
    env.reset(seed=42)

    zero_action = torch.zeros(env.action_space_shape, device=env._cuda_device)

    for i in range(N_INITIAL_DOMAINS):
        for mode in [EnvMode.TRAIN, EnvMode.VAL, EnvMode.TEST]:
            try:
                env._load_uncontrolled_episode(idx=i, mode=mode)
                logger.info(
                    f"Uncontrolled episode for mode {mode} and domain {i} already "
                    f"exists."
                )
                if not cfg.overwrite_existing:
                    logger.info("Skipping generation.")
                    continue
            except FileNotFoundError:
                pass

            logger.info(
                f"Collecting statistics for mode '{mode.value}' and domain "
                f"{i + 1}/{N_INITIAL_DOMAINS}"
            )
            env.reset()
            try:
                env.load_initial_domain(idx=i, mode=mode)
            except FileNotFoundError:
                logger.warning(
                    f"Initial domain for mode {mode} and domain {i} not found. "
                    f"Skipping."
                )
                continue

            trunc = False

            episode_metrics: dict[str, list[float]] = defaultdict(list)

            while not trunc:
                _, reward, _, trunc, info = env.step(zero_action)

                for metric in env.metrics:
                    # We take the mean to account for
                    # metric values for multiple substeps
                    metric_value = torch.mean(info[metric]).cpu().item()
                    episode_metrics[metric] += [metric_value]

                    # In case of the cylinder, we are interested
                    # in the absolute lift values as well
                    if (
                        isinstance(env, CylinderEnvBase)
                        or isinstance(env, AirfoilEnvBase)
                    ) and metric == "lift":
                        episode_metrics["abs_lift"] += [np.abs(metric_value)]

                # 2) Collect min/max velocity magnitude
                u = env.get_velocity()
                mag: torch.Tensor = torch.linalg.norm(u, dim=1)
                add_metrics(
                    all_metrics=episode_metrics, key="velocity_magnitude", data=mag
                )

                # 3) Collect min/max pressure
                p = env.get_pressure()
                add_metrics(all_metrics=episode_metrics, key="pressure", data=p)

                # 4) For some envs, we have vorticity
                if (
                    isinstance(env, TCF3DBottomEnv)
                    or isinstance(env, CylinderEnvBase)
                    or isinstance(env, AirfoilEnvBase)
                ):
                    vorticity = env.get_vorticity()

                    # If 3D, compute magnitude, for 2D we already have magnitude
                    if env.ndims == 3:
                        vorticity: torch.Tensor = torch.linalg.norm(vorticity, dim=1)

                    add_metrics(
                        all_metrics=episode_metrics,
                        key="vorticity_magnitude",
                        data=vorticity,
                    )

            episode_df = pd.DataFrame(episode_metrics)
            env._save_uncontrolled_episode(episode_df, idx=i, mode=mode)

            logger.info(
                f"Saved uncontrolled episode for mode '{mode.value}' and domain "
                f"{i + 1}/{N_INITIAL_DOMAINS}"
            )


if __name__ == "__main__":
    main()
