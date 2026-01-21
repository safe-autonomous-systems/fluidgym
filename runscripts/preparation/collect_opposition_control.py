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
from fluidgym.envs.fluid_env import N_INITIAL_DOMAINS, EnvMode
from fluidgym.envs.tcf.tcf_env import TCF3DBothEnv, TCF3DBottomEnv

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


def opposition_control(obs: torch.Tensor) -> torch.Tensor:
    u_y = obs[:, 1].clone()
    return -u_y.unsqueeze(-1)


def collect_uncontrolled_episode_metrics(cfg: DictConfig):
    env = fluidgym.make(
        cfg.env_id,
        randomize_initial_state=False,
    )

    if not isinstance(env, (TCF3DBottomEnv, TCF3DBothEnv)):
        raise ValueError("This script only supports TCF environments.")
    env.scale_actions = False

    logger.info("Initialized environment.")
    env.reset_marl(seed=42)

    for i in range(N_INITIAL_DOMAINS):
        for mode in [EnvMode.TRAIN, EnvMode.VAL, EnvMode.TEST]:
            try:
                env.load_opposition_control_episode(idx=i, mode=mode)
                logger.info(
                    f"Opposition control for mode {mode} and domain {i} already exists."
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
            obs, _ = env.reset_marl()
            env.load_initial_domain(idx=i, mode=mode)
            trunc = False

            episode_metrics: dict[str, list[float]] = defaultdict(list)
            actions: list[np.ndarray] = []

            while not trunc:
                action = opposition_control(obs)

                # Fair comparison: clamp actions to valid RL range
                action = torch.clamp(action, -env._u_wall, env._u_wall)
                action = action - action.mean()

                obs, _, _, trunc, info = env.step_marl(action)

                actions += [action[:64].cpu().numpy()]

                episode_metrics["reward"] += [info["global_reward"].item()]
                for metric in env.metrics:
                    # We take the mean to account for
                    # metric values for multiple substeps
                    metric_value = torch.mean(info[metric]).cpu().item()
                    episode_metrics[metric] += [metric_value]

            actions_arr = np.stack(actions, axis=0)
            episode_df = pd.DataFrame(episode_metrics)

            for a_idx in range(actions_arr.shape[1]):
                episode_df[f"action_{a_idx}"] = actions_arr[:, a_idx]

            env.save_opposition_control_episode(
                idx=i,
                mode=mode,
                df=episode_df,
            )

            logger.info(
                f"Saved opposition control episode for mode '{mode.value}' and domain "
                f"{i + 1}/{N_INITIAL_DOMAINS}"
            )


if __name__ == "__main__":
    main()
