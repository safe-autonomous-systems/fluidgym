import logging
import os
from datetime import datetime as dt

import hydra
import pandas as pd
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf

import fluidgym
from fluidgym.envs.rbc import RBCEnv2D
from fluidgym.envs.fluid_env import EnvState

logger = logging.getLogger("fluidgym.pd_rbc")


OmegaConf.register_new_resolver("eval", lambda x: eval(x))

local_data_path = os.environ.get("FLUIDGYM_LOCAL_DATA_PATH", "./local_data")
fluidgym.config.update("local_data_path", local_data_path)


class PDController:
    def __init__(self, env: RBCEnv2D, kp: float, kd: float):
        super().__init__()
        self.env = env

        self.kp = kp
        self.kd = kd
        self.e_prev = torch.zeros(env._n_sensors_x).to(env.device)

    def reset(self):
        self.e_prev = torch.zeros(self.env._n_sensors_x).to(self.env.device)

    def compute_e(self):
        u = self.env.get_velocity()

        u = u.permute(1, 2, 0)
        u = u[self.env._sensor_locations[1], self.env._sensor_locations[0], :]
        u = u.reshape(self.env._n_sensors_x, self.env._n_sensors_y, 2).permute(2, 1, 0)
        u_y = u[1].mean(dim=0) * -1

        return u_y

    def get_action(self) -> torch.Tensor:
        e_current = self.compute_e()
        e_grad = (e_current - self.e_prev) / self.env._step_length

        action = self.kp * e_current + self.kd * e_grad

        # average over segments
        action = action.view(12, 4).mean(dim=1)

        self.e_prev = e_current.detach().clone()

        return action

def evaluate(
    controller: PDController,
    env: RBCEnv2D,
    num_episodes: int,
    seed: int = 84,
    domain_idx: int = 0,
    enable_wandb: bool = True,
) -> pd.DataFrame:
    """
    Run num_episodes test episodes with a PD controller.
    Logging matches the format used in train_tdmpc.py evaluate().

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: reward, [metrics...], episode, step
    """
    from collections import defaultdict

    all_dfs = []

    for i in range(num_episodes):
        controller.reset()
        env.reset(seed=seed + i)
        env.load_initial_domain(idx=(domain_idx + i) % 10)

        if i == 0:
            env.render(save=True, filename="test_eval_episode_0_initial")

        ep_metrics = defaultdict(list)
        done = False
        step = 0

        while not done:
            action = controller.get_action()

            with torch.no_grad():
                _, reward, terminated, truncated, info = env.step(action)

            if i == 0:
                env.render()

            if enable_wandb:
                wandb.log(
                    {
                        f"eval/reward": reward.item(),
                        **{metric: info[metric].item() for metric in env.metrics},
                    },
                    step=i * 10000 + step,
                )

            print(f"Episode {i} | Step {step} | reward: {reward.item():.3f}")

            ep_metrics['reward'].append(reward.item())
            for metric in env.metrics:
                ep_metrics[metric].append(info[metric].item())
            ep_metrics['step'].append(step)

            done = terminated or truncated
            step += 1

        if i == 0:
            env.render(save=True, filename="test_eval_episode_0_final")
            env.save_gif("test_eval_episode_0")

        episode_df = pd.DataFrame(ep_metrics)
        episode_df['episode'] = i
        all_dfs.append(episode_df)

    return pd.concat(all_dfs, ignore_index=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="run_pd_rbc")
def main(cfg: DictConfig):
    logger.info("Training script started.")

    logger.info("Initializing environments...")
    env = fluidgym.make(
        cfg.env_id,
        randomize_initial_state=False,
        **cfg.env_kwargs,
    )
    assert isinstance(env, RBCEnv2D), "Expected env to be an instance of RBCEnv2D"

    env.test()
    logger.info("Done.")

    if cfg.wandb.enable:
        now = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(
            id="D-MPC-"
            + str(cfg.kp)
            + "-"
            + str(cfg.kd)
            + "-"
            + str(cfg.env_id)
            + "-"
            + str(cfg.seed)
            + "-"
            + now,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=env.id,
        )

    controller = PDController(env, kp=cfg.kp, kd=cfg.kd)

    logger.info("Running PD-RBC evaluation...")
    eval_results = evaluate(
        controller=controller,
        env=env,
        num_episodes=1,
        seed=cfg.seed + 84,
        domain_idx=cfg.seed % 10,
        enable_wandb=cfg.wandb.enable,
    )
    logger.info(f"Evaluation completed. Average reward: {eval_results['reward'].mean():.2f}")

    # Save metrics
    eval_results.to_csv("test_eval_episode_0.csv", index=False)

    if cfg.wandb.enable:
        wandb.finish()


if __name__ == "__main__":
    main()
