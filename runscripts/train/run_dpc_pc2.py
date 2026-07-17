import logging
import os
from datetime import datetime as dt

import hydra
import pandas as pd
import torch
import torch.nn as nn
import wandb
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

logger = logging.getLogger("fluidgym.dpc")

OmegaConf.register_new_resolver("eval", lambda x: eval(x))

class DPCPolicy(nn.Module):
    """
    A simple continuous control policy network for DPC.
    """
    def __init__(self, obs_dim: int, action_dim: int, action_low: torch.Tensor, action_high: torch.Tensor, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # Bounds outputs to [-1, 1]
        )
        # We need these to scale the [-1, 1] Tanh output to the environment's action bounds
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.net(obs)
        # Scale to environment bounds
        return x * self.action_scale + self.action_bias


def train_dpc_policy(
    seed: int,
    env_id: str,
    env_kwargs: dict,
    policy: DPCPolicy,
    optimizer: torch.optim.Optimizer,
    total_timesteps: int,
    horizon: int,
    discount_factor: float,
    max_grad_norm: float = 1.0,
    enable_wandb: bool = True,
):
    """
    Trains the DPC policy using Truncated Backpropagation Through Time (TBPTT).
    """
    import fluidgym
    import numpy as np
    np.random.seed(seed)

    training_logs = []

    policy.train()
    global_step = 0
    episode = 0

    while global_step < total_timesteps:
        episode_seed = np.random.randint(0, 10000)
        train_env = fluidgym.make(
            env_id,
            **env_kwargs,
            differentiable=True,
        )
        train_env.seed(int(episode_seed))
        device = train_env.cuda_device
        obs, _ = train_env.reset()
        
        done = False
        episode_reward = 0.0
        episode_metrics = defaultdict(list)
        
        while not done:
            optimizer.zero_grad()
            total_return = torch.zeros((), device=device)
            gamma = 1.0
            
            # Unroll simulation for 'horizon' steps to collect gradients
            for t in range(horizon):
                action = policy(obs)
                obs, reward, terminated, truncated, info = train_env.step(action)
                
                for metric in train_env.metrics:
                    episode_metrics[metric].append(info[metric].item())

                total_return = total_return + gamma * reward.squeeze()
                gamma = gamma * discount_factor
                episode_reward += reward.item()
                global_step += 1

                done = terminated or truncated
                if done or global_step >= total_timesteps:
                    break

            # Backpropagate through the environment simulator to the policy parameters
            loss = -total_return
            loss.backward()
            
            # Clip gradients to prevent exploding gradients from the fluid solver
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            
            optimizer.step()
            
            # Detach environment state to cut the computational graph for the next chunk
            train_env.detach()
            obs = obs.detach()

            metrics = {
                **{f"train/{metric}": np.mean(values) for metric, values in episode_metrics.items()},
                "train/mean_reward": total_return.item() / horizon,
                "train/loss": loss.item()
            }
            
            if enable_wandb:
                wandb.log(metrics, step=global_step)

            metrics["step"] = global_step
            training_logs.append(metrics)

            del total_return, loss, action, reward, terminated, truncated

        logger.info(f"Train Episode {episode} | Reward: {episode_reward:.3f}")
        if enable_wandb:
            wandb.log({"train/episode_reward": episode_reward}, step=global_step)

        training_logs_df = pd.DataFrame(training_logs)
        training_logs_df.to_csv("training_log.csv", index=False)
        episode += 1


def eval_dpc_policy(
    test_env,
    policy: DPCPolicy,
    render: bool = False,
    randomize: bool = True,
) -> pd.DataFrame:
    """
    Evaluates the trained DPC policy (no gradients).
    """
    device = test_env.cuda_device
    policy.eval()

    df_rows = []
    obs, _ = test_env.reset(randomize=randomize)
    if render:
        test_env.render(save=True, filename="test_eval_episode_0_initial")

    obs = torch.as_tensor(obs, device=device, dtype=torch.float32)

    step = 0
    done = False
    with torch.no_grad():
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, info = test_env.step(action)

            if render:            
                test_env.render()

            df_rows.append({
                "step": step,
                "reward": reward.item(),
                **{f"action_{i}": action[i].item() for i in range(min(action.shape[0], 12))},
                **{metric: info[metric].item() for metric in test_env.metrics},
            })
            
            step += 1
            done = terminated or truncated

    if render:
        test_env.render(save=True, filename="test_eval_episode_0_final")
        test_env.save_gif("test_eval_episode_0")

    logger.info(f"Eval Episode finished")

    return pd.DataFrame(df_rows)


@hydra.main(version_base="1.3", config_path="../configs", config_name="run_dpc")
def main(cfg: DictConfig):
    import fluidgym
    from fluidgym.integration.gymnasium import GymFluidEnv
	
    local_data_path = os.environ["FLUIDGYM_LOCAL_DATA_PATH"]
    fluidgym.config.update("local_data_path", local_data_path)

    logger.info("DPC Training script started.")
    logger.info("Initializing environments...")

    # Test env does not need to be differentiable for pure evaluation
    test_env = fluidgym.make(
        cfg.env_id,
        **cfg.env_kwargs,
    )

    test_env.test()
    test_env.reset(seed=cfg.seed + 84)
    logger.info("Done.")

    device = test_env.cuda_device
    
    # Setup Policy and Optimizer
    obs_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.shape[0]
    action_low = torch.as_tensor(test_env.action_space.low, device=device, dtype=torch.float32).flatten()
    action_high = torch.as_tensor(test_env.action_space.high, device=device, dtype=torch.float32).flatten()
    
    policy = DPCPolicy(obs_dim, action_dim, action_low, action_high).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    if cfg.wandb.enable:
        now = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(
            id=f"DPC-{cfg.lr}-{cfg.horizon}-{cfg.env_id}-{cfg.seed}-{now}",
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=test_env.id,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    logger.info("Starting DPC Training...")
    train_dpc_policy(
        seed=cfg.seed,
        env_id=cfg.env_id,
        env_kwargs=cfg.env_kwargs,
        policy=policy,
        optimizer=optimizer,
        total_timesteps=cfg.total_timesteps,
        horizon=cfg.horizon,
        discount_factor=cfg.discount_factor,
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        enable_wandb=cfg.wandb.enable,
    )

    # Save policy
    torch.save(policy.state_dict(), "dpc_policy.pth")

    logger.info("Running DPC Evaluation...")
    all_metrics = []
    for i in range(cfg.n_test_episodes):
        eval_metrics = eval_dpc_policy(
            test_env=test_env,
            policy=policy,
            render=i == 0,
            randomize=not (i == 0)  # Only randomize for episodes after the first one, which is rendered
        )
        eval_metrics["episode"] = i
        all_metrics.append(eval_metrics)
    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    
    # Save metrics
    output_path = Path("./test")
    output_path.mkdir(parents=True, exist_ok=True)

    all_metrics_df.to_csv(output_path / "test_eval_sequences.csv", index=False)
    sequence_0 = all_metrics_df[all_metrics_df["episode"] == 0]
    sequence_0.to_csv(output_path / "test_eval_episode_0.csv", index=False)
    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()