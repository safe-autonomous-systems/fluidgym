import logging
import os
from datetime import datetime as dt

import hydra
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

import fluidgym
from fluidgym.envs import FluidEnv
from fluidgym.envs.fluid_env import EnvState

logger = logging.getLogger("fluidgym.dmpc")


OmegaConf.register_new_resolver("eval", lambda x: eval(x))

local_data_path = os.environ.get("FLUIDGYM_LOCAL_DATA_PATH", "./local_data")
fluidgym.config.update("local_data_path", local_data_path)


def dmpc_optimize_action_sequence(
    env: FluidEnv,
    start_state: EnvState,
    horizon: int,
    n_iterations: int,
    lr: float,
    discount: float,
    previous_actions: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Differentiable Model Predictive Control (D-MPC) action selection.

    Performs gradient-based optimization over a finite-horizon action sequence
    to maximize the sum of discounted rewards.

    Parameters
    ----------
    env: FluidEnv
        A differentiable FluidEnv instance.

    start_state: EnvState
        State to start the predictive rollouts from (env.get_state()).

    horizon: int
        Number of steps to look ahead.

    n_iterations: int
        Number of gradient-descent iterations on the action sequence.

    lr: float
        Learning rate for the optimizer.

    discount: float
        Discount factor for rewards (1.0 = no discount).

    previous_actions: torch.Tensor | None
        If provided, use these actions as the initial guess for optimization.
        Shape should be (horizon, action_space_shape).

    Returns
    -------
    torch.Tensor
        Optimized action sequence of shape (horizon, action_space_shape).
    """
    device = env.cuda_device
    action_shape = env.action_space_shape

    if previous_actions is None:
        # Initial guess: zeros
        actions = torch.zeros(
            (horizon,) + action_shape,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
    else:
        # Use previous actions as initial guess
        actions = previous_actions.clone()

        # Shift actions to the left and pad with zeros at the end
        actions = torch.roll(actions, shifts=-1, dims=0)
        actions[-1] = torch.zeros(action_shape, device=device, dtype=torch.float32)

        actions = actions.to(device).requires_grad_(True)

    optimizer = torch.optim.Adam([actions], lr=lr)

    # Action bounds from env.action_space
    low = torch.as_tensor(env.action_space.low, device=device, dtype=torch.float32)
    high = torch.as_tensor(env.action_space.high, device=device, dtype=torch.float32)

    for _ in range(n_iterations):
        env.set_state(start_state)
        env.detach()

        optimizer.zero_grad()

        total_return = torch.zeros((), device=device)
        gamma = 1.0

        for t in range(horizon):
            a_t = actions[t]
            a_t_clamped = torch.clamp(a_t, low, high)

            _, reward, terminated, truncated, _ = env.step(a_t_clamped)

            total_return = total_return + gamma * reward.squeeze()
            gamma = gamma * discount

            if terminated or truncated:
                break

        loss = -total_return
        loss.backward()
        del loss
        optimizer.step()
        env.detach()

        del total_return, gamma, a_t, a_t_clamped, reward, terminated, truncated

        with torch.no_grad():
            actions[:] = torch.clamp(actions, low, high)

    env.detach()
    return actions.detach()


def run_mdpc_episode(
    train_env: FluidEnv,
    test_env: FluidEnv,
    n_iterations: int,
    lr: float,
    discount_factor: float,
    horizon: int,
    enable_wandb: bool = True,
) -> pd.DataFrame:
    """
    Run one episode using Differentiable Model Predictive Control (D-MPC).

    Parameters
    ----------
    train_env: FluidEnv
        A differentiable FluidEnv instance for training.

    test_env: FluidEnv
        A differentiable FluidEnv instance for testing.

    n_iterations: int
        Number of gradient-descent iterations on the action sequence.

    lr: float
        Learning rate for the action sequence optimizer.

    discount_factor: float
        Discount factor for rewards (1.0 = no discount).

    horizon: int
        Planning horizon for predictive control.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the collected metrics during the episode.

    """
    test_env.render(save=True)

    df_rows = []

    done = False
    actions = None
    step = 0

    while not done:
        # Save the current state for planning
        start_state = test_env.get_state()

        # Optimize an action sequence from the current state
        actions = dmpc_optimize_action_sequence(
            env=train_env,
            start_state=start_state,
            horizon=horizon,
            n_iterations=n_iterations,
            lr=lr,
            discount=discount_factor,
            previous_actions=actions,
        )
        test_env.detach()
        best_action = actions[0]

        with torch.no_grad():
            _, reward, terminated, truncated, info = test_env.step(best_action)
            test_env.render(save=True)

            if enable_wandb:
                wandb.log(
                    {
                        "reward": reward.item(),
                        **{
                            f"action_{i}": best_action[i].item()
                            for i in range(min(best_action.shape[0], 12))
                        },
                        **{metric: info[metric].item() for metric in test_env.metrics},
                    }
                )

            print(f"Step reward: {reward.item():.3f}")

        done = terminated or truncated

        df_rows.append(
            {
                **{metric: info[metric].item() for metric in test_env.metrics},
                **{
                    f"action_{i}": best_action[i].item()
                    for i in range(min(best_action.shape[0], 12))
                },
                "reward": reward.item(),
                "step": step,
            }
        )
        step += 1

    test_env.save_gif("test_episode.gif")

    return pd.DataFrame(df_rows)


@hydra.main(version_base="1.3", config_path="../configs", config_name="run_d-mpc")
def main(cfg: DictConfig):
    logger.info("Training script started.")

    logger.info("Initializing environments...")
    test_env = fluidgym.make(
        cfg.env_id,
        randomize_initial_state=False,
        **cfg.env_kwargs,
    )
    rollout_env = fluidgym.make(
        cfg.env_id,
        **cfg.env_kwargs,
        episode_length=test_env.episode_length * 2,  # Ensure rollout env is long enough
        randomize_initial_state=False,
        differentiable=True,
    )
    rollout_env.reset(seed=42)  # Seed does not matter for rollout env

    test_env.test()
    test_env.reset(seed=cfg.seed + 84)
    test_env.load_initial_domain(idx=cfg.seed % 10)
    logger.info("Done.")

    if cfg.wandb.enable:
        now = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(
            id="D-MPC-"
            + str(cfg.lr)
            + "-"
            + str(cfg.n_iterations)
            + "-"
            + str(cfg.horizon)
            + "-"
            + str(cfg.env_id)
            + "-"
            + str(cfg.seed)
            + "-"
            + now,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=rollout_env.id,
        )

    logger.info("Running D-MPC episode...")
    episode_metrics = run_mdpc_episode(
        train_env=rollout_env,
        test_env=test_env,
        n_iterations=cfg.n_iterations,
        lr=cfg.lr,
        discount_factor=cfg.discount_factor,
        horizon=cfg.horizon,
        enable_wandb=cfg.wandb.enable,
    )
    logger.info("DPC episode finished.")

    # Save metrics
    episode_metrics.to_csv("test_eval_episode_0.csv", index=False)

    if cfg.wandb.enable:
        wandb.finish()


if __name__ == "__main__":
    main()
