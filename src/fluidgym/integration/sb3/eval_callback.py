"""Custom evaluation callback for StableBaselines3 training with FluidGym
environments.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from stable_baselines3.common.callbacks import BaseCallback

from fluidgym.integration.gymnasium import GymFluidEnv
from fluidgym.integration.sb3.util import (
    evaluate_model,
    plot_eval_sequence,
)
from fluidgym.integration.sb3.vec_env import VecFluidEnv


class EvalCallback(BaseCallback):
    """Custom callback for evaluating and logging during training."""

    train_mode: str = "train"

    def __init__(
        self,
        env: GymFluidEnv | VecFluidEnv,
        eval_env: GymFluidEnv | VecFluidEnv,
        eval_freq: int,
        n_eval_episodes: int,
        use_wandb: bool,
        checkpoint_latest: bool,
        verbose: int = 1,
        save_eval_sequence: bool = True,
    ):
        """
        Initialize the EvalCallback.

        Parameters
        ----------
        env: GymFluidEnv | MultiAgentVecEnv
            The training environment.

        eval_env: GymFluidEnv | MultiAgentVecEnv
            The evaluation environment.

        eval_freq: int
            Frequency (in timesteps) at which to perform evaluations.

        n_eval_episodes: int
            Number of episodes to run during each evaluation.

        use_wandb: bool
            Whether to log results to Weights & Biases.

        checkpoint_best: bool
            Whether to save a checkpoint of the best model.

        checkpoint_latest: bool
            Whether to save a checkpoint of the latest model.

        verbose: int
            Verbosity level.

        save_eval_sequence: bool
            Whether to save the evaluation sequence plots and data.
        """
        super().__init__(verbose)
        self.env = env
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_freq = eval_freq // 10  # Log every 10% of the eval frequency
        self.n_eval_episodes = n_eval_episodes
        self.use_wandb = use_wandb
        self.checkpoint_latest = checkpoint_latest
        self.save_evaluation = save_eval_sequence

        assert self.env.action_space.shape is not None, (
            "Only Box action spaces are supported."
        )

        if isinstance(env, VecFluidEnv) and env.unwrapped.use_marl:
            self.num_actions = env.num_envs
            self.metrics = ["global_reward"] + env.unwrapped.metrics
        else:
            self.num_actions = int(self.env.action_space.shape[0])
            self.metrics = env.unwrapped.metrics

        self.last_eval_timesteps = 0
        self.last_log_timesteps = 0

        self.logged_reward: int | np.ndarray = 0
        self.logged_length = 0
        self.logged_metrics: dict[str, float] = defaultdict(float)

        self.logged_data: list[dict[str, float]] = []
        self.uncontrolled_sequence_df: pd.DataFrame | None = None

    @property
    def _num_env_steps(self) -> int:
        """Return the number of environment steps taken so far."""
        if isinstance(self.env, VecFluidEnv) and self.env.unwrapped.use_marl:
            return self.num_timesteps // self.env.num_envs
        else:
            return self.num_timesteps

    def _log(self, data: dict, step: int, tag: str) -> None:
        """Log data to console, CSV, and Weights & Biases."""
        data = {f"{tag}/{k}": v for k, v in data.items()}

        self.logged_data.append({"step": step, **data})

        self.logger.log(
            f"Step {step}: " + ", ".join([f"{k}={v:.4f}" for k, v in data.items()])
        )

        if self.use_wandb:
            wandb.log(data, step=step)

    def _on_step(self) -> bool:
        self.logged_reward += self.locals["rewards"]
        self.logged_length += 1

        infos = self.locals["infos"]
        for metric in self.metrics:
            metric_values = [info[metric] for info in infos]
            self.logged_metrics[metric] += float(np.mean(metric_values))

        if self._num_env_steps - self.last_log_timesteps >= self.log_freq:
            self.last_log_timesteps = self._num_env_steps

            self._log(
                {
                    "mean_reward": np.mean(self.logged_reward) / self.logged_length,
                    **{
                        f"mean_{metric}": self.logged_metrics[metric]
                        / self.logged_length
                        for metric in self.metrics
                    },
                },
                step=self._num_env_steps,
                tag="training",
            )

            self.logged_reward = 0
            self.logged_metrics = defaultdict(float)
            self.logged_length = 0

        # Check if it's time for evaluation
        if self._num_env_steps - self.last_eval_timesteps >= self.eval_freq:
            self.last_eval_timesteps = self._num_env_steps
            self._eval_step()

        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_start(self) -> None:
        self.uncontrolled_sequence_df = (
            self.env.unwrapped.get_uncontrolled_episode_metrics()
        )
        if self.uncontrolled_sequence_df is not None:
            if (
                len(self.uncontrolled_sequence_df)
                > self.eval_env.unwrapped.episode_length
            ):
                # Truncate to episode length
                self.uncontrolled_sequence_df = self.uncontrolled_sequence_df.iloc[
                    : self.eval_env.unwrapped.episode_length
                ]
            elif (
                len(self.uncontrolled_sequence_df)
                < self.eval_env.unwrapped.episode_length
            ):
                # Pad with NaNs to episode length
                self.uncontrolled_sequence_df = pd.concat(
                    [
                        self.uncontrolled_sequence_df,
                        pd.DataFrame(
                            np.full(
                                (
                                    self.eval_env.unwrapped.episode_length
                                    - len(self.uncontrolled_sequence_df),
                                    len(self.uncontrolled_sequence_df.columns),
                                ),
                                np.nan,
                            ),
                            columns=self.uncontrolled_sequence_df.columns,
                        ),
                    ],
                    ignore_index=True,
                )

    def _save_model(self) -> None:
        self.model.save("ckpt_latest")

    def _on_training_end(self) -> None:
        logged_df = pd.DataFrame(self.logged_data)

        if Path("training_log.csv").exists():
            existing_log = pd.read_csv("training_log.csv")
            existing_log.to_csv("training_log_backup.csv", index=False)
            combined_log = pd.concat([existing_log, logged_df], ignore_index=True)
            combined_log.to_csv("training_log.csv", index=False)
        else:
            logged_df.to_csv("training_log.csv", index=False)

        if self.checkpoint_latest:
            self._save_model()

    def _eval_step(self) -> None:
        """Perform an evaluation step and handle checkpointing."""
        mean_eval_reward = self._evaluate_model(
            env=self.eval_env, randomize=False, log=True, save=self.save_evaluation
        )

        if self.n_eval_episodes > 1:
            eval_rewards = [mean_eval_reward]
            for _ in range(self.n_eval_episodes - 1):
                reward = self._evaluate_model(
                    env=self.eval_env, randomize=True, log=False, save=False
                )
                eval_rewards.append(reward)
            mean_eval_reward = float(np.mean(eval_rewards))

        pd.DataFrame(self.logged_data).to_csv("training_log.csv", index=False)

        if self.checkpoint_latest:
            self._save_model()

    def _evaluate_model(
        self,
        env: GymFluidEnv | VecFluidEnv,
        randomize: bool,
        log: bool = False,
        save: bool = False,
    ) -> float:
        """Evaluate the model in the given environment.

        Parameters
        ----------
        env: GymFluidEnv | MultiAgentVecEnv
            The environment.

        randomize: bool
            Whether to randomize the initial state.

        log: bool
            Whether to log the evaluation metrics.

        save: bool
            Whether to save the evaluation sequence plots and data.

        Returns
        -------
        float
            The mean evaluation reward.
        """
        sequence_df, mean_eval_metrics = evaluate_model(
            env=env,
            model=self.model,
            randomize=randomize,
            save_name=f"eval_sequence_{self._num_env_steps}" if save else None,
        )

        if save:
            plot_eval_sequence(
                env=env,
                uncontrolled_sequence_df=self.uncontrolled_sequence_df,
                sequence_df=sequence_df,
                output_file=Path(".") / f"eval_sequence_{self._num_env_steps}.pdf",
            )

        if log:
            self._log(mean_eval_metrics, step=self._num_env_steps, tag="evaluation")

        return mean_eval_metrics["mean_reward"]
