"""Custom evaluation callback for StableBaselines3 training with FluidGym
environments.
"""

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
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
        eval_freq: int,
        log_freq: int,
        n_eval_episodes: int,
        use_wandb: bool,
        checkpoint_latest: bool,
        eval_env: GymFluidEnv | VecFluidEnv | None = None,
        verbose: int = 1,
        save_eval_sequence: bool = True,
        log_single_steps: bool = False,
        render_training: bool = False,
        continue_training: bool = False,
    ):
        """
        Initialize the EvalCallback.

        Parameters
        ----------
        env: GymFluidEnv | MultiAgentVecEnv
            The training environment.

        eval_freq: int
            Frequency (in timesteps) at which to perform evaluations.

        log_freq: int
            Frequency (in timsteps) at which to perform logging.

        n_eval_episodes: int
            Number of episodes to run during each evaluation.

        use_wandb: bool
            Whether to log results to Weights & Biases.

        checkpoint_best: bool
            Whether to save a checkpoint of the best model.

        checkpoint_latest: bool
            Whether to save a checkpoint of the latest model.

        eval_env: GymFluidEnv | MultiAgentVecEnv | None
            The evaluation environment. If None, no evaluation is performed and
            only periodic checkpointing and training logging are done.

        verbose: int
            Verbosity level.

        save_eval_sequence: bool
            Whether to save the evaluation sequence plots and data.

        log_single_steps: bool
            Whether to log per-environment-step reward, return and action to
            Weights & Biases, in addition to the aggregated logging at
            `log_freq`. Requires `use_wandb`.

        render_training: bool
            Whether to render the training environment after every step and save
            the frames of each completed episode as a GIF.

        continue_training: bool
            Whether this run resumes an earlier one. The rows of the existing
            `training_log.csv` are then kept in front of the rows of this run.
        """
        super().__init__(verbose)
        self.env = env
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_freq = log_freq
        self.n_eval_episodes = n_eval_episodes
        self.use_wandb = use_wandb

        self.checkpoint_latest = checkpoint_latest
        self.save_evaluation = save_eval_sequence
        self.log_single_steps = log_single_steps
        self.render_training = render_training
        self.continue_training = continue_training

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
        self.previous_logged_data: list[dict[str, float]] = []
        self.uncontrolled_sequence_df: pd.DataFrame | None = None

        self.episode_step = 0
        self.episode_return = 0.0
        self.render_episode_idx = 0
        self.render_episode_step = 0

        # Wall-clock time of the steps of the current logging window. The clock runs
        # from the end of one `_on_step` to the start of the next, so evaluating,
        # checkpointing and logging are not counted as step time
        self._step_start = time.perf_counter()
        self._window_time = 0.0
        self._window_steps = 0

    @property
    def _num_env_steps(self) -> int:
        """Return the number of environment steps taken so far."""
        if isinstance(self.env, VecFluidEnv) and self.env.unwrapped.use_marl:
            return self.num_timesteps // self.env.num_envs
        else:
            return self.num_timesteps

    def _mean_step_time(self) -> float:
        """Mean wall-clock time one step took over the current logging window."""
        step_time = self._window_time / max(1, self._window_steps)
        self._window_time = 0.0
        self._window_steps = 0
        return step_time

    def _log(self, data: dict, step: int, tag: str) -> None:
        """Log data to console, CSV, and Weights & Biases."""
        data = {f"{tag}/{k}": v for k, v in data.items()}

        self.logged_data.append({"step": step, **data})

        self.logger.log(
            f"Step {step}: " + ", ".join([f"{k}={v:.4f}" for k, v in data.items()])
        )

        if self.use_wandb:
            import wandb

            wandb.log(data, step=step)

    def _log_single_step(self) -> None:
        """Log reward, return and action of the current environment step to wandb."""
        import wandb

        reward = float(np.mean(self.locals["rewards"]))
        self.episode_return += reward

        infos = self.locals["infos"]
        step_data = {
            "train/step_reward": reward,
            "train/step_return": self.episode_return,
            "train/episode_step": self.episode_step,
            **{
                f"train/step_{metric}": float(np.mean([info[metric] for info in infos]))
                for metric in self.metrics
            },
        }

        # On-policy algorithms expose the rescaled action as `clipped_actions`,
        # off-policy ones already sample in environment space
        actions = np.asarray(
            self.locals.get("clipped_actions", self.locals["actions"]), dtype=float
        ).reshape(-1)
        if actions.size == 1:
            step_data["train/step_action"] = float(actions[0])
        else:
            for i, action in enumerate(actions):
                step_data[f"train/step_action_{i}"] = float(action)

        wandb.log(step_data, step=self._num_env_steps)

        self.episode_step += 1
        if np.any(self.locals["dones"]):
            self.episode_step = 0
            self.episode_return = 0.0

    def _render_training_step(self) -> None:
        """Render the current training frame and save a GIF per training episode.

        The environment buffers the rendered frames internally and clears them on
        reset. Since SB3 resets the environment as soon as an episode is done, the
        GIF is written on the last step before the episode ends, i.e. it contains
        every frame of the episode except the terminal one.
        """
        if np.any(self.locals["dones"]):
            # The environment has already been reset here, so this renders the
            # initial frame of the next episode into the freshly cleared buffer
            self.env.render(render_3d=False)
            self.render_episode_step = 0
            self.render_episode_idx += 1
            return

        self.env.render(render_3d=False)
        self.render_episode_step += 1

        # Episodes that terminate early do not reach this point and are not saved
        if self.render_episode_step == self.env.unwrapped.episode_length - 1:
            self.env.save_gif(f"train_episode_{self.render_episode_idx}")

    def _on_step(self) -> bool:
        self._window_time += time.perf_counter() - self._step_start
        self._window_steps += 1

        if self.render_training:
            self._render_training_step()

        if self.log_single_steps and self.use_wandb:
            self._log_single_step()

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
                    "time": self._mean_step_time(),
                    "mean_reward": np.mean(self.logged_reward) / self.logged_length,
                    **{
                        f"mean_{metric}": self.logged_metrics[metric]
                        / self.logged_length
                        for metric in self.metrics
                    },
                },
                step=self._num_env_steps,
                tag="train",
            )

            self.logged_reward = 0
            self.logged_metrics = defaultdict(float)
            self.logged_length = 0

            # Save current logged data
            self._write_log()

            # Additionally, save model when logging
            if self.checkpoint_latest:
                self._save_model()

        # Check if it's time for evaluation
        if self._num_env_steps - self.last_eval_timesteps >= self.eval_freq:
            self.last_eval_timesteps = self._num_env_steps
            self._eval_step()

        self._step_start = time.perf_counter()
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_start(self) -> None:
        self._restore_previous_log()

        # A resumed run starts at the step count of the checkpoint, so the
        # counters have to start there as well -- otherwise the first step
        # already crosses both thresholds and logs a window of a single step
        self.last_log_timesteps = self._num_env_steps
        self.last_eval_timesteps = self._num_env_steps

        self._window_time = 0.0
        self._window_steps = 0
        self._step_start = time.perf_counter()

        if self.render_training:
            # Initial frame of the first training episode
            self.env.render(render_3d=False)

        if self.eval_env is None:
            return

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

    def _restore_previous_log(self) -> None:
        """Pick up the log of the run being resumed.

        The rows written by the previous run are kept in front of the ones of this
        run, so the `time` column of a resumed run covers the whole training rather
        than only its last segment.
        """
        self.previous_logged_data = []

        if not self.continue_training or not Path("training_log.csv").exists():
            return

        existing_log = pd.read_csv("training_log.csv")
        existing_log.to_csv("training_log_backup.csv", index=False)
        self.previous_logged_data = existing_log.to_dict("records")  # type: ignore

    def _write_log(self) -> None:
        """Write the rows of the previous runs and of this one to the csv."""
        pd.DataFrame(self.previous_logged_data + self.logged_data).to_csv(
            "training_log.csv", index=False
        )

    def _save_model(self) -> None:
        self.model.save("ckpt_latest")

        # `model.save` excludes the replay buffer, so it has to be written out
        # separately for an off-policy run to be resumable (`sb3.load_buffer`)
        if hasattr(self.model, "save_replay_buffer"):
            self.model.save_replay_buffer("ckpt_latest_replay_buffer")

    def _on_training_end(self) -> None:
        self._write_log()

        if self.checkpoint_latest:
            self._save_model()

    def _eval_step(self) -> None:
        """Perform an evaluation step and handle checkpointing."""
        if self.eval_env is not None:
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
