"""Utility functions for StableBaselines3 integration."""

import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stable_baselines3.common.base_class import BaseAlgorithm

from fluidgym.integration.gymnasium import GymFluidEnv
from fluidgym.integration.sb3.vec_env import VecFluidEnv

logger = logging.getLogger("fluidgym.integration.sb3")


PLOT_MAX_ACTIONS = 64


def __get_sequence_df(
    env: GymFluidEnv | VecFluidEnv,
    metric_sequence: dict[str, list[np.ndarray]],
    action_sequence: list[np.ndarray],
) -> pd.DataFrame:
    """Convert the metric and action sequences to a pandas DataFrame.

    Parameters
    ----------
    env: GymFluidEnv | MultiAgentVecEnv
        The environment.

    metric_sequence: dict[str, list[np.ndarray]]
        The metric sequences.

    action_sequence: list[np.ndarray]
        The action sequences.

    Returns
    -------
    pd.DataFrame
        The sequence data as a pandas DataFrame.
    """
    sequence_dict = {
        **{
            metric: np.array(metric_sequence[metric])
            for metric in env.unwrapped.metrics
        },
    }

    action_sequence_arr = np.array(action_sequence)
    if action_sequence_arr.ndim == 1:
        sequence_dict["action"] = action_sequence_arr
    else:
        if action_sequence_arr.shape[1] <= PLOT_MAX_ACTIONS:
            sequence_dict.update(
                **{
                    f"action_{i}": action_sequence_arr[:, i]
                    for i in range(action_sequence_arr.shape[1])
                }
            )
        else:
            sequence_dict.update(
                **{
                    f"action_{i}": action_sequence_arr[:, i]
                    for i in range(PLOT_MAX_ACTIONS)
                }
            )

    if isinstance(env, VecFluidEnv):
        sequence_dict["local_reward"] = np.array(metric_sequence["local_reward"])

    sequence_dict["reward"] = np.array(metric_sequence["reward"]).flatten()

    return pd.DataFrame(sequence_dict)


def __env_step(
    env: GymFluidEnv | VecFluidEnv, action: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool, dict[str, np.ndarray]]:
    """Wraps the environment step function to handle both single-agent and multi-agent
    environments.

    Parameters
    ----------
    env: GymFluidEnv | MultiAgentVecEnv
        The environment.

    action: np.ndarray
        The action to take.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, bool, dict]
        A tuple containing the observation, reward, done flag, and info dictionary.
    """
    if isinstance(env, VecFluidEnv):
        action = action[:, None]
        obs, reward, dones, infos = env.step(action)
        done = bool(np.any(dones))
        info = infos[0]
    else:
        obs, scalar_reward, term, trunc, info = env.step(action)
        reward = np.array([scalar_reward])
        done = term or trunc

    assert isinstance(obs, np.ndarray)

    return obs, reward, done, info


def plot_eval_sequence(
    env: GymFluidEnv | VecFluidEnv,
    uncontrolled_sequence_df: pd.DataFrame | None,
    sequence_df: pd.DataFrame,
    output_file: Path,
) -> None:
    """Plot the evaluation sequence data.

    Parameters
    ----------
    env: GymFluidEnv | MultiAgentVecEnv
        The environment.

    uncontrolled_sequence_df: pd.DataFrame | None
        The uncontrolled sequence data.

    sequence_df: pd.DataFrame
        The controlled sequence data.

    output_file: Path
        The output file to save the plot to.
    """
    if isinstance(env, VecFluidEnv) and "reward_0" in sequence_df.columns:
        metrics = ["global_reward"] + env.unwrapped.metrics
    else:
        metrics = ["reward"] + env.unwrapped.metrics

    if uncontrolled_sequence_df is not None and len(uncontrolled_sequence_df) > len(
        sequence_df
    ):
        uncontrolled_sequence_df = uncontrolled_sequence_df.iloc[
            : len(sequence_df)
        ].reset_index(drop=True)

    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics + 1, 1, figsize=(10, 4 * (num_metrics + 1)))

    # same x axis for all plots
    timesteps = sequence_df.index * env.unwrapped.dt
    cur_ax = 0

    if env.num_actions == 1 or "action" in sequence_df.columns:
        g = sns.lineplot(
            data=sequence_df,
            x=timesteps,
            y="action_0",
            ax=axes[cur_ax],
            label="Action",
        )
        g.set_title("Action over Time")
        g.set_xlabel("Timestep")
        g.set_ylabel("Action Value")
        g.legend().remove()
        cur_ax += 1
    else:
        for i in range(min(env.num_actions, PLOT_MAX_ACTIONS)):
            g = sns.lineplot(
                data=sequence_df,
                x=timesteps,
                y=f"action_{i}",
                ax=axes[cur_ax],
                label=f"RL Action[{i}]",
            )
            g.set_title("Actions over Time")
            g.set_xlabel("Timestep")
            g.set_ylabel("Action Value")
            g.legend().remove()
        cur_ax += 1

    for i, metric in enumerate(metrics):
        # Plot unctronlled sequence if available
        if (
            uncontrolled_sequence_df is not None
            and metric in uncontrolled_sequence_df.columns
        ):
            sns.lineplot(
                data=uncontrolled_sequence_df,
                x=timesteps,
                y=metric,
                ax=axes[i + cur_ax],
                label="Uncontrolled",
                linestyle="--",
            )

        g = sns.lineplot(
            data=sequence_df,
            x=timesteps,
            y=metric,
            ax=axes[i + cur_ax],
            label="RL Control",
        )
        g.legend().remove()
        g.set_title(f"{metric} over Time")
        g.set_xlabel("Timestep")
        g.set_ylabel(metric)

    plt.tight_layout()
    plt.savefig(output_file, format="pdf")
    plt.close(fig)


def evaluate_model(
    env: GymFluidEnv | VecFluidEnv,
    model: BaseAlgorithm,
    randomize: bool,
    save_name: str | None = None,
    save_frames: bool = False,
    render_3d: bool = False,
    deterministic: bool = True,
    output_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate a trained model in the environment and collect metrics.

    Parameters
    ----------
    env: GymFluidEnv | MultiAgentVecEnv
        The environment.

    model: BaseAlgorithm
        The trained model to evaluate.

    randomize: bool
        Whether to randomize the initial state.

    save_name: str | None
        The base name to save the GIF and CSV files. If None, no files are saved.
        Defaults to None.

    save_frames: bool
        Whether to save individual rendered frames. Defaults to False.

    render_3d: bool
        Whether to use 3d rendering when saving the GIF. Defaults to False.

    deterministic: bool
        Whether to use deterministic actions during testing. Defaults to True.

    output_path: Path | None
        The path to save the evaluation data. If None, saves to the current directory.
        Defaults to None.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, float]]
        A tuple containing the sequence data as a pandas DataFrame and a dictionary of
        mean evaluation metrics.
    """
    if output_path is None:
        output_path = Path(".")

    is_marl = isinstance(env, VecFluidEnv)
    done = False
    episode_rewards: list[np.ndarray] = []
    episode_metrics: dict[str, float] = defaultdict(float)
    action_sequence: list[np.ndarray] = []
    metric_sequence: dict[str, list[np.ndarray]] = defaultdict(list)

    obs = env.reset(randomize=randomize)
    if isinstance(obs, tuple):
        obs = obs[0]
    assert isinstance(obs, np.ndarray)

    # Render initial frame
    if save_name is not None:
        env.unwrapped.render(
            save=save_frames,
            render_3d=render_3d,
            output_path=output_path,
            filename=save_name + "_initial",
        )

    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)  # type: ignore
        obs, reward, done, info = __env_step(env=env, action=action)

        if is_marl:
            metric_sequence["local_reward"] += [np.mean(reward)]
            metric_sequence["reward"] += [info["global_reward"].flatten()]
            episode_metrics["global_reward"] += float(info["global_reward"])
        else:
            metric_sequence["reward"] += [reward]
        action_sequence += [action.flatten()]

        if save_name is not None:
            env.unwrapped.render(
                save=False,
                render_3d=render_3d,
                output_path=output_path,
                filename=save_name + f"_step_{step:04d}",
            )

        episode_rewards += [reward]
        for metric in env.unwrapped.metrics:
            metric_sequence[metric] += [info[metric]]

            if isinstance(info[metric], np.ndarray):
                episode_metrics[metric] += float(np.mean(info[metric]))
            else:
                episode_metrics[metric] += float(info[metric])

        step += 1

    # Render final frame
    if save_name is not None:
        env.unwrapped.render(
            save=save_frames,
            render_3d=render_3d,
            output_path=output_path,
            filename=save_name + "_final",
        )

    episode_rewards_arr = np.array(episode_rewards)  # (steps, n_envs,)

    # Now we compute the mean over all episodes
    mean_episode_rewards = np.mean(episode_rewards_arr, axis=0)  # (n_envs,)
    mean_eval_reward = float(np.mean(mean_episode_rewards))

    mean_eval_metrics = {}
    for metric in env.unwrapped.metrics:
        mean_episode_metrics = episode_metrics[metric] / episode_rewards_arr.shape[0]
        mean_eval_metrics[f"mean_{metric}"] = float(np.mean(mean_episode_metrics))
    mean_eval_metrics["mean_reward"] = mean_eval_reward

    sequence_df = __get_sequence_df(
        env=env, metric_sequence=metric_sequence, action_sequence=action_sequence
    )

    if save_name is not None:
        gif_name = save_name + ".gif"
        env.save_gif(gif_name, output_path=output_path)

        csv_name = save_name + ".csv"
        sequence_df.to_csv(output_path / csv_name, index=False)

    return sequence_df, mean_eval_metrics


def test_model(
    model: BaseAlgorithm,
    test_env: GymFluidEnv | VecFluidEnv,
    n_episodes: int,
    save_frames: bool,
    render_3d: bool,
    deterministic: bool = True,
    output_path: Path | None = None,
) -> None:
    """Test a trained model in the test environment and collect metrics.

    Parameters
    ----------
    model: BaseAlgorithm
        The trained model to test.

    test_env: GymFluidEnv | MultiAgentVecEnv
        The test environment.

    n_episodes: int
        The number of episodes to run during testing.

    save_frames: bool
        Whether to save individual rendered frames.

    render_3d: bool
        Whether to use 3d rendering when saving the GIF.

    deterministic: bool
        Whether to use deterministic actions during testing. Defaults to True.

    output_path: Path | None
        The path to save the test evaluation data. If None, saves to the current
        directory. Defaults to None.
    """
    if output_path is None:
        output_path = Path(".")

    test_sequence_dfs = []
    sequence_df, _ = evaluate_model(
        env=test_env,
        model=model,
        randomize=False,
        save_name="test_eval_episode_0",
        save_frames=save_frames,
        render_3d=render_3d,
        output_path=output_path,
        deterministic=deterministic,
    )
    sequence_df["episode"] = 0
    sequence_df["step"] = np.arange(len(sequence_df))
    test_sequence_dfs.append(sequence_df)

    uncontrolled_test_df = test_env.unwrapped.get_uncontrolled_episode_metrics()
    plot_eval_sequence(
        env=test_env,
        uncontrolled_sequence_df=uncontrolled_test_df,
        sequence_df=sequence_df,
        output_file=output_path / "test_eval_sequence.pdf",
    )

    for i in range(1, n_episodes):
        sequence_df, _ = evaluate_model(
            env=test_env,
            model=model,
            randomize=True,
            save_frames=save_frames,
            render_3d=render_3d,
            output_path=output_path,
            deterministic=deterministic,
        )
        sequence_df["episode"] = i
        sequence_df["step"] = np.arange(len(sequence_df))
        test_sequence_dfs.append(sequence_df)

    all_test_sequences_df = pd.concat(test_sequence_dfs, ignore_index=True)
    all_test_sequences_df.to_csv(output_path / "test_eval_sequences.csv", index=False)
