import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as MplPath
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, Colormap

from rliable import library as rly
from rliable import metrics as rly_metrics

import fluidgym
from fluidgym.envs.fluid_env import EnvMode
from fluidgym.envs.rbc import RBCEnv3D
from fluidgym.envs.tcf import TCF3DBothEnv

mpl.rcParams.update(
    {
        "text.usetex": True,  # or False if you're okay with mathtext
        "pdf.fonttype": 42,  # TrueType in PDF
        "ps.fonttype": 42,  # TrueType in PS/EPS
    }
)

logger = logging.getLogger(__name__)

TRAINING_PATH = Path("./output/training")
RBC_PD_PATH = Path("./output/PD-RBC")
PLOTS_PATH = Path("./paper/plots")
PLOTS_PATH.mkdir(parents=True, exist_ok=True)
TABLES_PATH = Path("./paper/tables")
TABLES_PATH.mkdir(parents=True, exist_ok=True)

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
RL_MODES = {
    "CylinderRot2D": ["sarl"],
    "CylinderJet2D": ["sarl"],
    "CylinderJet3D": ["sarl", "marl"],
    "RBC2D": ["sarl", "marl"],
    "RBC3D": ["marl"],
    "Airfoil2D": ["sarl"],
    "Airfoil3D": ["sarl", "marl"],
    "TCFSmall3D-both": ["marl"],
    "TCFLarge3D-both": ["marl"],
}
ENV_ORDER = list(RL_MODES.keys())

ENV_ID_ORDER = []
for env_name in ENV_ORDER:
    for difficulty in DIFFICULTY_LEVELS:
        ENV_ID_ORDER.append(f"{env_name}-{difficulty}-v0")

SEEDS = range(5)
PD_RBC_SEEDS = range(3)
AIRFOIL3D_SEEDS = range(3)

FIGWIDTH = 10.0

ALGORITHMS = ["PPO", "SAC", "TD-MPC", "DPC"]
ALGORITHM_ORDER = ["PD", "PPO", "SAC", "MA-PPO", "MA-SAC", "DPC", "TD-MPC"]
ALGORITHM_COLOR_IDXS = {
    "PPO": 1,
    "SAC": 2,
    "MA-PPO": 3,
    "MA-SAC": 4,
    "DPC": 7,
    "TD-MPC": 6,
    "PD": 0,
}

ENV_CATEGORIES = {
    "CylinderJet2D": "Cylinder",
    "CylinderRot2D": "Cylinder",
    "CylinderJet3D": "Cylinder",
    "RBC2D": "RBC",
    "RBC3D": "RBC",
    "Airfoil2D": "Airfoil",
    "Airfoil3D": "Airfoil",
    "TCFSmall3D-both": "TCF",
    "TCFLarge3D-both": "TCF",
}

RENDER_KEYS = {
    "CylinderJet2D": "vorticity",
    "CylinderRot2D": "vorticity",
    "CylinderJet3D": "3d_vorticity",
    "RBC2D": "temperature",
    "RBC3D": "3d_temperature",
    "Airfoil2D": "vorticity",
    "Airfoil3D": "3d_vorticity",
    "TCFSmall3D": "3d_q_criterion",
    "TCFLarge3D": "3d_q_criterion",
}

METRICS = {
    "reward": "Mean Reward",
    "drag": r"$C_D$",
    "lift": r"$C_L$",
    "nusselt": r"$\mathrm{Nu}$",
}

TRAINING_COL_REPLACEMENT_MAP = {
    "training/mean_reward": "training/mean_local_reward",
    "training/mean_global_reward": "training/mean_reward",
}

fluidgym.config.update("local_data_path", "./local_data")

sns.set_style(
    style="whitegrid",
    rc={
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.bottom": True,
        "ytick.left": True,
    },
)
bright_colors = sns.color_palette(fluidgym.config.palette)
sns.set_palette(bright_colors)


# See https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html
def radar_factory(num_vars, frame="circle"):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return MplPath(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()  # type: ignore
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=MplPath.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def format_step(step: int) -> str:
    exp = len(str(step)) - 1
    base = step // (10**exp)
    return f"{base}e{exp}"


def format_axis(ax: Any, grid: bool | None = None, border: bool = False) -> None:
    major_length = 4
    minor_length = 2
    linewidth = 0.8
    grid_alpha = 0.8

    ax.minorticks_on()
    ax.tick_params(
        axis="x", which="major", length=major_length, width=linewidth, color="black"
    )
    ax.tick_params(
        axis="x", which="minor", length=minor_length, width=linewidth, color="black"
    )
    ax.tick_params(
        axis="y", which="major", length=major_length, width=linewidth, color="black"
    )
    ax.tick_params(
        axis="y", which="minor", length=minor_length, width=linewidth, color="black"
    )
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["right"].set_visible(border)
    ax.spines["top"].set_visible(border)

    if grid is True:
        ax.grid(
            which="major",
            linestyle="-",
            linewidth=linewidth,
            alpha=grid_alpha,
            color="lightgray",
            zorder=-1,
        )
        ax.grid(
            which="minor",
            linestyle="-",
            linewidth=linewidth,
            alpha=grid_alpha,
            color="lightgray",
            zorder=-1,
        )
    elif grid is False:
        ax.grid(False)


def iqm(data: pd.Series) -> float:
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqm_data = data[(data >= q1) & (data <= q3)]

    return iqm_data.mean()


def format_algorithm(algorithm: str, rl_mode: str) -> str:
    if rl_mode == "marl":
        return f"MA-{algorithm}"
    else:
        return algorithm


def load_single_training_run(
    env_id: str, rl_mode: str, algorithm: str, seed: int
) -> pd.DataFrame:
    training_path = (
        TRAINING_PATH / f"{rl_mode}/{env_id}/{algorithm}/{seed}/training_log.csv"
    )
    if not training_path.exists():
        return pd.DataFrame()

    training_run = pd.read_csv(training_path)
    # TD-MPC uses "train/" prefix instead of "training/", normalize columns
    training_run = training_run.rename(columns={
        col: col.replace("train/", "training/", 1)
        for col in training_run.columns
        if col.startswith("train/")
    })
    if "training/step" in training_run.columns and "step" not in training_run.columns:
        training_run = training_run.rename(columns={"training/step": "step"})
    training_run = (
        training_run.sort_values("step")  # ensure correct order
        .groupby("step", as_index=False)
        .agg(lambda col: col.dropna().iloc[0] if col.notna().any() else None)
    )

    # Apply smoothing
    if algorithm == "TD-MPC":
        smoothing_window = 20
    elif algorithm == "DPC":
        smoothing_window = 40
    else:
        smoothing_window = 3
    for col in training_run.columns:
        if col.startswith("training/mean_"):
            training_run[col] = (
                training_run[col].rolling(window=smoothing_window, min_periods=1).mean()
            )

    return training_run


def load_training_runs(
    env_name: str,
    additional_algorithms: list[str] | None = None,
    difficulty_levels: list[str] | None = None,
) -> pd.DataFrame:
    all_runs = []
    if "Airfoil3D" in env_name or "CylinderJet3D" in env_name:
        seeds = AIRFOIL3D_SEEDS
    else:
        seeds = SEEDS

    if difficulty_levels is None:
        difficulty_levels = DIFFICULTY_LEVELS

    algorithms = ALGORITHMS
    if additional_algorithms is not None:
        algorithms += additional_algorithms

    for rl_mode in RL_MODES[env_name]:
        for difficulty in difficulty_levels:
            env_id = f"{env_name}-{difficulty}-v0"
            for algorithm in algorithms:
                for seed in seeds:
                    run_df = load_single_training_run(
                        env_id=env_id,
                        rl_mode=rl_mode,
                        algorithm=algorithm,
                        seed=seed,
                    )
                    if run_df.empty:
                        logger.warning(
                            f"No training run found for {env_id}, "
                            f"{format_algorithm(algorithm, rl_mode)}, seed {seed}"
                        )
                        continue

                    run_df["env_id"] = env_id
                    run_df["difficulty"] = difficulty
                    run_df["seed"] = seed
                    run_df["algorithm"] = format_algorithm(algorithm, rl_mode)

                    all_runs.append(run_df)

    if not all_runs:
        return pd.DataFrame()

    all_runs_df = pd.concat(all_runs, ignore_index=True)

    return all_runs_df

def load_test_sequence(
    env_id: str,
    rl_mode: str,
    algorithm: str,
    seed: int,
    test_env_id: str | None = None
) -> pd.DataFrame:
    if test_env_id is None:
        test_path = (
            TRAINING_PATH
            / f"{rl_mode}/{env_id}/{algorithm}/{seed}/test/test_eval_episode_0.csv"
        )
    else:
        test_path = (
            TRAINING_PATH
            / f"{rl_mode}/{env_id}/{algorithm}/{seed}/transfer/{test_env_id}/"
            "test_eval_episode_0.csv"
        )

    if not test_path.exists():
        return pd.DataFrame()

    return pd.read_csv(test_path)


def load_single_test_render(
    env_id: str,
    rl_mode: str,
    algorithm: str,
    seed: int,
    test_env_id: str | None = None,
) -> dict:
    if test_env_id is None:
        if algorithm == "PD":
            test_path = RBC_PD_PATH / f"{env_id}/{seed}"
        else:
            test_path = TRAINING_PATH / f"{rl_mode}/{env_id}/{algorithm}/{seed}/test"
    else:
        test_path = (
            TRAINING_PATH
            / f"{rl_mode}/{env_id}/{algorithm}/{seed}/transfer/{test_env_id}/"
        )

    render_key = RENDER_KEYS[env_id.split("-")[0]]
    initial_png = test_path / f"{render_key}_test_eval_episode_0_initial.png"
    final_png = test_path / f"{render_key}_test_eval_episode_0_final.png"

    if final_png.exists() is False or initial_png.exists() is False:
        return {}

    CROP = 170

    def load_and_crop(path):
        img = plt.imread(path)
        return img[CROP:-CROP, ...]

    needs_crop = "3D" in env_id
    return {
        "initial": load_and_crop(initial_png)
        if needs_crop
        else plt.imread(initial_png),
        "final": load_and_crop(final_png) if needs_crop else plt.imread(final_png),
    }


def load_all_test_renders(env_name: str, seed: int) -> dict:
    algorithms = ALGORITHMS
    if "RBC2D" in env_name:
        algorithms =  ["PD"] + algorithms

    if "CylinderJet2D" in env_name or "RBC2D" in env_name:
        algorithms = algorithms + ["DPC", "TD-MPC"]
    else:
        algorithms = algorithms

    all_renders = defaultdict(dict)
    for rl_mode in RL_MODES[env_name]:
        for algorithm in algorithms:
            if rl_mode == "marl" and algorithm not in ["PPO", "SAC"]:
                continue

            for difficulty in DIFFICULTY_LEVELS:
                render_data = load_single_test_render(
                    env_id=f"{env_name}-{difficulty}-v0",
                    rl_mode=rl_mode,
                    algorithm=algorithm,
                    seed=seed,
                )
                if len(render_data) == 0:
                    continue

                if rl_mode == "sarl":
                    algorithm_name = algorithm
                else:
                    algorithm_name = f"MA-{algorithm}"

                all_renders[difficulty]["Baseflow"] = render_data["initial"]
                all_renders[difficulty][algorithm_name] = render_data["final"]

    return all_renders


def load_test_sequences(
    env_id: str,
    rl_mode: str,
    algorithm: str,
    seed: int,
    test_env_id: str | None = None,
    stress_test: bool = False,
) -> pd.DataFrame:
    if stress_test:
        test_path = (
            TRAINING_PATH
            / f"{rl_mode}/{env_id}/{algorithm}/{seed}/stress_test/test_eval_episode_0.csv"
        )  
    elif test_env_id is None:
        if algorithm == "PD":
            if rl_mode != "sarl":
                return pd.DataFrame()  # PD results only exist for SARL
            test_path = RBC_PD_PATH / f"{env_id}/{seed}/test_eval_episode_0.csv"
        else:
            test_path = (
                TRAINING_PATH
                / f"{rl_mode}/{env_id}/{algorithm}/{seed}/test/test_eval_sequences.csv"
            )
    else:
        test_path = (
            TRAINING_PATH
            / f"{rl_mode}/{env_id}/{algorithm}/{seed}/transfer/{test_env_id}/"
            "test_eval_sequences.csv"
        )

    if not test_path.exists():
        return pd.DataFrame()

    test_sequences = pd.read_csv(test_path)

    ref_env_id = env_id if test_env_id is None else test_env_id

    # Now, we add the relative improvement over the baseflow
    # Add rows for baseflows
    env = fluidgym.make(ref_env_id)
    stats = env._load_domain_statistics()

    # Add improvement column
    if "Cylinder" in ref_env_id:
        test_sequences["Improvement (%)"] = (
            100.0
            * (stats["drag"]["mean"] - test_sequences["drag"])
            / stats["drag"]["mean"]
        )
    elif "RBC" in ref_env_id:
        test_sequences["Improvement (%)"] = (
            100.0
            * (stats["nusselt"]["mean"] - test_sequences["nusselt"])
            / stats["nusselt"]["mean"]
        )
    elif "Airfoil" in ref_env_id:
        base_aero_efficiency = stats["lift"]["mean"] / stats["drag"]["mean"]
        test_sequences["aero_eff"] = test_sequences["lift"] / test_sequences["drag"]

        test_sequences["Improvement (%)"] = (
            100.0
            * (test_sequences["aero_eff"] - base_aero_efficiency)
            / base_aero_efficiency
        )
    else:  # env_type == "TCF":
        test_sequences["Improvement (%)"] = 100.0 * test_sequences["reward"]

    return test_sequences


def load_test_results(
    env_name: str,
    test_env_name: str | None = None,
    rl_modes: list[str] | None = None,
    average_over_episodes: bool = True,
) -> pd.DataFrame:
    all_results = []
    rl_modes = rl_modes or RL_MODES[env_name]

    difficulty_levels = DIFFICULTY_LEVELS
    seeds = SEEDS

    if "RBC2D" in env_name and test_env_name is None:
        algorithms = ALGORITHMS + ["PD"]
    else:
        algorithms = ALGORITHMS

    for rl_mode in rl_modes:
        for difficulty in difficulty_levels:
            env_id = f"{env_name}-{difficulty}-v0"
            for algorithm in algorithms:
                if algorithm == "PD":
                    _seeds = PD_RBC_SEEDS
                else:
                    _seeds = seeds

                # Skip invalid rl_mode / algorithm combinations
                if rl_mode == "marl" and algorithm not in ["PPO", "SAC"]:
                    continue

                for seed in _seeds:
                    result_df = load_test_sequences(
                        env_id=env_id,
                        rl_mode=rl_mode,
                        algorithm=algorithm,
                        seed=seed,
                        test_env_id=(
                            None
                            if test_env_name is None
                            else f"{test_env_name}-{difficulty}-v0"
                        ),
                    )
                    if result_df.empty:
                        logger.warning(
                            f"No test results found for {env_id}, "
                            f"{format_algorithm(algorithm, rl_mode)}, seed {seed}"
                        )
                        continue

                    drop_cols = [
                        col for col in result_df.columns if col.startswith("action")
                    ]
                    result_df = result_df.drop(columns=drop_cols)

                    # For PD, there is no episode column
                    if "episode" not in result_df.columns:
                        result_df["episode"] = 0

                    # Average per episode
                    if average_over_episodes:
                        result_df = result_df.groupby("episode").mean().reset_index()
                        if "step" in result_df.columns:
                            result_df = result_df.drop(columns=["step"])

                    result_df["env_id"] = env_id
                    result_df["category"] = ENV_CATEGORIES[env_name]
                    result_df["difficulty"] = difficulty
                    result_df["seed"] = seed
                    result_df["algorithm"] = format_algorithm(algorithm, rl_mode)

                    all_results.append(result_df)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def load_all_test_results(
    average_over_episodes: bool = True,
    include_transfer: bool = False
) -> pd.DataFrame:
    all_test_results_list = []

    # Cylinder
    all_test_results_list += [
        load_test_results(
            env_name="CylinderJet2D",
            rl_modes=["sarl"],
            average_over_episodes=average_over_episodes,
        )
    ]
    all_test_results_list += [
        load_test_results(env_name="CylinderRot2D", rl_modes=["sarl"], average_over_episodes=average_over_episodes,)
    ]
    all_test_results_list += [
        load_test_results(env_name="CylinderJet3D", rl_modes=["sarl", "marl"], average_over_episodes=average_over_episodes,)
    ]

    # RBC
    all_test_results_list += [
        load_test_results(env_name="RBC2D", rl_modes=["sarl", "marl"], average_over_episodes=average_over_episodes,)
    ]
    all_test_results_list += [load_test_results(env_name="RBC3D", rl_modes=["marl"], average_over_episodes=average_over_episodes,)]

    # Airfoil
    all_test_results_list += [
        load_test_results(env_name="Airfoil2D", rl_modes=["sarl"], average_over_episodes=average_over_episodes,)
    ]
    all_test_results_list += [
        load_test_results(env_name="Airfoil3D", rl_modes=["sarl", "marl"], average_over_episodes=average_over_episodes,)
    ]

    # TCF
    all_test_results_list += [
        load_test_results(env_name="TCFSmall3D-both", rl_modes=["marl"], average_over_episodes=average_over_episodes,)
    ]

    all_test_results_list += [
        load_test_results(env_name="TCFLarge3D-both", rl_modes=["marl"], average_over_episodes=average_over_episodes,)
    ]

    test_results = pd.concat(all_test_results_list, ignore_index=True)

    if include_transfer:
        test_results["test_env_id"] = test_results["env_id"]

        cylinder_transfer = load_test_results(
                env_name="CylinderJet2D",
                test_env_name="CylinderJet3D",
                rl_modes=["sarl"],
                average_over_episodes=average_over_episodes,
            )
        cylinder_transfer["test_env_id"] = cylinder_transfer["env_id"].str.replace("CylinderJet2D", "CylinderJet3D")

        tcf_transfer = load_test_results(
            env_name="TCFSmall3D-both",
            test_env_name="TCFLarge3D-both",
            rl_modes=["marl"],
            average_over_episodes=average_over_episodes,
        )
        tcf_transfer["test_env_id"] = tcf_transfer["env_id"].str.replace("TCFSmall3D-both", "TCFLarge3D-both")

        return pd.concat([test_results, cylinder_transfer, tcf_transfer], ignore_index=True)

    else:
        return test_results


def plot_cylinder_test_episode(env_id: str, seed: int) -> None:
    env = fluidgym.make(env_id)

    uncontrolled_df = env._load_uncontrolled_episode(idx=0, mode=EnvMode.TEST)
    ppo_df = load_test_sequence(
        env_id=env_id, rl_mode="sarl", algorithm="PPO", seed=seed
    )
    sac_df = load_test_sequence(
        env_id=env_id, rl_mode="sarl", algorithm="SAC", seed=seed
    )
    td_mcp_df = load_test_sequences(
        env_id=env_id, rl_mode="sarl", algorithm="TD-MPC", seed=seed
    )
    td_mcp_df = td_mcp_df[td_mcp_df["episode"] == 0].reset_index(drop=True)
    dpc_df = load_test_sequence(
        env_id=env_id, rl_mode="sarl", algorithm="DPC", seed=seed
    )

    ppo_df["step"] = ppo_df.index
    sac_df["step"] = sac_df.index
    uncontrolled_df["step"] = uncontrolled_df.index

    fig, axs = plt.subplots(1, 2, figsize=(FIGWIDTH * 0.45, 2.25), sharex=True)

    labels = [r"$a_t$", r"$C_D$"]
    metrics = ["action_0", "drag"]
    titles = ["Control Action", "Drag Coefficient"]

    # Compute the drag reduction for SAC
    mean_base_drag = uncontrolled_df["drag"].mean()
    sac_drag = sac_df["drag"].to_numpy()[-1]
    drag_reduction = (mean_base_drag - sac_drag) / mean_base_drag * 100
    print(f"Drag Reduction (SAC): {drag_reduction:.2f}%")

    for ax, metric in zip(axs, metrics, strict=False):
        if metric != "action_0" and metric != "reward":
            sns.lineplot(
                data=uncontrolled_df,
                x="step",
                y=metric,
                ax=ax,
                color=bright_colors[0],
                label="Baseflow",
                linestyle="-.",
            )
        sns.lineplot(
            data=ppo_df,
            x="step",
            y=metric,
            ax=ax,
            color=bright_colors[1],
            label="PPO",
        )
        sns.lineplot(
            data=sac_df,
            x="step",
            y=metric,
            ax=ax,
            color=bright_colors[2],
            label="SAC",
        )
        sns.lineplot(
            data=dpc_df,
            x="step",
            y=metric,
            ax=ax,
            color=bright_colors[ALGORITHM_COLOR_IDXS["DPC"]],
            label="DPC",
            zorder=10
        )
        sns.lineplot(
            data=td_mcp_df,
            x="step",
            y=metric,
            ax=ax,
            color=bright_colors[ALGORITHM_COLOR_IDXS["TD-MPC"]],
            label="TD-MPC",
        )
        ax.set_title(titles[metrics.index(metric)])
        ax.legend().remove()
        ax.set_xlabel(r"Episode Step $t$")
        ax.set_ylabel(labels[metrics.index(metric)])
        format_axis(ax, border=True)

    handles, y_labels = ax.get_legend_handles_labels()

    handles.insert(1, plt.Line2D([], [], color="white"))  # type: ignore
    y_labels.insert(1, "")

    # Swap positions of DPC and SAC
    # dpc_handle = handles.pop(-2)
    # dpc_label = y_labels.pop(-2)
    # sac_handle = handles.pop(3)
    # sac_label = y_labels.pop(3)
    # handles.insert(3, dpc_handle)
    # y_labels.insert(3, dpc_label)
    # handles.insert(4, sac_handle)
    # y_labels.insert(4, sac_label)


    fig.legend(
        handles,
        y_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.215),
        ncol=3,
        fancybox=False,
        shadow=False,
        frameon=False,
    )
    fig.subplots_adjust(
        top=0.88, bottom=0.39, left=0.11, right=0.98, hspace=0.25, wspace=0.35
    )

    plt.savefig(
        PLOTS_PATH / "individual" / f"CylinderJet2D_eval_sequence.pdf",
        format="pdf",
    )
    plt.close()

def plot_airfoil_transfer_episode(env_id: str, seed: int) -> None:
    transfer_env_id = env_id.replace("Airfoil2D", "Airfoil3D")
    env = fluidgym.make(transfer_env_id, load_domain_statistics=False)

    uncontrolled_df = env._load_uncontrolled_episode(idx=0, mode=EnvMode.TEST)
    ppo_df = load_test_sequence(
        env_id=env_id, rl_mode="sarl", algorithm="PPO", seed=seed, test_env_id=transfer_env_id
    )
    # sac_df = load_test_sequence(
    #     env_id=env_id, rl_mode="sarl", algorithm="SAC", seed=seed, test_env_id=transfer_env_id
    # )

    ppo_df["step"] = ppo_df.index
    ppo_df["aero_eff"] = ppo_df["lift"] / ppo_df["drag"]
    # sac_df["step"] = sac_df.index
    uncontrolled_df["step"] = uncontrolled_df.index
    uncontrolled_df["aero_eff"] = uncontrolled_df["lift"] / uncontrolled_df["drag"]

    fig, axs = plt.subplots(1, 4, figsize=(FIGWIDTH, 1.8), sharex=True)

    jet_colors = [bright_colors[2], bright_colors[3], bright_colors[4]]
    jet_labels = ["Jet 1", "Jet 2", "Jet 3"]

    ax_action = axs[0]
    jet_handles = []
    for jet_idx in range(3):
        color = jet_colors[jet_idx]
        col = f"action_{jet_idx}"
        line, = ax_action.plot(
            ppo_df["step"],
            ppo_df[col],
            color=color,
        )
        line.set_label(jet_labels[jet_idx])
        jet_handles.append(line)
    ax_action.set_title("Control Actions")
    ax_action.set_xlabel(r"Episode Step $t$")
    ax_action.set_ylabel(r"$a_t$")
    ax_action.legend().set_visible(False)
    format_axis(ax_action, border=True)

    # --- Drag Coefficient ---
    sns.lineplot(
        data=uncontrolled_df,
        x="step",
        y="drag",
        ax=axs[1],
        color=bright_colors[0],
        label="Baseflow",
        linestyle="-.",
    )
    sns.lineplot(
        data=ppo_df,
        x="step",
        y="drag",
        ax=axs[1],
        color=bright_colors[1],
        label="PPO",
    )
    axs[1].set_title("Drag Coefficient")
    axs[1].legend().remove()
    axs[1].set_xlabel(r"Episode Step $t$")
    axs[1].set_ylabel(r"$C_D$")
    format_axis(axs[1], border=True)

    # --- Lift Coefficient ---
    sns.lineplot(
        data=uncontrolled_df,
        x="step",
        y="lift",
        ax=axs[2],
        color=bright_colors[0],
        label="Baseflow",
        linestyle="-.",
    )
    sns.lineplot(
        data=ppo_df,
        x="step",
        y="lift",
        ax=axs[2],
        color=bright_colors[1],
        label="PPO",
    )
    axs[2].set_title("Lift Coefficient")
    axs[2].legend().remove()
    axs[2].set_xlabel(r"Episode Step $t$")
    axs[2].set_ylabel(r"$C_L$")
    format_axis(axs[2], border=True)

    # --- Aerodynamic Efficiency ---
    sns.lineplot(
        data=uncontrolled_df,
        x="step",
        y="aero_eff",
        ax=axs[3],
        color=bright_colors[0],
        label="Baseflow",
        linestyle="-.",
    )
    sns.lineplot(
        data=ppo_df,
        x="step",
        y="aero_eff",
        ax=axs[3],
        color=bright_colors[1],
        label="PPO",
    )
    axs[3].set_title("Aerodynamic Efficiency")
    axs[3].legend().remove()
    axs[3].set_xlabel(r"Episode Step $t$")
    axs[3].set_ylabel(r"$C_L / C_D$")
    format_axis(axs[3], border=True)

    aero_handles, aero_labels = axs[3].get_legend_handles_labels()
    all_handles = jet_handles + aero_handles
    all_labels = jet_labels + aero_labels
    fig.legend(
        all_handles,
        all_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.15),
        ncol=5,
        fancybox=False,
        shadow=False,
        frameon=False,
    )
    fig.subplots_adjust(
        top=0.87, bottom=0.34, left=0.07, right=0.98, hspace=0.25, wspace=0.35
    )

    plt.savefig(
        PLOTS_PATH / "individual" / f"{env_id}_transfer.pdf", format="pdf"
    )
    plt.close()


def plot_rbc_test_episode(env_id: str, algorithm: str, seed: int) -> None:
    env = fluidgym.make(env_id)
    assert isinstance(env, RBCEnv3D)

    eval_sequence_data = load_test_sequence(
        env_id=env_id,
        rl_mode="marl",
        algorithm=algorithm,
        seed=seed,
    )
    if eval_sequence_data.empty:
        raise ValueError("Evaluation sequence data is empty.")

    algorithm = format_algorithm(algorithm, rl_mode="marl")

    ts_index = env.episode_length - 26
    timestep = (ts_index + 1) * env.step_length

    if len(eval_sequence_data) > env.episode_length:
        eval_sequence_data = eval_sequence_data.iloc[
            :: len(eval_sequence_data) // env.episode_length, :
        ].reset_index(drop=True)

    uncontrolled_sequence_df = env._load_uncontrolled_episode(idx=0, mode=EnvMode.VAL)
    uncontrolled_sequence_df["timestep"] = (
        np.arange(env.episode_length) + 1
    ) * env.step_length
    eval_sequence_data["timestep"] = (
        np.arange(env.episode_length) + 1
    ) * env.step_length

    ts_data = eval_sequence_data.iloc[ts_index, :]
    actions = []
    for i in range(env.action_space_shape[0]):
        actions.append(ts_data[f"action_{i}"])

    actions = torch.as_tensor(actions).to(env._cuda_device)
    smoothed_actions = env._action_to_control(actions)

    vmax = env._T_hot + env._heater_limit
    vmin = env._T_cold

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(FIGWIDTH * 0.45, 1.8),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    sns.lineplot(
        data=uncontrolled_sequence_df,
        x="timestep",
        y="nusselt",
        ax=axs[0],
        color=bright_colors[0],
        label="Baseflow",
        linestyle="-.",
    )
    sns.lineplot(
        data=eval_sequence_data,
        x="timestep",
        y="nusselt",
        color=bright_colors[ALGORITHM_COLOR_IDXS[algorithm]],
        ax=axs[0],
        label=algorithm,
    )
    # center below plot
    axs[0].axvline(
        x=timestep,
        color=bright_colors[-1],
        linestyle="--",
        label=rf"$t={int(timestep)}$",
    )

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.13),
        fancybox=False,
        shadow=False,
        frameon=False,
        ncol=3,
    )

    axs[0].set_xlabel(r"Episode Step $t$")
    axs[0].set_ylabel(r"$\mathrm{Nu}_\mathrm{instant}$")
    axs[0].set_title("Test Episode")
    axs[0].legend().remove()
    format_axis(axs[0], border=True)

    axs[1].imshow(
        smoothed_actions.cpu().numpy(),
        cmap="rainbow",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )

    # add colorbar
    plt.colorbar(
        plt.cm.ScalarMappable(
            cmap="rainbow",
            norm=plt.Normalize(vmin=vmin, vmax=vmax),  # type: ignore
        ),
        label="Temperature",
        ax=axs[1],
        fraction=0.046,
        pad=0.04,
    )
    axs[1].grid(False)
    axs[1].set_title("Actuation at " + rf"$t={int(timestep)}$")
    axs[1].set_xlabel("Heater " + r"$x$" + " Index")
    axs[1].set_ylabel("Heater " + r"$z$" + " Index")

    # invert y axis
    axs[1].invert_yaxis()

    heaters = np.arange(0, env._n_heaters) * env._heater_width + env._heater_width / 2
    heater_labels = [str(i) for i in range(env._n_heaters)]
    axs[1].set_xticks(heaters)
    axs[1].set_xticklabels(heater_labels)
    axs[1].set_yticks(heaters)
    axs[1].set_yticklabels(heater_labels)

    fig.subplots_adjust(
        top=0.88, bottom=0.33, left=0.12, right=0.91, hspace=0.25, wspace=0.27
    )

    plt.savefig(PLOTS_PATH / "individual" / "rbc3d_marl_action.pdf", format="pdf")
    plt.close()


def plot_tcf_transfer(difficulty: str) -> None:
    small_env = fluidgym.make(f"TCFSmall3D-both-{difficulty}-v0")
    large_env = fluidgym.make(f"TCFLarge3D-both-{difficulty}-v0")

    assert isinstance(small_env, TCF3DBothEnv)
    assert isinstance(large_env, TCF3DBothEnv)

    algos_small = []
    algos_large = []

    for idx in range(10):
        opp_control = small_env.load_opposition_control_episode(
            idx=idx, mode=EnvMode.TEST
        )
        opp_control["step"] = opp_control.index
        opp_control = opp_control[opp_control["step"] < small_env.episode_length]
        opp_control["Improvement (%)"] = 100.0 * opp_control["reward"]
        opp_control = opp_control[["step", "Improvement (%)"]]
        opp_control["algorithm"] = "Opp. Control"
        algos_small += [opp_control]

        opp_control = large_env.load_opposition_control_episode(
            idx=idx, mode=EnvMode.TEST
        )
        opp_control["step"] = opp_control.index
        opp_control = opp_control[opp_control["step"] < large_env.episode_length]
        opp_control["Improvement (%)"] = 100.0 * opp_control["reward"]
        opp_control = opp_control[["step", "Improvement (%)"]]
        opp_control["algorithm"] = "Opp. Control"
        algos_large += [opp_control]

    for seed in SEEDS:
        sac = load_test_sequences(
            env_id=f"TCFSmall3D-both-{difficulty}-v0",
            rl_mode="marl",
            algorithm="SAC",
            seed=seed,
        )
        sac = sac[["step", "Improvement (%)"]]
        sac["algorithm"] = "MA-SAC (S)"
        sac["seed"] = seed
        algos_small += [sac]

        sac = load_test_sequences(
            env_id=f"TCFLarge3D-both-{difficulty}-v0",
            rl_mode="marl",
            algorithm="SAC",
            seed=seed,
        )
        sac = sac[["step", "Improvement (%)"]]
        sac["algorithm"] = "MA-SAC (L)"
        sac["seed"] = seed
        algos_large += [sac]

        sac = load_test_sequences(
            env_id=f"TCFSmall3D-both-{difficulty}-v0",
            test_env_id=f"TCFLarge3D-both-{difficulty}-v0",
            rl_mode="marl",
            algorithm="SAC",
            seed=seed,
        )
        sac = sac[["step", "Improvement (%)"]]
        sac["algorithm"] = "MA-SAC (S)"
        sac["seed"] = seed
        algos_large += [sac]

        ppo = load_test_sequences(
            env_id=f"TCFSmall3D-both-{difficulty}-v0",
            rl_mode="marl",
            algorithm="PPO",
            seed=seed,
        )
        ppo = ppo[["step", "Improvement (%)"]]
        ppo["algorithm"] = "MA-PPO (S)"
        ppo["seed"] = seed
        algos_small += [ppo]

        ppo = load_test_sequences(
            env_id=f"TCFLarge3D-both-{difficulty}-v0",
            rl_mode="marl",
            algorithm="PPO",
            seed=seed,
        )
        ppo = ppo[["step", "Improvement (%)"]]
        ppo["algorithm"] = "MA-PPO (L)"
        ppo["seed"] = seed
        algos_large += [ppo]

        ppo = load_test_sequences(
            env_id=f"TCFSmall3D-both-{difficulty}-v0",
            test_env_id=f"TCFLarge3D-both-{difficulty}-v0",
            rl_mode="marl",
            algorithm="PPO",
            seed=seed,
        )
        ppo = ppo[["step", "Improvement (%)"]]
        ppo["algorithm"] = "MA-PPO (S)"
        ppo["seed"] = seed
        algos_large += [ppo]

    algos_small_df = pd.concat(algos_small, ignore_index=True)
    algos_large_df = pd.concat(algos_large, ignore_index=True)

    hue_order = ["Opp. Control", "MA-PPO (S)", "MA-SAC (S)", "MA-PPO (L)", "MA-SAC (L)"]

    algos_small_df = algos_small_df[algos_small_df["step"] % 10 == 0]
    algos_large_df = algos_large_df[algos_large_df["step"] % 10 == 0]

    fig, axs = plt.subplots(
        1, 2, figsize=(FIGWIDTH * 0.45, 2.25), sharex=True, sharey=True
    )

    for ax, df, title in zip(
        axs,
        [algos_small_df, algos_large_df],
        ["Small Channel", "Large Channel"],
        strict=False,
    ):
        for algo in hue_order:
            df_algo = df[df["algorithm"] == algo]
            sns.lineplot(
                data=df_algo,
                x="step",
                y="Improvement (%)",
                ax=ax,
                label=algo,
                color=bright_colors[hue_order.index(algo)],
                linestyle="--" if algo == "Opp. Control" else "-",
                errorbar=("ci", 95),
            )
        ax.set_title(title)
        ax.legend().remove()
        ax.set_xlabel(r"Test Episode Step $t$")
        ax.set_ylabel("")
        ax.set_ylim(0, None)
        format_axis(ax, border=True)

    axs[0].set_ylabel("Drag Reduction " + r"(\%)")
    handles, y_labels = axs[-1].get_legend_handles_labels()

    # insert a dummy legend item to adjust spacing
    handles.insert(1, plt.Line2D([], [], color="white"))  # type: ignore
    y_labels.insert(1, "")

    fig.legend(
        handles,
        y_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.215),
        ncol=3,
        fancybox=False,
        shadow=False,
        frameon=False,
    )
    fig.subplots_adjust(
        top=0.9,
        bottom=0.39,
        left=0.11,
        right=0.98,
        hspace=0.3,
        wspace=0.18,
    )

    plt.savefig(
        PLOTS_PATH / "individual" / f"TCFSmall-{difficulty}-transfer.pdf", format="pdf"
    )
    plt.close()


def load_rbc_pd_mean(env_id: str) -> float:
    rewards = []
    for seed in PD_RBC_SEEDS:
        episode_path = RBC_PD_PATH / f"{env_id}/{seed}/test_eval_episode_0.csv"
        if not episode_path.exists():
            logger.warning(f"No PD episode found for {env_id}, seed {seed}")
            continue
        df = pd.read_csv(episode_path)
        rewards.append(df["reward"].mean())
    if not rewards:
        return float("nan")
    return float(np.mean(rewards))


def plot_training_results(env_name: str, metric: str = "reward", log: bool = False) -> None:
    training_runs = load_training_runs(env_name=env_name)

    algorithms = training_runs["algorithm"].unique()
    algorithms = sorted(algorithms, key=lambda x: ALGORITHM_ORDER.index(x))

    max_steps = int(training_runs["step"].max())

    palette = [bright_colors[ALGORITHM_COLOR_IDXS[alg]] for alg in algorithms]

    fig, axs = plt.subplots(1, 3, figsize=(FIGWIDTH, 2.5))

    for difficulty, ax in zip(DIFFICULTY_LEVELS, axs, strict=False):
        if difficulty is None:
            ax.axis("off")
            continue

        if log:
            ax.set_xscale("log")

        difficulty_data = training_runs[training_runs["difficulty"] == difficulty]

        if "RBC2D" in env_name and metric == "reward":
            pd_mean = load_rbc_pd_mean(f"{env_name}-{difficulty}-v0")
            ax.axhline(
                pd_mean,
                color=bright_colors[ALGORITHM_COLOR_IDXS["PD"]],
                linestyle="--",
                linewidth=1.5,
                label="PD",
            )

        sns.lineplot(
            data=difficulty_data,
            x="step",
            y=f"training/mean_{metric}",
            hue="algorithm",
            hue_order=algorithms,
            palette=palette,
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.set_title(f"{env_name}-{difficulty}-v0")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("")
        if not log:
            ax.set_xticks([0, max_steps])
            ax.set_xticklabels([0, format_step(max_steps)])
        ax.legend().remove()
        format_axis(ax, border=True)

    axs[0].set_ylabel(METRICS[metric])

    handles, labels = axs[0].get_legend_handles_labels()
    n_legend_cols = len(algorithms) + (1 if "RBC2D" in env_name and metric == "reward" else 0)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.1),
        ncol=n_legend_cols,
        fancybox=False,
        shadow=False,
        frameon=False,
    )
    fig.subplots_adjust(
        top=0.89, bottom=0.25, left=0.1, right=0.95, hspace=0.25, wspace=0.33
    )

    suffix = "_log" if log else ""
    plt.savefig(
        PLOTS_PATH / "quant_training" / f"{env_name}_training_{metric}{suffix}.pdf",
        format="pdf",
    )
    plt.close()


def plot_training_results_dpc(env_id_left: str, env_id_right: str) -> None:
    from matplotlib.lines import Line2D

    fig, axs = plt.subplots(1, 2, figsize=(FIGWIDTH * 0.45, 2.35))

    parts = env_id_right.rsplit("-", 2)
    right_runs = load_training_runs(env_name=parts[0])
    right_algorithms = sorted(
        right_runs[right_runs["difficulty"] == parts[1]]["algorithm"].unique(),
        key=lambda x: ALGORITHM_ORDER.index(x),
    )
    right_palette = [bright_colors[ALGORITHM_COLOR_IDXS[alg]] for alg in right_algorithms]

    env_ids = [env_id_left, env_id_right]

    for ax, env_id in zip(axs, env_ids):
        parts = env_id.rsplit("-", 2)
        env_name = parts[0]
        difficulty = parts[1]

        training_runs = load_training_runs(env_name=env_name)
        difficulty_data = training_runs[training_runs["difficulty"] == difficulty]

        algorithms = sorted(
            difficulty_data["algorithm"].unique(),
            key=lambda x: ALGORITHM_ORDER.index(x),
        )
        palette = [bright_colors[ALGORITHM_COLOR_IDXS[alg]] for alg in algorithms]

        ax.set_xscale("log")

        sns.lineplot(
            data=difficulty_data,
            x="step",
            y="training/mean_reward",
            hue="algorithm",
            hue_order=algorithms,
            palette=palette,
            errorbar=("ci", 95),
            ax=ax,
        )

        if env_id == "CylinderJet2D-easy-v0":
            dpc_color = bright_colors[ALGORITHM_COLOR_IDXS["DPC"]]
            non_dpc = [a for a in algorithms if a != "DPC"]
            non_dpc_palette = [bright_colors[ALGORITHM_COLOR_IDXS[a]] for a in non_dpc]

            # Clear the full plot and re-draw with DPC split at 10k
            ax.cla()
            ax.set_xscale("log")

            # All non-DPC algorithms: full range
            sns.lineplot(
                data=difficulty_data[difficulty_data["algorithm"].isin(non_dpc)],
                x="step",
                y="training/mean_reward",
                hue="algorithm",
                hue_order=non_dpc,
                palette=non_dpc_palette,
                errorbar=("ci", 95),
                ax=ax,
            )
            # DPC solid < 10k
            sns.lineplot(
                data=difficulty_data[
                    (difficulty_data["algorithm"] == "DPC") & (difficulty_data["step"] <= 10000)
                ],
                x="step",
                y="training/mean_reward",
                errorbar=("ci", 95),
                ax=ax,
                color=dpc_color,
            )
            # DPC dashed >= 10k
            sns.lineplot(
                data=difficulty_data[
                    (difficulty_data["algorithm"] == "DPC") & (difficulty_data["step"] >= 10000)
                ],
                x="step",
                y="training/mean_reward",
                errorbar=("ci", 95),
                ax=ax,
                color=dpc_color,
                alpha=0.3,
            )

        ax.set_title(env_id)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("")
        ax.legend().remove()
        format_axis(ax, border=True)

    axs[0].set_ylabel(METRICS["reward"])

    handles = [
        Line2D([0], [0], color=color, label=alg)
        for alg, color in zip(right_algorithms, right_palette)
    ]
    fig.legend(
        handles,
        right_algorithms,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.215),
        ncol=len(right_algorithms) // 2,
        fancybox=False,
        shadow=False,
        frameon=False,
    )
    fig.subplots_adjust(
        top=0.9,
        bottom=0.39,
        left=0.12,
        right=0.98,
        hspace=0.3,
        wspace=0.2,
    )

    plt.savefig(
        PLOTS_PATH / "quant_training" / f"dpc_{env_id_left}_{env_id_right}_training_reward.pdf",
        format="pdf",
    )
    plt.close()

def plot_training_results_dpc_combined(env_id_left: str, env_id_right: str, seed: int) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(FIGWIDTH * 0.45, 4.25))

    # ------------------------------------------------------------------------------
    # Training Runs
    # ------------------------------------------------------------------------------
    parts = env_id_right.rsplit("-", 2)
    top_env_ids = [env_id_left, env_id_right]

    for ax, env_id in zip(axs[0], top_env_ids):
        parts = env_id.rsplit("-", 2)
        env_name = parts[0]
        difficulty = parts[1]

        training_runs = load_training_runs(env_name=env_name, difficulty_levels=[difficulty])
        difficulty_data = training_runs[training_runs["difficulty"] == difficulty]

        algorithms = sorted(
            difficulty_data["algorithm"].unique(),
            key=lambda x: ALGORITHM_ORDER.index(x),
        )
        palette = [bright_colors[ALGORITHM_COLOR_IDXS[alg]] for alg in algorithms]

        ax.set_xscale("log")

        sns.lineplot(
            data=difficulty_data,
            x="step",
            y="training/mean_reward",
            hue="algorithm",
            hue_order=algorithms,
            palette=palette,
            errorbar=("ci", 95),
            ax=ax,
        )

        if env_id == "CylinderJet2D-easy-v0":
            dpc_color = bright_colors[ALGORITHM_COLOR_IDXS["DPC"]]
            non_dpc = [a for a in algorithms if a != "DPC"]
            non_dpc_palette = [bright_colors[ALGORITHM_COLOR_IDXS[a]] for a in non_dpc]

            # Clear the full plot and re-draw with DPC split at 10k
            ax.cla()
            ax.set_xscale("log")

            # All non-DPC algorithms: full range
            sns.lineplot(
                data=difficulty_data[difficulty_data["algorithm"].isin(non_dpc)],
                x="step",
                y="training/mean_reward",
                hue="algorithm",
                hue_order=non_dpc,
                palette=non_dpc_palette,
                errorbar=("ci", 95),
                ax=ax,
            )
            # DPC solid < 10k
            sns.lineplot(
                data=difficulty_data[
                    (difficulty_data["algorithm"] == "DPC") & (difficulty_data["step"] <= 10000)
                ],
                x="step",
                y="training/mean_reward",
                errorbar=("ci", 95),
                ax=ax,
                color=dpc_color,
            )
            # DPC dashed >= 10k
            sns.lineplot(
                data=difficulty_data[
                    (difficulty_data["algorithm"] == "DPC") & (difficulty_data["step"] >= 10000)
                ],
                x="step",
                y="training/mean_reward",
                errorbar=("ci", 95),
                ax=ax,
                color=dpc_color,
                alpha=0.3,
            )

        ax.set_title(env_id)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("")
        ax.legend().remove()
        format_axis(ax, border=True)

    axs[0, 0].set_ylabel(METRICS["reward"])


    # ------------------------------------------------------------------------------
    # Cylinder Test Episode
    # ------------------------------------------------------------------------------
    env = fluidgym.make(env_id_left)

    uncontrolled_df = env._load_uncontrolled_episode(idx=0, mode=EnvMode.TEST)
    ppo_df = load_test_sequence(
        env_id=env_id_left, rl_mode="sarl", algorithm="PPO", seed=seed
    )
    sac_df = load_test_sequence(
        env_id=env_id_left, rl_mode="sarl", algorithm="SAC", seed=seed
    )
    td_mcp_df = load_test_sequences(
        env_id=env_id_left, rl_mode="sarl", algorithm="TD-MPC", seed=seed
    )
    td_mcp_df = td_mcp_df[td_mcp_df["episode"] == 0].reset_index(drop=True)
    dpc_df = load_test_sequence(
        env_id=env_id_left, rl_mode="sarl", algorithm="DPC", seed=seed
    )

    ppo_df["step"] = ppo_df.index
    sac_df["step"] = sac_df.index
    uncontrolled_df["step"] = uncontrolled_df.index

    labels = [r"$a_t$", r"$C_D$"]
    metrics = ["action_0", "drag"]
    titles = ["Control Action", "Drag Coefficient"]

    # Compute the drag reduction for SAC
    mean_base_drag = uncontrolled_df["drag"].mean()
    sac_drag = sac_df["drag"].to_numpy()[-1]
    dpc_drag = dpc_df["drag"].to_numpy()[-1]
    sac_drag_reduction = (mean_base_drag - sac_drag) / mean_base_drag * 100
    dpc_drag_reduction = (mean_base_drag - dpc_drag) / mean_base_drag * 100

    print(f"Drag Reduction (SAC): {sac_drag_reduction:.2f}%")
    print(f"Drag Reduction (DPC): {dpc_drag_reduction:.2f}%")

    for ax, metric in zip(axs[1], metrics, strict=False):
        if metric != "action_0" and metric != "reward":
            sns.lineplot(
                data=uncontrolled_df,
                x="step",
                y=metric,
                ax=ax,
                color=bright_colors[0],
                label="Baseflow",
                linestyle="-.",
            )
        sns.lineplot(
            data=ppo_df,
            x="step",
            y=metric,
            ax=ax,
            color=bright_colors[1],
            label="PPO",
        )
        sns.lineplot(
            data=sac_df,
            x="step",
            y=metric,
            ax=ax,
            color=bright_colors[2],
            label="SAC",
        )
        sns.lineplot(
            data=dpc_df,
            x="step",
            y=metric,
            ax=ax,
            color=bright_colors[ALGORITHM_COLOR_IDXS["DPC"]],
            label="DPC",
            zorder=10
        )
        sns.lineplot(
            data=td_mcp_df,
            x="step",
            y=metric,
            ax=ax,
            color=bright_colors[ALGORITHM_COLOR_IDXS["TD-MPC"]],
            label="TD-MPC",
        )
        ax.set_title(titles[metrics.index(metric)])
        ax.legend().remove()
        ax.set_xlabel(r"Episode Step $t$")
        ax.set_ylabel(labels[metrics.index(metric)])
        format_axis(ax, border=True)

    train_handles, train_labels = axs[0, 1].get_legend_handles_labels()
    test_handles, test_labels = axs[1, 1].get_legend_handles_labels()

    # Add Baseflow
    train_handles.insert(0, test_handles[0])
    train_labels.insert(0, test_labels[0])  

    # Add blank below Baseflow
    train_handles.insert(1, plt.Line2D([], [], color="white"))  # type: ignore
    train_labels.insert(1, "")

    fig.legend(
        train_handles,
        train_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.115),
        ncol=len(train_handles) // 2,
        fancybox=False,
        shadow=False,
        frameon=False,
    )
    fig.subplots_adjust(
        top=0.93,
        bottom=0.22,
        left=0.12,
        right=0.98,
        hspace=0.72,
        wspace=0.35,
    )

    plt.savefig(
        PLOTS_PATH / "individual" / f"dpc_combined.pdf",
        format="pdf",
    )
    plt.close()

def plot_training_results_dpc_combined_landscape(env_id_left: str, env_id_right: str, seed: int) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(FIGWIDTH, 2.25))

    # ------------------------------------------------------------------------------
    # Training Runs
    # ------------------------------------------------------------------------------
    parts = env_id_right.rsplit("-", 2)
    top_env_ids = [env_id_left, env_id_right]

    for ax, env_id in zip(axs[:2], top_env_ids):
        parts = env_id.rsplit("-", 2)
        env_name = parts[0]
        difficulty = parts[1]

        training_runs = load_training_runs(env_name=env_name, difficulty_levels=[difficulty])
        difficulty_data = training_runs[training_runs["difficulty"] == difficulty]

        algorithms = sorted(
            difficulty_data["algorithm"].unique(),
            key=lambda x: ALGORITHM_ORDER.index(x),
        )
        palette = [bright_colors[ALGORITHM_COLOR_IDXS[alg]] for alg in algorithms]

        ax.set_xscale("log")

        sns.lineplot(
            data=difficulty_data,
            x="step",
            y="training/mean_reward",
            hue="algorithm",
            hue_order=algorithms,
            palette=palette,
            errorbar=("ci", 95),
            ax=ax,
        )

        if env_id == "CylinderJet2D-easy-v0":
            dpc_color = bright_colors[ALGORITHM_COLOR_IDXS["DPC"]]
            non_dpc = [a for a in algorithms if a != "DPC"]
            non_dpc_palette = [bright_colors[ALGORITHM_COLOR_IDXS[a]] for a in non_dpc]

            # Clear the full plot and re-draw with DPC split at 10k
            ax.cla()
            ax.set_xscale("log")

            # All non-DPC algorithms: full range
            sns.lineplot(
                data=difficulty_data[difficulty_data["algorithm"].isin(non_dpc)],
                x="step",
                y="training/mean_reward",
                hue="algorithm",
                hue_order=non_dpc,
                palette=non_dpc_palette,
                errorbar=("ci", 95),
                ax=ax,
            )
            # DPC solid < 10k
            sns.lineplot(
                data=difficulty_data[
                    (difficulty_data["algorithm"] == "DPC") & (difficulty_data["step"] <= 10000)
                ],
                x="step",
                y="training/mean_reward",
                errorbar=("ci", 95),
                ax=ax,
                color=dpc_color,
            )
            # DPC dashed >= 10k
            sns.lineplot(
                data=difficulty_data[
                    (difficulty_data["algorithm"] == "DPC") & (difficulty_data["step"] >= 10000)
                ],
                x="step",
                y="training/mean_reward",
                errorbar=("ci", 95),
                ax=ax,
                color=dpc_color,
                alpha=0.3,
            )

        ax.set_title(env_id)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("")
        ax.legend().remove()
        format_axis(ax, border=True)

    axs[0].set_ylabel("Mean Training Reward")
    axs[1].set_ylabel("Mean Training Reward")

    # ------------------------------------------------------------------------------
    # Cylinder Test Episode
    # ------------------------------------------------------------------------------
    env = fluidgym.make(env_id_left)

    uncontrolled_df = env._load_uncontrolled_episode(idx=0, mode=EnvMode.TEST)
    ppo_df = load_test_sequence(
        env_id=env_id_left, rl_mode="sarl", algorithm="PPO", seed=seed
    )
    sac_df = load_test_sequence(
        env_id=env_id_left, rl_mode="sarl", algorithm="SAC", seed=seed
    )
    td_mcp_df = load_test_sequences(
        env_id=env_id_left, rl_mode="sarl", algorithm="TD-MPC", seed=seed
    )
    td_mcp_df = td_mcp_df[td_mcp_df["episode"] == 0].reset_index(drop=True)
    dpc_df = load_test_sequence(
        env_id=env_id_left, rl_mode="sarl", algorithm="DPC", seed=seed
    )

    ppo_df["step"] = ppo_df.index
    sac_df["step"] = sac_df.index
    uncontrolled_df["step"] = uncontrolled_df.index

    labels = [r"$a_t$", r"$C_D$"]
    metrics = ["action_0", "drag"]
    titles = ["Control Action", "Drag Coefficient"]

    # Compute the drag reduction for SAC
    mean_base_drag = uncontrolled_df["drag"].mean()
    sac_drag = sac_df["drag"].to_numpy()[-1]
    dpc_drag = dpc_df["drag"].to_numpy()[-1]
    sac_drag_reduction = (mean_base_drag - sac_drag) / mean_base_drag * 100
    dpc_drag_reduction = (mean_base_drag - dpc_drag) / mean_base_drag * 100

    print(f"Drag Reduction (SAC): {sac_drag_reduction:.2f}%")
    print(f"Drag Reduction (DPC): {dpc_drag_reduction:.2f}%")

    for ax, metric in zip(axs[2:], metrics, strict=False):
        if metric != "action_0" and metric != "reward":
            sns.lineplot(
                data=uncontrolled_df,
                x="step",
                y=metric,
                ax=ax,
                color=bright_colors[0],
                label="Baseflow",
                linestyle="-.",
            )
        sns.lineplot(
            data=ppo_df,
            x="step",
            y=metric,
            ax=ax,
            color=bright_colors[1],
            label="PPO",
        )
        sns.lineplot(
            data=sac_df,
            x="step",
            y=metric,
            ax=ax,
            color=bright_colors[2],
            label="SAC",
        )
        sns.lineplot(
            data=dpc_df,
            x="step",
            y=metric,
            ax=ax,
            color=bright_colors[ALGORITHM_COLOR_IDXS["DPC"]],
            label="DPC",
            zorder=10
        )
        sns.lineplot(
            data=td_mcp_df,
            x="step",
            y=metric,
            ax=ax,
            color=bright_colors[ALGORITHM_COLOR_IDXS["TD-MPC"]],
            label="TD-MPC",
        )
        ax.set_title(titles[metrics.index(metric)])
        ax.legend().remove()
        ax.set_xlabel(r"Test Episode Step $t$")
        ax.set_ylabel(labels[metrics.index(metric)])
        format_axis(ax, border=True)

    train_handles, train_labels = axs[1].get_legend_handles_labels()
    test_handles, test_labels = axs[-1].get_legend_handles_labels()

    # Add Baseflow
    train_handles.insert(0, test_handles[0])
    train_labels.insert(0, test_labels[0])  

    fig.legend(
        train_handles,
        train_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.115),
        ncol=len(train_handles),
        fancybox=False,
        shadow=False,
        frameon=False,
    )
    fig.subplots_adjust(
        top=0.89,
        bottom=0.3,
        left=0.07,
        right=0.99,
        wspace=0.37,
    )

    plt.savefig(
        PLOTS_PATH / "quant_training" / f"dpc_combined_landscape_{env_id_left}_{env_id_right}.png", format="png", dpi=500 #, transparent=True
    )
    plt.close()

def plot_test_results(env_name: str, metric: str = "reward") -> None:
    test_results = load_test_results(env_name=env_name)

    algorithms = list(test_results["algorithm"].unique())
    algorithms = sorted(algorithms, key=lambda x: ALGORITHM_ORDER.index(x))
    palette = [bright_colors[ALGORITHM_COLOR_IDXS[alg]] for alg in algorithms]

    if len(algorithms) > 4:
        height = 3
    else:
        height = 2.5
    fig, axs = plt.subplots(1, 3, figsize=(FIGWIDTH, height))

    difficulty_levels = DIFFICULTY_LEVELS

    for difficulty, ax in zip(difficulty_levels, axs, strict=False):
        if difficulty is None:
            ax.axis("off")
            continue

        difficulty_data = test_results[test_results["difficulty"] == difficulty]

        sns.boxplot(
            data=difficulty_data,
            x="algorithm",
            y=metric,
            order=algorithms,
            hue_order=algorithms,
            palette=palette,
            ax=ax,
        )
        ax.set_title(f"{env_name}-{difficulty}-v0")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("")
        if len(algorithms) > 4:
            ax.tick_params(axis='x', labelrotation=45)

    axs[0].set_ylabel(METRICS[metric])

    fig.subplots_adjust(
        top=0.89,
        bottom=0.28 if len(algorithms) > 4 else 0.17,
        left=0.1,
        right=0.98,
        hspace=0.25,
        wspace=0.3
    )

    plt.savefig(
        PLOTS_PATH / "quant_test" / f"{env_name}_test_{metric}.pdf", format="pdf"
    )
    plt.close()


def _load_stress_test_data(env_id: str, algorithm: str) -> pd.DataFrame:
    parts = env_id.rsplit("-", 2)
    env_name = parts[0]
    rl_modes = RL_MODES.get(env_name, ["sarl"])
    seeds = range(5)

    normal_sequences = []
    stress_sequences = []

    for rl_mode in rl_modes:
        for seed in seeds:
            normal_df = load_test_sequences(
                env_id=env_id,
                rl_mode=rl_mode,
                algorithm=algorithm,
                seed=seed,
                stress_test=False,
            )
            if not normal_df.empty:
                drop_cols = [col for col in normal_df.columns if col.startswith("action")]
                normal_df = normal_df.drop(columns=drop_cols)
                if "episode" not in normal_df.columns:
                    normal_df["episode"] = 0
                normal_df = normal_df.groupby("episode").mean().reset_index()
                if "step" in normal_df.columns:
                    normal_df = normal_df.drop(columns=["step"])
                normal_df["condition"] = "Default"
                normal_sequences.append(normal_df)

            stress_df = load_test_sequences(
                env_id=env_id,
                rl_mode=rl_mode,
                algorithm=algorithm,
                seed=seed,
                stress_test=True,
            )
            if not stress_df.empty:
                drop_cols = [col for col in stress_df.columns if col.startswith("action")]
                stress_df = stress_df.drop(columns=drop_cols)
                if "episode" not in stress_df.columns:
                    stress_df["episode"] = 0
                stress_df = stress_df.groupby("episode").mean().reset_index()
                if "step" in stress_df.columns:
                    stress_df = stress_df.drop(columns=["step"])
                stress_df["condition"] = "CFL=0.1"
                stress_sequences.append(stress_df)

    if not normal_sequences and not stress_sequences:
        return pd.DataFrame()

    return pd.concat(normal_sequences + stress_sequences, ignore_index=True)


def plot_stability_analyis_results(env_name: str, algorithm: str, metric: str) -> None:
    condition_colors = [bright_colors[0], bright_colors[3]]

    fig, axs = plt.subplots(1, 2, figsize=(FIGWIDTH * 0.5, 2.5), sharey=True)

    for ax, difficulty in zip(axs, ["easy", "hard"]):
        env_id = f"{env_name}-{difficulty}-v0"
        all_data = _load_stress_test_data(env_id, algorithm)
        if all_data.empty:
            logger.warning(f"No data found for {env_id}, algorithm {algorithm}")
            continue

        sns.boxplot(
            data=all_data,
            x="condition",
            y=metric,
            palette=condition_colors,
            ax=ax,
            order=["Default", "CFL=0.1"],
        )
        ax.set_title(env_id)
        ax.set_xlabel("")
        ax.set_ylabel(METRICS.get(metric, metric))
        format_axis(ax, border=True)

    fig.suptitle("Policy Stability Analysis")
    fig.subplots_adjust(top=0.82, bottom=0.15, left=0.13, right=0.98, wspace=0.3)

    safe_alg = algorithm.replace("/", "_").replace(" ", "_")
    plt.savefig(
        PLOTS_PATH / "individual" / f"{env_name}_{safe_alg}_stability_analysis.pdf", format="pdf"
    )
    plt.close()


def plot_transfer_results_cylinder_dimensionality() -> None:
    test_results_sarl = load_test_results(env_name="CylinderJet3D", rl_modes=["sarl"])
    test_results_marl = load_test_results(env_name="CylinderJet3D", rl_modes=["marl"])

    transfer_results = load_test_results(
        env_name="CylinderJet2D", test_env_name="CylinderJet3D", rl_modes=["sarl"]
    )
    transfer_results["env_id"] = transfer_results["env_id"].str.replace("2D", "3D")
    transfer_results = transfer_results.drop(columns=["local_reward"])

    # DEBUG mockup data
    transfer_results = transfer_results[
        [
            "reward",
            "drag",
            "lift",
            "algorithm",
            "difficulty",
            "env_id",
            "Improvement (%)",
            "seed",
        ]
    ]

    transfer_results["mode"] = r"SARL 2D $\rightarrow$ MARL 3D"
    test_results_sarl["mode"] = "SARL 3D"
    test_results_marl["mode"] = "MARL 3D"

    all_results = pd.concat(
        [test_results_sarl, test_results_marl, transfer_results], ignore_index=True
    )

    difficulty_order = ["easy", "medium", "hard"]
    all_results["difficulty"] = pd.Categorical(
        all_results["difficulty"], categories=difficulty_order, ordered=True
    )

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(FIGWIDTH * 0.45, 1.9), sharey=True)

    for algorithm, ax in zip(["PPO", "SAC"], axs, strict=True):
        algo_data = all_results[all_results["algorithm"].str.contains(algorithm)]
        sns.lineplot(
            data=algo_data,
            x="difficulty",
            y="Improvement (%)",
            hue="mode",
            marker="o",
            linewidth=2,
            ax=ax,
            errorbar=("ci", 95),
        )

        ax.legend().remove()
        ax.set_title(f"{algorithm}")
        ax.set_xlabel("Difficulty")
        ax.set_ylabel("Drag Reduction " + r"(\%)")
        ax.set_xticks(range(len(DIFFICULTY_LEVELS)))
        ax.set_xticklabels(DIFFICULTY_LEVELS)
        format_axis(ax, border=True)

    fig.legend(
        *axs[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.135),
        ncol=3,
        fancybox=False,
        shadow=False,
        frameon=False,
    )

    fig.subplots_adjust(
        top=0.87, bottom=0.33, left=0.12, right=0.99, hspace=0.25, wspace=0.2
    )

    plt.savefig(
        PLOTS_PATH / "individual" / "CylinderJet_transfer2d-3d.pdf", format="pdf"
    )
    plt.close()


def plot_performance_profile() -> None:
    test_results = load_all_test_results()
    test_results = test_results[["episode", "reward", "algorithm", "env_id", "seed"]]

    # Compute mean reward per (episode, algorithm, env_id, seed)
    test_results = (
        test_results.groupby(["algorithm", "env_id", "seed"]).mean().reset_index()
    )
    test_results = test_results[["reward", "algorithm", "env_id", "seed"]]

    # Min-max normalize rewards per env_id
    test_results["normalized_reward"] = test_results.groupby(["env_id"])[
        "reward"
    ].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Convert to matrix dictionary: one matrix per algorithm
    alg_to_matrix = {
        str(alg): group.pivot_table(
            index="seed", columns="env_id", values="normalized_reward", aggfunc="first"
        )
        .sort_index()
        .sort_index(axis=1)
        .values
        for alg, group in test_results.groupby("algorithm")
    }

    tau = np.linspace(-0.1, 1.1, 101)
    profiles, profile_cis = rly.create_performance_profile(
        alg_to_matrix, tau, reps=2000
    )

    fig, ax = plt.subplots(figsize=(FIGWIDTH * 0.45, 2.5))

    algorithms = test_results["algorithm"].unique()
    algorithms = sorted(algorithms, key=lambda x: ALGORITHM_ORDER.index(x))
    palette = [bright_colors[ALGORITHM_COLOR_IDXS[alg]] for alg in algorithms]

    for i, algorithm in enumerate(algorithms):
        profile = profiles[algorithm]
        sns.lineplot(x=tau, y=profile, color=palette[i], linewidth=2.0, label=algorithm)
        lower_ci, upper_ci = profile_cis[algorithm]
        ax.fill_between(tau, lower_ci, upper_ci, color=palette[i], alpha=0.3)
    ax.legend().remove()
    ax.set_xlabel(r"Min-Max Normalized Score ($\tau$) $\uparrow$")
    ax.set_ylabel(
        r"Fraction of runs with score $> \tau$",
    )
    ax.set_title("FluidGym Score Distributions")
    format_axis(ax, border=True)

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.1),
        ncol=len(profiles),
        fancybox=False,
        shadow=False,
        frameon=False,
    )

    fig.subplots_adjust(
        top=0.88, bottom=0.25, left=0.15, right=0.95, hspace=0.25, wspace=0.3
    )

    plt.savefig(PLOTS_PATH / "performance_profile.pdf", format="pdf")


def plot_env_categories_and_difficulties() -> None:
    test_results = load_all_test_results()
    test_results = test_results[
        ["episode", "reward", "algorithm", "env_id", "seed", "category", "difficulty"]
    ]

    # Compute mean reward per (episode, algorithm, env_id, seed)
    test_results = (
        test_results.groupby(["algorithm", "env_id", "seed", "category", "difficulty"])
        .mean()
        .reset_index()
    )
    test_results = test_results[
        ["reward", "algorithm", "env_id", "seed", "category", "difficulty"]
    ]

    # Min-max normalize rewards per env_id
    test_results["normalized_reward"] = test_results.groupby(["env_id"])[
        "reward"
    ].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    iqm_per_category = (
        test_results.groupby(["algorithm", "category"])["normalized_reward"]
        .apply(lambda x: rly_metrics.aggregate_iqm(x.values))
        .reset_index(name="iqm")
    )
    iqm_per_difficulty = (
        test_results.groupby(["algorithm", "difficulty"])["normalized_reward"]
        .apply(lambda x: rly_metrics.aggregate_iqm(x.values))
        .reset_index(name="iqm")
    )
    categories = list(np.unique(list(ENV_CATEGORIES.values())))
    difficulties = DIFFICULTY_LEVELS
    algorithms = test_results["algorithm"].unique()
    algorithms = sorted(algorithms, key=lambda x: ALGORITHM_ORDER.index(x))
    colors = [bright_colors[ALGORITHM_COLOR_IDXS[alg]] for alg in algorithms]

    cat_iqm = {
        algo: (
            iqm_per_category[iqm_per_category["algorithm"] == algo]
            .set_index("category")
            .reindex(categories)["iqm"]
            .fillna(0)
            .values
        )
        for algo in algorithms
    }

    diff_iqm = {
        algo: (
            iqm_per_difficulty[iqm_per_difficulty["algorithm"] == algo]
            .set_index("difficulty")
            .reindex(difficulties)["iqm"]
            .fillna(0)
            .values
        )
        for algo in algorithms
    }

    fig, axs = plt.subplots(
        figsize=(FIGWIDTH * 0.45, 2.5),
        nrows=1,
        ncols=2,
        subplot_kw=dict(projection="polar"),
    )

    theta_cat = radar_factory(len(categories))
    theta_diff = radar_factory(len(difficulties))

    ax = axs[0]
    for i, algo in enumerate(algorithms):
        vals = cat_iqm[algo]
        theta_closed = np.concatenate([theta_cat, theta_cat[:1]])
        vals_closed = np.concatenate([vals, vals[:1]])
        color = colors[i]

        ax.plot(theta_closed, vals_closed, label=algo, color=color)
        ax.fill(theta_closed, vals_closed, alpha=0.2, color=color)

    ax.set_xticks(theta_cat)
    ax.set_xticklabels(categories)
    ax.legend().remove()
    ax.set_rlabel_position(90)

    ax = axs[1]
    for i, algo in enumerate(algorithms):
        vals = diff_iqm[algo]
        theta_closed = np.concatenate([theta_diff, theta_diff[:1]])
        vals_closed = np.concatenate([vals, vals[:1]])
        color = colors[i]

        ax.plot(theta_closed, vals_closed, label=algo, color=color)
        ax.fill(theta_closed, vals_closed, alpha=0.2, color=color)

    ax.set_xticks(theta_diff)
    ax.set_xticklabels(difficulties)
    ax.set_theta_offset(np.deg2rad(90))
    ax.set_rlabel_position(0)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.1),
        ncol=len(algorithms),
        fancybox=False,
        shadow=False,
        frameon=False,
    )

    for ax in axs:
        ax.spines["polar"].set_visible(False)
        ax.tick_params(axis="y", labelsize=8, pad=0, colors="gray")
        ax.tick_params(axis="x", labelsize=10, pad=7)
        for lbl in ax.get_yticklabels():
            lbl.set_verticalalignment("bottom")
            lbl.set_horizontalalignment("center")

    fig.suptitle("IQM over Environment Categories and Difficulties")
    fig.subplots_adjust(
        top=0.75, bottom=0.2, left=0.07, right=0.95, hspace=0.25, wspace=0.3
    )

    plt.savefig(PLOTS_PATH / "env_categories_and_difficulties_iqm.pdf", format="pdf")


def plot_results_combined() -> None:
    test_results = load_all_test_results()
    test_results = test_results[
        [
            "episode",
            "algorithm",
            "env_id",
            "seed",
            "category",
            "difficulty",
            "Improvement (%)",
        ]
    ]

    # Filter out Airfoil3D-hard-v0
    test_results = test_results[test_results["env_id"] != "Airfoil3D-hard-v0"]

    # Compute mean reward per (episode, algorithm, env_id, seed)
    test_results = (
        test_results.groupby(["algorithm", "env_id", "seed", "category", "difficulty"])
        .mean()
        .reset_index()
    )
    test_results = test_results[
        ["algorithm", "env_id", "seed", "category", "difficulty", "Improvement (%)"]
    ]

    # Min-max normalize improvements per env_id
    test_results["score"] = test_results.groupby(["env_id"])[
        "Improvement (%)"
    ].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Convert to matrix dictionary: one matrix per algorithm
    alg_to_matrix = {
        str(alg): group.pivot_table(
            index="seed", columns="env_id", values="score", aggfunc="first"
        )
        .sort_index()
        .sort_index(axis=1)
        .values
        for alg, group in test_results.groupby("algorithm")
    }

    tau = np.linspace(-0.1, 1.1, 101)
    profiles, profile_cis = rly.create_performance_profile(
        alg_to_matrix, tau, reps=2000
    )

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(FIGWIDTH, 2.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], projection="polar")
    ax2 = fig.add_subplot(gs[0, 2], projection="polar")

    axs = [ax0, ax1, ax2]

    algorithms = test_results["algorithm"].unique()

    # Remove PD from overview plots
    remove_algos = ["PD", "DPC", "TD-MPC"]
    algorithms = [alg for alg in algorithms if alg not in remove_algos]
    algorithms = sorted(algorithms, key=lambda x: ALGORITHM_ORDER.index(x))

    palette = [bright_colors[ALGORITHM_COLOR_IDXS[alg]] for alg in algorithms]

    ax = axs[0]
    for i, algorithm in enumerate(algorithms):
        profile = profiles[algorithm]
        sns.lineplot(
            x=tau, y=profile, color=palette[i], linewidth=2.0, label=algorithm, ax=ax
        )
        lower_ci, upper_ci = profile_cis[algorithm]
        ax.fill_between(tau, lower_ci, upper_ci, color=palette[i], alpha=0.3)
    ax.legend().remove()
    ax.set_xlabel(r"Min-Max Normalized Score $\tau$ ($\uparrow$)")
    ax.set_ylabel(
        r"Fraction of runs with score $> \tau$",
    )
    ax.set_title("FluidGym Score Distributions")
    format_axis(ax, border=True)

    iqm_per_category = (
        test_results.groupby(["algorithm", "category"])["score"]
        .apply(lambda x: rly_metrics.aggregate_iqm(x.values))
        .reset_index(name="iqm")
    )
    iqm_per_difficulty = (
        test_results.groupby(["algorithm", "difficulty"])["score"]
        .apply(lambda x: rly_metrics.aggregate_iqm(x.values))
        .reset_index(name="iqm")
    )
    categories = list(np.unique(list(ENV_CATEGORIES.values())))
    difficulties = DIFFICULTY_LEVELS

    cat_iqm = {
        algo: (
            iqm_per_category[iqm_per_category["algorithm"] == algo]
            .set_index("category")
            .reindex(categories)["iqm"]
            .fillna(0)
            .values
        )
        for algo in algorithms
    }

    diff_iqm = {
        algo: (
            iqm_per_difficulty[iqm_per_difficulty["algorithm"] == algo]
            .set_index("difficulty")
            .reindex(difficulties)["iqm"]
            .fillna(0)
            .values
        )
        for algo in algorithms
    }

    theta_cat = radar_factory(len(categories))
    theta_diff = radar_factory(len(difficulties))

    ax = axs[1]
    for i, algo in enumerate(algorithms):
        vals = cat_iqm[algo]
        theta_closed = np.concatenate([theta_cat, theta_cat[:1]])
        vals_closed = np.concatenate([vals, vals[:1]])
        color = palette[i]

        ax.plot(theta_closed, vals_closed, label=algo, color=color)
        ax.fill(theta_closed, vals_closed, alpha=0.2, color=color)

    ax.set_xticks(theta_cat)
    ax.set_xticklabels(categories)
    ax.legend().remove()
    ax.set_rlabel_position(90)  # type: ignore

    ax = axs[2]
    for i, algo in enumerate(algorithms):
        vals = diff_iqm[algo]
        theta_closed = np.concatenate([theta_diff, theta_diff[:1]])
        vals_closed = np.concatenate([vals, vals[:1]])
        color = palette[i]

        ax.plot(theta_closed, vals_closed, label=algo, color=color)
        ax.fill(theta_closed, vals_closed, alpha=0.1, color=color)

    ax.set_xticks(theta_diff)
    ax.set_xticklabels(difficulties)
    ax.legend().remove()
    ax.set_theta_offset(np.deg2rad(90))  # type: ignore
    ax.set_rlabel_position(0)  # type: ignore

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.1),
        ncol=len(algorithms),
        fancybox=False,
        shadow=False,
        frameon=False,
    )

    for ax in axs[1:]:
        ax.spines["polar"].set_visible(False)
        ax.tick_params(axis="y", labelsize=8, pad=0, colors="gray")
        ax.tick_params(axis="x", labelsize=10, pad=7)
        for lbl in ax.get_yticklabels():
            lbl.set_verticalalignment("bottom")
            lbl.set_horizontalalignment("center")

    # fig.suptitle("IQM over Environment Categories and Difficulties")
    fig.subplots_adjust(
        top=0.85, bottom=0.27, left=0.07, right=0.95, hspace=0.25, wspace=0.3
    )

    plt.savefig(PLOTS_PATH / "results_overview.pdf", format="pdf")


def get_cmap_and_range(env_name: str) -> tuple[Colormap, float, float]:
    if "CylinderJet2D" in env_name or "CylinderRot2D" in env_name:
        cmap = plt.get_cmap("icefire")
        vmin, vmax = -10.0, 10.0
    elif "CylinderJet3D" in env_name:
        cmap = plt.get_cmap("rainbow")
        vmin, vmax = 0.0, 1.0
    elif "RBC2D" in env_name or "RBC3D" in env_name:
        cmap = plt.get_cmap("rainbow")
        vmin, vmax = 0.0, 1.75
    elif "Airfoil2D" in env_name:
        cmap = plt.get_cmap("icefire")
        vmin, vmax = -1.0, 1.0
    elif "Airfoil3D" in env_name:
        cmap = plt.get_cmap("rainbow")
        vmin, vmax = 0.0, 1.0
    elif "TCF" in env_name:
        cmap = plt.get_cmap("rainbow")
        vmin, vmax = 0.0, 0.9
    else:
        raise ValueError(f"Unknown env_name: {env_name}")

    return cmap, vmin, vmax


def plot_qualitative_results(env_name: str, seed: int) -> None:
    images = load_all_test_renders(env_name=env_name, seed=seed)
    algo_order = ["Baseflow", "PPO", "SAC", "MA-PPO", "MA-SAC"]
    if env_name == "RBC2D":
        algo_order.insert(1, "PD")
    if env_name == "CylinderJet2D" or env_name == "RBC2D":
        algo_order += ["DPC", "TD-MPC"]
    elif env_name == "CylinderRot2D":
        algo_order += ["TD-MPC"]
    algo_order = [algo for algo in algo_order if algo in images["easy"]]

    n_rows = len(images["easy"])
    n_cols = len(DIFFICULTY_LEVELS)

    if n_rows == 0:
        return

    ref_shape = next(
        (img.shape for d in images.values() for img in d.values() if img is not None),
        None,
    )

    _, axes = plt.subplots(
        n_rows, n_cols, figsize=(FIGWIDTH, 1.6 * n_rows), squeeze=False
    )

    for row, algo in enumerate(algo_order):
        for col, diff in enumerate(DIFFICULTY_LEVELS):
            ax = axes[row, col]

            if diff is None:
                ax.axis("off")
                continue

            img = images.get(diff, {}).get(algo, None)
            if img is not None:
                ax.imshow(img)
            elif ref_shape is not None:
                ax.set_aspect(ref_shape[0] / ref_shape[1])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if row == 0:
                ax.set_title(diff.capitalize())

            if col == 0:
                ax.set_ylabel(algo, rotation=90, va="center")
            else:
                ax.set_ylabel("")

        # We add a colorbar
        cmap, vmin, vmax = get_cmap_and_range(env_name)
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cax = axes[row, -1].inset_axes([1.02, 0.0, 0.03, 1.0])
        plt.colorbar(sm, cax=cax)

    plt.tight_layout()
    plt.savefig(
        PLOTS_PATH / "qual_test" / f"{env_name}_qualitative_seed{seed}.pdf",
        format="pdf",
    )


def plot_envs():
    output_path = Path("./plots/envs")

    fluidgym.make("CylinderJet2D-easy-v0").plot(output_path=output_path)
    fluidgym.make("RBC2D-easy-v0").plot(output_path=output_path)
    fluidgym.make("RBC2D-wide-easy-v0").plot(output_path=output_path)
    fluidgym.make("Airfoil2D-easy-v0").plot(output_path=output_path)
    fluidgym.make("TCFSmall3D-both-easy-v0").plot(output_path=output_path)
    fluidgym.make("TCFLarge3D-both-easy-v0").plot(output_path=output_path)


def create_result_table(env_type: str) -> None:
    test_results = load_all_test_results()

    if env_type == "Cylinder":
        relevant_metrics = ["drag", "lift"]
    elif env_type == "RBC":
        relevant_metrics = ["nusselt"]
    elif env_type == "Airfoil":
        relevant_metrics = ["aero_eff"]
    elif env_type == "TCF":
        relevant_metrics = ["wall_stress"]
    else:
        raise ValueError(f"Unknown env_type: {env_type}")

    # Filter Cylinder metrics
    test_results = test_results[test_results["env_id"].str.contains(env_type)]

    # Compute IQM per algorithm and env_id
    iqm_results = (
        test_results.groupby(["algorithm", "env_id"])
        .agg(
            {
                "reward": lambda x: rly_metrics.aggregate_iqm(x.values),
                **{
                    metric: lambda x: rly_metrics.aggregate_iqm(x.values)
                    for metric in relevant_metrics
                },
                "Improvement (%)": lambda x: rly_metrics.aggregate_iqm(x.values),
            }
        )
        .reset_index()
    )

    # Add rows for baseflows
    baseflow_rows = []
    for env_id in iqm_results["env_id"].unique():
        env = fluidgym.make(env_id)
        stats = env._load_domain_statistics()

        if "Airfoil" in env_id:
            stats["aero_eff"] = {"mean": stats["lift"]["mean"] / stats["drag"]["mean"]}

        row = {
            "env_id": env_id,
            "algorithm": "Baseflow",
            "reward": np.nan,
            **{metric: stats[metric]["mean"] for metric in relevant_metrics},
        }
        baseflow_rows += [row]

    baseflow_df = pd.DataFrame(baseflow_rows)
    iqm_results = pd.concat([iqm_results, baseflow_df], ignore_index=True)

    iqm_results["env_id"] = pd.Categorical(
        iqm_results["env_id"], categories=ENV_ID_ORDER, ordered=True
    )

    algorithms = ["Baseflow"] + ALGORITHM_ORDER
    if env_type == "Cylinder":
        algorithms += ["D-MPC"]

    iqm_results["algorithm"] = pd.Categorical(
        iqm_results["algorithm"],
        categories=algorithms,
        ordered=True,
    )
    iqm_results = iqm_results.sort_values(by=["env_id", "algorithm"])
    iqm_results.style.format(na_rep="")

    col_order = (
        ["env_id", "algorithm", "reward"] + relevant_metrics + ["Improvement (%)"]
    )
    iqm_results_reordered = iqm_results[col_order]
    if env_type == "TCF":
        iqm_results["wall_stress"] = iqm_results["wall_stress"].apply(
            lambda x: f"${x:.3e}$"
        )

    # Latex table
    iqm_results_reordered.to_latex(
        buf=TABLES_PATH / f"{env_type}_results_table.tex",
        index=False,
        float_format="$%.3f$",
        na_rep="-",
    )


def merge_all_training_runs() -> None:
    env_names = [
        "CylinderJet2D",
        "CylinderRot2D",
        "CylinderJet3D",
        "RBC2D",
        "RBC3D",
        "Airfoil2D",
        "Airfoil3D",
        "TCFSmall3D-both",
        "TCFLarge3D-both",
    ]

    runs_list = []
    for env_name in env_names:
        run = load_training_runs(env_name=env_name)
        runs_list += [run]

    all_runs = pd.concat(runs_list, ignore_index=True)
    ordered_cols = [
        'env_id', 'difficulty', 'step', 'seed', 'algorithm', 
        'training/mean_global_reward', 'training/mean_reward',
        'training/mean_drag', 'training/mean_lift',
        'training/mean_nusselt', 'training/mean_wall_stress',
        'training/mean_wall_stress_bottom', 'training/mean_wall_stress_top',
        'evaluation/mean_reward','evaluation/mean_drag', 'evaluation/mean_lift',
        'evaluation/mean_nusselt', 'evaluation/mean_wall_stress',
        'evaluation/mean_wall_stress_bottom', 'evaluation/mean_wall_stress_top'
    ]
    all_runs = all_runs[ordered_cols]
    all_runs.to_csv("training.csv", index=False)


def merge_all_test_sequences() -> None:
    all_episodes = load_all_test_results(average_over_episodes=False, include_transfer=True)
    ordered_cols = [
        'env_id', 'test_env_id', 'category', 'difficulty', 'algorithm', 'seed', 
        'episode', 'step', 'reward',  'local_reward', 'Improvement (%)', 'drag', 'lift',  
        'nusselt',  'aero_eff', 'wall_stress', 'wall_stress_bottom', 'wall_stress_top', 
    ]
    all_episodes = all_episodes[ordered_cols]
    all_episodes.to_csv("test.csv", index=False)


def plot_ppo_ablation(env_id: str) -> None:
    algorithms = ["[64, 64]", "[256, 256]"]
    palette = [bright_colors[ALGORITHM_COLOR_IDXS["PPO"]], bright_colors[ALGORITHM_COLOR_IDXS["DPC"]]]

    # Load training data
    all_runs = []
    for algorithm, label in [("PPO", "[64, 64]"), ("PPO-large", "[256, 256]")]:
        for seed in SEEDS:
            run_df = load_single_training_run(
                env_id=env_id,
                rl_mode="sarl",
                algorithm=algorithm,
                seed=seed,
            )
            if run_df.empty:
                logger.warning(f"No training run found for {env_id}, {label}, seed {seed}")
                continue
            run_df["seed"] = seed
            run_df["algorithm"] = label
            all_runs.append(run_df)

    # Load test data
    all_results = []
    for algorithm, label in [("PPO", "[64, 64]"), ("PPO-large", "[256, 256]")]:
        for seed in SEEDS:
            result_df = load_test_sequences(
                env_id=env_id,
                rl_mode="sarl",
                algorithm=algorithm,
                seed=seed,
            )
            if result_df.empty:
                logger.warning(f"No test results found for {env_id}, {label}, seed {seed}")
                continue

            drop_cols = [col for col in result_df.columns if col.startswith("action")]
            result_df = result_df.drop(columns=drop_cols)

            if "episode" not in result_df.columns:
                result_df["episode"] = 0

            result_df = result_df.groupby("episode").mean().reset_index()
            result_df = result_df.drop(columns=["step"])

            result_df["seed"] = seed
            result_df["algorithm"] = label

            all_results.append(result_df)

    if not all_runs and not all_results:
        logger.warning(f"No PPO ablation results found for {env_id}")
        return

    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(FIGWIDTH * 0.5, 2.5), sharey=True)

    # Training subplot
    if all_runs:
        training_runs = pd.concat(all_runs, ignore_index=True)
        max_steps = int(training_runs["step"].max())

        sns.lineplot(
            data=training_runs,
            x="step",
            y="training/mean_reward",
            hue="algorithm",
            hue_order=algorithms,
            palette=palette,
            errorbar=("ci", 95),
            ax=ax_train,
        )
        ax_train.set_title("Training")
        ax_train.set_xlabel("Training Step")
        ax_train.set_ylabel(METRICS["reward"])
        ax_train.set_xticks([0, max_steps])
        ax_train.set_xticklabels([0, format_step(max_steps)])
        ax_train.legend().remove()
        format_axis(ax_train, border=True)
    else:
        logger.warning(f"No PPO ablation training results found for {env_id}")

    # Test subplot
    if all_results:
        test_results = pd.concat(all_results, ignore_index=True)

        sns.boxplot(
            data=test_results,
            x="algorithm",
            y="reward",
            order=algorithms,
            hue_order=algorithms,
            palette=palette,
            ax=ax_test,
        )
        ax_test.set_title("Test")
        ax_test.set_xlabel("Network Size")
        ax_test.set_ylabel(METRICS["reward"])
        format_axis(ax_test, border=True)
    else:
        logger.warning(f"No PPO ablation test results found for {env_id}")

    handles, labels = ax_train.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.11),
        ncol=len(algorithms),
        fancybox=False,
        shadow=False,
        frameon=False,
    )

    fig.suptitle(f"{env_id} — PPO Network Size Ablation")

    fig.subplots_adjust(top=0.82, bottom=0.26, left=0.13, right=0.98, wspace=0.1)

    plt.savefig(
        PLOTS_PATH / "quant_test" / f"{env_id}_ppo_ablation.pdf",
        format="pdf",
    )
    plt.close()


if __name__ == "__main__":
    # Quantiative test result tables
    # create_result_table(env_type="Cylinder")
    # create_result_table(env_type="RBC")
    create_result_table(env_type="Airfoil")
    # create_result_table(env_type="TCF")

    # Summary plots
    # plot_results_combined()

    # Quantitative training results
    # plot_training_results(env_name="CylinderJet2D")
    # plot_training_results(env_name="CylinderRot2D")
    # plot_training_results(env_name="CylinderJet3D")
    # plot_training_results(env_name="RBC2D")
    # plot_training_results(env_name="RBC3D")
    # plot_training_results(env_name="Airfoil2D")
    # plot_training_results(env_name="Airfoil3D")
    # plot_training_results(env_name="TCFSmall3D-both")
    # plot_training_results(env_name="TCFLarge3D-both")

    # # Quantitative test results
    # plot_test_results(env_name="CylinderJet2D")
    # plot_test_results(env_name="CylinderRot2D")
    # plot_test_results(env_name="CylinderJet3D")
    # plot_test_results(env_name="RBC2D")
    # plot_test_results(env_name="RBC3D")
    # plot_test_results(env_name="Airfoil2D")
    # plot_test_results(env_name="Airfoil3D")
    # plot_test_results(env_name="TCFSmall3D-both")
    # plot_test_results(env_name="TCFLarge3D-both")

    # Individual plots
    # plot_rbc_test_episode(env_id="RBC3D-easy-v0", algorithm="PPO", seed=0)
    # plot_tcf_transfer(difficulty="easy")
    # plot_transfer_results_cylinder_dimensionality()

    # plot_qualitative_results(env_name="CylinderJet2D", seed=0)
    # plot_qualitative_results(env_name="CylinderRot2D", seed=0)
    # plot_qualitative_results(env_name="CylinderJet3D", seed=0)
    # plot_qualitative_results(env_name="RBC2D", seed=0)
    # plot_qualitative_results(env_name="RBC3D", seed=0)
    # plot_qualitative_results(env_name="Airfoil2D", seed=0)
    # plot_qualitative_results(env_name="Airfoil3D", seed=0)
    # plot_qualitative_results(env_name="TCFSmall3D-both", seed=0)
    # plot_qualitative_results(env_name="TCFLarge3D-both", seed=0)

    # DPC training curves
    # plot_training_results_dpc_combined(env_id_left="CylinderJet2D-easy-v0", env_id_right="RBC2D-hard-v0", seed=0)

    # PPO ablations
    # plot_ppo_ablation(env_id="CylinderJet2D-easy-v0")
    # plot_ppo_ablation(env_id="RBC2D-hard-v0")
   
    # Stability Analysis (aka. stress test)
    # plot_stability_analyis_results(
    #     env_name="RBC2D",
    #     algorithm="SAC",
    #     metric="nusselt"
    # )