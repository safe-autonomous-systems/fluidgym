import logging
import re
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any

import fluidgym

mpl.rcParams.update(
    {
        "text.usetex": True,  # or False if you're okay with mathtext
        "pdf.fonttype": 42,  # TrueType in PDF
        "ps.fonttype": 42,  # TrueType in PS/EPS
    }
)

logger = logging.getLogger(__name__)

EPISODE_LENGTHS = {
    "CylinderRot2D": 80,
    "CylinderJet2D": 80,
    "CylinderJet3D": 80,
    "RBC2D": 200,
    "RBC3D": 200,
    "Airfoil2D": 300,
    "Airfoil3D": 200,
    "TCFSmall3D-both": 1000,
    "TCFLarge3D-both": 1000,
}

TRAINING_PATH = Path("./output/training")
PLOTS_PATH = Path("./paper/plots")
PLOTS_PATH.mkdir(parents=True, exist_ok=True)
RUNTIME_PLOTS_PATH = PLOTS_PATH / "runtimes"
RUNTIME_PLOTS_PATH.mkdir(parents=True, exist_ok=True)
TABLES_PATH = Path("./paper/tables")
TABLES_PATH.mkdir(parents=True, exist_ok=True)

FIGWIDTH = 10.0

ENV_ORDER = [
    "CylinderRot2D",
    "CylinderJet2D",
    "CylinderJet3D",
    "RBC2D",
    "RBC3D",
    "Airfoil2D",
    "Airfoil3D",
    "TCFSmall3D-both",
    "TCFLarge3D-both",
]

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

CATEGORY_ORDER = ["Cylinder", "RBC", "Airfoil", "TCF"]

ALGORITHM_ORDER = ["PPO", "SAC", "MA-PPO", "MA-SAC", "DPC", "TD-MPC"]
BASELINE_ALGORITHMS = ["PPO", "SAC", "MA-PPO", "MA-SAC"]  # shown in the global plot
ALGORITHM_COLOR_IDXS = {
    "PPO": 1,
    "SAC": 2,
    "MA-PPO": 3,
    "MA-SAC": 4,
    "TD-MPC": 6,
    "DPC": 7,
}

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

TIMESTAMP_RE = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\]"
)
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S,%f"


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
            axis="y",
            which="major",
            linestyle="-",
            linewidth=linewidth,
            alpha=grid_alpha,
            color="lightgray",
            zorder=-1,
        )
        ax.grid(
            axis="y",
            which="minor",
            linestyle="-",
            linewidth=linewidth,
            alpha=grid_alpha,
            color="lightgray",
            zorder=-1,
        )
        ax.grid(axis="x", visible=False)
    elif grid is False:
        ax.grid(False)

def _parse_timestamp(line: str) -> datetime | None:
    m = TIMESTAMP_RE.match(line)
    if m:
        return datetime.strptime(m.group(1), TIMESTAMP_FMT)
    return None


def _parse_log_runtime(
    log_path: Path,
    start_keyword: str = "Starting training...",
    finish_keyword: str = "Training finished.",
    check_continuation: bool = True,
) -> float | None:
    if not log_path.exists():
        return None

    text = log_path.read_text()

    if check_continuation and "Continuing training" in text:
        return None

    lines = text.splitlines()

    # Find the LAST finish_keyword line
    last_finish_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if finish_keyword in lines[i]:
            last_finish_idx = i
            break

    if last_finish_idx is None:
        return None

    finish_ts = _parse_timestamp(lines[last_finish_idx])
    if finish_ts is None:
        return None

    # Find the last start_keyword BEFORE last_finish_idx
    start_ts = None
    for i in range(last_finish_idx - 1, -1, -1):
        if start_keyword in lines[i]:
            start_ts = _parse_timestamp(lines[i])
            break

    if start_ts is None:
        return None

    runtime_hours = (finish_ts - start_ts).total_seconds() / 3600.0
    return runtime_hours


def _format_algorithm(algo: str, rl_mode: str) -> str:
    if rl_mode == "marl" and algo not in ("DPC", "TD-MPC"):
        return f"MA-{algo}"
    return algo


def _get_total_steps(seed_dir: Path) -> int | None:
    """Read training_log.csv and return the maximum step value."""
    csv_path = seed_dir / "training_log.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, nrows=0)
        step_col = "step" if "step" in df.columns else "train/step" if "train/step" in df.columns else None
        if step_col is None:
            return None
        df = pd.read_csv(csv_path, usecols=[step_col])
        return int(df[step_col].max())
    except Exception:
        return None


def _resolve_log_path(seed_dir: Path, algo: str, rl_mode: str) -> tuple[Path | None, str, str, bool]:
    if algo == "DPC":
        for name in ("run_dpc.log", "run_dpc_pc2.log"):
            p = seed_dir / name
            if p.exists():
                return p, "Starting DPC Training...", "Running DPC Evaluation...", False
        return None, "", "", False
    elif algo == "TD-MPC":
        candidates = ["train_tdmpc.log", "train_tdmpc_pc2.log", "train_tdmpc_marl.log"]
        for name in candidates:
            p = seed_dir / name
            if p.exists():
                return p, "Starting training...", "Training finished.", True
        return None, "", "", False
    else:
        return seed_dir / "train_sb3.log", "Starting training...", "Training finished.", True


def read_training_runtimes() -> pd.DataFrame:
    records = []

    for rl_mode_dir in sorted(TRAINING_PATH.iterdir()):
        if not rl_mode_dir.is_dir():
            continue
        rl_mode = rl_mode_dir.name  # "sarl" or "marl"

        for env_dir in sorted(rl_mode_dir.iterdir()):
            if not env_dir.is_dir():
                continue
            env_id = env_dir.name  # e.g. "CylinderJet2D-easy-v0"

            for algo_dir in sorted(env_dir.iterdir()):
                if not algo_dir.is_dir():
                    continue
                algo = algo_dir.name  # e.g. "PPO", "SAC", "DPC", "TD-MPC"

                for seed_dir in sorted(algo_dir.iterdir()):
                    if not seed_dir.is_dir():
                        continue
                    try:
                        seed = int(seed_dir.name)
                    except ValueError:
                        continue

                    log_path, start_kw, finish_kw, check_cont = _resolve_log_path(seed_dir, algo, rl_mode)
                    if log_path is None:
                        logger.debug(f"Skipped {rl_mode}/{env_id}/{algo}/{seed}: no log file found")
                        continue

                    runtime = _parse_log_runtime(log_path, start_kw, finish_kw, check_cont)
                    if runtime is None:
                        logger.debug(
                            f"Skipped {rl_mode}/{env_id}/{algo}/{seed}: "
                            "no log, continued run, or incomplete"
                        )
                        continue

                    total_steps = _get_total_steps(seed_dir)
                    if total_steps is None or total_steps == 0:
                        logger.debug(f"Skipped {rl_mode}/{env_id}/{algo}/{seed}: no training_log.csv or zero steps")
                        continue

                    runtime_per_step = (runtime * 3600.0) / total_steps  # seconds per step

                    # DPC and TD-MPC don't run evaluations during training;
                    # add the equivalent of 10 evaluation episodes to match RL baselines.
                    if algo in ("DPC", "TD-MPC"):
                        env_without_version = env_id.removesuffix("-v0")
                        parts_tmp = env_without_version.rsplit("-", 1)
                        env_name_tmp = parts_tmp[0] if len(parts_tmp) == 2 and parts_tmp[1] in ("easy", "medium", "hard") else env_without_version
                        episode_length = EPISODE_LENGTHS.get(env_name_tmp)
                        if episode_length is not None:
                            eval_overhead_hours = (10 * episode_length * runtime_per_step) / 3600.0
                            runtime += eval_overhead_hours
                            runtime_per_step = (runtime * 3600.0) / total_steps

                    algorithm = _format_algorithm(algo, rl_mode)

                    # Split env_id into name + difficulty, e.g. "CylinderJet2D-easy-v0"
                    env_without_version = env_id.removesuffix("-v0")
                    parts = env_without_version.rsplit("-", 1)
                    if len(parts) == 2 and parts[1] in ("easy", "medium", "hard"):
                        env_name = parts[0]
                        difficulty = parts[1]
                    else:
                        env_name = env_without_version
                        difficulty = None

                    category = ENV_CATEGORIES.get(env_name, env_name)

                    records.append(
                        {
                            "rl_mode": rl_mode,
                            "env_id": env_id,
                            "env_name": env_name,
                            "category": category,
                            "difficulty": difficulty,
                            "algorithm": algorithm,
                            "seed": seed,
                            "runtime_hours": runtime,
                            "total_steps": total_steps,
                            "runtime_per_step": runtime_per_step,
                        }
                    )

    df = pd.DataFrame(records)

    if df.empty:
        return df

    env_order_present = [e for e in ENV_ORDER if e in df["env_name"].unique()]
    df["env_name"] = pd.Categorical(df["env_name"], categories=env_order_present, ordered=True)
    cat_order_present = [c for c in CATEGORY_ORDER if c in df["category"].unique()]
    df["category"] = pd.Categorical(df["category"], categories=cat_order_present, ordered=True)
    algo_order_present = [a for a in ALGORITHM_ORDER if a in df["algorithm"].unique()]
    df["algorithm"] = pd.Categorical(df["algorithm"], categories=algo_order_present, ordered=True)

    return df.sort_values(["category", "env_name", "difficulty", "algorithm", "seed"]).reset_index(drop=True)



def _draw_runtime_bars(ax: plt.Axes, agg: pd.DataFrame, group_col: str, groups: list, algorithms: list, metric: str) -> None:
    palette = [bright_colors[ALGORITHM_COLOR_IDXS[alg]] for alg in algorithms]
    x = np.arange(len(groups))
    bar_width = 0.8 / len(algorithms)

    for i, (algo, color) in enumerate(zip(algorithms, palette)):
        algo_data = agg[agg["algorithm"] == algo].set_index(group_col)
        means = np.array([algo_data.loc[g, "mean"] if g in algo_data.index else np.nan for g in groups], dtype=float)
        sems = np.array([algo_data.loc[g, "sem"] if g in algo_data.index else np.nan for g in groups], dtype=float)

        offset = (i - (len(algorithms) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            means,
            width=bar_width,
            color=color,
            label=algo,
            yerr=sems,
            capsize=3,
            error_kw={"linewidth": 0.8, "ecolor": "black"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(groups)


def plot_runtimes_all_envs(
    df: pd.DataFrame,
    per_step: bool = True,
    exclude_algorithms: list[str] | None = None,
) -> None:
    if df.empty:
        logger.warning("Empty dataframe — nothing to plot.")
        return

    metric = "runtime_per_step" if per_step else "runtime_hours"
    ylabel = "Runtime per Step (s)" if per_step else "Runtime (hours)"
    step_part = "_per_step" if per_step else ""
    out_path = RUNTIME_PLOTS_PATH / f"runtime_all{step_part}.pdf"

    exclude = set(exclude_algorithms or [])
    algorithms_wanted = [a for a in ["PPO", "SAC", "MA-PPO", "MA-SAC", "TD-MPC"] if a not in exclude]
    df_base = df[df["algorithm"].isin(algorithms_wanted)].copy()
    algorithms = [a for a in algorithms_wanted if a in df_base["algorithm"].values]

    agg = (
        df_base.groupby(["category", "algorithm"], observed=True)[metric]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])

    categories = [c for c in CATEGORY_ORDER if c in agg["category"].values]

    fig, ax = plt.subplots(figsize=(FIGWIDTH * 0.45, 1.8))
    _draw_runtime_bars(ax, agg, "category", categories, algorithms, metric)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.15),
               ncol=len(algorithms), fancybox=False, shadow=False, frameon=False)
    format_axis(ax, grid=True, border=True)
    ax.tick_params(axis="x", which="both", length=0)
    fig.subplots_adjust(top=0.95, bottom=0.22, left=0.15, right=0.99)
    
    plt.savefig(out_path, format="pdf")
    plt.close(fig)

def plot_runtimes_env_comparison(
    df: pd.DataFrame,
    per_step: bool = True,
    exclude_algorithms: list[str] | None = None,
) -> None:
    if df.empty:
        logger.warning("Empty dataframe — nothing to plot.")
        return

    metric = "runtime_per_step" if per_step else "runtime_hours"
    ylabel = "Runtime per Step (s)" if per_step else "Runtime (hours)"
    step_part = "_per_step" if per_step else ""
    out_path = RUNTIME_PLOTS_PATH / f"runtime_cyl_rbc{step_part}.pdf"

    exclude = set(exclude_algorithms or [])
    algorithms = [a for a in ["PPO", "SAC", "MA-PPO", "MA-SAC", "DPC", "TD-MPC"] if a not in exclude]
    envs = ["CylinderJet2D", "RBC2D"]
    df_base = df[df["env_name"].isin(envs) & df["algorithm"].isin(algorithms)].copy()

    # DPC on CylinderJet2D-easy-v0 was only trained for 10k steps; extrapolate to 50k
    if not per_step:
        mask = (df_base["algorithm"] == "DPC") & (df_base["env_name"] == "CylinderJet2D") & (df_base["difficulty"] == "easy")
        df_base.loc[mask, "runtime_hours"] = df_base.loc[mask, "runtime_hours"] * (50_000 / df_base.loc[mask, "total_steps"])

    agg = (
        df_base.groupby(["env_name", "algorithm"], observed=True)[metric]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])

    env_labels = [e for e in envs if e in agg["env_name"].values]
    algorithms = [a for a in algorithms if a in agg["algorithm"].values]

    fig, ax = plt.subplots(figsize=(FIGWIDTH * 0.45, 2.0))
    _draw_runtime_bars(ax, agg, "env_name", env_labels, algorithms, metric)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.23),
               ncol=len(algorithms) // 2, fancybox=False, shadow=False, frameon=False)
    format_axis(ax, grid=True, border=True)
    ax.tick_params(axis="x", which="both", length=0)
    fig.subplots_adjust(top=0.95, bottom=0.3, left=0.15, right=0.99)
    
    plt.savefig(out_path, format="pdf")
    plt.close(fig)

def plot_runtimes_combined(
    df: pd.DataFrame,
    per_step: bool = True,
    exclude_algorithms_top: list[str] | None = None,
) -> None:
    """
    Two-panel runtime plot:
      top    — runtime per category (all envs), with optional algorithm exclusions
      bottom — runtime for CylinderJet2D and RBC2D with PPO/SAC/MA-PPO/MA-SAC/DPC/TD-MPC
    A single shared legend is placed below both panels using the bottom plot's labels.
    """
    if df.empty:
        logger.warning("Empty dataframe — nothing to plot.")
        return

    metric = "runtime_per_step" if per_step else "runtime_hours"
    ylabel = "Runtime per Step (s)" if per_step else "Runtime (h)"
    step_part = "_per_step" if per_step else ""
    out_path = RUNTIME_PLOTS_PATH / f"runtime_combined{step_part}.pdf"

    _merge_map = {} #{"MA-PPO": "PPO", "MA-SAC": "SAC"}

    # --- top subplot data ---
    exclude_top = set(exclude_algorithms_top or [])
    top_algos_wanted = [a for a in ["PPO", "SAC", "MA-PPO", "MA-SAC", "TD-MPC"] if a not in exclude_top]
    df_top = df[df["algorithm"].isin(top_algos_wanted)].copy()
    df_top["algorithm"] = df_top["algorithm"].map(lambda a: _merge_map.get(str(a), str(a)))
    top_algos_merged = [_merge_map.get(a, a) for a in top_algos_wanted]
    top_algos = list(dict.fromkeys(a for a in top_algos_merged if a in df_top["algorithm"].values))
    agg_top = (
        df_top.groupby(["category", "algorithm"], observed=False)[metric]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    agg_top["sem"] = agg_top["std"] / np.sqrt(agg_top["count"])
    categories = [c for c in CATEGORY_ORDER if c in agg_top["category"].values]

    # --- bottom subplot data ---
    bot_algos_wanted = ["PPO", "SAC", "MA-PPO", "MA-SAC", "DPC", "TD-MPC"]
    envs = ["CylinderJet2D", "RBC2D"]
    df_bot = df[df["env_name"].isin(envs) & df["algorithm"].isin(bot_algos_wanted)].copy()
    if not per_step:
        mask = (df_bot["algorithm"] == "DPC") & (df_bot["env_name"] == "CylinderJet2D") & (df_bot["difficulty"] == "easy")
        df_bot.loc[mask, "runtime_hours"] = df_bot.loc[mask, "runtime_hours"] * (50_000 / df_bot.loc[mask, "total_steps"])
    df_bot["algorithm"] = df_bot["algorithm"].map(lambda a: _merge_map.get(str(a), str(a)))
    bot_algos_merged = [_merge_map.get(a, a) for a in bot_algos_wanted]
    bot_algos_ordered = list(dict.fromkeys(bot_algos_merged))
    agg_bot = (
        df_bot.groupby(["env_name", "algorithm"], observed=False)[metric]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    agg_bot["sem"] = agg_bot["std"] / np.sqrt(agg_bot["count"])
    env_labels = [e for e in envs if e in agg_bot["env_name"].values]
    bot_algos = [a for a in bot_algos_ordered if a in agg_bot["algorithm"].values]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(FIGWIDTH * 0.45, 2.8))

    _draw_runtime_bars(ax_top, agg_top, "category", categories, top_algos, metric)
    ax_top.set_ylabel(ylabel)
    ax_top.set_xlabel("")
    format_axis(ax_top, grid=True, border=True)
    ax_top.tick_params(axis="x", which="both", length=0)

    _draw_runtime_bars(ax_bot, agg_bot, "env_name", env_labels, bot_algos, metric)
    ax_bot.set_ylabel(ylabel)
    ax_bot.set_xlabel("")
    format_axis(ax_bot, grid=True, border=True)
    ax_bot.tick_params(axis="x", which="both", length=0)

    handles, labels = ax_bot.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.16),
               ncol=len(bot_algos) // 2, fancybox=False, shadow=False, frameon=False)
    fig.subplots_adjust(top=0.97, bottom=0.21, left=0.15, right=0.97, hspace=0.35)
    plt.savefig(out_path, format="pdf")
    plt.close(fig)


def _make_runtime_table(
    df: pd.DataFrame,
    algorithms: list[str],
    index_col: str,
    index_order: list[str],
    metric: str,
    decimals: int,
    caption: str,
    label: str,
    out_path: Path,
) -> None:
    agg = (
        df.groupby([index_col, "algorithm"], observed=True)[metric]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    def fmt(row):
        mean, std = row["mean"], row["std"]
        if pd.isna(mean):
            return "-"
        if pd.isna(std) or std == 0:
            return f"${mean:.{decimals}f}$"
        return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"

    agg["formatted"] = agg.apply(fmt, axis=1)
    pivot = agg.pivot_table(index=index_col, columns="algorithm", values="formatted", aggfunc="first")
    pivot.columns.name = None
    for algo in algorithms:
        if algo not in pivot.columns:
            pivot[algo] = "-"
    pivot = pivot[algorithms]
    pivot = pivot.reindex([r for r in index_order if r in pivot.index])
    pivot.to_latex(buf=out_path, index=True, escape=False, na_rep="-", caption=caption, label=label)
    logger.info(f"Runtime table saved to {out_path}")


def create_runtime_table_all_envs(df: pd.DataFrame, per_step: bool = True) -> None:
    if df.empty:
        logger.warning("Empty dataframe — no table created.")
        return

    metric = "runtime_per_step" if per_step else "runtime_hours"
    decimals = 3 if per_step else 1
    caption_unit = "seconds per step" if per_step else "hours"
    step_part = "_per_step" if per_step else ""
    out_path = TABLES_PATH / f"runtime_all{step_part}.tex"

    algorithms_wanted = ["PPO", "SAC", "MA-PPO", "MA-SAC", "TD-MPC"]
    df_base = df[df["algorithm"].isin(algorithms_wanted)].copy()
    algorithms = [a for a in algorithms_wanted if a in df_base["algorithm"].values]
    _make_runtime_table(
        df_base, algorithms,
        index_col="category", index_order=CATEGORY_ORDER,
        metric=metric, decimals=decimals,
        caption=f"Mean $\\pm$ std training runtime ({caption_unit}) per category.",
        label="tab:runtimes_all_envs",
        out_path=out_path,
    )


def create_runtime_table_env_comparison(df: pd.DataFrame, per_step: bool = True) -> None:
    if df.empty:
        logger.warning("Empty dataframe — no table created.")
        return

    metric = "runtime_per_step" if per_step else "runtime_hours"
    decimals = 3 if per_step else 1
    caption_unit = "seconds per step" if per_step else "hours"
    step_part = "_per_step" if per_step else ""
    out_path = TABLES_PATH / f"runtime_cyl_rbc{step_part}.tex"

    algorithms = ["PPO", "SAC", "DPC", "TD-MPC"]
    envs = ["CylinderJet2D", "RBC2D"]
    df_base = df[df["env_name"].isin(envs) & df["algorithm"].isin(algorithms)]
    _make_runtime_table(
        df_base, algorithms,
        index_col="env_name", index_order=envs,
        metric=metric, decimals=decimals,
        caption=f"Mean $\\pm$ std training runtime ({caption_unit}) for CylinderJet2D and RBC2D.",
        label="tab:runtimes_env_comparison",
        out_path=out_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = read_training_runtimes()
    plot_runtimes_combined(df, per_step=False, exclude_algorithms_top=["TD-MPC"])
