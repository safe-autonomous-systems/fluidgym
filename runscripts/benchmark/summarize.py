import json
from pathlib import Path

import pandas as pd

BENCHMARK_DIR = Path("./output/benchmark")
TABLES_PATH = Path("./paper/tables")
TABLES_PATH.mkdir(parents=True, exist_ok=True)


ENV_ORDER = [
    "CylinderRot2D",
    "CylinderJet2D",
    "CylinderJet3D",
    "RBC2D",
    "RBC2D-wide",
    "RBC3D",
    "RBC3D-wide",
    "Airfoil2D",
    "Airfoil3D",
    "TCFSmall3D-both",
    "TCFSmall3D-bottom",
    "TCFLarge3D-both",
    "TCFLarge3D-bottom",
]
DIFFICULTY_ORDER = ["easy", "medium", "hard"]


def read_results() -> pd.DataFrame:
    results = []
    for env_dir in BENCHMARK_DIR.iterdir():
        if env_dir.is_dir():
            result_file = env_dir / "result.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                    if "avg_time_seconds" not in data:
                        continue

                    data["env_name"] = "-".join(env_dir.name.split("-")[:-2])
                    difficulty = env_dir.name.split("-")[-2]
                    data["difficulty"] = difficulty
                    results.append(data)

    results_df = pd.DataFrame(results)
    results_df["sec_per_step"] = results_df["avg_time_seconds"] / results_df["steps"]

    return results_df


def print_benchmark_table():
    results_df = read_results()

    mean = (  # type: ignore
        results_df.groupby("env_name", as_index=False)["sec_per_step"]
        .mean()
        .round(2)
        .sort_values("env_name", key=lambda s: s.map(lambda x: ENV_ORDER.index(x)))
    )

    print("Average seconds per step over all difficulties:")
    print(mean)


def print_gpu_hours():
    results_df = read_results()

    # Only consider trained envs
    results_df = results_df[results_df["env_name"].isin(ENV_ORDER)]

    results_df = results_df.drop(columns=["avg_time_seconds", "steps"])

    # Default is 50,000 steps
    results_df["#steps"] = 50_000
    results_df["#seeds"] = 5

    results_df["#algorithms"] = 2  # PPO and SAC
    results_df.loc[
        results_df["env_name"].str.contains("CylinderJet3D"), "#algorithms"
    ] = 4
    results_df.loc[results_df["env_name"].str.contains("RBC2D"), "#algorithms"] = 4
    results_df.loc[results_df["env_name"].str.contains("RBC3D"), "#algorithms"] = 2
    results_df.loc[results_df["env_name"].str.contains("Airfoil3D"), "#algorithms"] = 4
    results_df.loc[results_df["env_name"].str.contains("TCF"), "#algorithms"] = 2

    # Adapt total steps for specific envs
    results_df.loc[results_df["env_name"].str.contains("Airfoil"), "#steps"] = 20_000
    results_df.loc[results_df["env_name"].str.contains("TCF"), "#steps"] = 100_000

    # Adapt num seeds for specific envs
    results_df.loc[results_df["env_name"].str.contains("Airfoil3D"), "#seeds"] = 3
    results_df.loc[results_df["env_name"].str.contains("CylinderJet3D"), "#seeds"] = 3

    # Remove env variations we don't run experiments for
    results_df["actual_steps"] = results_df["#steps"].copy()
    results_df.loc[results_df["env_name"].str.contains("bottom"), "actual_steps"] = 0
    results_df.loc[results_df["env_name"].str.contains("wide"), "actual_steps"] = 0

    # We do not run medium/hard Airfoil3D experiments
    results_df.loc[
        results_df["env_name"].str.contains("Airfoil3D") & \
        ~results_df["difficulty"].str.contains("easy"),
        "actual_steps"
    ] = 0

    results_df["gpu_hours"] = (
        results_df["sec_per_step"]
        * results_df["actual_steps"]
        * results_df["#seeds"]
        * results_df["#algorithms"]
        / 3600
    )

    # Sort by 1) ENV_ORDER and 2) difficulty
    results_df["env_order"] = results_df["env_name"].apply(lambda x: ENV_ORDER.index(x))
    results_df["difficulty"] = pd.Categorical(
        results_df["difficulty"], categories=DIFFICULTY_ORDER, ordered=True
    )
    results_df = results_df.sort_values(by=["env_order", "difficulty"])

    results_df = results_df.drop(columns=["env_order"])

    table = results_df[
        [
            "env_name",
            "difficulty",
            "#steps",
            "#seeds",
            "sec_per_step",
            "#algorithms",
            "gpu_hours",
        ]
    ]

    # 2. Round GPU hours
    table.loc[:, "gpu_hours"] = table["gpu_hours"].round(2)

    # 3. Add total row
    total_gpu_hours = table["gpu_hours"].sum().round(2)
    total_row = pd.DataFrame(
        [["\\textbf{Total}", "", "", "", "", "", total_gpu_hours]],
        columns=table.columns,
    )

    table = pd.concat([table, total_row], ignore_index=True)

    table.to_latex(
        buf=TABLES_PATH / f"experiments.tex",
        index=False,
        escape=False,
        float_format="%.3f",
        na_rep="-",
    )


if __name__ == "__main__":
    print_benchmark_table()
    print_gpu_hours()
