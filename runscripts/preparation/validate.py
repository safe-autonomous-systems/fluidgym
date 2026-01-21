import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

import fluidgym
from fluidgym.envs.rbc.rbc_env_base import RBCEnvBase
from fluidgym.util.data_utils import load_statistics

logger = logging.getLogger("fluidgym.preparation")
logging.basicConfig(level=logging.WARNING)


fluidgym.config.update("local_data_path", "./local_data")


def get_rbc_2d_filtered_nusselt(
    stats: dict[str, dict[str, float]], all_metrics: pd.DataFrame
) -> dict[str, float]:
    # First, we compute the mean nusselt per domain idx
    mean_df = all_metrics.groupby("domain_idx").mean().reset_index()

    # Now, we compute the overall median nusselt number across domains
    nusselt_p25 = stats["nusselt"]["p25"]

    # We use the 25th percentile to filter out domains with a lower nusselt number
    filtered_df = mean_df[(mean_df["nusselt"] < nusselt_p25)]

    return {
        "mean": filtered_df["nusselt"].mean(),
        "min": filtered_df["nusselt"].min(),
        "max": filtered_df["nusselt"].max(),
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="preparation")
def main(cfg: DictConfig):
    try:
        run_experiment(cfg)
    except Exception as e:
        logger.exception("Job crashed with an exception")
        logger.error(e)
        raise


def run_experiment(cfg: DictConfig):
    base_dir = fluidgym.config.initial_domains_path

    all_metrics = []

    for resolution in cfg.resolutions:
        env = fluidgym.make(
            cfg.env_id,
            resolution=resolution,
            load_initial_domain=False,
            load_domain_statistics=False,
            randomize_initial_state=False,
        )
        env.reset(seed=42)
        logger.info(f"Initialized environment with resolution {resolution}.")

        domain_dir = base_dir / env.initial_domain_id

        print(domain_dir)

        try:
            statistics = load_statistics(domain_dir)
        except FileNotFoundError:
            print(
                f"Statistics file not found for domain {env.initial_domain_id} at "
                f"{domain_dir}. Skipping."
            )
            continue

        metrics = {**{key: statistics[key]["mean"] for key in env.metrics}}

        # For RBC, we use the median nusselt number across domains
        # to filter out domains with low heat transfer
        if isinstance(env, RBCEnvBase):
            metrics = {"nusselt": statistics["nusselt"]["p50"]}

        metrics["resolution"] = resolution
        metrics["size"] = env._domain.getTotalSize()

        all_metrics += [metrics]

    all_metrics_df = pd.DataFrame(all_metrics)

    print("Collected domain metrics:")
    print(all_metrics_df)


if __name__ == "__main__":
    main()
