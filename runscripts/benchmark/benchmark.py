import os
import json
import logging
import time

import hydra
import numpy as np
from omegaconf import DictConfig

import fluidgym

logger = logging.getLogger("fluidgym.benchmark")
logging.basicConfig(level=logging.WARNING)


local_data_path = os.environ.get("FLUIDGYM_LOCAL_DATA_PATH", "./local_data")
fluidgym.config.update("local_data_path", local_data_path)


@hydra.main(version_base="1.3", config_path="../configs", config_name="benchmark")
def main(cfg: DictConfig):
    try:
        benchmark(cfg)
    except Exception as e:
        logger.exception("Job crashed with an exception")
        logger.error(e)
        raise


def benchmark(cfg: DictConfig):
    steps = cfg.steps

    if "Airfoil3D" in cfg.env_id and ("medium" in cfg.env_id or "hard" in cfg.env_id):
        steps = steps // 10  # Reduce steps for 3D medium/hard envs due to long runtimes
        env = fluidgym.make(
            cfg.env_id,
            **cfg.env_kwargs,
            load_initial_domain=False,
            load_domain_statistics=False,
        )
    else:
        env = fluidgym.make(cfg.env_id, **cfg.env_kwargs)
    logger.info("Initialized environment.")

    runtimes = []
    for i in range(cfg.n_iterations):
        env.reset(seed=42 * i)

        logger.info(f"Starting benchmark iteration {i + 1} for {steps} steps...")
        start_time = time.perf_counter()
        for _ in range(steps):
            action = env.sample_action()
            env.step(action)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        runtimes += [total_time]

    mean_time = np.mean(runtimes)

    with open("result.json", "w") as f:
        json.dump(
            {
                "iterations": cfg.n_iterations,
                "steps": steps,
                "avg_time_seconds": mean_time,
            },
            f,
            indent=4,
        )
    logger.info(f"Completed {cfg.steps} steps in {mean_time:.2f} seconds on average.")


if __name__ == "__main__":
    main()
