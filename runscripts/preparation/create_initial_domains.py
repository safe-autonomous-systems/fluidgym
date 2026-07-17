import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging

import hydra
from omegaconf import DictConfig

import fluidgym

logger = logging.getLogger("fluidgym.preparation")
logging.basicConfig(level=logging.WARNING)


fluidgym.config.update("local_data_path", "./local_data")


@hydra.main(version_base="1.3", config_path="../configs", config_name="preparation")
def main(cfg: DictConfig):
    try:
        run_experiment(cfg)
    except Exception as e:
        logger.exception("Job crashed with an exception")
        logger.error(e)
        raise


def run_experiment(cfg: DictConfig):
    env = fluidgym.make(
        cfg.env_id,
        load_initial_domain=False,
        load_domain_statistics=False,
        **cfg.env_kwargs,
    )
    logger.info("Initialized environment.")

    logger.info("Creating initial domains...")
    if cfg.get("domain_idxs", False):
        domain_idxs = cfg.domain_idxs
    else:
        domain_idxs = None
    
    env.init(domain_idxs=domain_idxs)
    logger.info("Initial domains created.")


if __name__ == "__main__":
    main()
