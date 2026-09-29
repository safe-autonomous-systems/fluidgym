import logging
from datetime import datetime as dt

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm

import fluidgym
from fluidgym.integration.gymnasium import GymFluidEnv
from fluidgym.integration.sb3 import EvalCallback, load_buffer
from fluidgym.wrappers import FlattenObservation

logger = logging.getLogger("fluidgym.training")


OmegaConf.register_new_resolver("eval", lambda x: eval(x))


@hydra.main(version_base="1.3", config_path="configs", config_name="train_sb3")
def main(cfg: DictConfig):
    try:
        run_experiment(cfg)
    except Exception as e:
        logger.exception("Job crashed with an exception")
        logger.error(e)
        raise


def run_experiment(cfg: DictConfig):
    logger.info("Training script started.")

    logger.info("Initializing environments...")
    fluid_env = fluidgym.make(cfg.env_id, **cfg.env_kwargs, randomize_initial_state=False, load_domain_statistics=False)
    fluid_eval_env = fluidgym.make(cfg.env_id, **cfg.eval_env_kwargs, randomize_initial_state=False, load_domain_statistics=False)

    fluid_env = FlattenObservation(fluid_env)
    fluid_eval_env = FlattenObservation(fluid_eval_env)

    env = GymFluidEnv(fluid_env)
    eval_env = GymFluidEnv(fluid_eval_env)

    env.seed(cfg.seed)
    eval_env.seed(cfg.seed + 42)
    eval_env.val()
    logger.info("Done.")

    now = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{cfg.algorithm.name}_{cfg.env_id}_{cfg.seed}_{now}"
    run_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    assert isinstance(run_config, dict), "Run config must be a dictionary."

    eval_callback: EvalCallback = hydra.utils.instantiate(
        cfg.eval_callback,
        env=env,
        eval_env=eval_env,
    )

    logger.info("Initializing model...")
    model: BaseAlgorithm = hydra.utils.instantiate(cfg.algorithm.obj, env=env)
    if cfg.continue_training:
        logger.info("Continuing training from latest checkpoint...")
        model = model.load("ckpt_latest", env=env, device=cfg.rl_device)
        if isinstance(model, SAC):
            load_buffer(model, "ckpt_latest")
    logger.info("Done.")

    if cfg.wandb.enable:
        logger.info("Initializing Weights & Biases...")
        wandb.init(
            id=run_id,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=run_config,  # type: ignore
            group=fluid_env.id,
        )
        logger.info("Done.")

    logger.info("Starting training...")

    remaining_timesteps = cfg.total_timesteps - model.num_timesteps
    model.learn(
        total_timesteps=remaining_timesteps,
        progress_bar=True,
        callback=eval_callback,
        reset_num_timesteps=False,
    )
    if cfg.wandb.enable:
        wandb.finish()
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
