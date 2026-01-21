import logging
import os
import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.base_class import BaseAlgorithm

import fluidgym
from fluidgym.envs.multi_agent_fluid_env import MultiAgentFluidEnv
from fluidgym.integration.gymnasium import GymFluidEnv
from fluidgym.integration.sb3 import MultiAgentVecEnv, test_model

logger = logging.getLogger("fluidgym.testing")


OmegaConf.register_new_resolver("eval", lambda x: eval(x))


local_data_path = os.environ.get("FLUIDGYM_LOCAL_DATA_PATH", "./local_data")
fluidgym.config.update("local_data_path", local_data_path)


@hydra.main(version_base="1.3", config_path="../configs", config_name="test_sb3")
def main(cfg: DictConfig):
    try:
        run_experiment(cfg)
    except Exception as e:
        logger.exception("Job crashed with an exception")
        logger.error(e)
        raise


def run_experiment(cfg: DictConfig):
    logger.info("Training script started.")

    logger.info("Initializing test environment...")
    fluid_test_env = fluidgym.make(cfg.test_env_id, **cfg.test_env_kwargs)

    if cfg.test_rl_mode == "marl":
        assert isinstance(fluid_test_env, MultiAgentFluidEnv)
        test_env = MultiAgentVecEnv(fluid_test_env, auto_reset=False)
    elif cfg.test_rl_mode == "sarl":
        test_env = GymFluidEnv(fluid_test_env)
    else:
        raise ValueError(f"Unknown test rl_mode: {cfg.test_rl_mode}")
    test_env.seed(cfg.seed + 84)
    test_env.test()
    logger.info("Done.")

    logger.info("Initializing model...")
    with warnings.catch_warnings():
        # Ignore the warning 'len(rollout_buffer) % batch_size warning != 0'
        # when transfering a MARL policy to an env with a different number of agents
        warnings.simplefilter("ignore", category=UserWarning)
        model: BaseAlgorithm = hydra.utils.instantiate(cfg.algorithm.obj, env=test_env)
    model = model.load("ckpt_latest", env=test_env, device=cfg.rl_device)
    logger.info("Done.")

    if cfg.env_id != cfg.test_env_id:
        output_path = Path(f"./transfer/{cfg.test_env_id}/")
    else:
        output_path = Path("./test/")
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting evaluation on test environment...")
    test_model(
        model=model,
        test_env=test_env,
        n_episodes=cfg.n_test_episodes,
        output_path=output_path,
        save_frames=cfg.save_frames,
        render_3d=cfg.render_3d,
        deterministic=True,
    )
    logger.info("Evaluation on test environment finished.")


if __name__ == "__main__":
    main()
