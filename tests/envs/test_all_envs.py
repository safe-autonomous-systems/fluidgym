import torch
import fluidgym
import pytest
from fluidgym.envs.multi_agent_fluid_env import MultiAgentFluidEnv

all_env_ids = list(fluidgym.registry.registry.ids)

@pytest.mark.parametrize("env_id", all_env_ids)
def test_env(env_id: str):
    if env_id == "Airfoil3D-hard-v0":
        pytest.skip("Airfoil3D-hard-v0 is too computationally expensive for CI.")

    env = fluidgym.make(env_id)
    env.seed(42)
    
    obs, info = env.reset()

    assert obs.shape == env.observation_space_shape, \
        f"Observation shape {obs.shape} does not match " \
        f"expected {env.observation_space_shape}"
    assert isinstance(info, dict), "Info should be a dictionary"

    obs, reward, terminated, truncated, info = env.step(env.sample_action())
    assert obs.shape == env.observation_space_shape, \
        f"Observation shape {obs.shape} does not match " \
        f"expected {env.observation_space_shape}"
    assert isinstance(reward, torch.Tensor), "Reward should be a float"
    assert isinstance(terminated, bool), "Terminated should be a boolean"
    assert isinstance(truncated, bool), "Truncated should be a boolean"
    assert isinstance(info, dict), "Info should be a dictionary"
    for metric in env.metrics:
        assert metric in info, f"Metric '{metric}' missing from info"
        assert isinstance(info[metric], torch.Tensor), \
            f"Metric '{metric}' should be a tensor"
        
    # Additional checks for multi-agent environments
    if isinstance(env, MultiAgentFluidEnv):
        num_agents = env.n_agents

        obs, info = env.reset_marl()
        assert obs.shape[0] == num_agents, \
            f"Number of agents in observation {obs.shape[1]} does not match "\
            f"expected {num_agents}"
        
        obs, reward, terminated, truncated, info = env.step_marl(env.sample_action())
        assert obs.shape[0] == num_agents, \
            f"Number of agents in observation {obs.shape[1]} does not match "\
            f"expected {num_agents}"
        assert reward.shape[0] == num_agents, \
            f"Number of agents in reward {reward.shape[0]} does not match "\
            f"expected {num_agents}"
        
        assert obs[0].shape == env.local_observation_space_shape, \
            f"Local observation shape {obs[1:].shape} does not match " \
            f"expected {env.local_observation_space_shape}"
        
        assert "global_reward" in info, \
            "Global reward missing from info"
        assert isinstance(info["global_reward"], torch.Tensor), \
            "Global reward should be a tensor"  