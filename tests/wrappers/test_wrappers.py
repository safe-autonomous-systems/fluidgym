import fluidgym
import torch
import pytest
import numpy as np

from fluidgym.wrappers import (
    FlattenObservation,
    ObsExtraction,
    ActionNoise,
    SensorNoise,
)
from gymnasium import spaces


ids = fluidgym.registry.registry.ids
env_names = ["-".join(id.split("-")[:-2]) for id in ids]
env_names = sorted(list(set(env_names)))


@pytest.mark.parametrize("env_name", env_names)
def test_flatten_observation_wrapper(env_name: str) -> None:
    """Test the FlattenObservation wrapper for all environments that support it."""
    env_id = f"{env_name}-easy-v0"
    env = fluidgym.make(env_id, use_marl=False)

    original_obs_space = env.observation_space
    assert isinstance(original_obs_space, spaces.Dict)

    target_size = 0
    for key in ["temperature", "velocity"]:
        if key in original_obs_space.spaces:
            space = original_obs_space.spaces[key]
            assert isinstance(space, spaces.Box)
            target_size += int(np.prod(space.shape))
    target_shape = (target_size,)

    env = FlattenObservation(env)
    new_obs_space = env.observation_space

    assert isinstance(new_obs_space, spaces.Box), (
        "Observation space should be of type spaces.Box"
    )
    assert new_obs_space.shape == target_shape, (
        f"Flattened observation shape {new_obs_space.shape} does not match target "
        f"shape {target_shape}."
    )

    obs, _ = env.reset(seed=42)

    assert isinstance(obs, torch.Tensor), "Observation should be a torch.Tensor"
    assert obs.shape == target_shape, (
        f"Observation shape {obs.shape} does not match target shape {target_shape}."
    )

    action = env.sample_action()
    obs, _, _, _, _ = env.step(action)

    assert isinstance(obs, torch.Tensor), "Observation should be a torch.Tensor"
    assert obs.shape == target_shape, (
        f"Observation shape {obs.shape} does not match target shape {target_shape}."
    )


@pytest.mark.parametrize("env_name", env_names)
def test_obs_extraction_wrapper(env_name: str) -> None:
    env_id = f"{env_name}-easy-v0"
    env = fluidgym.make(env_id, use_marl=False)

    original_obs_space = env.observation_space
    assert isinstance(original_obs_space, spaces.Dict)
    original_keys = list(original_obs_space.spaces.keys())

    np.random.seed(42)
    keys = ["pressure"]

    env = ObsExtraction(env, keys=keys)
    new_obs_space = env.observation_space

    assert isinstance(new_obs_space, spaces.Dict), (
        "Observation space should be of type spaces.Dict"
    )
    assert list(new_obs_space.spaces.keys()) == keys, (
        f"Observation space keys {list(new_obs_space.spaces.keys())} do not match "
        f"target keys {keys}."
    )

    assert "pressure" in list(new_obs_space.spaces.keys()), (
        "Observation space should contain 'pressure' key"
    )

    obs, _ = env.reset(seed=42)
    assert set(obs.keys()) == set(keys), (
        f"Observation keys {list(obs.keys())} do not match target keys {keys}."
    )
    action = env.sample_action()
    obs, _, _, _, _ = env.step(action)
    assert set(obs.keys()) == set(keys), (
        f"Observation keys {list(obs.keys())} do not match target keys {keys}."
    )


@pytest.mark.parametrize("env_name", env_names)
def test_action_sensor_noise_wrappers(env_name: str) -> None:
    env_id = f"{env_name}-easy-v0"
    env = fluidgym.make(env_id, use_marl=False)
    original_obs_space = env.observation_space
    assert isinstance(original_obs_space, spaces.Dict)

    env = ActionNoise(env, sigma=0.1, seed=42)
    env = SensorNoise(env, sigma=0.1, seed=42)

    obs, _ = env.reset(seed=42)
    assert isinstance(obs, dict), "Observation should be a dict"

    for k, v in obs.items():
        assert isinstance(v, torch.Tensor), (
            f"Observation for key '{k}' should be a torch.Tensor"
        )
        assert v.shape == original_obs_space.spaces[k].shape, (
            f"Observation shape for key '{k}' is {v.shape}, expected "
            f"{original_obs_space.spaces[k].shape}"
        )

    action = env.sample_action()
    obs, _, _, _, _ = env.step(action)
    assert isinstance(obs, dict), "Observation should be a dict"

    for k, v in obs.items():
        assert isinstance(v, torch.Tensor), (
            f"Observation for key '{k}' should be a torch.Tensor"
        )
        assert v.shape == original_obs_space.spaces[k].shape, (
            f"Observation shape for key '{k}' is {v.shape}, expected "
            f"{original_obs_space.spaces[k].shape}"
        )
