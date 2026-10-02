import torch
import pytest
from fluidgym.envs.cylinder.jet_cylinder_env_2d import (
    CylinderJetEnv2D,
    CYLINDER_JET_2D_DEFAULT_CONFIG,
)


def test_sampling_before_reset():
    env = CylinderJetEnv2D(**CYLINDER_JET_2D_DEFAULT_CONFIG)

    with pytest.raises(RuntimeError) as excinfo:
        action = env.sample_action()
        env.step(action)

    assert "Environment must be seeded before sampling actions" in str(excinfo.value)


def test_step_before_reset():
    env = CylinderJetEnv2D(**CYLINDER_JET_2D_DEFAULT_CONFIG)

    with pytest.raises(RuntimeError) as excinfo:
        action = torch.zeros(env.action_space.shape, device=env.cuda_device)
        env.step(action)

    assert "Environment must be reset before stepping" in str(excinfo.value)


def test_action_bound_check():
    env = CylinderJetEnv2D(**CYLINDER_JET_2D_DEFAULT_CONFIG)
    low = torch.as_tensor(env.action_space.low, device=env.cuda_device)
    high = torch.as_tensor(env.action_space.high, device=env.cuda_device)

    # Bounds are inclusive
    assert env._is_action_valid(low)
    assert env._is_action_valid(high)
    assert env._is_action_valid((low + high) / 2)

    assert not env._is_action_valid(low - 1.0)
    assert not env._is_action_valid(high + 1.0)
    assert not env._is_action_valid(torch.full_like(low, float("nan")))


def test_action_bound_check_vectorized():
    config = {**CYLINDER_JET_2D_DEFAULT_CONFIG, "n_envs": 2}
    env = CylinderJetEnv2D(**config)
    high = torch.as_tensor(env.action_space.high, device=env.cuda_device)

    action = torch.stack([high, high])
    assert env._is_action_valid(action)

    # A single out-of-bounds environment invalidates the whole batch
    action[1] += 1.0
    assert not env._is_action_valid(action)


def test_step_with_out_of_bounds_action():
    env = CylinderJetEnv2D(**CYLINDER_JET_2D_DEFAULT_CONFIG)
    env.reset(seed=42)

    high = torch.as_tensor(env.action_space.high, device=env.cuda_device)

    with pytest.raises(ValueError) as excinfo:
        env.step(high + 1.0)

    assert "outside of the action bounds" in str(excinfo.value)
