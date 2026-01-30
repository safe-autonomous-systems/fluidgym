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
