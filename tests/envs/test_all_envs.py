import torch
import fluidgym
import pytest
from fluidgym.envs import FluidEnv
from gymnasium import spaces

all_env_ids = list(fluidgym.registry.registry.ids)
all_env_ids.remove("Airfoil3D-hard-v0")


def _check_obs(env: FluidEnv, obs: dict[str, torch.Tensor], marl: bool) -> None:
    obs_space = env.observation_space

    assert isinstance(obs_space, spaces.Dict), (
        "Observation space should be of type spaces.Dict"
    )

    for key, space in obs_space.spaces.items():
        assert key in obs, f"Observation missing key: {key}"
        if marl:
            _obs = obs[key][0]
        else:
            _obs = obs[key]

        assert isinstance(_obs, torch.Tensor), (
            f"Observation for key '{key}' should be a torch.Tensor"
        )
        assert _obs.shape == space.shape, (
            f"Observation shape for key '{key}' is {_obs.shape}, expected {space.shape}"
        )


def _check_action(env: FluidEnv, action: torch.Tensor, marl: bool) -> None:
    action_space = env.action_space

    if marl:
        _action = action[0]
    else:
        _action = action

    assert isinstance(action_space, spaces.Box), (
        "Action space should be of type spaces.Dict"
    )

    assert isinstance(_action, torch.Tensor), f"Action should be a torch.Tensor"
    assert _action.shape == action_space.shape, (
        f"Action shape is {_action.shape}, expected {action_space.shape}"
    )


@pytest.mark.parametrize("env_id", all_env_ids)
def test_env_sarl(env_id: str) -> None:
    env = fluidgym.make(env_id, use_marl=False)
    env.seed(42)

    obs, info = env.reset()

    action = env.sample_action()
    _check_action(env, action, marl=False)

    obs, reward, terminated, truncated, info = env.step(env.sample_action())

    _check_obs(env, obs, marl=False)
    assert isinstance(reward, torch.Tensor), "Reward should be a float"
    assert isinstance(terminated, bool), "Terminated should be a boolean"
    assert isinstance(truncated, bool), "Truncated should be a boolean"
    assert isinstance(info, dict), "Info should be a dictionary"
    for metric in env.metrics:
        assert metric in info, f"Metric '{metric}' missing from info"
        assert isinstance(info[metric], torch.Tensor), (
            f"Metric '{metric}' should be a tensor"
        )


@pytest.mark.parametrize("env_id", all_env_ids)
def test_env_marl(env_id: str) -> None:
    try:
        env = fluidgym.make(env_id, use_marl=True)
    except ValueError:
        return  # Env does not support MARL

    env.seed(42)

    obs, info = env.reset()

    action = env.sample_action()
    _check_action(env, action, marl=True)

    obs, reward, terminated, truncated, info = env.step(env.sample_action())

    _check_obs(env, obs, marl=True)
    assert reward.shape[0] == env.n_agents, (
        f"Number of agents in reward {reward.shape[0]} does not match "
        f"expected {env.n_agents}"
    )
    assert "global_reward" in info, "Global reward missing from info"
    assert isinstance(info["global_reward"], torch.Tensor), (
        "Global reward should be a tensor"
    )


@pytest.mark.parametrize("env_id", ["CylinderJet3D-easy-v0", "Airfoil3D-easy-v0"])
def test_env_local_2d_obs(env_id: str) -> None:
    env_2d = fluidgym.make(env_id.replace("3D", "2D"))
    env_3d = fluidgym.make(env_id, use_marl=True, local_2d_obs=True)

    env_2d.seed(42)
    env_3d.seed(42)

    obs_2d, _ = env_2d.reset()
    obs_3d, _ = env_3d.reset()

    for k in obs_2d.keys():
        assert obs_2d[k].shape == obs_3d[k].shape[1:], (
            f"Observation shape for key '{k}' does not match between 2D and 3D envs: "
            f"{obs_2d[k].shape} vs {obs_3d[k].shape[1:]}"
        )

    obs_2d, _, _, _, _ = env_2d.step(env_2d.sample_action())
    obs_3d, _, _, _, _ = env_3d.step(env_3d.sample_action())

    for k in obs_2d.keys():
        assert obs_2d[k].shape == obs_3d[k].shape[1:], (
            f"Observation shape for key '{k}' does not match between 2D and 3D envs: "
            f"{obs_2d[k].shape} vs {obs_3d[k].shape[1:]}"
        )
