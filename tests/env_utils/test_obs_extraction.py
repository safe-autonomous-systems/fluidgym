import pytest
import torch
from fluidgym.envs.util.obs_extraction import (
    extract_moving_window_2d,
    extract_moving_window_2d_x_z,
    extract_moving_window_3d
)

torch.manual_seed(0)

@pytest.mark.parametrize(
    "n_agents, agent_width, n_agents_per_window",
    [
        (8, 12, 1),
        (8, 12, 3),
        (8, 12, 5),
    ],
)
def test_moving_window_2d(n_agents: int, agent_width: int, n_agents_per_window: int):
    y = 10
    field_2d = torch.rand((y, n_agents * agent_width))

    half_window_width = n_agents_per_window // 2

    windows = extract_moving_window_2d(
        field_2d,
        n_agents=n_agents,
        agent_width=agent_width,
        n_agents_per_window=n_agents_per_window
    )
    assert windows.shape == (n_agents, y, n_agents_per_window * agent_width), \
        f"Unexpected window shape: {windows.shape}"

    for agent_idx in range(n_agents):
        start = (agent_idx - half_window_width) * agent_width
        end = (agent_idx + half_window_width + 1) * agent_width

        # Handle wrapping around
        if start < 0:
            expected_window = torch.cat((
                field_2d[:, start % (n_agents * agent_width):],
                field_2d[:, :end]
            ), dim=1)
        elif end > n_agents * agent_width:
            expected_window = torch.cat((
                field_2d[:, start:],
                field_2d[:, :end % (n_agents * agent_width)]
            ), dim=1)
        else:
            expected_window = field_2d[:, start:end]

        assert torch.allclose(windows[agent_idx], expected_window), \
            f"Window for agent {agent_idx} does not match expected."

@pytest.mark.parametrize(
    "n_agents_x, n_agents_z, agent_width, n_agents_per_window_x, n_agents_per_window_z, pad_x, pad_z",
    [
        (10, 20, 2, 1, 1, 0, 0),
        (20, 40, 2, 5, 3, 4, 1),
        (10, 20, 4, 5, 5, 4, 4),
    ],
)
def test_moving_window_2d_x_z(
    n_agents_x: int, 
    n_agents_z: int,
    agent_width: int,
    n_agents_per_window_x: int,
    n_agents_per_window_z: int,
    pad_x: int,
    pad_z: int,
):
    field = torch.rand((n_agents_z * agent_width, n_agents_x * agent_width))
    field[
        :n_agents_per_window_z * agent_width,
        :n_agents_per_window_x * agent_width
    ] = 1.0

    result = extract_moving_window_2d_x_z(
        field=field,
        n_agents_x=n_agents_x,
        n_agents_z=n_agents_z,
        agent_width=agent_width,
        n_agents_per_window_x=n_agents_per_window_x,
        n_agents_per_window_z=n_agents_per_window_z,
        pad_x=pad_x,
        pad_z=pad_z,
    )
    assert result.shape == (
        n_agents_z * n_agents_x,
        n_agents_per_window_z,
        n_agents_per_window_x
    ), f"Unexpected result shape: {result.shape}"


    expected_field = field[
        :n_agents_per_window_z * agent_width,
        :n_agents_per_window_x * agent_width
    ]

    # Average over agent_width to get local observation
    expected_obs = expected_field.view(
        n_agents_per_window_z, agent_width,
        n_agents_per_window_x, agent_width
    ).mean(dim=(1, 3))
    
    agent_idx = pad_x * n_agents_z + pad_z
    assert torch.allclose(result[agent_idx], expected_obs), \
        "Top-left corner observation does not match expected."


@pytest.mark.parametrize(
    "n_agents, agent_width, n_agents_per_window",
    [
        (10, 1, 1),
        (10, 2, 3),
        (20, 4, 1),
        (20, 4, 3),
    ],
)
def test_moving_window_3d(
    n_agents: int,
    agent_width: int,
    n_agents_per_window: int
):
    y = 10

    field = torch.rand((n_agents * agent_width, y, n_agents * agent_width))
    field[
        :n_agents_per_window * agent_width,
        :n_agents_per_window * agent_width
    ] = 1.0

    result = extract_moving_window_3d(
        field=field,
        n_agents=n_agents,
        agent_width=agent_width,
        n_agents_per_window=n_agents_per_window
    )
    assert result.shape == (
        n_agents * n_agents,
        agent_width * n_agents_per_window,
        y,
        agent_width * n_agents_per_window
    ), f"Unexpected result shape: {result.shape}"

    expected_obs = field[
        :n_agents_per_window * agent_width,
        :,
        :n_agents_per_window * agent_width
    ]

    agent_idx = (n_agents_per_window // 2) * n_agents + (n_agents_per_window // 2)
    assert torch.allclose(result[agent_idx], expected_obs), \
        "Left corner observation does not match expected."
