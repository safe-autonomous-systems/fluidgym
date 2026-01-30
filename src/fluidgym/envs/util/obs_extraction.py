"""Utility functions for extracting observation windows for agents."""

import torch
import torch.nn.functional as F

from fluidgym.envs import FluidEnv
from fluidgym.simulation.pict.util.output import _resample_block_data


def extract_global_2d_obs(
    env: FluidEnv, sensor_locations: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Extracts global 2D observations for agents arranged in a 2D domain.

    Parameters
    ----------
    env: FluidEnv
        The fluid environment.

    sensor_locations: torch.Tensor
        Tensor of shape [2, n_agents * n_sensors_per_agent] containing the
        (x, y) coordinates of the sensors.

    Returns
    -------
        Global observations with shape [n_agents * n_sensors_per_agent, ...].
    """
    u_list = [block.velocity for block in env._domain.getBlocks()]
    p_list = [block.pressure for block in env._domain.getBlocks()]

    u = _resample_block_data(
        u_list,
        env._sim.output_resampling_coords,
        env._sim.output_resampling_shape,
        env._ndims,
        fill_max_steps=env._sim.output_resampling_fill_max_steps,
    )
    u = u.squeeze()
    u = u.permute(1, 2, 0)
    u = u[sensor_locations[1], sensor_locations[0], :]

    p = _resample_block_data(
        p_list,
        env._sim.output_resampling_coords,
        env._sim.output_resampling_shape,
        env._ndims,
        fill_max_steps=env._sim.output_resampling_fill_max_steps,
    )
    p = p.squeeze()
    p = p[sensor_locations[1], sensor_locations[0]]

    return {
        "velocity": u,
        "pressure": p,
    }


def extract_global_3d_obs(
    env: FluidEnv,
    sensor_locations: torch.Tensor,
    n_agents: int,
    n_sensors_per_agent: int,
    n_sensors_z: int,
    local_2d_obs: bool = False,
) -> dict[str, torch.Tensor]:
    """Extracts global 3D observations for agents arranged in a 3D domain.

    Parameters
    ----------
    env: FluidEnv
        The fluid environment.

    sensor_locations: torch.Tensor
        Tensor of shape [3, n_agents * n_sensors_per_agent] containing the
        (x, y, z) coordinates of the sensors.

    n_agents: int
        Number of agents.

    n_sensors_per_agent: int
        Number of sensors per agent.

    n_sensors_z: int
        Number of sensors along the z-axis.

    local_2d_obs: bool
        Whether the local observations are 2D (True) or 3D (False).

    Returns
    -------
        Global observations with shape [n_agents * n_sensors_per_agent, ...].
    """
    u_list = [block.velocity for block in env._domain.getBlocks()]
    p_list = [block.pressure for block in env._domain.getBlocks()]

    u: torch.Tensor = _resample_block_data(
        u_list,
        env._sim.output_resampling_coords,
        env._sim.output_resampling_shape,
        env._ndims,
        fill_max_steps=env._sim.output_resampling_fill_max_steps,
    )
    u = u.squeeze()
    u = u.permute(1, 2, 3, 0)

    p: torch.Tensor = _resample_block_data(
        p_list,
        env._sim.output_resampling_coords,
        env._sim.output_resampling_shape,
        env._ndims,
        fill_max_steps=env._sim.output_resampling_fill_max_steps,
    )
    p = p.squeeze()

    sensor_locations = sensor_locations.flatten(start_dim=1)

    if local_2d_obs:
        u = u[:, :, :, :2]
        velocity_dims = 2
    else:
        velocity_dims = 3

    u = u[
        sensor_locations[2],
        sensor_locations[1],
        sensor_locations[0],
        :,
    ]

    u = u.view(n_sensors_z, velocity_dims, -1)
    u = u.view(n_agents, n_sensors_per_agent, velocity_dims, -1)

    if local_2d_obs:
        u = u.permute(0, 1, 3, 2)

    p = p[
        sensor_locations[2],
        sensor_locations[1],
        sensor_locations[0],
    ]
    p = p.view(n_sensors_z, -1)
    p = p.view(n_agents, n_sensors_per_agent, -1)

    return {
        "velocity": u,
        "pressure": p,
    }


def transform_global_to_local_obs_3d(
    global_obs: dict[str, torch.Tensor],
    local_obs_window: int,
    n_agents: int,
    local_2d_obs: bool = False,
) -> dict[str, torch.Tensor]:
    """Transforms global observations into local observations for agents arranged
    in a 3D domain.

    Parameters
    ----------
    global_obs: dict[str, torch.Tensor]
        Global observations with shape [n_agents * n_sensors_per_agent, ...].

    local_obs_window: int
        Size of the local observation window (in number of agents).

    n_agents: int
        Number of agents.

    local_2d_obs: bool
        Whether the local observations are 2D (True) or 3D (False).

    Returns
    -------
        Local observations with shape [n_agents, local_obs_window, ...].

    """
    offset = local_obs_window // 2

    local_obs = {}
    for k, v in global_obs.items():
        # First, shift the global obs to start with the first agents sensor
        # window at zero
        shifted_obs = torch.roll(v, shifts=offset, dims=0)

        local_obs_list = []
        for _ in range(n_agents):
            window = shifted_obs[:local_obs_window]

            if local_2d_obs:
                window = window.squeeze()

            local_obs_list += [window]

            shifted_obs = torch.roll(shifted_obs, shifts=-1, dims=0)

        local_obs[k] = torch.stack(local_obs_list, dim=0)

    return local_obs


def extract_moving_window_2d(
    field: torch.Tensor, n_agents: int, agent_width: int, n_agents_per_window: int
) -> torch.Tensor:
    """Extracts local 2D observation windows for agents arranged in a single row.

    Parameters
    ----------
    field: torch.Tensor
        [Y, X] tensor.

    n_agents: int
        Number of agents along X.

    agent_width: int
        Spatial width per agent (in X).

    n_agents_per_window: int
        Number of neighboring agents per window along X.

    Returns
    -------
        Tensor of shape [n_agents, Y, window_size_x].
    """
    if field.ndim != 2:
        raise ValueError("field must be a 2D tensor with shape (Y, X)")

    Y, X = field.shape
    assert X == n_agents * agent_width, "X must equal n_agents * agent_width"

    # Reshape into per-agent blocks: [Y, n_agents, agent_width]
    field_agents = field.view(Y, n_agents, agent_width)

    # Pad along the agent dimension (circularly)
    pad = n_agents_per_window // 2
    field_padded = F.pad(field_agents, (0, 0, pad, pad), mode="circular")

    window_list = []
    for i in range(n_agents):
        start = i
        end = i + n_agents_per_window
        window = field_padded[:, start:end, :]  # [Y, window_agents, agent_width]

        # Flatten the local agent window along the X dimension
        local_obs = window.reshape(Y, n_agents_per_window * agent_width)
        window_list.append(local_obs)

    return torch.stack(window_list, dim=0)  # [n_agents, Y, window_size_x]


def extract_moving_window_2d_x_z(
    field: torch.Tensor,
    n_agents_x: int,
    n_agents_z: int,
    agent_width: int,
    n_agents_per_window_x: int,
    n_agents_per_window_z: int,
    pad_x: int,
    pad_z: int,
) -> torch.Tensor:
    """Extracts local 2D observation windows for agents arranged in both X and Z
    directions.

    Parameters
    ----------
    field: torch.Tensor
        [Z, X] tensor.

    n_agents_x: int
        Number of agents along X.

    n_agents_z: int
        Number of agents along Z.

    agent_width: int
        Spatial width per agent (in X and Z).

    n_agents_per_window_x: int
        Number of neighboring agents per window along X.

    n_agents_per_window_z: int
        Number of neighboring agents per window along Z.

    pad_x: int
        Padding along X axis.

    pad_z: int
        Padding along Z axis.

    Returns
    -------
        Tensor of shape [n_agents_z * n_agents_x, Z_local, X_local].
    """
    if field.ndim != 2:
        raise ValueError("field must be a 3D tensor with shape (Z, Y, X)")

    Z, X = field.shape
    assert X == n_agents_x * agent_width, "X must equal n_agents_x * agent_width"
    assert Z == n_agents_z * agent_width, "Z must equal n_agents_z * agent_width"

    if pad_x < 0 or pad_x > n_agents_per_window_x:
        raise ValueError("pad_x must be in range [0, n_agents_per_window_x]")

    if pad_z < 0 or pad_z > n_agents_per_window_z:
        raise ValueError("pad_z must be in range [0, n_agents_per_window_z]")

    # Split field into per-agent spatial blocks
    field_agents = field.view(
        n_agents_z, agent_width, n_agents_x, agent_width
    )  # [n_agents_z, agent_width_z, n_agents_x, agent_width_x]
    field_agents = field_agents.permute(0, 2, 1, 3).contiguous()

    # First, we pad s.t. the first agent has a full window
    field_agents = torch.roll(field_agents, shifts=(pad_z, pad_x), dims=(0, 1))

    windows = []
    # Then, we start to extract windows and roll again
    for _ in range(n_agents_x):
        for _ in range(n_agents_z):
            local_window = field_agents[:n_agents_per_window_z, :n_agents_per_window_x]

            # Bring back to [Z, X] shape
            local_window = local_window.mean(dim=(2, 3))
            # local_window = local_window.permute(0, 2, 1, 3)
            # local_window = local_window.view(
            #     n_agents_per_window_z, agent_width, n_agents_per_window_x, agent_width
            # )
            # local_window = local_window.permute(0, 2, 1, 3).mean(dim=(2, 3))

            # Flip x and z to match original field orientation
            # local_window = local_window.flip(dims=(0,))

            windows += [local_window]

            field_agents = torch.roll(field_agents, shifts=-1, dims=0)
        field_agents = torch.roll(field_agents, shifts=-1, dims=1)

    return torch.stack(windows, dim=0)


def extract_moving_window_3d(
    field: torch.Tensor,
    n_agents: int,
    agent_width: int,
    n_agents_per_window: int,
) -> torch.Tensor:
    """
    Extracts local 3D observation windows for agents arranged in both X and Z
    directions.

    Parameters
    ----------
    field: torch.Tensor
        [Z, Y, X] tensor.

    n_agents: int
        Number of agents along X and Z.

    agent_width: int
        Spatial width per agent (in X and Z).

    n_agents_per_window: int
        Number of neighboring agents per window along X and Z.

    Returns
    -------
        Tensor of shape [n_agents_z * n_agents_x, Z_local, Y, X_local]
    """
    if field.ndim != 3:
        raise ValueError("field must be a 3D tensor with shape (Z, Y, X)")

    Z, Y, X = field.shape
    if X != n_agents * agent_width:
        raise ValueError("X must equal n_agents_x * agent_width")

    if Z != n_agents * agent_width:
        raise ValueError("Z must equal n_agents_z * agent_width")

    # Split field into per-agent spatial blocks
    field_agents = field.view(
        n_agents, agent_width, Y, n_agents, agent_width
    )  # [n_agents_z, agent_width_z, Y, n_agents_x, agent_width_x]
    field_agents = field_agents.permute(0, 2, 3, 1, 4).contiguous()

    pad = n_agents_per_window // 2

    # First, we pad s.t. the first agent has a full window
    field_agents = torch.roll(field_agents, shifts=(pad, pad), dims=(0, 2))

    windows = []
    # Then, we start to extract windows and roll again
    for _ in range(n_agents):
        for _ in range(n_agents):
            local_window = field_agents[:n_agents_per_window, :, :n_agents_per_window]

            # Bring back to [Z, Y, X] shape
            local_window = local_window.permute(0, 3, 1, 2, 4).contiguous()
            local_window = local_window.view(
                n_agents_per_window * agent_width, Y, n_agents_per_window * agent_width
            )
            windows += [local_window]

            field_agents = torch.roll(field_agents, shifts=-1, dims=2)
        field_agents = torch.roll(field_agents, shifts=-1, dims=0)

    return torch.stack(windows, dim=0)
