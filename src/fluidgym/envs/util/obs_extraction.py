"""Utility functions for extracting observation windows for agents.

Every function works per environment: the global observations come back with the
env dim first, and the window extractors accept any number of leading batch dims
(e.g. the env dim of a batched environment) and keep them in front.
"""

import torch

from fluidgym.envs import FluidEnv


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
        Global observations per environment with shape
        [E, n_agents * n_sensors_per_agent, ...].

    """
    u_list = [block.velocity for block in env._domain.getBlocks()]
    p_list = [block.pressure for block in env._domain.getBlocks()]

    # Straight to the sensor cells: never materialises the full grid, and stays
    # differentiable where the compiled kernel would detach. Readings are [E, C, R]
    u = env._resample_blocks_at(u_list, sensor_locations).transpose(1, 2)
    p = env._resample_blocks_at(p_list, sensor_locations)[:, 0]

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
        Global observations per environment with shape
        [E, n_agents, n_sensors_per_agent, ...].

    """
    u_list = [block.velocity for block in env._domain.getBlocks()]
    p_list = [block.pressure for block in env._domain.getBlocks()]

    sensor_locations = sensor_locations.flatten(start_dim=1)

    # See extract_global_2d_obs. [E, C, R] and [E, R]
    u: torch.Tensor = env._resample_blocks_at(u_list, sensor_locations)
    p: torch.Tensor = env._resample_blocks_at(p_list, sensor_locations)[:, 0]
    n_envs = u.size(0)

    if local_2d_obs:
        u = u[:, :2]
        velocity_dims = 2
    else:
        velocity_dims = 3

    # [E, C, R] -> [E, R, C], the layout the reshapes below expect
    u = u.transpose(1, 2).contiguous()

    u = u.view(n_envs, n_sensors_z, velocity_dims, -1)
    u = u.view(n_envs, n_agents, n_sensors_per_agent, velocity_dims, -1)

    if local_2d_obs:
        u = u.permute(0, 1, 2, 4, 3)

    p = p.view(n_envs, n_sensors_z, -1)
    p = p.view(n_envs, n_agents, n_sensors_per_agent, -1)

    return {
        "velocity": u,
        "pressure": p,
    }


def transform_global_to_local_obs_3d(
    global_obs: dict[str, torch.Tensor],
    local_obs_window: int,
    n_agents: int,
    local_2d_obs: bool = False,
    batch_dims: int = 0,
) -> dict[str, torch.Tensor]:
    """Transforms global observations into local observations for agents arranged
    in a 3D domain.

    Parameters
    ----------
    global_obs: dict[str, torch.Tensor]
        Global observations with shape [*batch, n_agents * n_sensors_per_agent, ...].

    local_obs_window: int
        Size of the local observation window (in number of agents).

    n_agents: int
        Number of agents.

    local_2d_obs: bool
        Whether the local observations are 2D (True) or 3D (False).

    batch_dims: int
        Number of leading batch dims, e.g. 1 for the env dim of a batched
        environment; the agent dim follows them. Defaults to 0.

    Returns
    -------
        Local observations with shape [*batch, n_agents, local_obs_window, ...].

    """
    offset = local_obs_window // 2
    agent_dim = batch_dims

    local_obs = {}
    for k, v in global_obs.items():
        # First, shift the global obs to start with the first agents sensor
        # window at zero
        shifted_obs = torch.roll(v, shifts=offset, dims=agent_dim)

        local_obs_list = []
        for _ in range(n_agents):
            window = shifted_obs.narrow(agent_dim, 0, local_obs_window)

            if local_2d_obs:
                # Drop the singleton dims of the window itself, never a batch dim
                batch, rest = window.shape[:batch_dims], window.shape[batch_dims:]
                window = window.reshape(*batch, *[n for n in rest if n != 1])

            local_obs_list += [window]

            shifted_obs = torch.roll(shifted_obs, shifts=-1, dims=agent_dim)

        local_obs[k] = torch.stack(local_obs_list, dim=agent_dim)

    return local_obs


def extract_moving_window_2d(
    field: torch.Tensor, n_agents: int, agent_width: int, n_agents_per_window: int
) -> torch.Tensor:
    """Extracts local 2D observation windows for agents arranged in a single row.

    Parameters
    ----------
    field: torch.Tensor
        [*batch, Y, X] tensor, with any number of leading batch dims.

    n_agents: int
        Number of agents along X.

    agent_width: int
        Spatial width per agent (in X).

    n_agents_per_window: int
        Number of neighboring agents per window along X.

    Returns
    -------
        Tensor of shape [*batch, n_agents, Y, window_size_x].
    """
    if field.ndim < 2:
        raise ValueError("field must be a tensor with shape (*batch, Y, X)")

    *batch, Y, X = field.shape
    assert X == n_agents * agent_width, "X must equal n_agents * agent_width"

    # Reshape into per-agent blocks: [*batch, Y, n_agents, agent_width]
    field_agents = field.reshape(*batch, Y, n_agents, agent_width)

    # Pad along the agent dimension (circularly)
    pad = n_agents_per_window // 2
    if pad > 0:
        field_padded = torch.cat(
            [field_agents[..., -pad:, :], field_agents, field_agents[..., :pad, :]],
            dim=-2,
        )
    else:
        field_padded = field_agents

    window_list = []
    for i in range(n_agents):
        start = i
        end = i + n_agents_per_window
        window = field_padded[..., start:end, :]  # [*batch, Y, window_agents, width]

        # Flatten the local agent window along the X dimension
        local_obs = window.reshape(*batch, Y, n_agents_per_window * agent_width)
        window_list.append(local_obs)

    return torch.stack(window_list, dim=-3)  # [*batch, n_agents, Y, window_size_x]


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
        [*batch, Z, X] tensor, with any number of leading batch dims.

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
        Tensor of shape [*batch, n_agents_z * n_agents_x, Z_local, X_local].
    """
    if field.ndim < 2:
        raise ValueError("field must be a tensor with shape (*batch, Z, X)")

    *batch, Z, X = field.shape
    nb = len(batch)
    assert X == n_agents_x * agent_width, "X must equal n_agents_x * agent_width"
    assert Z == n_agents_z * agent_width, "Z must equal n_agents_z * agent_width"

    if pad_x < 0 or pad_x > n_agents_per_window_x:
        raise ValueError("pad_x must be in range [0, n_agents_per_window_x]")

    if pad_z < 0 or pad_z > n_agents_per_window_z:
        raise ValueError("pad_z must be in range [0, n_agents_per_window_z]")

    # Split field into per-agent spatial blocks
    field_agents = field.reshape(
        *batch, n_agents_z, agent_width, n_agents_x, agent_width
    )  # [*batch, n_agents_z, agent_width_z, n_agents_x, agent_width_x]
    field_agents = field_agents.permute(
        *range(nb), nb, nb + 2, nb + 1, nb + 3
    ).contiguous()

    # First, we pad s.t. the first agent has a full window
    field_agents = torch.roll(field_agents, shifts=(pad_z, pad_x), dims=(nb, nb + 1))

    windows = []
    # Then, we start to extract windows and roll again
    for _ in range(n_agents_x):
        for _ in range(n_agents_z):
            local_window = field_agents[
                ..., :n_agents_per_window_z, :n_agents_per_window_x, :, :
            ]

            # Bring back to [Z, X] shape
            local_window = local_window.mean(dim=(-2, -1))

            windows += [local_window]

            field_agents = torch.roll(field_agents, shifts=-1, dims=nb)
        field_agents = torch.roll(field_agents, shifts=-1, dims=nb + 1)

    return torch.stack(windows, dim=nb)


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
        [*batch, Z, Y, X] tensor, with any number of leading batch dims.

    n_agents: int
        Number of agents along X and Z.

    agent_width: int
        Spatial width per agent (in X and Z).

    n_agents_per_window: int
        Number of neighboring agents per window along X and Z.

    Returns
    -------
        Tensor of shape [*batch, n_agents_z * n_agents_x, Z_local, Y, X_local]
    """
    if field.ndim < 3:
        raise ValueError("field must be a tensor with shape (*batch, Z, Y, X)")

    *batch, Z, Y, X = field.shape
    nb = len(batch)
    if X != n_agents * agent_width:
        raise ValueError("X must equal n_agents_x * agent_width")

    if Z != n_agents * agent_width:
        raise ValueError("Z must equal n_agents_z * agent_width")

    # Split field into per-agent spatial blocks
    field_agents = field.reshape(
        *batch, n_agents, agent_width, Y, n_agents, agent_width
    )  # [*batch, n_agents_z, agent_width_z, Y, n_agents_x, agent_width_x]
    field_agents = field_agents.permute(
        *range(nb), nb, nb + 2, nb + 3, nb + 1, nb + 4
    ).contiguous()

    pad = n_agents_per_window // 2

    # First, we pad s.t. the first agent has a full window
    field_agents = torch.roll(field_agents, shifts=(pad, pad), dims=(nb, nb + 2))

    windows = []
    # Then, we start to extract windows and roll again
    for _ in range(n_agents):
        for _ in range(n_agents):
            local_window = field_agents[
                ..., :n_agents_per_window, :, :n_agents_per_window, :, :
            ]

            # Bring back to [Z, Y, X] shape
            local_window = local_window.permute(
                *range(nb), nb, nb + 3, nb + 1, nb + 2, nb + 4
            ).contiguous()
            local_window = local_window.view(
                *batch,
                n_agents_per_window * agent_width,
                Y,
                n_agents_per_window * agent_width,
            )
            windows += [local_window]

            field_agents = torch.roll(field_agents, shifts=-1, dims=nb + 2)
        field_agents = torch.roll(field_agents, shifts=-1, dims=nb)

    return torch.stack(windows, dim=nb)
