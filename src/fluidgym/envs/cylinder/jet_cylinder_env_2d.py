"""2D Environment for flow around a cylinder with jet actuation."""

import torch

from fluidgym.envs.cylinder.cylinder_env_base import CylinderEnvBase
from fluidgym.envs.util.profiles import get_jet_profile

CYLINDER_JET_2D_DEFAULT_CONFIG = {
    "reynolds_number": 1e2,
    "resolution": 24,
    "dt": 1e-2,
    "adaptive_cfl": 0.8,
    "step_length": 0.25,
    "episode_length": 80,
    "lift_penalty": 1.0,
    "use_marl": False,
    "dtype": torch.float32,
    "load_initial_domain": True,
    "load_domain_statistics": True,
    "randomize_initial_state": True,
    "enable_actions": True,
    "differentiable": False,
}


class CylinderJetEnv2D(CylinderEnvBase):
    """Environment for flow around a cylinder with jet actuation.

    Parameters
    ----------
    reynolds_number: float
        The Reynolds number of the flow.

    resolution: int
        The resolution of the simulation grid. Corresponds to the angular resolution
        around the cylinder.

    dt: float
        The time step size to use in the simulation.

    adaptive_cfl: float
        The adaptive CFL number to use in the simulation.

    step_length: float
        The non-dimensional time length of each environment step.

    episode_length: int
        The number of steps per episode.

    lift_penalty: float
        The penalty factor for lift in the reward calculation.

    use_marl: bool
        Whether to enable multi-agent reinforcement learning mode.

    dtype: torch.dtype
        The data type to use for the simulation. Defaults to torch.float32.

    cuda_device: torch.device | None
        The CUDA device to use for the simulation. If None, the default cuda device is
        used. Defaults to None.

    load_initial_domain: bool
        Whether to load initial domain states from disk. Defaults to True.

    load_domain_statistics: bool
        Whether to load domain statistics from disk. Defaults to True.

    randomize_initial_state: bool
        Whether to randomize the initial state on reset. Defaults to True.

    enable_actions: bool
        Whether to enable actions. If False, the environment will be run in
        uncontrolled mode. Defaults to True.

    differentiable: bool
        Whether to enable differentiable simulation mode. Defaults to False.

    References
    ----------
    [1] J. Rabault, M. Kuchta, A. Jensen, U. Réglade, and N. Cerardi, “Artificial neural
    networks trained through deep reinforcement learning discover control strategies for
    active flow control,” Journal of Fluid Mechanics, vol. 865, pp. 281-302, Apr. 2019,
    doi: 10.1017/jfm.2019.62.

    [2] F. Ren, J. Rabault, and H. Tang, “Applying deep reinforcement learning to active
    flow control in weakly turbulent conditions,” Physics of Fluids, vol. 33, no. 3,
    p. 037121, Mar. 2021, doi: 10.1063/5.0037371.
    """

    _jet_angle: float = 10.0  # degrees

    def __init__(
        self,
        reynolds_number: float,
        resolution: int,
        adaptive_cfl: float,
        dt: float,
        step_length: float,
        episode_length: int,
        lift_penalty: float,
        use_marl: bool,
        dtype: torch.dtype = torch.float32,
        cuda_device: torch.device | None = None,
        load_initial_domain: bool = True,
        load_domain_statistics: bool = True,
        randomize_initial_state: bool = True,
        enable_actions: bool = True,
        differentiable: bool = False,
    ):
        super().__init__(
            reynolds_number=reynolds_number,
            resolution=resolution,
            dt=dt,
            adaptive_cfl=adaptive_cfl,
            step_length=step_length,
            episode_length=episode_length,
            ndims=2,
            lift_penalty=lift_penalty,
            use_marl=use_marl,
            dtype=dtype,
            cuda_device=cuda_device,
            load_initial_domain=load_initial_domain,
            randomize_initial_state=randomize_initial_state,
            load_domain_statistics=load_domain_statistics,
            enable_actions=enable_actions,
            differentiable=differentiable,
        )

    def _additional_initialization(self) -> None:
        super()._additional_initialization()
        self._top_velocity, self._bottom_velocity = self._get_boundary_velocities()
        assert self._top_velocity.grad_fn is None
        assert self._bottom_velocity.grad_fn is None

    def _get_boundary_velocities(self) -> tuple[torch.Tensor, torch.Tensor]:
        def coords_to_velocities(
            coords_boundary: torch.Tensor, direction: str
        ) -> torch.Tensor:
            centers = 0.5 * (coords_boundary[:, :-1] + coords_boundary[:, 1:])

            if direction == "top":
                angles = torch.pi / 2 - torch.atan2(centers[1, :], centers[0, :])
            else:
                angles = -torch.pi / 2 - torch.atan2(centers[1, :], centers[0, :])

            angles_deg = torch.rad2deg(angles)
            angles_deg_abs = torch.abs(angles_deg)

            angles_deg_abs[angles_deg_abs > self._jet_angle] = 0.0
            min_idx, max_idx = torch.where(angles_deg_abs > 0.0)[0][[0, -1]]
            min_idx = min_idx - 1
            max_idx = max_idx + 1

            profile_width = max_idx - min_idx + 1
            u_profile = get_jet_profile(
                h=int(profile_width), dtype=self._dtype, device=self._cuda_device
            )

            velocities = torch.zeros_like(centers)
            for i, u_magn in zip(range(min_idx, max_idx + 1), u_profile, strict=False):
                angle = angles_deg[i]
                velocities[0, i] = u_magn * torch.sin(torch.deg2rad(angle))
                velocities[1, i] = u_magn * torch.cos(torch.deg2rad(angle))

            return velocities

        vertex_coords = self._domain.getVertexCoordinates()
        top_coords = vertex_coords[self._top_block_idx]
        bottom_coords = vertex_coords[self._bottom_block_idx]

        top_coords_boundary = top_coords[0, :, 0, :]
        bottom_coords_boundary = bottom_coords[0, :, -1, :]

        top_velocities = coords_to_velocities(top_coords_boundary, direction="top")
        bottom_velocities = coords_to_velocities(
            bottom_coords_boundary, direction="bottom"
        )

        top_velocities = top_velocities[None, :, None, :]
        bottom_velocities = bottom_velocities[None, :, None, :]

        return top_velocities, bottom_velocities

    def _apply_action(self, action: torch.Tensor) -> None:
        """Apply the given action to the simulation."""
        self._top_boundary.setVelocity(self._top_velocity * action)
        self._bottom_boundary.setVelocity(self._bottom_velocity * action)

    @property
    def id(self) -> str:
        """Unique identifier for the environment."""
        return f"JetCylinder2D_Re{self._reynolds_number}"
