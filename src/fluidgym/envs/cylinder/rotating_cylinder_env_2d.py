"""Environment for flow around a cylinder with rotating cylinder actuation."""

import torch

from fluidgym.envs.cylinder.cylinder_env_base import CylinderEnvBase

CYLINDER_ROT_2D_DEFAULT_CONFIG = {
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


class CylinderRotEnv2D(CylinderEnvBase):
    """Environment for flow around a cylinder with rotating cylinder actuation.

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
    [1] M. Tokarev, E. Palkin, and R. Mullyadzhanov, “Deep Reinforcement Learning
    Control of Cylinder Flow Using Rotary Oscillations at Low Reynolds Number,”
    Energies, vol. 13, no. 22, Art. no. 22, Jan. 2020, doi: 10.3390/en13225920.
    """

    def __init__(
        self,
        reynolds_number: float,
        resolution: int,
        dt: float,
        adaptive_cfl: float,
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
            ndims=2,
            resolution=resolution,
            reynolds_number=reynolds_number,
            dt=dt,
            adaptive_cfl=adaptive_cfl,
            step_length=step_length,
            episode_length=episode_length,
            lift_penalty=lift_penalty,
            use_marl=use_marl,
            dtype=dtype,
            cuda_device=cuda_device,
            load_initial_domain=load_initial_domain,
            load_domain_statistics=load_domain_statistics,
            randomize_initial_state=randomize_initial_state,
            enable_actions=enable_actions,
            differentiable=differentiable,
        )

    def _additional_initialization(self) -> None:
        super()._additional_initialization()
        (
            self._left_velocity,
            self._top_velocity,
            self._bottom_velocity,
            self._right_velocity,
        ) = self._get_boundary_velocities()

    def _get_boundary_velocities(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        def coords_to_velocities(coords_boundary: torch.Tensor) -> torch.Tensor:
            centers = 0.5 * (coords_boundary[:, :-1] + coords_boundary[:, 1:])

            velocities = torch.zeros_like(centers)
            theta = torch.atan2(centers[1, :], centers[0, :])
            velocities[0], velocities[1] = torch.sin(theta), -torch.cos(theta)

            return velocities

        vertex_coords = self._domain.getVertexCoordinates()
        left_coords = vertex_coords[self._left_block_idx]
        top_coords = vertex_coords[self._top_block_idx]
        right_coords = vertex_coords[self._right_block_idx]
        bottom_coords = vertex_coords[self._bottom_block_idx]

        left_coords_boundary = left_coords[0, :, :, -1]
        top_coords_boundary = top_coords[0, :, 0, :]
        right_coords_boundary = right_coords[0, :, :, 0]
        bottom_coords_boundary = bottom_coords[0, :, -1, :]

        left_velocities = coords_to_velocities(left_coords_boundary)
        top_velocities = coords_to_velocities(top_coords_boundary)
        right_velocities = coords_to_velocities(right_coords_boundary)
        bottom_velocities = coords_to_velocities(bottom_coords_boundary)

        left_velocities = left_velocities[None, :, :, None]
        top_velocities = top_velocities[None, :, None, :]
        right_velocities = right_velocities[None, :, :, None]
        bottom_velocities = bottom_velocities[None, :, None, :]

        return left_velocities, top_velocities, bottom_velocities, right_velocities

    def _apply_action(self, action: torch.Tensor) -> None:
        """Apply the given action to the simulation.

        Parameters
        ----------
        action: torch.Tensor
            The action to apply, representing the rotation speed of the cylinder wall.
        """
        self._left_boundary.setVelocity(self._left_velocity * action)
        self._top_boundary.setVelocity(self._top_velocity * action)
        self._right_boundary.setVelocity(self._right_velocity * action)
        self._bottom_boundary.setVelocity(self._bottom_velocity * action)

    @property
    def id(self) -> str:
        """Unique identifier for the environment."""
        return f"RotatingCylinder{self._ndims}D_Re{self._reynolds_number}"
