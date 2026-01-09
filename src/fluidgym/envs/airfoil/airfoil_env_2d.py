"""Environment for 2D airfoil aerodynamic efficiency improvement."""

import torch

from fluidgym.envs.airfoil.airfoil_env_base import AirfoilEnvBase
from fluidgym.simulation.pict.util.output import _resample_block_data

AIRFOIL_2D_DEFAULT_CONFIG = {
    "reynolds_number": 3e3,
    "dt": 0.05,
    "step_length": 0.25,
    "adaptive_cfl": 0.8,
    "episode_length": 300,
    "attack_angle_deg": 10.0,
    "dtype": torch.float32,
    "load_initial_domain": True,
    "load_domain_statistics": True,
    "randomize_initial_state": True,
    "enable_actions": True,
    "differentiable": False,
}


class AirfoilEnv2D(AirfoilEnvBase):
    """Environment for 2D airfoil aerodynamic efficiency improvement.

    References
    ----------
    [1] X. Garcia et al.,
    “Deep-reinforcement-learning-based separation control in a two-dimensional airfoil,”
    Feb. 24, 2025, arXiv: arXiv:2502.16993. doi: 10.48550/arXiv.2502.16993.

    [2] Y.-Z. Wang, Y.-F. Mei, N. Aubry, Z. Chen, P. Wu, and W.-T. Wu, “Deep
    reinforcement learning based synthetic jet control on disturbed flow over airfoil,”
    Physics of Fluids, 2022, [Online]. Available: https://doi.org/10.1063/5.0080922

    Parameters
    ----------
    reynolds_number: float
        The Reynolds number of the flow.

    adaptive_cfl: float
        Target CFL number for adaptive time stepping.

    step_length: float
        The non-dimensional time length of each environment step.

    episode_length: int
        The number of steps per episode.

    dt: float
        The time step size to use in the simulation.

    attack_angle_deg: float
        The angle of attack of the airfoil in degrees.

    dtype: torch.dtype
        The data type to use for the simulation.

    cuda_device: torch.device | None
        The CUDA device to use for the simulation. If None, the default cuda device is
        used. Defaults to None.

    debug: bool
        Whether to enable debug mode. Defaults to False.

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
    """

    def __init__(
        self,
        reynolds_number: float,
        adaptive_cfl: float,
        step_length: float,
        episode_length: int,
        dt: float,
        attack_angle_deg: float,
        dtype: torch.dtype = torch.float32,
        cuda_device: torch.device | None = None,
        debug: bool = False,
        load_initial_domain: bool = True,
        load_domain_statistics: bool = True,
        randomize_initial_state: bool = True,
        enable_actions: bool = True,
        differentiable: bool = False,
    ):
        super().__init__(
            ndims=2,
            reynolds_number=reynolds_number,
            adaptive_cfl=adaptive_cfl,
            step_length=step_length,
            episode_length=episode_length,
            dt=dt,
            attack_angle_deg=attack_angle_deg,
            dtype=dtype,
            cuda_device=cuda_device,
            debug=debug,
            load_initial_domain=load_initial_domain,
            load_domain_statistics=load_domain_statistics,
            randomize_initial_state=randomize_initial_state,
            enable_actions=enable_actions,
            differentiable=differentiable,
        )

    @property
    def action_space_shape(self) -> tuple[int, ...]:
        """The shape of the action space."""
        return (self._n_jets,)

    @property
    def observation_space_shape(self) -> tuple[int, ...]:
        """The shape of the observation space."""
        return (self._sensors_locations.shape[-1] * self._ndims,)

    def _get_global_obs(self) -> torch.Tensor:
        """Return the current observation."""
        u_list = [block.velocity for block in self._domain.getBlocks()]

        u = _resample_block_data(
            u_list,
            self._sim.output_resampling_coords,
            self._sim.output_resampling_shape,
            self._ndims,
            fill_max_steps=self._sim.output_resampling_fill_max_steps,
        )
        u = u.squeeze()
        u = u.permute(1, 2, 0)
        u = u[self._sensors_locations[1], self._sensors_locations[0], :]
        u = u.view(-1)

        return u

    def _action_to_control(self, action: torch.Tensor) -> torch.Tensor:
        assert self._jet_locations_top is not None

        v_action = action - action.mean()

        # Ensure max abs. value of 1.0
        max_v = torch.max(torch.abs(v_action))
        if max_v > 1.0:
            v_action = v_action / max_v

        top_profile = self._top_base_profile.clone()

        for i in range(self._n_jets):
            start_idx_top, end_idx_top = self._jet_locations_top[i]

            top_profile[
                0,
                :,
                0,
                start_idx_top : end_idx_top + 1,
            ] *= v_action[i]

        return top_profile

    def reset(
        self,
        seed: int | None = None,
        randomize: bool | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Resets the environment to an initial internal state, returning an initial
        observation and info.

        Parameters
        ----------
        seed: int | None
            The seed to use for random number generation. If None, the current seed is
            used.

        randomize: bool | None
            Whether to randomize the initial state. If None, the default behavior is
            used.

        Returns
        -------
        tuple[torch.Tensor, dict]
            A tuple containing the initial observation and an info dictionary.
        """
        obs, info = super().reset(seed=seed, randomize=randomize)
        info.pop("full_obs")
        return obs, info
