"""Environment for 2D airfoil aerodynamic efficiency improvement."""

import numpy as np
import torch
from gymnasium import spaces

from fluidgym.envs.airfoil.airfoil_env_base import AirfoilEnvBase
from fluidgym.envs.util.obs_extraction import extract_global_2d_obs

AIRFOIL_2D_DEFAULT_CONFIG = {
    "reynolds_number": 3e3,
    "dt": 0.05,
    "step_length": 0.25,
    "adaptive_cfl": 0.8,
    "episode_length": 300,
    "attack_angle_deg": 10.0,
    "use_marl": False,
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

    use_marl: bool
        Whether to enable multi-agent reinforcement learning mode.

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
        use_marl: bool,
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
            use_marl=use_marl,
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
    def n_agents(self) -> int:
        """The number of agents in the environment."""
        return self._n_jets

    def _get_action_space(self) -> spaces.Box:
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._n_jets,),
            dtype=np.float32,
        )

    def _get_observation_space(self) -> spaces.Dict:
        n_sensors_x_y = self._sensor_locations.shape[-1]

        return spaces.Dict(
            {
                "velocity": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        n_sensors_x_y,
                        self._ndims,
                    ),
                    dtype=np.float32,
                ),
                "pressure": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(n_sensors_x_y,),
                    dtype=np.float32,
                ),
            }
        )

    def _get_global_obs(self) -> dict[str, torch.Tensor]:
        return extract_global_2d_obs(
            env=self,
            sensor_locations=self._sensor_locations,
        )

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
