Adding Custom Environments to FluidGym
======================================

FluidGym allows users to create and integrate their own custom fluid dynamics
environments. This is particularly useful for researchers who want to experiment
with specific fluid scenarios not covered by the built-in environments.

To create a custom environment, you need to subclass ``FluidEnv`` from the 
``fluidgym.envs`` module.

Here is a basic example of how to create a custom environment:

.. code-block:: python

    import torch
    from fluidgym.envs.fluid_env import FluidEnv
    from fluidgym.simulation.extensions import PISOtorch
    from gymnasium import spaces


    class CustomFluidEnv(FluidEnv):
        # Here, you can define whether your environment supports multi-agent RL.
        # If not, set this to False. Otherwise, set to True and implement the 
        # _step_marl_impl and _get_local_obs methods.
        _supports_marl: bool = False


        def __init__(
            self,
            dt: float,
            adaptive_cfl: float,
            step_length: float,
            episode_length: int,
            ndims: int,
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
                dt=dt,
                adaptive_cfl=adaptive_cfl,
                step_length=step_length,
                episode_length=episode_length,
                ndims=ndims,
                use_marl=use_marl,
                dtype=dtype,
                cuda_device=cuda_device,
                load_initial_domain=load_initial_domain,
                load_domain_statistics=load_domain_statistics,
                randomize_initial_state=randomize_initial_state,
                enable_actions=enable_actions,
                differentiable=differentiable

            )

        def _get_observation_space(self) -> spaces.Space:
            # Define and return the observation space for your environment
            pass

        def _get_action_space(self) -> spaces.Space:
            # Define and return the action space for your environment
            pass

        def _get_domain(self) -> PISOtorch.Domain:
            # Define and return your custom PISOtorch domain here
            pass

        def _get_prep_fn(self, domain: PISOtorch.Domain) -> dict[str, Any]:
            # Define and return any preprocessing functions needed for your environment
            pass

        def _get_simulation(self, domain: PISOtorch.Domain, prep_fn: dict[str, Any]) -> Simulation:         
            # Define and return the PISOtorch simulation setup for your environment
            pass

        def _additional_initialization(self) -> None:
            # Any optional additional initialization steps for your environment
            pass

        def _randomize_domain(self) -> None:
            # Logic to randomize the domain if needed
            pass

        def render_shape(self) -> tuple[int, int, int]:
            # Define the shape of the rendered environment  
            pass

        def id(self) -> str:
            # Unique identifier for the environment
            pass

        def initial_domain_id(self) -> str:
            # Identifier for the initial domain configuration
            pass

        def _step_impl(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
            # Implement the step logic for your environment
            pass

        def _step_marl_impl(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
            # Implement the step logic for your environment in case it supports multi-agent RL
            pass

        def _get_local_obs(self) -> torch.Tensor:
            # Implement the logic to get local observations for multi-agent RL
            pass
