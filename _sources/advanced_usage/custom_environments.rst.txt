Adding Custom Environments to FluidGym
======================================

FluidGym allows users to create and integrate their own custom fluid dynamics
environments. This is particularly useful for researchers who want to experiment
with specific fluid scenarios not covered by the built-in environments.

To create a custom environment, you need to subclass either ``FluidEnv`` or
``MultiAgentFluidEnv`` from the ``fluidgym.envs`` module, depending on whether
your environment is single-agent or multi-agent.

Here is a basic example of how to create a custom single-agent environment:

.. code-block:: python

    import torch
    from fluidgym.envs.fluid_env import FluidEnv
    from fluidgym.simulation.extensions import PISOtorch


    class CustomFluidEnv(FluidEnv):
        def __init__(
            self,
            dt: float,
            adaptive_cfl: float,
            step_length: float,
            episode_length: int,
            ndims: int,
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
                dtype=dtype,
                cuda_device=cuda_device,
                load_initial_domain=load_initial_domain,
                load_domain_statistics=load_domain_statistics,
                randomize_initial_state=randomize_initial_state,
                enable_actions=enable_actions,
                differentiable=differentiable

            )

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

For multi-agent environments, the process is similar, but you would subclass
``MultiAgentFluidEnv`` instead and additionally implement ``step_marl`` and ``reset_marl`` methods.