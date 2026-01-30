TorchRL
=======

FluidGym environments are compatible with TorchRL, allowing seamless 
integration with various RL libraries that support TorchRL. This is especially 
useful to leverage PyTorch's automatic differentiation capabilities for RL.

Due to the complexity of the TorchRL interface, we provide only a minimal example here
from ``examples/interfaces/torchrl.py``:

.. code-block:: python

    import fluidgym

    from fluidgym.integration.torchrl import TorchRLFluidEnv

    env = fluidgym.make("CylinderJet2D-easy-v0")
    env.seed(42)

    # For the TorchRL interface, wrap the FluidGym environment
    trl_env = TorchRLFluidEnv(env)

    # use with torchrl ...
