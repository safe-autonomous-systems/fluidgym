FluidGym Environments
=====================

FluidGym environments are implemented as subclasses of the ``FluidEnv`` base class,
which is similar to the standard Gymnasium ``Env`` interface but tailored to flow 
control tasks. The ``FluidEnv`` class defines the core methods and attributes that
all FluidGym environments must implement. Here is an overview of its API:

.. autosummary::
   :toctree: generated/

   fluidgym.envs.fluid_env.FluidEnv