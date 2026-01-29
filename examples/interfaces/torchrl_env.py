import fluidgym

from fluidgym.integration.torchrl import TorchRLFluidEnv

env = fluidgym.make("CylinderJet2D-easy-v0")
env.seed(42)

trl_env = TorchRLFluidEnv(env)

# use with torchrl ...
