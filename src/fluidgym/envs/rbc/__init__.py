"""Rayleigh-Bénard Convection (RBC) Environments."""

from .rbc_env_2d import RBC_2D_DEFAULT_CONFIG, RBCEnv2D
from .rbc_env_3d import RBC_3D_DEFAULT_CONFIG, RBCEnv3D

__all__ = ["RBCEnv2D", "RBCEnv3D", "RBC_2D_DEFAULT_CONFIG", "RBC_3D_DEFAULT_CONFIG"]
