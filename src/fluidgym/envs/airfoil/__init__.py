"""Flow Past Airfoil Environments."""

from .airfoil_env_2d import AIRFOIL_2D_DEFAULT_CONFIG, AirfoilEnv2D
from .airfoil_env_3d import AIRFOIL_3D_DEFAULT_CONFIG, AirfoilEnv3D
from .airfoil_env_base import AirfoilEnvBase

__all__ = [
    "AirfoilEnvBase",
    "AirfoilEnv2D",
    "AIRFOIL_2D_DEFAULT_CONFIG",
    "AirfoilEnv3D",
    "AIRFOIL_3D_DEFAULT_CONFIG",
]
