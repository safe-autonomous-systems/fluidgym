"""Turbulent Channel Flow (TCF) Environments."""

from .tcf_env import (
    LARGE_TCF_3D_DEFAULT_CONFIG,
    SMALL_TCF_3D_DEFAULT_CONFIG,
    TCF3DBothEnv,
    TCF3DBottomEnv,
)

__all__ = [
    "TCF3DBottomEnv",
    "TCF3DBothEnv",
    "SMALL_TCF_3D_DEFAULT_CONFIG",
    "LARGE_TCF_3D_DEFAULT_CONFIG",
]
