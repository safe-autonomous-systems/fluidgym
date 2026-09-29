from .config import config
from .envs.airfoil import (
    AIRFOIL_2D_DEFAULT_CONFIG,
    AIRFOIL_3D_DEFAULT_CONFIG,
    AirfoilEnv2D,
    AirfoilEnv3D,
)
from .envs.cylinder import (
    CYLINDER_JET_2D_DEFAULT_CONFIG,
    CYLINDER_JET_3D_DEFAULT_CONFIG,
    CYLINDER_ROT_2D_DEFAULT_CONFIG,
    CylinderJetEnv2D,
    CylinderJetEnv3D,
    CylinderRotEnv2D,
)
from .envs.rbc import RBC_2D_DEFAULT_CONFIG, RBC_3D_DEFAULT_CONFIG, RBCEnv2D, RBCEnv3D
from .envs.tcf import (
    LARGE_TCF_3D_DEFAULT_CONFIG,
    SMALL_TCF_3D_DEFAULT_CONFIG,
    TCF3DBothEnv,
    TCF3DBottomEnv,
)
from .registry import make, register

# ------------------------------------------------------------------------
# 2D Cylinder Environments
# ------------------------------------------------------------------------
register(
    id="CylinderJet2D-easy-v0",
    entry_point=CylinderJetEnv2D,
    defaults=CYLINDER_JET_2D_DEFAULT_CONFIG,
    reynolds_number=100,
    resolution=24,
)

register(
    id="CylinderJet2D-medium-v0",
    entry_point=CylinderJetEnv2D,
    defaults=CYLINDER_JET_2D_DEFAULT_CONFIG,
    reynolds_number=250,
    resolution=32,
)

register(
    id="CylinderJet2D-hard-v0",
    entry_point=CylinderJetEnv2D,
    defaults=CYLINDER_JET_2D_DEFAULT_CONFIG,
    reynolds_number=500,
    resolution=32,
)

register(
    id="CylinderRot2D-easy-v0",
    entry_point=CylinderRotEnv2D,
    defaults=CYLINDER_ROT_2D_DEFAULT_CONFIG,
    reynolds_number=100,
    resolution=24,
)

register(
    id="CylinderRot2D-medium-v0",
    entry_point=CylinderRotEnv2D,
    defaults=CYLINDER_ROT_2D_DEFAULT_CONFIG,
    reynolds_number=250,
    resolution=32,
)

register(
    id="CylinderRot2D-hard-v0",
    entry_point=CylinderRotEnv2D,
    defaults=CYLINDER_ROT_2D_DEFAULT_CONFIG,
    reynolds_number=500,
    resolution=32,
)

# ------------------------------------------------------------------------
# 3D Cylinder Environments
# ------------------------------------------------------------------------
register(
    id="CylinderJet3D-easy-v0",
    entry_point=CylinderJetEnv3D,
    defaults=CYLINDER_JET_3D_DEFAULT_CONFIG,
    reynolds_number=100,
    resolution=24,
)

register(
    id="CylinderJet3D-medium-v0",
    entry_point=CylinderJetEnv3D,
    defaults=CYLINDER_JET_3D_DEFAULT_CONFIG,
    reynolds_number=250,
    resolution=32,
)

register(
    id="CylinderJet3D-hard-v0",
    entry_point=CylinderJetEnv3D,
    defaults=CYLINDER_JET_3D_DEFAULT_CONFIG,
    reynolds_number=500,
    resolution=48,
)

# ------------------------------------------------------------------------
# 2D Rayleigh-Bénard Convection Environments
# ------------------------------------------------------------------------
register(
    id="RBC2D-easy-v0",
    entry_point=RBCEnv2D,
    defaults=RBC_2D_DEFAULT_CONFIG,
    rayleigh_number=8e4,
    adaptive_cfl=0.8,
)

register(
    id="RBC2D-medium-v0",
    entry_point=RBCEnv2D,
    defaults=RBC_2D_DEFAULT_CONFIG,
    rayleigh_number=4e5,
    adaptive_cfl=0.5,
)

register(
    id="RBC2D-hard-v0",
    entry_point=RBCEnv2D,
    defaults=RBC_2D_DEFAULT_CONFIG,
    rayleigh_number=8e5,
    adaptive_cfl=0.5,
)

register(
    id="RBC2D-wide-easy-v0",
    entry_point=RBCEnv2D,
    defaults=RBC_2D_DEFAULT_CONFIG,
    aspect_ratio=2,
    n_heaters=24,
    rayleigh_number=8e4,
)

register(
    id="RBC2D-wide-medium-v0",
    entry_point=RBCEnv2D,
    defaults=RBC_2D_DEFAULT_CONFIG,
    aspect_ratio=2,
    n_heaters=24,
    rayleigh_number=4e5,
    adaptive_cfl=0.5,
)

register(
    id="RBC2D-wide-hard-v0",
    entry_point=RBCEnv2D,
    defaults=RBC_2D_DEFAULT_CONFIG,
    aspect_ratio=2,
    n_heaters=24,
    rayleigh_number=8e5,
    adaptive_cfl=0.5,
)

# ------------------------------------------------------------------------
# 3D Rayleigh-Bénard Convection Environments
# ------------------------------------------------------------------------
register(
    id="RBC3D-easy-v0",
    entry_point=RBCEnv3D,
    defaults=RBC_3D_DEFAULT_CONFIG,
    rayleigh_number=6e3,
    adaptive_cfl=0.5,
)

register(
    id="RBC3D-medium-v0",
    entry_point=RBCEnv3D,
    defaults=RBC_3D_DEFAULT_CONFIG,
    rayleigh_number=8e3,
    adaptive_cfl=0.5,
)

register(
    id="RBC3D-hard-v0",
    entry_point=RBCEnv3D,
    defaults=RBC_3D_DEFAULT_CONFIG,
    rayleigh_number=1e4,
    adaptive_cfl=0.5,
)

register(
    id="RBC3D-wide-easy-v0",
    entry_point=RBCEnv3D,
    defaults=RBC_3D_DEFAULT_CONFIG,
    aspect_ratio=2,
    n_heaters=16,
    rayleigh_number=6e3,
    adaptive_cfl=0.5,
)

register(
    id="RBC3D-wide-medium-v0",
    entry_point=RBCEnv3D,
    defaults=RBC_3D_DEFAULT_CONFIG,
    aspect_ratio=2,
    n_heaters=16,
    rayleigh_number=8e3,
    adaptive_cfl=0.5,
)

register(
    id="RBC3D-wide-hard-v0",
    entry_point=RBCEnv3D,
    defaults=RBC_3D_DEFAULT_CONFIG,
    aspect_ratio=2,
    n_heaters=16,
    rayleigh_number=1e4,
    adaptive_cfl=0.5,
)

# ------------------------------------------------------------------------
# 3D Small Channel Flow Environments
# ------------------------------------------------------------------------
register(
    id="TCFSmall3D-bottom-easy-v0",
    entry_point=TCF3DBottomEnv,
    defaults=SMALL_TCF_3D_DEFAULT_CONFIG,
    reynolds_number_wall=180,
)

register(
    id="TCFSmall3D-bottom-medium-v0",
    entry_point=TCF3DBottomEnv,
    defaults=SMALL_TCF_3D_DEFAULT_CONFIG,
    reynolds_number_wall=330,
)

register(
    id="TCFSmall3D-bottom-hard-v0",
    entry_point=TCF3DBottomEnv,
    defaults=SMALL_TCF_3D_DEFAULT_CONFIG,
    reynolds_number_wall=550,
)

register(
    id="TCFSmall3D-both-easy-v0",
    entry_point=TCF3DBothEnv,
    defaults=SMALL_TCF_3D_DEFAULT_CONFIG,
    reynolds_number_wall=180,
)

register(
    id="TCFSmall3D-both-medium-v0",
    entry_point=TCF3DBothEnv,
    defaults=SMALL_TCF_3D_DEFAULT_CONFIG,
    reynolds_number_wall=330,
)

register(
    id="TCFSmall3D-both-hard-v0",
    entry_point=TCF3DBothEnv,
    defaults=SMALL_TCF_3D_DEFAULT_CONFIG,
    reynolds_number_wall=550,
)

# ------------------------------------------------------------------------
# 3D Large TCF Environments
# ------------------------------------------------------------------------
register(
    id="TCFLarge3D-bottom-easy-v0",
    entry_point=TCF3DBottomEnv,
    defaults=LARGE_TCF_3D_DEFAULT_CONFIG,
    reynolds_number_wall=180,
)

register(
    id="TCFLarge3D-bottom-medium-v0",
    entry_point=TCF3DBottomEnv,
    defaults=LARGE_TCF_3D_DEFAULT_CONFIG,
    reynolds_number_wall=330,
)

register(
    id="TCFLarge3D-bottom-hard-v0",
    entry_point=TCF3DBottomEnv,
    defaults=LARGE_TCF_3D_DEFAULT_CONFIG,
    reynolds_number_wall=550,
)

register(
    id="TCFLarge3D-both-easy-v0",
    entry_point=TCF3DBothEnv,
    defaults=LARGE_TCF_3D_DEFAULT_CONFIG,
    reynolds_number_wall=180,
)

register(
    id="TCFLarge3D-both-medium-v0",
    entry_point=TCF3DBothEnv,
    defaults=LARGE_TCF_3D_DEFAULT_CONFIG,
    reynolds_number_wall=330,
)

register(
    id="TCFLarge3D-both-hard-v0",
    entry_point=TCF3DBothEnv,
    defaults=LARGE_TCF_3D_DEFAULT_CONFIG,
    reynolds_number_wall=550,
)

# ------------------------------------------------------------------------
# 2D Airfoil Environments
# ------------------------------------------------------------------------
register(
    id="Airfoil2D-easy-v0",
    entry_point=AirfoilEnv2D,
    defaults=AIRFOIL_2D_DEFAULT_CONFIG,
    reynolds_number=1e3,
)

register(
    id="Airfoil2D-medium-v0",
    entry_point=AirfoilEnv2D,
    defaults=AIRFOIL_2D_DEFAULT_CONFIG,
    reynolds_number=3e3,
)

register(
    id="Airfoil2D-hard-v0",
    entry_point=AirfoilEnv2D,
    defaults=AIRFOIL_2D_DEFAULT_CONFIG,
    reynolds_number=5e3,
)

# ------------------------------------------------------------------------
# 3D Airfoil Environments
# ------------------------------------------------------------------------
register(
    id="Airfoil3D-easy-v0",
    entry_point=AirfoilEnv3D,
    defaults=AIRFOIL_3D_DEFAULT_CONFIG,
    reynolds_number=1e3,
)

register(
    id="Airfoil3D-medium-v0",
    entry_point=AirfoilEnv3D,
    defaults=AIRFOIL_3D_DEFAULT_CONFIG,
    reynolds_number=3e3,
)

register(
    id="Airfoil3D-hard-v0",
    entry_point=AirfoilEnv3D,
    defaults=AIRFOIL_3D_DEFAULT_CONFIG,
    reynolds_number=5e3,
)


__all__ = ["config", "make"]
