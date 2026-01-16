"""Setup script for FluidGym package with CUDA extensions."""

import os
from setuptools import find_packages, setup
from torch.utils import cpp_extension
import torch


torch_version = torch.__version__.split("+")[0].replace(".", "")[:2]


def _get_extensions():
    # Main PISOtorch extension
    pisotorch_sources = [
        "src/fluidgym/simulation/extensions/resampling.cu",
        "src/fluidgym/simulation/extensions/PISOtorch.cpp",
        "src/fluidgym/simulation/extensions/domain_structs.cpp",
        "src/fluidgym/simulation/extensions/PISO_multiblock_cuda_kernel.cu",
        "src/fluidgym/simulation/extensions/cg_solver_kernel.cu",
        "src/fluidgym/simulation/extensions/bicgstab_solver_kernel.cu",
        "src/fluidgym/simulation/extensions/grid_gen.cu",
        "src/fluidgym/simulation/extensions/eigenvalue.cu",
        "src/fluidgym/simulation/extensions/ortho_basis.cu",
        "src/fluidgym/simulation/extensions/matrix_vector_ops.cu",
        "src/fluidgym/simulation/extensions/matrix_vector_ops_grads.cu",
    ]
    pisotorch_macros = [("PYTHON_EXTENSION_BUILD", "1")]

    pisotorch_ext = cpp_extension.CUDAExtension(
        name="fluidgym.simulation.extensions.PISOtorch",
        sources=pisotorch_sources,
        include_dirs=["src/fluidgym/simulation/extensions"],
        extra_compile_args={
            "cxx": [
                "-fvisibility=hidden",
                "-Ofast",
                "-w",  # Suppress all warnings for GCC/Clang
            ],
            "nvcc": [
                f"--threads=2",
                "-O3",
                "--use_fast_math",
                "--compiler-options=-w",  # Pass -w to the host compiler (GCC/Clang)
            ],
        },
        extra_link_args=[],
        define_macros=pisotorch_macros,
    )

    # SimplexNoiseVariations extension
    noise_sources = [
        "src/fluidgym/simulation/extensions/noise/simplex_noise.cu",
        "src/fluidgym/simulation/extensions/noise/SimplexNoiseVariations.cpp",
    ]

    noise_ext = cpp_extension.CUDAExtension(
        name="fluidgym.simulation.extensions.SimplexNoiseVariations",
        sources=noise_sources,
        include_dirs=[
            "src/fluidgym/simulation/extensions/noise",
            "src/fluidgym/simulation/extensions",
        ],
        extra_compile_args={"cxx": ["-fvisibility=hidden"]},
    )

    # Only build the noise extension if the environment variable is set
    if os.environ.get("FLUIDGYM_BUILD_NOISE_EXT", "0") == "1":
        return [pisotorch_ext, noise_ext]
    else:
        return [pisotorch_ext]


setup(
    name="FluidGym",
    version="0.0.2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=_get_extensions(),
    cmdclass={
        "build_ext": cpp_extension.BuildExtension,
    },
    install_requires=[
        f"torch==2.9.*",
        "numpy",
        "scipy",
        "pandas",
    ],
)
