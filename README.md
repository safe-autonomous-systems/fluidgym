<p align="center">
    <a href="./docs/images/logo_lm.png#gh-light-mode-only">
        <img src="./docs/source/_static/img/logo_lm.png#gh-light-mode-only" alt="FluidGym Logo" width="50%"/>
    </a>
    <a href="./docs/images/logo_dm.png#gh-dark-mode-only">
        <img src="./docs/source/_static/img/logo_dm.png#gh-dark-mode-only" alt="FluidGym Logo" width="50%"/>
    </a>
</p>

<div align="center">
    
[![PyPI version](https://badge.fury.io/py/fluidgym.svg)](https://badge.fury.io/py/fluidgym)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fluidgym)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-%2376B900)
![License](https://img.shields.io/badge/License-MIT-orange)
[![Linters](https://github.com/safe-autonomous-systems/fluidgym/actions/workflows/linters.yml/badge.svg?branch=main)](https://github.com/safe-autonomous-systems/fluidgym/actions/workflows/linters.yml)
    
</div>

<div align="center">
    <h3>
      <a href="#-installation">Installation</a> |
      <a href="#-getting-started">Getting Started</a> |
      <a href="#-reproducing-experiments">Reproducing Experiments</a> |
      <a href="https://safe-autonomous-systems.github.io/fluidgym">Documentation</a> | 
      <a href="#-license-&-citation">License & Citation</a>
    </h3>
</div>

---

## Installation

### 📦 Installation from PyPi

1. Ensure the correct PyTorch version is installed (compatible with CUDA 12.8):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

2. Install 

```bash
pip install fluidgym
```

### 🐳 Using Docker 

Instead of installing FluidGym you can use one of our Docker containers:

- [fluidgym-runtime](https://hub.docker.com/repository/docker/becktepe/fluidgym-runtime) for running FluidGym
- [fluidgym-devel](https://hub.docker.com/repository/docker/becktepe/fluidgym-devel) for development

Both containers come with the following Miniconda environments:
- ```py310```: Python 3.10
- ```py311```: Python 3.11
- ```py312```: Python 3.12
- ```py313```: Python 3.13

Start the containers with:
```bash
docker run -it --gpus all fluidgym-runtime bash
docker run -it --gpus all fluidgym-devel bash
```

### 🧱 Build from Source (GitHub)

1. Create a new conda environment and activate it:
```bash
conda create -n fluidgym python=3.10
conda activate fluidgym
```

2. Install gcc:
```bash
conda install pip "gcc_linux-64>=6.0,<=11.5" "gxx_linux-64>=6.0,<=11.5"
```

3. Install the latest Pytorch for CUDA 12.8 via pip:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

4. Install the matching cuda toolkit via conda:
```bash
conda install cuda-toolkit=12.8 -c nvidia/label/cuda-12.8.1
```

5. Clone the repository and enter the directory, then compile the custom CUDA kernels and install the package (this might take several minutes):
```bash
make install
```

## Getting Started

For an easy start refer to our [documentation](https://safe-autonomous-systems.github.io/fluidgym/) and the [`examples`](examples) directory.
FluidGym provides a ```gymnasium```-like interface that can be used as follows:

```python
import fluidgym

env = fluidgym.make(
    "JetCylinder2D-easy-v0",
)
obs, info = env.reset(seed=42)

for _ in range(50):
    action = env.sample_action()
    obs, reward, term, trunc, info = env.step(action)
    env.render()

    if term or trunc:
        break
```

## Reproducing Experiments

All commands to reproduce the experiments of the paper can be found in [`experiments.md`](experiments.md).
The steps to generate the initial domain snapshots and statistics are stated in [`initial_domain_generation.md`](initial_domain_generation.md).

## License & Citation

This repository is published under the MIT license.
