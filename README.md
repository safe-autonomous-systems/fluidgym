<p align="center">
    <a href="./docs/images/logo_lm.png#gh-light-mode-only">
        <img src="./docs/source/_static/img/logo_lm.png#gh-light-mode-only" alt="FluidGym Logo" width="50%"/>
    </a>
    <a href="./docs/images/logo_dm.png#gh-dark-mode-only">
        <img src="./docs/source/_static/img/logo_dm.png#gh-dark-mode-only" alt="FluidGym Logo" width="50%"/>
    </a>
</p>

<div align="center">
    
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-%2376B900)
![License](https://img.shields.io/badge/License-MIT-orange)
    
</div>

<div align="center">
    <h3>
      <a href="#-installation">Installation</a> |
      <a href="#-getting-started">Getting Started</a> |
      <a href="#-reproducing-experiments">Reproducing Experiments</a> |
      <a href="#-license-&-citation">License & Citation</a>
    </h3>
</div>

---

## Installation

### 📦 Installation from PyPi

Not available in anonymized version.

### 🐳 Using Docker 

Not available in anonymized version.

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

For an easy start refer to our documentation and the [`examples`](examples) directory.
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

<div style="border: 2px solid red; padding: 10px; border-radius: 5px;">
<strong>Note:</strong> 


The download of initial domain snapshots is not possible without accessing our huggingface repository.
Therefore, during the double-blind review phase, environments can only be used with the following arguments:
```python
import fluidgym

env = fluidgym.make(
    "JetCylinder2D-easy-v0",
    load_initial_domain=False,
    load_domain_statistics=False
)
obs, info = env.reset(seed=42)

for _ in range(50):
    action = env.sample_action()
    obs, reward, term, trunc, info = env.step(action)
    env.render()

    if term or trunc:
        break
```

However, rewards are not normalized with uncontrolled values and episodes start with an 
un-initialized flow field. 

Alternatively, you can create the initial domain snapshots yourself as explained in [`initial_domain_generation.md`](initial_domain_generation.md).
We note that, depending on the environment, this might take a while.

</div>

## License & Citation

This repository is published under the MIT license.
