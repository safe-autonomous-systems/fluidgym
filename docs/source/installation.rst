Installation
============

There are two main ways to install FluidGym: via PyPI or from the source code on
GitHub. Regardless of the installation method, it is recommended to set up a
dedicated Python virtual environment using tools like `venv` or `conda` to avoid
dependency conflicts.

FluidGym requires Linux, an NVIDIA GPU with a driver for CUDA 12.8 (≥ 570), and
Python 3.11–3.13.

1. Using PyPI
-------------

This is the simplest way to install FluidGym:

.. code-block:: bash

    pip install fluidgym

This also installs the GPU-accelerated solver
`phiPICT <https://github.com/safe-autonomous-systems/phiPICT>`_ together with
the PyTorch version it was built for (PyTorch 2.10 with CUDA 12.8). There is no
need to install PyTorch first: phiPICT's compiled extension only works with the
PyTorch version it was built against, so pip replaces any other version.

To use a different PyTorch or CUDA version, install a matching phiPICT build
first and then FluidGym, see the
`phiPICT installation docs <https://safe-autonomous-systems.github.io/phiPICT/installation.html>`_.

2. From GitHub
--------------

This is the best way to install the latest version of FluidGym. FluidGym itself
is pure Python; the solver still comes with phiPICT from PyPI. Clone the
repository and install the package:

.. code-block:: bash

    git clone https://github.com/safe-autonomous-systems/fluidgym.git
    cd fluidgym
    pip install .

To develop FluidGym, install it in editable mode together with the development
tools instead (needs pip ≥ 25.1):

.. code-block:: bash

    make install-dev
