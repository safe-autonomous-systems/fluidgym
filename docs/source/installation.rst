Installation
============

There are two main ways to install FluidGym: via PyPI or by downloading the source code
from GitHub. Regardless of the installation method, it is recommended to set up a dedicated
Python virtual environment using tools like `venv` or `conda` to avoid dependency conflicts.

Before installing FluidGym, ensure that you have Python 3.10 or higher installed on your system.
To enable the GPU-accelerated solver, PyTorch with CUDA 12.8 is required. It can be 
installed using the following command:

.. code-block:: bash

    pip install torch --index-url https://download.pytorch.org/whl/cu128

Then, follow one of the methods below to install FluidGym.

1. Using PyPI
-------------

This is the simplest way to install FluidGym. After setting up the python environment,
just run the following command:

.. code-block:: bash

    # Not available in anonymized version

2. Downloading from GitHub
--------------------------

This is the best way to install the latest version of FluidGym. Additionally, this method allows
you to compile FluidGym from source on architectures/operating systems that are not supported
by the PyPI binaries.
First, clone the FluidGym repository from GitHub:

.. code-block:: bash

    # Not available in anonymized version

Then, install the package:

.. code-block:: bash

    make install

Depending on whether you want to reproduce our experiments or develop new features, you can install
FluidGym with different sets of dependencies:

- To install FluidGym for development purposes, run:

  .. code-block:: bash

       make install-dev

- To install FluidGym for reproducing experiments, run:

  .. code-block:: bash

       make install-exp
