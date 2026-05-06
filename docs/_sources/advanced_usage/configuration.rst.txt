Additional Configuration Options
================================

FluidGym provides a global configuration system that allows users to customize various aspects of the
library's behavior. Using ``fluidgym.config``, users can access and set the following options:

- ``local_data_path``: Specifies the directory where FluidGym stores local data, such as initial domain snapshots. By default, this is set to the user's local data directory (e.g., ``/home/username/.fluidgym/data`` on Linux systems). Users can change this path to a different location if desired.
- ``hf_intial_domains_repo_id``: Specifies the Hugging Face repository ID from which FluidGym fetches initial domain snapshots. This allows users to customize the source of initial domain data if they have their own repository.
- ``dtype``: Sets the default data type for tensors used in FluidGym simulations. This can be set to either ``torch.float32`` or ``torch.float64``, depending on the precision requirements of the user's application.

The general mechanism to access and modify these configuration options is as follows:

.. code-block:: python

    import fluidgym
    from pathlib import Path

    # Configuration options can be accessed as direct properties
    local_data_path = fluidgym.config.local_data_path
    hf_repo_id = fluidgym.config.hf_intial_domains_repo_id
    default_dtype = fluidgym.config.dtype

    # To update a configuration option, simply use the update method
    fluidgym.config.update("local_data_path", Path("/new/path/to/data"))
    fluidgym.config.update("hf_intial_domains_repo_id", "username/custom-repo")
    fluidgym.config.update("dtype", torch.float64)

    # Optionally, options can also be accessed via the get method
    local_data_path = fluidgym.config.get("local_data_path")
    hf_repo_id = fluidgym.config.get("hf_intial_domains_repo_id")
    default_dtype = fluidgym.config.get("dtype")
