Vectorized Environments: Batching and Multiple GPUs
===================================================

FluidGym can simulate many environments at once, on two levels:

- **Batching** runs several environments in *one* simulation on *one* GPU. The
  environments share the grid, the solver setup and every kernel launch, so a
  batch of ``B`` small environments costs little more than a single one.
  Throughput grows almost linearly with ``B`` until the GPU is saturated.
- **GPU parallelization** runs one worker process per GPU (or several per GPU),
  each simulating its own batch. This scales beyond a single GPU once its memory
  or compute is exhausted.

Both give the same vector API, so a training script does not need to know which
one it is running on. Batching alone is the cheapest option and should be the
first choice. Add workers only when one GPU is not enough.

Batching on a Single GPU
------------------------

Pass ``n_envs`` to :func:`fluidgym.make` (or call :func:`fluidgym.make_vec`
without ``devices``, which is equivalent). Here is a simple example from
``examples/advanced_usage/parallelization/single_gpu.py``:

.. code-block:: python

    import fluidgym

    # Four environments batched in one simulation on one GPU. They share the grid,
    # the solver setup and every kernel launch, so a small batch costs little more
    # than a single environment. fluidgym.make_vec(id, n_envs=4) is equivalent
    env = fluidgym.make("CylinderJet2D-easy-v0", n_envs=4)
    env.seed(42)

    # Everything has a leading env dim: [n_envs, ...], or [n_envs, n_agents, ...]
    # for MARL
    obs, info = env.reset()
    action = env.sample_action()

    obs, reward, terminated, truncated, info = env.step(action)
    # reward: [4], terminated/truncated: bool tensors [4]

    # The batch shares the episode clock, but every env can start from its own
    # initial domain
    obs, info = env.reset(domain_idx=[0, 1, 2, 3])

The vector API
~~~~~~~~~~~~~~

With ``n_envs`` given (even ``n_envs=1``), everything carries a leading env dim:

=====================  ==============================  ===================================
                       single-agent                    MARL (``use_marl=True``)
=====================  ==============================  ===================================
action                 ``[E, *action_space.shape]``     ``[E, n_agents, *action_space.shape]``
observation (per key)  ``[E, *space.shape]``            ``[E, n_agents, *space.shape]``
reward                 ``[E]``                          ``[E, n_agents]``
terminated, truncated  bool ``[E]``                     bool ``[E]``
info (per key)         ``[E, ...]``                     ``[E, ...]``
=====================  ==============================  ===================================

- ``action_space`` and ``observation_space`` stay those of a single environment
  (and agent), like ``single_action_space`` in gymnasium's vector envs.
- ``n_envs`` / ``num_envs`` give the number of environments, ``vectorized`` is
  True.
- Without ``n_envs`` (the default), nothing changes: a single environment with
  exactly the previous API.

Properties of a batch
~~~~~~~~~~~~~~~~~~~~~

**Shared episodes.** The environments of a batch share the episode clock. They
are reset together, are truncated at the same step, and never terminate early.
Each environment has its own state, initial condition and actions:

.. code-block:: python

    env.reset(domain_idx=[0, 3, 5, 7, 1, 2, 4, 6])  # one initial domain per env
    env.reset(randomize=True)                        # every env draws its own

**Solver settings.** Solver settings are shared, but relative tolerances are
resolved per environment. Every environment converges exactly as it would on its
own, so a batched environment reproduces the single environments up to
round-off. The time step is common to all environments (the smallest adaptive
time step of the batch).

**Rendering.** ``render()`` returns the frames of all environments,
``[E, H, W, 3]`` (the layout of vectorized-env video recorders). Pass
``env_ids`` to choose which: one index returns a single frame ``[H, W, 3]``, a
list returns a stack. The ``VideoRecorder`` wrapper takes ``env_ids`` as
well and saves one GIF per recorded environment (``<filename>_env<i>_ep<k>.gif``).

**Choosing the batch size.** The best batch size depends on the environment
and the GPU. Increase ``n_envs`` until the throughput (env steps per second)
stops growing or the GPU runs out of memory. Beyond that point, more
environments need more GPUs.

Parallelization on Multiple GPUs
--------------------------------

.. warning::

    Experimental feature. If you encounter issues, please report them on our GitHub
    issues page.

Pass ``devices`` to :func:`fluidgym.make_vec` to get a
:class:`~fluidgym.envs.parallel_env.ParallelFluidEnv`. It starts one worker
process per entry of ``devices``, and every worker batches
``n_envs / len(devices)`` environments on its GPU. Here is a simple example from
``examples/advanced_usage/parallelization/multi_gpu.py``:

.. code-block:: python

    import fluidgym

    # Since the worker processes are spawned, we need to protect the entry point
    if __name__ == "__main__":
        # 128 environments: one worker process per GPU, each simulating 64
        # environments batched together. n_envs must be divisible by len(devices)
        env = fluidgym.make_vec("CylinderJet2D-easy-v0", n_envs=128, devices=[0, 1])
        try:
            # Everything has a leading env dim: [n_envs, ...], or
            # [n_envs, n_agents, ...] for MARL. Worker r is seeded with seed + r
            env.seed(42)

            obs, info = env.reset()
            action = env.sample_action()

            obs, reward, terminated, truncated, info = env.step(action)

            # Results of a ParallelFluidEnv are returned on the CPU
        finally:
            env.close()

- **Workers and devices.** ``devices`` lists the CUDA device of every worker. A
  device may occur several times, e.g. ``devices=[0, 0, 1, 1]`` runs two
  workers per GPU. ``n_envs`` must be divisible by ``len(devices)``.
- **Env order.** The environments are split in order: worker ``r`` simulates
  environments ``r * n_envs / len(devices)`` to
  ``(r + 1) * n_envs / len(devices) - 1``. Actions, observations, rewards and
  ``domain_idx`` in ``reset`` follow the same order.
- **Seeding.** ``seed(s)`` and ``reset(seed=s)`` seed worker ``r`` with
  ``s + r``, so the workers do not simulate identical batches.
- **Process start.** The workers are spawned, so the script's entry point must
  be protected by ``if __name__ == "__main__":``. Call ``close()`` (or use the
  environment as a context manager) to shut the workers down.
- **Data transfer.** Actions are sent to the workers and results returned as CPU
  tensors, with the leading env dim of all workers concatenated.
- **Limitations.** ``ParallelFluidEnv`` does not support differentiable
  environments. Use a batched environment on one GPU for gradient-based methods.

.. note::

    ``ParallelFluidEnv`` now follows the vector API. Its MARL observations are
    ``[E, n_agents, ...]`` (previously flattened to ``[E * n_agents, ...]``), and
    its flags and infos are tensors with a leading env dim (previously lists).

Interfaces
----------

- **Gymnasium**: :class:`~fluidgym.integration.gymnasium.GymVecFluidEnv` is a
  ``gymnasium.vector.VectorEnv`` with batched spaces. Its autoreset mode is
  ``SAME_STEP``: the last observations are in ``info["final_obs"]``.
  ``GymFluidEnv`` wraps single environments only.
- **Stable-Baselines3**: :class:`~fluidgym.integration.sb3.VecFluidEnv` accepts
  vectorized (and MARL) environments. Every agent of every environment is one
  SB3 environment (``num_envs = n_envs * n_agents``, environment-major).
- **TorchRL**: :class:`~fluidgym.integration.torchrl.TorchRLFluidEnv` has batch
  size ``(E,)``, or ``(E, n_agents)`` in MARL mode. ``from_pixels=True`` adds the
  rendered frames as ``pixels`` (e.g. for TorchRL's ``VideoRecorder``).
- **PettingZoo** supports single environments only.
- **Wrappers** keep the env dim: ``FlattenObservation`` flattens behind it, and
  ``PowerPenalty`` charges every environment (and agent) its own action power.
