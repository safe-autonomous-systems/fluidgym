Differentiable Simulation
=========================

With ``differentiable=True``, a FluidGym environment keeps the autograd graph of
every simulation step. Observations and rewards returned by ``step()`` are then
differentiable functions of the action and of the flow state, and PyTorch's
autograd can backpropagate through the solver. The
:doc:`../basic_usage/interfaces/gradient_based_methods` page shows how to
differentiate the reward w.r.t. the action. This page goes one level deeper and
differentiates w.r.t. the flow state itself.

The flow state
--------------

The differentiable state of the fluid is the velocity field of every block of
the simulation domain (plus the passive scalar field, e.g. the temperature, in
domains that transport one). To differentiate w.r.t. it, mark these tensors as
leaves with ``requires_grad_(True)`` before calling ``step()``.

Computing the observation Jacobian
----------------------------------

The Jacobian ``d(obs) / d(velocity)`` describes how every sensor reading
responds to the flow field. Here is a simple example from
``examples/advanced_usage/differentiable/compute_obs_jacobian.py``:

.. code-block:: python

    import torch

    import fluidgym

    # Create a FluidGym environment. With differentiable=True the observation
    # returned by env.step() keeps a gradient graph back to the velocity field, so we
    # can differentiate observations w.r.t. the flow state.
    env = fluidgym.make(
        "CylinderJet2D-easy-v0",
        differentiable=True,  # This flag enables backpropagation through the environment
    )
    env.reset(seed=42)

    action = env.sample_action()

    # The differentiable "state" of the fluid is the per-block velocity field. Mark
    # the velocity components of every block as leaves we want gradients for.
    velocity = [block.velocity.requires_grad_(True) for block in env._domain.getBlocks()]

    # Step the simulation. new_obs["velocity"] has shape [n_sensors, 2] and is now a
    # differentiable function of the velocity components marked above.
    new_obs, reward, terminated, truncated, info = env.step(action)
    obs = new_obs["velocity"]

    # Compute the Jacobian d(obs) / d(velocity) for the first block, one output row
    # at a time. Row i holds the gradient of obs.flatten()[i] w.r.t. the (flattened)
    # velocity field of block 0.
    outputs = obs.reshape(-1)
    jacobian = torch.stack(
        [
            torch.autograd.grad(output, velocity[0], retain_graph=True)[0].reshape(-1)
            for output in outputs
        ]
    )  # shape: [n_sensors * 2, 2 * H * W]

    print("Observation shape    :", tuple(obs.shape))
    print("Velocity field shape :", tuple(velocity[0].shape))
    print("Jacobian shape       :", tuple(jacobian.shape))

    # Detach the environment from the computation graph before the next step to avoid
    # accumulating the graph across steps:
    env.detach()

Every row of the Jacobian costs one backward pass through the simulation step,
so the full Jacobian is only affordable for a small number of outputs.

Computing a vector-Jacobian product
-----------------------------------

When only the gradient of a scalar (or of a weighted sum of outputs) is needed,
a vector-Jacobian product (VJP) gives it in a single backward pass, independent
of the number of outputs. Here is a simple example from
``examples/advanced_usage/differentiable/compute_state_vjp.py``, which computes
the VJP of the state after one step w.r.t. the state before it:

.. code-block:: python

    import torch

    import fluidgym
    from fluidgym.envs.util.diff_tools import (
        get_flat_state,
        mark_state_differentiable,
    )

    env = fluidgym.make(
        "CylinderJet2D-easy-v0",
        differentiable=True,  # This flag enables backpropagation through the environment
    )
    env.reset(seed=42)

    action = env.sample_action()

    # The differentiable state of the fluid is the per-block velocity field, plus the
    # per-block passive scalar field for domains that transport one. Mark all of them
    # as leaves we want gradients for.
    inputs = mark_state_differentiable(env)

    env.step(action)

    outputs = get_flat_state(env)
    cotangent = torch.ones_like(outputs)

    grad = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=cotangent,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
        materialize_grads=True,
    )[0]

    print("State dim :", outputs.numel())
    print("VJP shape :", tuple(grad.shape))  # [input_dim], one backward pass

    # Detach the env as soon as a new horizon is entered
    env.detach()

The cotangent ``v`` selects what is differentiated: ``grad`` is ``v^T J``, where
``J`` is the Jacobian of the new state w.r.t. the old one. A one-hot ``v``
yields a single row of ``J``.

Detaching the environment
-------------------------

The graph grows with every step. Call ``env.detach()`` whenever a new horizon
begins (e.g. after each backward pass) to cut the graph and free its memory.
Otherwise, memory usage grows until the GPU runs out.

.. note::

    Differentiable environments run in a single process. They are not supported
    by :class:`~fluidgym.envs.parallel_env.ParallelFluidEnv`, see
    :doc:`parallelization`.
