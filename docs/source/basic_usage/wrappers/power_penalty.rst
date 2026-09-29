PowerPenalty Wrapper
====================

Actuation is not free: jets, rotating cylinders and heaters all draw power. To
learn control policies that achieve their goal with little actuation effort,
FluidGym provides the ``PowerPenalty`` wrapper. It charges the reward for the
power of the action as

.. math::

    r' = r - \lambda \, \lVert a \rVert_2^2,

where ``penalty`` sets the coefficient :math:`\lambda`. For vectorized
environments, every environment is charged its own action power, and in MARL
mode every agent is charged for its own action, so the penalty has the same
shape as the reward.

FluidGym wrappers can be used by importing them from the ``fluidgym.wrappers`` module
and wrapping the environment instance.

Here is a simple example from ``examples/basic_usage/wrappers/power_penalty.py``:

.. code-block:: python

    import fluidgym
    from fluidgym.wrappers import PowerPenalty

    env = fluidgym.make(
        "CylinderJet2D-easy-v0",
    )

    # Now, we charge the reward for the power the actuation draws
    env = PowerPenalty(env, penalty=0.1)

    obs, info = env.reset(seed=42)

    action = env.sample_action()

    # Now, if we take a step, the reward is reduced by 0.1 * ||action||^2
    obs, reward, terminated, truncated, info = env.step(action)
