Flow Past Airfoil
=================

Flow past a stationary NACA 0012 airfoil at an angle of attack of 20°. Variations in
Reynolds number influence flow separation and vortex dynamics. The objective is to
improve aerodynamic efficiency by increasing the lift-to-drag ratio.

Task difficulty is set by the Reynolds number, with higher values producing sharper flow
separation and stronger turbulence.

Environment List
----------------

2D Airfoil

+---------------------------+----------+
| Environment ID            | Re       |
+===========================+==========+
| ``Airfoil2D-easy-v0``     | 1×10³    |
+---------------------------+----------+
| ``Airfoil2D-medium-v0``   | 3×10³    |
+---------------------------+----------+
| ``Airfoil2D-hard-v0``     | 5×10³    |
+---------------------------+----------+

3D Airfoil

+---------------------------+----------+
| Environment ID            | Re       |
+===========================+==========+
| ``Airfoil3D-easy-v0``     | 1×10³    |
+---------------------------+----------+
| ``Airfoil3D-medium-v0``   | 3×10³    |
+---------------------------+----------+
| ``Airfoil3D-hard-v0``     | 5×10³    |
+---------------------------+----------+

Reward
------

The reward at step :math:`t` maximizes aerodynamic efficiency:

.. math::

   r_t = \frac{\langle C_L \rangle_{T_{\mathrm{act}}}}{\langle C_D \rangle_{T_{\mathrm{act}}}} - \frac{C_{L,\mathrm{ref}}}{C_{D,\mathrm{ref}}}

where :math:`\langle \cdot \rangle_{T_{\mathrm{act}}}` denotes the average over the
actuation interval and the reference values correspond to the uncontrolled baseline.

Action Space
------------

Actuation uses surface-mounted synthetic jet actuators placed on top of the airfoil.
Zero net-mass-flux is enforced across actuators.

In 3D, the domain is extended spanwise (depth :math:`D = 1.4`), yielding four spanwise
jet segments and 12 individual actuators in total.

In **MARL mode**, each agent controls a group of three adjacent jets (one spanwise
segment), enabling decentralized control over independent surface regions.

As in the cylinder environment, the raw control signal is temporally smoothed via
exponential filtering with :math:`\alpha = 0.1`.

Observation Space
-----------------

Observations consist of velocity components at sensor locations distributed around the
airfoil surface (analogous to the cylinder setup). The 3D configuration follows the same
layout as the 2D case but extends sensor placement spanwise.

Difficulty Levels
-----------------

Difficulty is controlled by the Reynolds number:

+------------+------------------+
| Level      | Reynolds number  |
+============+==================+
| Easy       | Re = 1×10³       |
+------------+------------------+
| Medium     | Re = 3×10³       |
+------------+------------------+
| Hard       | Re = 5×10³       |
+------------+------------------+

Higher Reynolds numbers lead to more abrupt flow separation and stronger turbulence,
which increases the challenge of effective flow control.

API Reference
-------------

.. autosummary::
   :toctree: generated/

   fluidgym.envs.airfoil.AirfoilEnv2D
   fluidgym.envs.airfoil.AirfoilEnv3D
