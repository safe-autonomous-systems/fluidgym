Rayleigh-Bénard Convection (RBC)
================================

Rayleigh-Bénard Convection models buoyancy-driven flow between a heated bottom plate
and a cooled top plate. This leads to convective fluid motion and the formation of
thermal plumes with complex, potentially chaotic patterns.

The system is governed by two dimensionless parameters: the Prandtl number
:math:`\mathrm{Pr}` (a material property of the fluid) and the Rayleigh number
:math:`\mathrm{Ra}` (controls the intensity of buoyancy-driven convection).

Environment List
----------------

2D RBC

+-----------------------------+------------+--------------+--------------+
| Environment ID              | Rayleigh   | Aspect Ratio | Notes        |
+=============================+============+==============+==============+
| ``RBC2D-easy-v0``           | 8×10⁴      | 1            | 12 heaters   |
+-----------------------------+------------+--------------+--------------+
| ``RBC2D-medium-v0``         | 4×10⁵      | 1            | 12 heaters   |
+-----------------------------+------------+--------------+--------------+
| ``RBC2D-hard-v0``           | 8×10⁵      | 1            | 12 heaters   |
+-----------------------------+------------+--------------+--------------+
| ``RBC2D-wide-easy-v0``      | 8×10⁴      | 2            | 24 heaters   |
+-----------------------------+------------+--------------+--------------+
| ``RBC2D-wide-medium-v0``    | 4×10⁵      | 2            | 24 heaters   |
+-----------------------------+------------+--------------+--------------+
| ``RBC2D-wide-hard-v0``      | 8×10⁵      | 2            | 24 heaters   |
+-----------------------------+------------+--------------+--------------+

3D RBC

+-----------------------------+------------+--------------+--------------+
| Environment ID              | Rayleigh   | Aspect Ratio | Notes        |
+=============================+============+==============+==============+
| ``RBC3D-easy-v0``           | 6×10³      | 1            | 64 heaters   |
+-----------------------------+------------+--------------+--------------+
| ``RBC3D-medium-v0``         | 8×10³      | 1            | 64 heaters   |
+-----------------------------+------------+--------------+--------------+
| ``RBC3D-hard-v0``           | 1×10⁴      | 1            | 64 heaters   |
+-----------------------------+------------+--------------+--------------+
| ``RBC3D-wide-easy-v0``      | 6×10³      | 2            | 256 heaters  |
+-----------------------------+------------+--------------+--------------+
| ``RBC3D-wide-medium-v0``    | 8×10³      | 2            | 256 heaters  |
+-----------------------------+------------+--------------+--------------+
| ``RBC3D-wide-hard-v0``      | 1×10⁴      | 2            | 256 heaters  |
+-----------------------------+------------+--------------+--------------+

Reward
------

The objective is to reduce convective heat transfer. The reward uses the instantaneous
Nusselt number:

.. math::

   \mathrm{Nu}_{\mathrm{instant}} = \sqrt{\mathrm{Ra} \, \mathrm{Pr}} \, \langle u_y T \rangle_V

where :math:`u_y` is the vertical fluid velocity, :math:`T` the temperature field, and
:math:`\langle \cdot \rangle_V` denotes volumetric averaging. The reward is:

.. math::

   r_t = \mathrm{Nu}_{\mathrm{ref}} - \mathrm{Nu}_{\mathrm{instant}}

where :math:`\mathrm{Nu}_{\mathrm{ref}}` is the Nusselt number of the uncontrolled
baseline.

PD Controller Baseline
~~~~~~~~~~~~~~~~~~~~~~

A linear proportional-derivative (PD) controller is provided as a reference baseline:

.. math::

   a(x,t) = k_p \, E(x,t) + k_d \, \frac{\Delta E(x,t)}{\Delta t}

where :math:`E(x,t) = \langle u_y(x,y,t) \rangle` and :math:`k_p = 970`,
:math:`k_d = 2000`.

Action Space
------------

Control is applied via localized heaters at the bottom boundary. The heater temperature
actions are:

- Normalized and clipped to ensure a mean equal to the default bottom temperature and a
  maximum heater temperature of 1.75.
- Spatially smoothed to avoid hard transitions between neighboring heaters.

In 2D, 12 heaters are used (24 for wide-domain variants). In 3D, 64 heaters are used
(256 for wide-domain variants). Centralized (SARL) control is supported in 2D;
decentralized (MARL) control is used for 3D due to the large number of actuators.

Observation Space
-----------------

Observations include all velocity components and temperature at sensor locations.

- **2D**: The default observation window contains sensors above 11 heaters, centered
  around the currently actuated heater.
- **3D**: Each agent observes a local window of :math:`3 \times 3` heaters and their
  associated sensors.

Difficulty Levels
-----------------

Difficulty is controlled by the Rayleigh number:

+------------+--------------------+--------------------+
| Level      | 2D Rayleigh        | 3D Rayleigh        |
+============+====================+====================+
| Easy       | Ra = 8×10⁴         | Ra = 6×10³         |
+------------+--------------------+--------------------+
| Medium     | Ra = 4×10⁵         | Ra = 8×10³         |
+------------+--------------------+--------------------+
| Hard       | Ra = 8×10⁵         | Ra = 1×10⁴         |
+------------+--------------------+--------------------+

Higher Rayleigh numbers lead to stronger plume interactions and increasingly chaotic
convection patterns.

API Reference
-------------

.. autosummary::
   :toctree: generated/

   fluidgym.envs.rbc.RBCEnv2D
   fluidgym.envs.rbc.RBCEnv3D
