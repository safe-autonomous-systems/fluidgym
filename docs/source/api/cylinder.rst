Flow Past Cylinder
==================

The *von Kármán vortex street* is a canonical setup in which flow separation behind a
cylinder induces periodic vortex shedding and fluctuating forces on the cylinder.
The objective is to reduce the drag coefficient :math:`C_D` while keeping the lift
:math:`C_L` small.

The system is parametrized via the Reynolds number
:math:`\mathrm{Re} = \frac{\overline{U} D}{\nu}`, where :math:`\overline{U}` is the
mean incoming velocity, :math:`D` the cylinder diameter, and :math:`\nu` the kinematic
viscosity.

Environment List
----------------

+-----------------------------+-----------+-------------+------------+
| Environment ID              | Actuation | Reynolds    | Resolution |
+=============================+===========+=============+============+
| ``CylinderJet2D-easy-v0``   | Jet       | 100         | 24         |
+-----------------------------+-----------+-------------+------------+
| ``CylinderJet2D-medium-v0`` | Jet       | 250         | 32         |
+-----------------------------+-----------+-------------+------------+
| ``CylinderJet2D-hard-v0``   | Jet       | 500         | 32         |
+-----------------------------+-----------+-------------+------------+
| ``CylinderRot2D-easy-v0``   | Rotation  | 100         | 24         |
+-----------------------------+-----------+-------------+------------+
| ``CylinderRot2D-medium-v0`` | Rotation  | 250         | 32         |
+-----------------------------+-----------+-------------+------------+
| ``CylinderRot2D-hard-v0``   | Rotation  | 500         | 32         |
+-----------------------------+-----------+-------------+------------+
| ``CylinderJet3D-easy-v0``   | Jet       | 100         | 24         |
+-----------------------------+-----------+-------------+------------+
| ``CylinderJet3D-medium-v0`` | Jet       | 250         | 32         |
+-----------------------------+-----------+-------------+------------+
| ``CylinderJet3D-hard-v0``   | Jet       | 500         | 48         |
+-----------------------------+-----------+-------------+------------+

Reward
------

The reward at step :math:`t` is:

.. math::

   r_t = C_{D,\mathrm{ref}} - \langle C_D \rangle_{T_{\mathrm{act}}} - \omega \, |\langle C_L \rangle_{T_{\mathrm{act}}}|

where :math:`\langle \cdot \rangle_{T_{\mathrm{act}}}` is the temporal average over the
actuation interval, :math:`C_{D,\mathrm{ref}}` is the reference drag coefficient of the
uncontrolled flow, and the lift regularization weight :math:`\omega = 1.0`.

The drag and lift coefficients are computed as:

.. math::

   C_D = \frac{F_D}{\frac{1}{2} \rho \overline{U}^2 D}, \qquad
   C_L = \frac{F_L}{\frac{1}{2} \rho \overline{U}^2 D}

with forces acting on the cylinder surface :math:`S`:

.. math::

   F_D = \int_S (\boldsymbol{\sigma} \cdot \mathbf{n}) \cdot \mathbf{e}_x \, \mathrm{d}S, \qquad
   F_L = \int_S (\boldsymbol{\sigma} \cdot \mathbf{n}) \cdot \mathbf{e}_y \, \mathrm{d}S

where :math:`\boldsymbol{\sigma}` is the Cauchy stress tensor and :math:`\mathbf{n}` the
outward unit normal at the cylinder surface.

MARL Reward
~~~~~~~~~~~

In MARL mode, the reward for agent :math:`i` is a weighted combination of a local and
global term:

.. math::

   r_t^i = \beta \, r_t^{i,\mathrm{local}} + (1 - \beta) \, r_t^{\mathrm{global}}

The local reward is computed over the cylinder segment controlled by agent :math:`i`,
while the global reward covers the full cylinder. The local weight :math:`\beta = 0.8`.

Action Space
------------

Two actuation modes are available:

- **Jet actuation** (``CylinderJet``): opposing synthetic jets on the top and bottom
  surfaces with a parabolic velocity profile and a total deflection angle of 10°.
  In 3D, the domain is extended spanwise, yielding eight individual jet pairs. The
  maximum jet velocity is :math:`\overline{U}`.
- **Rotation actuation** (``CylinderRot``): rigid cylinder rotation with a maximum
  absolute angular velocity of :math:`\overline{U}`.

The raw control signal is temporally smoothed via exponential filtering:

.. math::

   c_s = c_{s-1} + \alpha \, (a_t - c_{s-1}), \qquad \alpha = 0.1

where :math:`c_s` is the applied value at simulation sub-step :math:`s` and :math:`a_t`
the action at episode step :math:`t`.

Observation Space
-----------------

Observations consist of the horizontal and vertical velocity components at sensor
locations surrounding the cylinder (indicated by pink dots in 2D / pink planes in 3D).
In 3D, the spanwise velocity component is additionally included. To enable policy
transfer from 2D to 3D, the number of sensor planes and included components can be
configured to match the 2D observation layout.

Difficulty Levels
-----------------

Difficulty is controlled by the Reynolds number:

+------------+------------------+
| Level      | Reynolds number  |
+============+==================+
| Easy       | Re = 100         |
+------------+------------------+
| Medium     | Re = 250         |
+------------+------------------+
| Hard       | Re = 500         |
+------------+------------------+

Higher Reynolds numbers increase turbulence intensity and flow unsteadiness. The medium
and hard settings introduce three-dimensional flow interactions.

API Reference
-------------

.. autosummary::
   :toctree: generated/

   fluidgym.envs.cylinder.CylinderJetEnv2D
   fluidgym.envs.cylinder.CylinderRotEnv2D
   fluidgym.envs.cylinder.CylinderJetEnv3D
