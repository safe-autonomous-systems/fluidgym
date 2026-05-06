Turbulent Channel Flow (TCF)
============================

The turbulent channel flow describes flow between two parallel infinite plates and is a
classic experiment for studying wall-bounded turbulence. Most active flow control
strategies aim to reduce wall shear stress via spatially distributed blowing and suction
actuators at the walls.

Environment List
----------------

Small Channel

+-----------------------------------+--------+-------+
| Environment ID                    | Walls  | Re_τ  |
+===================================+========+=======+
| ``TCFSmall3D-bottom-easy-v0``     | Bottom | 180   |
+-----------------------------------+--------+-------+
| ``TCFSmall3D-bottom-medium-v0``   | Bottom | 330   |
+-----------------------------------+--------+-------+
| ``TCFSmall3D-bottom-hard-v0``     | Bottom | 550   |
+-----------------------------------+--------+-------+
| ``TCFSmall3D-both-easy-v0``       | Both   | 180   |
+-----------------------------------+--------+-------+
| ``TCFSmall3D-both-medium-v0``     | Both   | 330   |
+-----------------------------------+--------+-------+
| ``TCFSmall3D-both-hard-v0``       | Both   | 550   |
+-----------------------------------+--------+-------+

Large Channel

+-----------------------------------+--------+-------+
| Environment ID                    | Walls  | Re_τ  |
+===================================+========+=======+
| ``TCFLarge3D-bottom-easy-v0``     | Bottom | 180   |
+-----------------------------------+--------+-------+
| ``TCFLarge3D-bottom-medium-v0``   | Bottom | 330   |
+-----------------------------------+--------+-------+
| ``TCFLarge3D-bottom-hard-v0``     | Bottom | 550   |
+-----------------------------------+--------+-------+
| ``TCFLarge3D-both-easy-v0``       | Both   | 180   |
+-----------------------------------+--------+-------+
| ``TCFLarge3D-both-medium-v0``     | Both   | 330   |
+-----------------------------------+--------+-------+
| ``TCFLarge3D-both-hard-v0``       | Both   | 550   |
+-----------------------------------+--------+-------+

Reward
------

The reward is based on the instantaneous reduction of wall shear stress relative to the
uncontrolled reference:

.. math::

   r_t = 1 - \frac{\tau_{\mathrm{wall}}}{\tau_{\mathrm{wall},\mathrm{ref}}}

The wall shear stress is computed as:

.. math::

   \tau_{\mathrm{wall}} = \nu \left.\frac{\partial u_x}{\partial y}\right|_{y=0}

For single-wall (bottom) actuation, only the bottom wall stress is used; for dual-wall
actuation, the stress is averaged across both walls.

Opposition Control Baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A pre-computed opposition control baseline is provided. Wall-normal velocity at the wall
is prescribed as:

.. math::

   v_{\mathrm{wall}}(x,z,t) = -\alpha \, v'(x,y_s,z,t)

where :math:`v'(x,y_s,z,t)` is the wall-normal velocity fluctuation at the detection
plane (:math:`y^+ = 15`) and :math:`\alpha = 1.0`.

Action Space
------------

Control is applied via wall-normal blowing and suction at the boundary using spatially
distributed actuators. Zero net-mass-flux is enforced. Boundary velocities are scaled
by the friction velocity :math:`u_+` as the maximum value.

- **Small channel**: :math:`32 \times 32` actuators per wall.
- **Large channel**: :math:`64 \times 64` actuators per wall.

Two configurations are available: bottom-wall actuation only, or dual-wall actuation
(both walls simultaneously).

Observation Space
-----------------

Observations consist of local velocity fluctuations :math:`\mathbf{u}' = (u', v')`
defined as deviations from the instantaneous spatial mean:

.. math::

   u' = u - \langle u \rangle_V, \qquad v' = v - \langle v \rangle_V

where :math:`\langle \cdot \rangle_V` denotes volumetric averaging. Fluctuations are
sampled at a detection plane located at wall-normal distance :math:`y^+ = 15`, directly
above the corresponding actuator. In the bottom-actuation variant, only observations
from the bottom wall are provided.

Difficulty Levels
-----------------

Difficulty is controlled by the friction Reynolds number :math:`\mathrm{Re}_\tau`:

+------------+---------------------+
| Level      | Re_τ                |
+============+=====================+
| Easy       | Re_τ = 180          |
+------------+---------------------+
| Medium     | Re_τ = 330          |
+------------+---------------------+
| Hard       | Re_τ = 550          |
+------------+---------------------+

Higher friction Reynolds numbers correspond to more intense turbulence, making drag
reduction increasingly difficult.

API Reference
-------------

.. autosummary::
   :toctree: generated/

   fluidgym.envs.tcf.TCF3DBottomEnv
   fluidgym.envs.tcf.TCF3DBothEnv
