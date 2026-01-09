Turbulent Channel Flow (TCF)
============================

3D wall-bounded turbulent channel flows with bottom or both-wall actuation.

Small Channel

+-----------------------------------+--------+-------+
| Environment ID                    | Walls  | Re    |
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
| Environment ID                    | Walls  | Re    |
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

The individual environment classes are documented below:

.. autosummary::
   :toctree: generated/

   fluidgym.envs.tcf.TCF3DBottomEnv
   fluidgym.envs.tcf.TCF3DBothEnv