Flow Past Cylinder 
==================

Flow past a cylinder with active control via jet or rotational actuation. The following
table summarizes the available environments:

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

The individual environment classes are documented below:

.. autosummary::
   :toctree: generated/

   fluidgym.envs.cylinder.CylinderJetEnv2D
   fluidgym.envs.cylinder.CylinderRotEnv2D
   fluidgym.envs.cylinder.CylinderJetEnv3D