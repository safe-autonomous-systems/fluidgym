Flow Past Airfoil
=================

Flow past a NACA 0012 airfoil at various Reynolds numbers with the goal of
improving the aerodynamic efficiency.

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

The individual environment classes are documented below:

.. autosummary::
   :toctree: generated/

   fluidgym.envs.airfoil.AirfoilEnv2D
   fluidgym.envs.airfoil.AirfoilEnv3D