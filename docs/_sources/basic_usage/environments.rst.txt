Environments
============

FluidGym provides 35+ environments across four fluid physics domains, each available in
easy, medium, and hard difficulty levels.

Flow Past Cylinder
------------------

Active flow control around a cylinder via jet or rotational actuation. The goal is to
suppress vortex shedding and reduce drag.

.. list-table::
   :widths: 33 33 33
   :header-rows: 1

   * - Easy (Re = 100)
     - Medium (Re = 250)
     - Hard (Re = 500)
   * - .. image:: /_static/img/envs/cylinder_easy.png
          :alt: Cylinder Easy
          :width: 100%
     - .. image:: /_static/img/envs/cylinder_medium.png
          :alt: Cylinder Medium
          :width: 100%
     - .. image:: /_static/img/envs/cylinder_hard.png
          :alt: Cylinder Hard
          :width: 100%

See :doc:`/api/cylinder` for the full environment list and API reference.

Flow Past Airfoil
-----------------

Flow past a NACA 0012 airfoil with the goal of improving aerodynamic efficiency by
maximizing the lift-to-drag ratio.

.. list-table::
   :widths: 33 33 33
   :header-rows: 1

   * - Easy (Re = 1×10³)
     - Medium (Re = 3×10³)
     - Hard (Re = 5×10³)
   * - .. image:: /_static/img/envs/airfoil_easy.png
          :alt: Airfoil Easy
          :width: 100%
     - .. image:: /_static/img/envs/airfoil_medium.png
          :alt: Airfoil Medium
          :width: 100%
     - .. image:: /_static/img/envs/airfoil_hard.png
          :alt: Airfoil Hard
          :width: 100%

See :doc:`/api/airfoil` for the full environment list and API reference.

Rayleigh-Bénard Convection (RBC)
---------------------------------

Thermal convection driven by buoyancy between a heated bottom plate and a cooled top
plate. Heaters along the bottom wall are controlled to enhance heat transfer.

.. list-table::
   :widths: 33 33 33
   :header-rows: 1

   * - Easy (Ra = 8×10⁴)
     - Medium (Ra = 4×10⁵)
     - Hard (Ra = 8×10⁵)
   * - .. image:: /_static/img/envs/rbc_easy.png
          :alt: RBC Easy
          :width: 100%
     - .. image:: /_static/img/envs/rbc_medium.png
          :alt: RBC Medium
          :width: 100%
     - .. image:: /_static/img/envs/rbc_hard.png
          :alt: RBC Hard
          :width: 100%

See :doc:`/api/rbc` for the full environment list and API reference.

Turbulent Channel Flow (TCF)
-----------------------------

3D wall-bounded turbulent channel flow with blowing/suction actuation on the bottom or
both walls to reduce skin-friction drag.

.. list-table::
   :widths: 33 33 33
   :header-rows: 1

   * - Easy (Re = 180)
     - Medium (Re = 330)
     - Hard (Re = 550)
   * - .. image:: /_static/img/envs/tcf_easy.png
          :alt: TCF Easy
          :width: 100%
     - .. image:: /_static/img/envs/tcf_medium.png
          :alt: TCF Medium
          :width: 100%
     - .. image:: /_static/img/envs/tcf_hard.png
          :alt: TCF Hard
          :width: 100%

See :doc:`/api/tcf` for the full environment list and API reference.
