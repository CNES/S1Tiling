.. include:: <isoamsa.txt>

.. _dataflow-eia:

.. index:: Incidence Angle data flow

======================================================================
Incidence Angle data flow
======================================================================

.. contents:: Contents:
   :local:
   :depth: 3

This dataflow is implemented in :ref:`S1IAMap program <S1IAMap>`.

S1 Tiling processes by looping on all required S2 tiles. For each S2 tile:

1. It :ref:`downloads precise orbit files (EOF) <downloading_eof>` that cover
   the specified time range and that match the specified S1 platform.
   The download is done on condition the requested relative orbit number is not
   found in the EOF files already available in the :ref:`eof data cache
   <paths.eof_dir>`.

2. Then, it makes sure the requested :ref:`associated IA maps <ia-files>`
   exist, it:

   1. :ref:`produces an image of ECEF coordinates for the ellipsoid surface
      points and their associated satellite positions
      <compute_wgs4_xyz_n_sat_s2-proc>` in the S2 geometry,
   2. :ref:`computes the normal <compute_normals_on_ellipsoid-proc>` of each
      ellipsoid surface point,
   3. :ref:`computes the requested IA maps <compute_eia-proc>` of each point.


.. _eia-processings:

Incidence Angle specific processings
------------------------------------

.. graphviz::
    :name: graph_IA
    :caption: Tasks for generating Ellipsoid Incidence Angle map on 31TCH, orbit 110
    :alt: Complete task flow for generating Ellipsoid Incidence Angle map on 31TCH, orbit 110
    :align: center

     digraph "sphinx-ext-graphviz" {
         rankdir="LR";
         graph [fontname="Verdana", fontsize="12"];
         node [fontname="Verdana", fontsize="12", shape="note", target="_top", style=filled];
         edge [fontname="Sans", fontsize="9"];

         # =====[ Inputs nodes
         eof_dx      [label="EOF (110)", href="configuration.html#paths-eof-dir", shape="doublecircle", fillcolor=cyan]

         # =====[ IA workflow
         xyz_d1_t1     [label="ellipsoid+satellite XYZ on 31TCH obt 110", href="files.html#wgs84-surface-and-sat-s2-files", fillcolor=palegoldenrod];
         normals_on_S2 [label="ellipsoid normals on 31TCH obt 110",                                   fillcolor=palegoldenrod];
         tch_ia        [label="sin(IA) on 31TCH obt 110",                 href="files.html#ia-files", fillcolor="gold" ]

         eof_dx        -> xyz_d1_t1;
         normals_on_S2 -> tch_ia;
         xyz_d1_t1     -> tch_ia;
     }



.. _compute_wgs4_xyz_n_sat_s2-proc:
.. index:: Project ellipsoid coordinates onto S2 tile

Compute ECEF Ellipsoid surface and satellite positions on S2
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

:Inputs:         A :ref:`matching EOF file <downloading_eof>`
:Output:         :ref:`ECEF WGS84 ellipsoid surface and satellite positions
                 <wgs84_surface_and_sat_s2-files>` on the S2 tile.
:OTBApplication: `SARComputeGroundAndSatPositionsOnEllipsoid
                 <https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0>`_
                 (developed for the purpose of this project)

                 .. note::
                     Beware, this OTB application isn't distributed with OTB
                     yet. It has to be installed specifically on your machine.
                     It will be already installed in the :ref:`docker images
                     <docker>` though.
:StepFactory:    :class:`s1tiling.libs.otbwrappers.ComputeGroundAndSatPositionsOnEllipsoid`

This step computes the WGS84 ellipsoid surface positions of the pixels in the
S2 geometry, and searches their associated zero doppler to also issue the
coordinates of the SAR sensor.

All coordinates are stored in `ECEF
<https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system>`__.

.. _compute_normals_on_ellipsoid-proc:
.. index:: Normals computation on Earth Ellipsoid

Normals computation on Earth Ellipsoid
++++++++++++++++++++++++++++++++++++++

:Input:          None
:Output:         None: chained in memory with :ref:`IA maps computation <compute_eia-proc>`
:OTBApplication: `ExtractNormalVectorToEllipsoid OTB application
                 <https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0>`__
                 (developed for the purpose of this project)

                 .. note::
                     Beware, this OTB application isn't distributed with OTB
                     yet. It has to be installed specifically on your machine.
                     It will be already installed in the :ref:`docker images
                     <docker>` though.
:StepFactory:    :class:`s1tiling.libs.otbwrappers.ComputeEllipsoidNormalsOnS2`

This step computes the normal vectors on Earth ellipsoid surface, in the MGRS
S2 geometry.

.. _ellipsoid_normals_computation-maths:
.. index:: Ellipsoid Normals computation

Details about Ellipsoid normal vector computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ellipsoid quadratic surface is defined in Cartesian coordinates as
(`wikipedia <https://en.wikipedia.org/wiki/Ellipsoid#Standard_equation>`__):

.. math:: \frac{x^2}{a^2} + \frac{y^2}{b^2} + \frac{z^2}{c^2} = 1
   :label: Ellipsoid Standard Equation

where :math:`a`, :math:`b`, and :math:`c` are the lengths of the semi-axes.

At point :math:`P \left( \begin{smallmatrix}X \\ Y \\ Z
\end{smallmatrix}\right)`, surface normal is parallel to:

.. math:: n(P) = \left( \begin{matrix} \frac{X}{a^2} \\ \frac{Y}{b^2} \\ \frac{Z}{c^2} \end{matrix} \right)
   :label: Ellipsoid Normal

The WGS84 datum surface is an `oblate spheroid
<https://en.wikipedia.org/wiki/Spheroid#Oblate_spheroids>`__ -- `wikipedia
<https://en.wikipedia.org/wiki/World_Geodetic_System#Definition>`__

This implies that in :eq:`Ellipsoid Standard Equation` that :math:`a = b`. Also,
:math:`c = a(1-f)` where :math:`f` is the flattening.
Both :math:`a` and :math:`1/f` are precisely defined for WGS84.

We also `know
<https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates>`__
that given latitude :math:`\phi` and longitude :math:`\lambda`, we can compute
the ECEF coordinates in the following way:

.. math::
   :label: ECEF Coordinates

   \begin{eqnarray}
   X & = & \left(N(\phi) + h\right) \cos{\phi} \cos{\lambda} \\
   Y & = & \left(N(\phi) + h\right) \cos{\phi} \sin{\lambda} \\
   Z & = & \left( (1 - f)^2 N(\phi) + h\right)\sin{\phi}
   \end{eqnarray}

Injecting :eq:`ECEF Coordinates` into :eq:`Ellipsoid Normal` for :math:`h=0`,
we obtain:

.. math::
   n(P) = \left( \begin{matrix} \frac{X}{a^2} \\ \frac{Y}{a^2} \\ \frac{Z}{a^2
   (1-f)^2} \end{matrix} \right)

.. math::
   n(P) = \frac{1}{a^2} \left(
   \begin{matrix}
   N(\phi) \cos{\phi} \cos{\lambda} \\
   N(\phi) \cos{\phi} \sin{\lambda} \\
   \left( \frac{(1 - f)^2}{(1 - f)^2} N(\phi) \right)\sin{\phi}
   \end{matrix}
   \right)

.. math::
   n(P) = \frac{N(\phi)}{a^2} \left(
   \begin{matrix}
   \cos{\phi} \cos{\lambda} \\
   \cos{\phi} \sin{\lambda} \\
   \sin{\phi}
   \end{matrix}
   \right)

As :math:`(\cos{\phi} \cos{\lambda})^2 + (\cos{\phi} \sin{\lambda})^2 +
(\sin{\phi})^2 = 1)`, the normal vector is:

.. math::
   :label: Normal vector to Earth Ellipsoid

   N(\phi,\lambda) = \left(
   \begin{matrix}
   \cos{\phi} \cos{\lambda} \\
   \cos{\phi} \sin{\lambda} \\
   \sin{\phi}
   \end{matrix}
   \right)

Which matches `Converting latitude/longitude to n-vector (wikipedia)
<https://en.wikipedia.org/wiki/N-vector#Converting_latitude/longitude_to_n-vector>`__

.. _compute_eia-proc:
.. index:: Compute Ellipsoid IA maps

Ellipsoid IA maps computation
+++++++++++++++++++++++++++++

:Input:          - A :ref:`XYZ Cartesian coordinates file
                   <wgs84_surface_and_sat_S2-files>` of ellipsoid surface
                   positions, and of satellite positions
                 - and the associated normals, chained in memory from
                   :ref:`WGS84 Normals computation
                   <compute_normals_on_ellipsoid-proc>`
:Output:         :ref:`Incidence Angle map, and/or cosine, sine and tangent IA
                 maps <ia-files>`
:OTBApplication: `SARComputeIncidenceAngle OTB application
                 <https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0>`__
                 (developed for the purpose of this project)

                 .. note::
                     Beware, this OTB application isn't distributed with OTB
                     yet. It has to be installed specifically on your machine.
                     It will be already installed in the :ref:`docker images
                     <docker>` though.
:StepFactory:    :class:`s1tiling.libs.otbwrappers.ComputeIAOnS2`

It computes the :ref:`Incidence Angle map, and/or cosine, sine and tangent IA
maps <ia-files>` between the ellipsoid surface normal projected in range
plane :math:`\overrightarrow{n}` (plane defined by S, T, and Earth's centre)
and :math:`\overrightarrow{TS}` -- where T is the target point on Earth's
surface, and S the SAR sensor position.
