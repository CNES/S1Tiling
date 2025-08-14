.. include:: <isoamsa.txt>

.. _dataflow-lia:

.. index:: Normlim data flow

======================================================================
:math:`σ^0_{T}` Normlim data flow
======================================================================

.. contents:: Contents:
   :local:
   :depth: 3

Two data flows are possible:

- with :ref:`S1LIAMap` only LIA maps are produced,
- with :ref:`S1Processor` LIA maps are produced if not found, then
  :math:`σ^0_{T}` NORMLIM calibrated and orthorectified files are produced.

Normlim global processing
-------------------------

The following processing described is the new processing starting from S1Tiling
v1.2.

S1 Tiling processes by looping on all required S2 tiles within the time range.

For each S2 tile,

1. It :ref:`downloads S1 images <downloading_s1>` (S1Processor scenario only).
   The download is done on condition the images are not already available
   in :ref:`input data cache <paths.s1_images>`.

2. It :ref:`downloads precise orbit files (EOF) <downloading_eof>` that cover
   the specified time range and that match the specified S1 platform.
   The download is done on condition the requested relative orbit number is not
   found in the EOF files already available in the :ref:`eof data cache
   <paths.eof_dir>`.

3. Then, it makes sure the :ref:`associated sine LIA map <lia-files>`
   exists (all scenarios), it:

   1. :ref:`prepares a VRT <prepare_VRT_s2-proc>` of the DEM files that
      cover the S2 image,
   2. :ref:`projects DEM information <project_dem_to_s2-proc>` (from
      the VRT) on the S2 geometry,
   3. :ref:`project GEOID information <project_geoid_to_s2-proc>` on
      the S2 geometry,
   4. :ref:`sums both elevation information
      <sum_dem_geoid_on_s2-proc>` on the S2 geometry,
   5. produces a `image` of ECEF coordinates for the ground points and their
      associated satellite positions in the S2 geometry,
   6. :ref:`computes the normal <compute_normals-proc>` of each ground point,
   7. :ref:`computes the sine LIA map <compute_lia-proc>` of each ground point.

3. Then, for each polarisation (S1Processor scenario only),

   1. It :ref:`calibrates with β° LUT <calibration-proc>`, :ref:`cuts
      <cutting-proc>` and :ref:`orthorectifies <orthorectification-proc>` all
      the S1 images onto the S2 grid,
   2. It :ref:`superposes (concatenates) <concatenation-proc>` the
      orthorectified images into a single S2 tile,
   3. It :ref:`multiplies <apply_lia-proc>` the β° orthorectified image with
      the sine LIA map.


As with the main dataflow for all other calibrations (β°, γ°, or σ°), these
tasks are done :ref:`in parallel <parallelization>` in respect of all the
dependencies.

.. _lia-processings:

LIA specific processings
------------------------

.. graphviz::
    :name: graph_LIA
    :caption: Tasks for processing 33NWC and 33NWB with NORMLIM calibration -- v1.1 workflow
    :alt: Complete task flow for processing 33NWC and 33NWB with NORMLIM calibration
    :align: center

     digraph "sphinx-ext-graphviz" {
         rankdir="LR";
         graph [fontname="Verdana", fontsize="12"];
         node [fontname="Verdana", fontsize="12", shape="note", target="_top", style=filled];
         edge [fontname="Sans", fontsize="9"];

         # =====[ Inputs nodes
         raw_d1_t1t2 [label="Raw d1 t1-t2",   href="configuration.html#paths-s1-images", shape="folder", fillcolor=green]
         raw_d1_t2t3 [label="Raw d1 t2-t3",   href="configuration.html#paths-s1-images", shape="folder", fillcolor=green]

         raw_d2_t1t2 [label="Raw d2 t1'-t2'", href="configuration.html#paths-s1-images", shape="folder", fillcolor=green]
         raw_d2_t2t3 [label="Raw d2 t2'-t3'", href="configuration.html#paths-s1-images", shape="folder", fillcolor=green]

         raw_dn_t1t2 [label="Raw dn t1'-t2'", href="configuration.html#paths-s1-images", shape="folder", fillcolor=green]
         raw_dn_t2t3 [label="Raw dn t2'-t3'", href="configuration.html#paths-s1-images", shape="folder", fillcolor=green]

         eof_dx      [label="EOF dn",         href="configuration.html#paths-eof-dir",   shape="doublecircle", fillcolor=cyan]

         # =====[ Classic workflow
         # β° calibrated + orthorectified nodes
         o_nwb_d1_t1 [label="Orthorectified β° 33NWB d1 t1", href="files.html#orthorectified-files", fillcolor=lightyellow]
         o_nwb_d1_t2 [label="Orthorectified β° 33NWB d1 t2", href="files.html#orthorectified-files", fillcolor=lightyellow]

         o_nwb_d2_t1 [label="Orthorectified β° 33NWB d2 t'1", href="files.html#orthorectified-files", fillcolor=lightyellow]
         o_nwb_d2_t2 [label="Orthorectified β° 33NWB d2 t'2", href="files.html#orthorectified-files", fillcolor=lightyellow]

         o_nwb_dn_t1 [label="Orthorectified β° 33NWB dn t'1", href="files.html#orthorectified-files", fillcolor=lightyellow]
         o_nwb_dn_t2 [label="Orthorectified β° 33NWB dn t'2", href="files.html#orthorectified-files", fillcolor=lightyellow]

         # Concatenated β° calibrated + orthorectified nodes
         nwb_d1_b0 [label="S2 β° 33NWB d1", href="files.html#full-s2-tiles", fillcolor=pink]
         nwb_d2_b0 [label="S2 β° 33NWB d2", href="files.html#full-s2-tiles", fillcolor=pink]
         nwb_dn_b0 [label="S2 β° 33NWB dn", href="files.html#full-s2-tiles", fillcolor=pink]

         # Classic workflow up to concatenated β° calibrated + orthorectified nodes
         raw_d1_t1t2 -> o_nwb_d1_t1 [label="β° cal | noise | cut | ortho"];
         raw_d1_t2t3 -> o_nwb_d1_t2 [label="β° cal | noise | cut | ortho"];
         raw_d2_t1t2 -> o_nwb_d2_t1 [label="β° cal | noise | cut | ortho"];
         raw_d2_t2t3 -> o_nwb_d2_t2 [label="β° cal | noise | cut | ortho"];
         raw_dn_t1t2 -> o_nwb_dn_t1 [label="β° cal | noise | cut | ortho"];
         raw_dn_t2t3 -> o_nwb_dn_t2 [label="β° cal | noise | cut | ortho"];

         o_nwb_d1_t1 -> nwb_d1_b0 [label="concatenation"];
         o_nwb_d1_t2 -> nwb_d1_b0 [label="concatenation"];
         o_nwb_d2_t1 -> nwb_d2_b0 [label="concatenation"];
         o_nwb_d2_t2 -> nwb_d2_b0 [label="concatenation"];
         o_nwb_dn_t1 -> nwb_dn_b0 [label="concatenation"];
         o_nwb_dn_t2 -> nwb_dn_b0 [label="concatenation"];

         # ===================================
         # =====[ LIA workflow
         vrt_nwb       [label="DEM VRT 33NWB",                 href="files.html#dem_vrt_on_s2-files",           fillcolor=palegoldenrod];

         DEM_on_S2     [label="DEM projected on 33NWB",        href="files.html#dem-on-s2-files",         fillcolor=palegoldenrod];
         heights_on_S2 [label="geoid|DEM+geoid on 33NWB",      href="files.html#height-on-s2-files",      fillcolor=palegoldenrod];
         xyz_d1_t1     [label="ground+satellite XYZ on 33NWB", href="files.html#ground-and-sat-s2-files", fillcolor=palegoldenrod];
         normals_on_S2 [label="ground normals on 33NWB",       fillcolor=palegoldenrod];

         nwb_lia       [label="sin(LIA) on 33NWB",             href="files.html#lia-files", fillcolor="gold" ]

         mult_d1       [label="X", shape="circle"]
         mult_d2       [label="X", shape="circle"]
         mult_dn       [label="X", shape="circle"]

         nwb_d1        [label="S2 σ° NORMLIM 33NWB d1", href="files.html#full-s2-tiles", fillcolor=lightblue];
         nwb_d2        [label="S2 σ° NORMLIM 33NWB d2", href="files.html#full-s2-tiles",fillcolor=lightblue];
         nwb_dn        [label="S2 σ° NORMLIM 33NWB dn", href="files.html#full-s2-tiles",fillcolor=lightblue];

         vrt_nwb       -> DEM_on_S2;
         DEM_on_S2     -> heights_on_S2;

         eof_dx        -> xyz_d1_t1;
         heights_on_S2 -> xyz_d1_t1;
         xyz_d1_t1     -> normals_on_S2;
         normals_on_S2 -> nwb_lia;
         xyz_d1_t1     -> nwb_lia;

         nwb_lia   -> mult_d1;
         nwb_lia   -> mult_d2;
         nwb_lia   -> mult_dn;
         nwb_d1_b0 -> mult_d1;
         nwb_d2_b0 -> mult_d2;
         nwb_dn_b0 -> mult_dn;

         mult_d1 -> nwb_d1;
         mult_d2 -> nwb_d2;
         mult_dn -> nwb_dn;

         # =====[ Align
         {
             rank = same ;
             vrt_nwb raw_d1_t1t2 raw_d1_t2t3 raw_d2_t1t2 raw_d2_t2t3 raw_dn_t1t2 raw_dn_t2t3 eof_dx
             edge[ style=invis];
             eof_dx -> vrt_nwb -> raw_d1_t1t2 -> raw_d1_t2t3 -> raw_d2_t1t2 -> raw_d2_t2t3 -> raw_dn_t1t2 -> raw_dn_t2t3

         }
     }

.. _downloading_eof:
.. index:: downloading

Downloading of EOF precise orbit files
++++++++++++++++++++++++++++++++++++++

Precise orbit files will be searched in :ref:`[PATHS].eof_dir <paths.eof_dir>`
in NORMLIM related scenarios.

An EOF file is a match when:

- it matches the requested S1 :ref:`platform <datasource.platform_list>`,
- and `Orbit State Vectors` matching the requested :ref:`relative orbit number
  <datasource.relative_orbit_list>` are found in the file -- actually a ± 1
  margin is taken into account in order to handle tiles around the ANX when the
  orbit number changes.

If no matching EOF file is found, then on the condition
:ref:`[Datasource].download <datasource.download>` is ``True``, all EOF files
matching the platform and within the requested :ref:`time range
<datasource.first_date>` will be downloaded.
EOF files are downloaded with `sentineleof
<https://github.com/scottstanie/sentineleof>`_ on Copernicus Dataspace or on
Earthdata. See the FAQ regarding how credentials should be configured:
“:ref:`faq.eof`”.

Downloaded files are stored into the directory specified by
:ref:`[Paths].eof_dir <Paths.eof_dir>` option. If the directory doesn't exist,
it's created on the fly.

.. _prepare_VRT_s2-proc:
.. index:: Agglomerate DEMs over S2 tile

Agglomerate DEM files in a VRT that covers S2 footprint
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

:Inputs:      All DEM files that intersect the target S2 tile
:Output:      A :ref:`VRT file <dem_vrt_on_s2-files>`
:Function:    :func:`osgeo.gdal.BuildVRT`
:StepFactory: :class:`s1tiling.libs.otbwrappers.AgglomerateDEMOnS2`

All DEM files that intersect the target S2 tile are agglomerated in a :ref:`VRT
file <dem_vrt_on_s2-files>`.


.. _project_dem_to_s2-proc:
.. index:: Project DEM on S2 tile

Project DEM on S2 tile
++++++++++++++++++++++

:Inputs:         The :ref:`DEM VRT file <dem_vrt_on_s2-files>` over the S2 tile
:Output:         The :ref:`DEM projected on S2 tile <dem_on_S2-files>`
:Program:        :external:std:doc:`programs/gdalwarp`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.ProjectDEMToS2Tile`

This step projects the :ref:`DEM VRT file <dem_vrt_on_s2-files>` on the S2 geometry.

.. _project_geoid_to_s2-proc:
.. index:: Project GEOID on S2 tile

Project GEOID on S2 tile
++++++++++++++++++++++++

:Inputs:         - The :ref:`DEM projected on S2 tile <dem_on_S2-files>` as
                   reference
                 - The :ref:`GEOID file <paths.geoid_file>`
:Output:         None: chained in memory with :ref:`Height computation
                 <sum_dem_geoid_on_s2-proc>`
:OTBApplication: :external+OTB:std:doc:`OTB Superimpose
                 <Applications/app_Superimpose>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.ProjectGeoidToS2Tile`

This step projects the :ref:`GEOID file <paths.geoid_file>` on the S2 geometry.

.. _sum_dem_geoid_on_s2-proc:
.. index:: Compute full height elevation on S2

Compute full height elevation on S2
+++++++++++++++++++++++++++++++++++

:Inputs:         - The :ref:`DEM projected on S2 tile <dem_on_S2-files>`
                 - The projected GEOID on S2 tile -- chained in memory
                   from :ref:`GEOID projection step
                   <project_geoid_to_s2-proc>`
:Output:         The :ref:`Height projected on S2 tile
                 <height_on_s2-files>`
:OTBApplication: :external+OTB:std:doc:`OTB BandMath
                 <Applications/app_BandMath>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.SumAllHeights`

This step sums both DEM and GEOID information projected in S2 tile geometry.

.. _sardemproject_s2-proc:
.. index:: Project SAR coordinates onto S2 tile

Compute ECEF ground and satellite positions on S2
+++++++++++++++++++++++++++++++++++++++++++++++++

:Inputs:         - A :ref:`matching EOF file <downloading_eof>`
                 - The :ref:`height information <height_on_s2-files>` of the S2
                   tile.
:Output:         :ref:`ECEF Ground and satellite positions
                 <ground_and_sat_s2-files>` on the S2 tile.
:OTBApplication: `SARComputeGroundAndSatPositionsOnDEM
                 <https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0>`_
                 (developed for the purpose of this project)

                 .. note::
                     Beware, this OTB application isn't distributed with OTB
                     yet. It has to be installed specifically on your machine.
                     It will be already installed in the :ref:`docker images
                     <docker>` though.
:StepFactory:    :class:`s1tiling.libs.otbwrappers.ComputeGroundAndSatPositionsOnDEMFromEOF`

This step computes the ground positions of the pixels in the S2 geometry, and
searches their associated zero doppler to also issue the coordinates of the
SAR sensor.

All coordinates are stored in `ECEF
<https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system>`_.

.. _compute_normals-proc:
.. index:: Normals computation

Normals computation
+++++++++++++++++++

:Input:          A :ref:`XYZ Cartesian coordinates file <xyz-files>`
:Output:         None: chained in memory with :ref:`LIA maps computation <compute_lia-proc>`
:OTBApplication: `ExtractNormalVector OTB application
                 <https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0>`_
                 (developed for the purpose of this project)

                 .. note::
                     Beware, this OTB application isn't distributed with OTB
                     yet. It has to be installed specifically on your machine.
                     It will be already installed in the :ref:`docker images
                     <docker>` though.
:StepFactory:    :class:`s1tiling.libs.otbwrappers.ComputeNormalsOnS2` (or
                 :class:`s1tiling.libs.otbwrappers.ComputeNormalsOnS1` in the
                 deprecated workflow)

This step computes the normal vectors to the ground, in the MGRS S2 geometry.


.. _compute_lia-proc:
.. index:: Compute LIA maps

LIA maps computation
++++++++++++++++++++

:Input:          - A :ref:`XYZ Cartesian coordinates file
                   <ground_and_sat_S2-files>` of ground positions, and of
                   satellite positions (or that
                   contains satellite trajectory -- deprecated workflow)
                 - and the associated normals, chained in memory from
                   :ref:`Normals computation <compute_normals-proc>`
:Output:         :ref:`Local Incidence Angle map, and sine LIA map
                 <lia-files>` (or :ref:`the equivalent <lia-s1-files>` in the
                 deprecated workflow)
:OTBApplication: `SARComputeIncidenceAngle OTB application
                 <https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0>`_
                 (developed for the purpose of this project)

                 .. note::
                     Beware, this OTB application isn't distributed with OTB
                     yet. It has to be installed specifically on your machine.
                     It will be already installed in the :ref:`docker images
                     <docker>` though.
:StepFactory:    :class:`s1tiling.libs.otbwrappers.ComputeLIAOnS2` (or
                 :class:`s1tiling.libs.otbwrappers.ComputeLIAOnS1` in the
                 deprecated workflow)

It computes the :ref:`Local Incidence Angle map, and sine LIA map
<lia-files>` between the ground normal projected in range plane
:math:`\overrightarrow{n}` (plane defined by S, T, and Earth's centre) and
:math:`\overrightarrow{TS}` -- where T is the target point on Earth's surface,
and S the SAR sensor position.


.. _apply_lia-proc:
.. index:: Application of LIA maps

Application of LIA maps to β° calibrated S2 images
++++++++++++++++++++++++++++++++++++++++++++++++++

:Inputs:         - The :ref:`sine LIA map file <lia-files>` associated to the
                   S2 grid
                 - A β° calibrated, cut and orthorectified image on the S2 grid
:Output:         :ref:`final S2 tiles <full-S2-tiles>`, :math:`σ^0_{T}`
                 calibrated
:OTBApplication: :external+OTB:std:doc:`BandMath <Applications/app_BandMath>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.ApplyLIACalibration`

This final step multiplies the sine LIA map (in S2 grid geometry) with β°
calibrated files orthorectified on the S2 grid.


Normlim deprecated global processing
------------------------------------

The following processing was the one supported in v1.0 of S1Tiling.

S1 Tiling processes by looping on all required S2 tiles within the time range.

For each S2 tile,

1. It :ref:`downloads <downloading_s1>` the necessary S1 images that intersect
   the S2 tile, within the specified time range, that are not already available
   in :ref:`input data cache <paths.s1_images>`
   (all scenarios)

2. Then, it makes sure the :ref:`associated sine LIA map <lia-files>` exists
   (all scenarios),

   0. It selects a pair of :ref:`input S1 images <paths.s1_images>` that
      intersect the S2 tile,
   1. For each :ref:`input S1 image <paths.s1_images>`, It:

       1. :ref:`prepares a VRT <prepare_VRT_s1-proc>` of the DEM files that
          cover the image,
       2. :ref:`projects <sardemproject_s1-proc>` the coordinates of the input
          S1 image onto the geometry of the VRT,
       3. :ref:`projects <sarcartesianmeanestimation>` back the Cartesian
          coordinates of each ground point in the origin S1 image geometry,
       4. :ref:`computes the normal <compute_normals-proc>` of each ground point,
       5. :ref:`computes the sine LIA map <compute_lia-proc>` of each ground point,
       6. :ref:`orthorectifies the sine LIA map <ortho_lia-proc>` to the S2 tile.

   2. It :ref:`concatenates <concat_lia-proc>` both files into a single sine
      LIA map for the S2 tile.

3. Then, for each polarisation (S1Processor scenario only),

   1. It :ref:`calibrates with β° LUT <calibration-proc>`, :ref:`cuts
      <cutting-proc>` and :ref:`orthorectifies <orthorectification>` all the S1
      images onto the S2 grid,
   2. It :ref:`superposes (concatenates) <concatenation-proc>` the
      orthorectified images into a single S2 tile,
   3. It :ref:`multiplies <apply_lia-proc>` the β° orthorectified image with
      the sine LIA map.


As with the main dataflow for all other calibrations (β°, γ°, or σ°), these
tasks are done :ref:`in parallel <parallelization>` in respect of all the
dependencies.



.. _lia-deprecated-processings:

LIA specific deprecated processings
-----------------------------------

.. graphviz::
    :name: graph_LIA_v1
    :caption: Tasks for processing 33NWC and 33NWB with NORMLIM calibration -- v1.0 deprecated workflow
    :alt: Complete task flow for processing 33NWC and 33NWB with NORMLIM calibration
    :align: center

     digraph "sphinx-ext-graphviz" {
         rankdir="LR";
         graph [fontname="Verdana", fontsize="12"];
         node [fontname="Verdana", fontsize="12", shape="note", target="_top", style=filled];
         edge [fontname="Sans", fontsize="9"];

         # =====[ Inputs nodes
         raw_d1_t1t2 [label="Raw d1 t1-t2", href="configuration.html#paths-s1-images", shape="folder", fillcolor=green]
         raw_d1_t2t3 [label="Raw d1 t2-t3", href="configuration.html#paths-s1-images", shape="folder", fillcolor=green]

         raw_d2_t1t2 [label="Raw d2 t1'-t2'", href="configuration.html#paths-s1-images", shape="folder", fillcolor=green]
         raw_d2_t2t3 [label="Raw d2 t2'-t3'", href="configuration.html#paths-s1-images", shape="folder", fillcolor=green]

         raw_dn_t1t2 [label="Raw dn t1'-t2'", href="configuration.html#paths-s1-images", shape="folder", fillcolor=green]
         raw_dn_t2t3 [label="Raw dn t2'-t3'", href="configuration.html#paths-s1-images", shape="folder", fillcolor=green]

         { rank = same ;  raw_d1_t1t2 raw_d1_t2t3 raw_d2_t1t2 raw_d2_t2t3 raw_dn_t1t2 raw_dn_t2t3}

         # =====[ Classic workflow
         o_nwb_d1_t1 [label="Orthorectified β° 33NWB d1 t1", href="files.html#orthorectified-files", fillcolor=lightyellow]
         o_nwb_d1_t2 [label="Orthorectified β° 33NWB d1 t2", href="files.html#orthorectified-files", fillcolor=lightyellow]

         o_nwb_d2_t1 [label="Orthorectified β° 33NWB d2 t'1", href="files.html#orthorectified-files", fillcolor=lightyellow]
         o_nwb_d2_t2 [label="Orthorectified β° 33NWB d2 t'2", href="files.html#orthorectified-files", fillcolor=lightyellow]

         o_nwb_dn_t1 [label="Orthorectified β° 33NWB dn t'1", href="files.html#orthorectified-files", fillcolor=lightyellow]
         o_nwb_dn_t2 [label="Orthorectified β° 33NWB dn t'2", href="files.html#orthorectified-files", fillcolor=lightyellow]

         # Concatenated β° calibrated + orthorectified nodes
         nwb_d1_b0 [label="S2 β° 33NWB d1", href="files.html#full-s2-tiles", fillcolor=pink]
         nwb_d2_b0 [label="S2 β° 33NWB d2", href="files.html#full-s2-tiles", fillcolor=pink]
         nwb_dn_b0 [label="S2 β° 33NWB dn", href="files.html#full-s2-tiles", fillcolor=pink]

         # Classic workflow up to concatenated β° calibrated + orthorectified nodes
         raw_d1_t1t2 -> o_nwb_d1_t1 [label="β° cal | noise | cut | ortho"];
         raw_d1_t2t3 -> o_nwb_d1_t2 [label="β° cal | noise | cut | ortho"];
         raw_d2_t1t2 -> o_nwb_d2_t1 [label="β° cal | noise | cut | ortho"];
         raw_d2_t2t3 -> o_nwb_d2_t2 [label="β° cal | noise | cut | ortho"];
         raw_dn_t1t2 -> o_nwb_dn_t1 [label="β° cal | noise | cut | ortho"];
         raw_dn_t2t3 -> o_nwb_dn_t2 [label="β° cal | noise | cut | ortho"];

         o_nwb_d1_t1 -> nwb_d1_b0 [label="concatenation"];
         o_nwb_d1_t2 -> nwb_d1_b0 [label="concatenation"];
         o_nwb_d2_t1 -> nwb_d2_b0 [label="concatenation"];
         o_nwb_d2_t2 -> nwb_d2_b0 [label="concatenation"];
         o_nwb_dn_t1 -> nwb_dn_b0 [label="concatenation"];
         o_nwb_dn_t2 -> nwb_dn_b0 [label="concatenation"];

         # ===================================
         # ====[ LIA workflow
         vrt_d1_t1t2 [label="DEM VRT d1 t1-t2", fillcolor=palegoldenrod];
         vrt_d1_t2t3 [label="DEM VRT d1 t2-t3", fillcolor=palegoldenrod];

         S1_on_DEM_d1_t1t2 [label="S1 on DEM d1 t1-t2", fillcolor=palegoldenrod];
         S1_on_DEM_d1_t2t3 [label="S1 on DEM d1 t2-t3", fillcolor=palegoldenrod];

         xyz_d1_t1t2 [label="XYZ d1 t1-t2", fillcolor=palegoldenrod];
         xyz_d1_t2t3 [label="XYZ d1 t2-t3", fillcolor=palegoldenrod];

         # LIA on S1
         lia_d1_t1t2 [label="sin(LIA) d1 t1-t2", fillcolor=palegoldenrod];
         lia_d1_t2t3 [label="sin(LIA) d1 t2-t3", fillcolor=palegoldenrod];

         # LIA orthorectified on S2
         o_lia_d1_t1 [label="sin(LIA) d1 t1 on 33NWB", fillcolor=palegoldenrod];
         o_lia_d1_t2 [label="sin(LIA) d1 t2 on 33NWB", fillcolor=palegoldenrod];
         # LIA concatenated a selected (best coverage)
         nwb_lia     [label="sin(LIA) on 33NWB", fillcolor=gold];

         nwb_d1      [label="S2 σ° NORMLIM 33NWB d1", fillcolor=lightblue];
         nwb_d2      [label="S2 σ° NORMLIM 33NWB d2", fillcolor=lightblue];
         nwb_dn      [label="S2 σ° NORMLIM 33NWB dn", fillcolor=lightblue];

         mult_d1     [label="X", shape="circle"]
         mult_d2     [label="X", shape="circle"]
         mult_dn     [label="X", shape="circle"]

         raw_d1_t1t2 -> vrt_d1_t1t2 [label=""];
         raw_d1_t2t3 -> vrt_d1_t2t3 [label=""];

         vrt_d1_t1t2 -> S1_on_DEM_d1_t1t2;
         vrt_d1_t2t3 -> S1_on_DEM_d1_t2t3;
         raw_d1_t1t2 -> S1_on_DEM_d1_t1t2;
         raw_d1_t2t3 -> S1_on_DEM_d1_t2t3;

         vrt_d1_t1t2       -> xyz_d1_t1t2;
         vrt_d1_t2t3       -> xyz_d1_t2t3;
         raw_d1_t1t2       -> xyz_d1_t1t2;
         raw_d1_t2t3       -> xyz_d1_t2t3;
         S1_on_DEM_d1_t1t2 -> xyz_d1_t1t2;
         S1_on_DEM_d1_t2t3 -> xyz_d1_t2t3;

         xyz_d1_t1t2 -> lia_d1_t1t2 [label=""];
         xyz_d1_t2t3 -> lia_d1_t2t3 [label=""];

         lia_d1_t1t2 -> o_lia_d1_t1 [label="ortho"];
         lia_d1_t2t3 -> o_lia_d1_t2 [label="ortho"];

         o_lia_d1_t1 -> nwb_lia [label="concatenation"];
         o_lia_d1_t2 -> nwb_lia [label="concatenation"];

         nwb_lia   -> mult_d1;
         nwb_lia   -> mult_d2;
         nwb_lia   -> mult_dn;
         nwb_d1_b0 -> mult_d1;
         nwb_d2_b0 -> mult_d2;
         nwb_dn_b0 -> mult_dn;

         mult_d1 -> nwb_d1;
         mult_d2 -> nwb_d2;
         mult_dn -> nwb_dn;
     }


.. _prepare_VRT_s1-proc:
.. index:: Agglomerate DEM

Agglomerate DEM files in a VRT that covers S1 footprint (LIA) -- deprecated
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

:Inputs:      All DEM files that intersect an original :ref:`input S1 image <paths.s1_images>`
:Output:      A :ref:`VRT file <dem_vrt_on_s2-files>`
:Function:    :func:`osgeo.gdal.BuildVRT`
:StepFactory: :class:`s1tiling.libs.otbwrappers.AgglomerateDEMOnS1`

All DEM files that intersect an original :ref:`input S1 image
<paths.s1_images>` are agglomerated in a :ref:`VRT file <dem_vrt_on_s2-files>`.


.. _sardemproject_s1-proc:
.. index:: Project SAR coordinates onto DEM

Project SAR coordinates onto DEM (LIA) -- deprecated
++++++++++++++++++++++++++++++++++++++++++++++++++++

:Inputs:         - An original :ref:`input S1 image <paths.s1_images>` (geometry)
                 - The associated :ref:`VRT file <dem_vrt_on_s2-files>`
:Output:         A :ref:`SAR DEM projected file <S1_on_dem-files>`
:OTBApplication: :external:std:doc:`Our patched version of DiapOTB
                 SARDEMProjection <Applications/app_SARDEMProjection>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.SARDEMProjection`

This step projects the coordinates of original :ref:`input S1 image
<paths.s1_images>` in the geometry of the DEM VRT file.


.. _sarcartesianmeanestimation-proc:
.. index:: Project XYZ coordinates onto SAR

Project XYZ coordinates onto SAR -- deprecated
++++++++++++++++++++++++++++++++++++++++++++++

:Inputs:         - An original :ref:`input S1 image <paths.s1_images>` (geometry)
                 - The associated :ref:`VRT file <dem_vrt_on_s2-files>`
                 - The associated :ref:`SAR DEM projected file <S1_on_dem-files>`
:Output:         A :ref:`XYZ Cartesian coordinates file <xyz-files>`
:OTBApplication: :external:std:doc:`Our patched version of DiapOTB
                 SARCartesianMeanEstimation
                 <Applications/app_SARCartesianMeanEstimation>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.SARCartesianMeanEstimation`

This step estimates the XYZ Cartesian coordinates on the ground in the geometry
of the original :ref:`input S1 image <paths.s1_images>`.


.. _ortho_lia-proc:
.. index:: Orthorectification of LIA maps

Orthorectification of LIA maps -- deprecated
++++++++++++++++++++++++++++++++++++++++++++

:Inputs:      A :ref:`Sine Local Incidence Angle map, and an optional degrees
              LIA map <lia-s1-files>` in the original S1 image geometry
:Output:      The associated :ref:`LIA map file(s) <lia-s2-half-files>`
              orthorectified on the target S2 tile.
:OTBApplication: :external+OTB:std:doc:`Orthorectification
                 <Applications/app_OrthoRectification>`
:StepFactory: :class:`s1tiling.libs.otbwrappers.OrthoRectifyLIA`

This steps orthorectifies the LIA map image(s) in S1 geometry to S2 grid.

It uses the following parameters from the request configuration file:

- :ref:`[Processing].orthorectification_gridspacing
  <Processing.orthorectification_gridspacing>`
- :ref:`[Processing].orthorectification_interpolation_method
  <Processing.orthorectification_interpolation_method>`
- :ref:`[Paths].dem_dir <paths.dem_dir>`
- :ref:`[Paths].geoid_file <paths.geoid_file>`


.. _concat_lia-proc:
.. index:: Concatenation of LIA maps

Concatenation of LIA maps -- deprecated
+++++++++++++++++++++++++++++++++++++++

:Inputs:         A pair of :ref:`LIA map files <lia-s2-half-files>` (sines or
                 degrees) orthorectified on the target S2 tile.
:Output:         The :ref:`LIA map file(s) <lia-files>` associated to the S2 grid
:OTBApplication: :external+OTB:std:doc:`Synthetize <Applications/app_Synthetize>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.ConcatLIA`

This step merges all the images of the orthorectified S1 LIA maps on a given S2
grid. As all orthorectified images are almost exclusive, they are concatenated
by taking the first non-null pixel.


.. _lia-data-caches:
.. index:: Data caches (LIA)

LIA specific data caches
------------------------

As with main dataflow, two kinds of data are cached, but only one is regularly
cleaned-up by S1 Tiling. The other kind is left along as the software cannot
really tell whether they could be reused later on or not.

.. important:: This means that you may have to regularly clean up this space.
