.. include:: <isoamsa.txt>

.. _dataflow-lia:

.. index:: Normlim data flow

======================================================================
Normlim data flow
======================================================================

.. contents:: Contents:
   :local:
   :depth: 3

Two data flows are possibles:

- with :program:`S1LIAMap` only LIA maps are produced,
- with :program:`S1Processor` LIA maps are produced is not found, then
  :math:`σ^0_{RTC}` orthorectified files are produced.

NormLim global processing
-------------------------

S1 Tiling processes by looping on all required S2 tiles within the time range.

For each S2 tile,

1. It :ref:`downloads <downloading>` the necessary S1 images that intersect the
   S2 tile, within the specified time range, that are not already available in
   :ref:`input data cache <paths.s1_images>`
   (all scenarios)

2. Then, it makes sure the :ref:`associated sine LIA map <lia-files>` exists
   (all scenarios),

   0. It selects a pair of :ref:`input S1 images <paths.s1_images>` that
      intersect the S2 tile,
   1. For each :ref:`input S1 image <paths.s1_images>`

       1. It :ref:`prepares a VRT <prepare-VRT>` of the SRTM files that cover
          the image,
       2. It :ref:`projects <sardemproject>` the coordinates of the input S1
          image onto the geometry of the VRT,
       3. It :ref:`projects <sarcartesianmeanestimation>` back the cartesian
          coordinates of each ground point in the origin S1 image geometry,
       4. It :ref:`computes the normal <compute-normals>` of each ground point,
       5. It :ref:`computes the sine LIA map <compute-lia>` of each ground point,
       6. It :ref:`orthorectifies the sine LIA map <ortho-lia>` to the S2 tile

   2. It :ref:`concatenates <concat-lia>` both files into a single sine LIA map
      for the S2 tile.

3. Then, for each polarisation (S1Processor scenario only),

   1. It :ref:`calibrates with β° LUT <calibration>`, :ref:`cuts <cutting>` and
      :ref:`orthorectifies <orthorectification>` all the S1 images onto the S2
      grid,
   2. It :ref:`superposes (concatenates) <concatenation>` the orthorectified
      images into a single S2 tile,
   3. It :ref:`multiplies <apply-lia>` the β° orthorectified image with the
      sine LIA map.


As with the main dataflow for all other calibrations (β°, γ°, or σ°), these
tasks are done :ref:`in parallel <parallelization>` in respect of all the
dependencies.



.. _lia-processings:

LIA specific processings
------------------------

.. graphviz::
    :name: graph_LIA
    :caption: Tasks for processing 33NWC and 33NWB with NORMLIM calibration
    :alt: Complete task flow for processing 33NWC and 33NWB with NORMLIM calibration
    :align: center

     digraph "sphinx-ext-graphviz" {
         rankdir="LR";
         graph [fontname="Verdana", fontsize="12"];
         node [fontname="Verdana", fontsize="12", shape="note", target="_top", style=filled];
         edge [fontname="Sans", fontsize="9"];

         raw_d1_t1t2 [label="Raw d1 t1-t2", href="files.html#inputs", shape="folder", fillcolor=green]
         raw_d1_t2t3 [label="Raw d1 t2-t3", href="files.html#inputs", shape="folder", fillcolor=green]

         raw_d2_t1t2 [label="Raw d2 t1'-t2'", href="files.html#inputs", shape="folder", fillcolor=green]
         raw_d2_t2t3 [label="Raw d2 t2'-t3'", href="files.html#inputs", shape="folder", fillcolor=green]

         raw_dn_t1t2 [label="Raw dn t1'-t2'", href="files.html#inputs", shape="folder", fillcolor=green]
         raw_dn_t2t3 [label="Raw dn t2'-t3'", href="files.html#inputs", shape="folder", fillcolor=green]

         { rank = same ;  raw_d1_t1t2 raw_d1_t2t3 raw_d2_t1t2 raw_d2_t2t3 raw_dn_t1t2 raw_dn_t2t3}

         o_nwb_d1_t1 [label="Orthorectified β° 33NWB d1 t1", href="files.html#orthorectified-files", fillcolor=lightyellow]
         o_nwb_d1_t2 [label="Orthorectified β° 33NWB d1 t2", href="files.html#orthorectified-files", fillcolor=lightyellow]

         o_nwb_d2_t1 [label="Orthorectified β° 33NWB d2 t'1", href="files.html#orthorectified-files", fillcolor=lightyellow]
         o_nwb_d2_t2 [label="Orthorectified β° 33NWB d2 t'2", href="files.html#orthorectified-files", fillcolor=lightyellow]

         o_nwb_dn_t1 [label="Orthorectified β° 33NWB dn t'1", href="files.html#orthorectified-files", fillcolor=lightyellow]
         o_nwb_dn_t2 [label="Orthorectified β° 33NWB dn t'2", href="files.html#orthorectified-files", fillcolor=lightyellow]

         nwb_d1_b0 [label="S2 β° 33NWB d1", href="files.html#full-S2-tiles", fillcolor=pink]
         nwb_d2_b0 [label="S2 β° 33NWB d2", href="files.html#full-S2-tiles", fillcolor=pink]
         nwb_dn_b0 [label="S2 β° 33NWB dn", href="files.html#full-S2-tiles", fillcolor=pink]

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
         vrt_d1_t1t2 [label="DEM VRT d1 t1-t2", fillcolor=palegoldenrod];
         vrt_d1_t2t3 [label="DEM VRT d1 t2-t3", fillcolor=palegoldenrod];

         S1_on_DEM_d1_t1t2 [label="S1 on DEM d1 t1-t2", fillcolor=palegoldenrod];
         S1_on_DEM_d1_t2t3 [label="S1 on DEM d1 t2-t3", fillcolor=palegoldenrod];

         xyz_d1_t1t2 [label="XYZ d1 t1-t2", fillcolor=palegoldenrod];
         xyz_d1_t2t3 [label="XYZ d1 t2-t3", fillcolor=palegoldenrod];

         lia_d1_t1t2 [label="sin(LIA) d1 t1-t2", fillcolor=palegoldenrod];
         lia_d1_t2t3 [label="sin(LIA) d1 t2-t3", fillcolor=palegoldenrod];

         o_lia_d1_t1 [label="sin(LIA) d1 t1 on 33NWB", fillcolor=palegoldenrod];
         o_lia_d1_t2 [label="sin(LIA) d1 t2 on 33NWB", fillcolor=palegoldenrod];
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

         vrt_d1_t1t2 -> xyz_d1_t1t2;
         vrt_d1_t2t3 -> xyz_d1_t2t3;
         raw_d1_t1t2 -> xyz_d1_t1t2;
         raw_d1_t2t3 -> xyz_d1_t2t3;
         S1_on_DEM_d1_t1t2 -> xyz_d1_t1t2;
         S1_on_DEM_d1_t2t3 -> xyz_d1_t2t3;

         xyz_d1_t1t2 -> lia_d1_t1t2 [label=""];
         xyz_d1_t2t3 -> lia_d1_t2t3 [label=""];

         lia_d1_t1t2 -> o_lia_d1_t1;
         lia_d1_t2t3 -> o_lia_d1_t2;

         o_lia_d1_t1 -> nwb_lia;
         o_lia_d1_t2 -> nwb_lia;

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


.. _prepare-VRT:
.. index:: Agglomerate SRTM

Agglomerate SRTM files in a VRT
+++++++++++++++++++++++++++++++

:Inputs:      All SRTM files that intersect an original :ref:`input S1 image <paths.s1_images>`
:Output:      A :ref:`VRT file <dem-vrt-files>`
:Program:     :std:doc:`programs/gdalbuildvrt`
:StepFactory: :class:`s1tiling.libs.otbwrappers.AgglomerateDEM`

All SRTM files that intersect an original :ref:`input S1 image
<paths.s1_images>` are agglomerated in a :ref:`VRT file <dem-vrt-files>`.


.. _sardemproject:
.. index:: Project SAR coordinates onto DEM

Project SAR coordinates onto DEM
++++++++++++++++++++++++++++++++

:Inputs:         - An original :ref:`input S1 image <paths.s1_images>` (geometry)
                 - The associated :ref:`VRT file <dem-vrt-files>`
:Output:         A :ref:`SAR DEM projected file <S1_on_dem-files>`
:OTBApplication: :std:doc:`DiapOTB SARDEMProjection <Applications/app_SARDEMProjection>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.SARDEMProjection`

This step projects the coordinates of original :ref:`input S1 image
<paths.s1_images>` in the geometry of the SRTM VRT file.


.. _sarcartesianmeanestimation:
.. index:: Project XYZ coordinates onto SAR

Project XYZ coordinates onto SAR
++++++++++++++++++++++++++++++++

:Inputs:         - An original :ref:`input S1 image <paths.s1_images>` (geometry)
                 - The associated :ref:`VRT file <dem-vrt-files>`
                 - The associated :ref:`SAR DEM projected file <S1_on_dem-files>`
:Output:         A :ref:`XYZ Cartesian coordinates file <xyz-files>`
:OTBApplication: :std:doc:`Our patched version of DiapOTB SARCartesianMeanEstimation
                 <Applications/app_SARCartesianMeanEstimation2>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.SARCartesianMeanEstimation`

This step estimates the XYZ Cartesian coordinates on the ground in the geometry
of the original :ref:`input S1 image <paths.s1_images>`.


.. _compute-normals:
.. index:: Normals computation

Normals computation
+++++++++++++++++++

:Input:          A :ref:`XYZ Cartesian coordinates file <xyz-files>`
:Output:         None: chained in memory with :ref:`LIA maps computation <compute-lia>`
:OTBApplication: `ExtractNormalVector OTB application
                 <https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0>`_
                 (developed for the purpose of this project)

                 .. note::
                     Beware, this OTB application isn't distributed with OTB
                     yet. It has to be installed specifically on your machine.
                     It will be already installed in the :ref:`docker images
                     <docker>` though.
:StepFactory:    :class:`s1tiling.libs.otbwrappers.ComputeNormals`

This step computes the normal vectors to the ground, in the original
:ref:`input S1 image <paths.s1_images>` geometry.


.. _compute-lia:
.. index:: Compute LIA maps

LIA maps computation
++++++++++++++++++++

:Input:          None: chained in memory from :ref:`Normals computation
                 <compute-normals>`
:Output:         :ref:`Local Incidence Angle map, and sine LIA map <lia-s1-files>`
:OTBApplication: `SARComputeLocalIncidenceAngle OTB application
                 <https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0>`_
                 (developed for the purpose of this project)

                 .. note::
                     Beware, this OTB application isn't distributed with OTB
                     yet. It has to be installed specifically on your machine.
                     It will be already installed in the :ref:`docker images
                     <docker>` though.
:StepFactory:    :class:`s1tiling.libs.otbwrappers.ComputeLIA`

It computes the :ref:`Local Incidence Angle map, and sine LIA map
<lia-s1-files>` between the between the ground normal projected in range plane
:math:`\overrightarrow{n}` (plane defined by S, T, and Earth's center) and
:math:`\overrightarrow{TS}`, in the original :ref:`input S1 image
<paths.s1_images>` geometry -- where T is the target point on Earth's surface,
and S the SAR sensor position.


.. _ortho-lia:
.. index:: Orthorectification of LIA maps

Orthorectification of LIA maps
++++++++++++++++++++++++++++++

:Inputs:      A :ref:`Sine Local Incidence Angle map, and an optional degrees
              LIA map <lia-s1-files>` in the original S1 image geometry
:Output:      The associated :ref:`LIA map file(s) <lia-s2-half-files>`
              orthorectified on the target S2 tile.
:OTBApplication: :std:doc:`Orthorectification
                 <Applications/app_OrthoRectification>`
:StepFactory: :class:`s1tiling.libs.otbwrappers.OrthoRectifyLIA`

This steps ortho-rectifies the LIA map image(s) in S1 geometry to S2 grid.

It uses the following parameters from the request configuration file:

- :ref:`[Processing].orthorectification_gridspacing
  <Processing.orthorectification_gridspacing>`
- :ref:`[Processing].orthorectification_interpolation_method
  <Processing.orthorectification_interpolation_method>`
- :ref:`[Paths].srtm <paths.srtm>`
- :ref:`[Paths].geoid_file <paths.geoid_file>`


.. _concat-lia:
.. index:: Concatenation of LIA maps

Concatenation of LIA maps
+++++++++++++++++++++++++

:Inputs:         A pair of :ref:`LIA map files <lia-s2-half-files>` (sines or
                 degrees) orthorectified on the target S2 tile.
:Output:         The :ref:`LIA map file(s) <lia-files>` associated to the S2 grid
:OTBApplication: :std:doc:`Synthetize <Applications/app_Synthetize>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.ConcatLIA`

This step merges all the images of the orthorectified S1 LIA maps on a given S2
grid. As all orthorectified images are almost exclusive, they are concatenated
by taking the first non null pixel.


.. _apply-lia:
.. index:: Application of LIA maps

Application of LIA maps to β° calibrated S2 images
++++++++++++++++++++++++++++++++++++++++++++++++++

:Inputs:         - The :ref:`sine LIA map file <lia-files>` associated to the
                   S2 grid
                 - A β° calibrated, cut and orthorectified image on the S2 grid
:Output:         :ref:`final S2 tiles <full-S2-tiles>`, :math:`σ^0_{RTC}`
                 calibrated
:OTBApplication: :std:doc:`Synthetize <Applications/app_BandMath>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.ApplyLIACalibration`

This final step multiplies the sine LIA map (in S2 grid geometry) with β0
calibrated files orthorectified on the S2 grid.


.. _lia-data-caches:

.. index:: Data caches (LIA)

LIA specified data caches
-------------------------

As with main dataflow, two kinds of data are cached, but only one is regularly
cleaned-up by S1 Tiling. The other kind is left along as the software cannot
really tell whether they could be reused later on or not.

.. important:: This means that you may have to regularly clean up this space.
