.. include:: <isoamsa.txt>

.. _dataflow-gammanaughtrtc:

.. index:: GammaNaughtRTC data flow

======================================================================
GammaNaughtRTC data flow
======================================================================

.. contents:: Contents:
   :local:
   :depth: 3

Two data flows are possibles:

- with :program:`S1GammaAreaMap` only GAMMA AREA maps are produced,
- with :program:`S1Processor` GAMMA AREA maps are produced is not found, then
  :math:`γ^0_{RTC}` orthorectified files are produced.

GammaNaughtRTC global processing
------------------------------------

The following processing was the one supported in v1.0 of S1Tiling.

S1 Tiling processes by looping on all required S2 tiles within the time range.

For each S2 tile,

1. It :ref:`downloads <downloading>` the necessary S1 images that intersect the
   S2 tile, within the specified time range, that are not already available in
   :ref:`input data cache <paths.s1_images>`
   (all scenarios)

2. Then, it makes sure the :ref:`associated GAMMA_AREA map <gamma_area-files>` exists
   (all scenarios),

   0. It selects a pair of :ref:`input S1 images <paths.s1_images>` that
      intersect the S2 tile,
   1. For each :ref:`input S1 image <paths.s1_images>`

       1. It :ref:`prepares a VRT <prepare_VRT_s1-proc>` of the DEM files that
          cover the image,
       2. It :ref:`projects <sardemproject_s1-proc>` the coordinates of the
          input S1 image onto the geometry of the VRT,
       3. It :ref:`computes the GAMMA_AREA map <compute_gamma_area-proc>` of each ground point,
       4. It :ref:`orthorectifies the GAMMA_AREA map <ortho_gamma_area-proc>` to the S2 tile

   2. It :ref:`concatenates <concat_gamma_area-proc>` both files into a single sine
      GAMMA_AREA map for the S2 tile.

3. Then, for each polarisation (S1Processor scenario only),

   1. It :ref:`calibrates with β° LUT <calibration-proc>`, :ref:`cuts
      <cutting-proc>` and :ref:`orthorectifies <orthorectification>` all the S1
      images onto the S2 grid,
   2. It :ref:`superposes (concatenates) <concatenation-proc>` the
      orthorectified images into a single S2 tile,
   3. It :ref:`normlizes <apply_gamma_area-proc>` the β° orthorectified image with
      the GAMMA_AREA map.


As with the main dataflow for all other calibrations (β°, γ°, or σ°), these
tasks are done :ref:`in parallel <parallelization>` in respect of all the
dependencies.



.. _gamma_area-processings:

GAMMA_AREA specific processings
-----------------------------------

.. graphviz::
    :name: graph_GAMMA_AREA_v1
    :caption: Tasks for processing 33NWC and 33NWB with GammaNaughtRTC calibration -- v1.0 workflow
    :alt: Complete task flow for processing 33NWC and 33NWB with GammaNaughtRTC calibration
    :align: center

     digraph "sphinx-ext-graphviz" {
         rankdir="LR";
         graph [fontname="Verdana", fontsize="12"];
         node [fontname="Verdana", fontsize="12", shape="note", target="_top", style=filled];
         edge [fontname="Sans", fontsize="9"];

         # =====[ Inputs nodes
         raw_d1_t1t2 [label="Raw d1 t1-t2", href="files.html#inputs", shape="folder", fillcolor=green]
         raw_d1_t2t3 [label="Raw d1 t2-t3", href="files.html#inputs", shape="folder", fillcolor=green]

         raw_d2_t1t2 [label="Raw d2 t1'-t2'", href="files.html#inputs", shape="folder", fillcolor=green]
         raw_d2_t2t3 [label="Raw d2 t2'-t3'", href="files.html#inputs", shape="folder", fillcolor=green]

         raw_dn_t1t2 [label="Raw dn t1'-t2'", href="files.html#inputs", shape="folder", fillcolor=green]
         raw_dn_t2t3 [label="Raw dn t2'-t3'", href="files.html#inputs", shape="folder", fillcolor=green]

         { rank = same ;  raw_d1_t1t2 raw_d1_t2t3 raw_d2_t1t2 raw_d2_t2t3 raw_dn_t1t2 raw_dn_t2t3}

         # =====[ Classic workflow
         o_nwb_d1_t1 [label="Orthorectified β° 33NWB d1 t1", href="files.html#orthorectified-files", fillcolor=lightyellow]
         o_nwb_d1_t2 [label="Orthorectified β° 33NWB d1 t2", href="files.html#orthorectified-files", fillcolor=lightyellow]

         o_nwb_d2_t1 [label="Orthorectified β° 33NWB d2 t'1", href="files.html#orthorectified-files", fillcolor=lightyellow]
         o_nwb_d2_t2 [label="Orthorectified β° 33NWB d2 t'2", href="files.html#orthorectified-files", fillcolor=lightyellow]

         o_nwb_dn_t1 [label="Orthorectified β° 33NWB dn t'1", href="files.html#orthorectified-files", fillcolor=lightyellow]
         o_nwb_dn_t2 [label="Orthorectified β° 33NWB dn t'2", href="files.html#orthorectified-files", fillcolor=lightyellow]

         # Concatenated β° calibrated + orthorectified nodes
         nwb_d1_b0 [label="S2 β° 33NWB d1", href="files.html#full-S2-tiles", fillcolor=pink]
         nwb_d2_b0 [label="S2 β° 33NWB d2", href="files.html#full-S2-tiles", fillcolor=pink]
         nwb_dn_b0 [label="S2 β° 33NWB dn", href="files.html#full-S2-tiles", fillcolor=pink]

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
         # ====[ GAMMA_AREA workflow
         vrt_d1_t1t2 [label="DEM VRT d1 t1-t2", fillcolor=palegoldenrod];
         vrt_d1_t2t3 [label="DEM VRT d1 t2-t3", fillcolor=palegoldenrod];

         S1_on_DEM_d1_t1t2 [label="S1 on DEM d1 t1-t2", fillcolor=palegoldenrod];
         S1_on_DEM_d1_t2t3 [label="S1 on DEM d1 t2-t3", fillcolor=palegoldenrod];

         lia_d1_t1t2 [label="GAMMA_AREA d1 t1-t2", fillcolor=palegoldenrod];
         lia_d1_t2t3 [label="GAMMA_AREA d1 t2-t3", fillcolor=palegoldenrod];

         o_lia_d1_t1 [label="GAMMA_AREA d1 t1 on 33NWB", fillcolor=palegoldenrod];
         o_lia_d1_t2 [label="GAMMA_AREA d1 t2 on 33NWB", fillcolor=palegoldenrod];
         nwb_gamma_area     [label="GAMMA_AREA on 33NWB", fillcolor=gold];

         nwb_d1      [label="S2 γ° GAMMA_AREA 33NWB d1", fillcolor=lightblue];
         nwb_d2      [label="S2 γ° GAMMA_AREA 33NWB d2", fillcolor=lightblue];
         nwb_dn      [label="S2 γ° GAMMA_AREA 33NWB dn", fillcolor=lightblue];

         mult_d1     [label="X", shape="circle"]
         mult_d2     [label="X", shape="circle"]
         mult_dn     [label="X", shape="circle"]

         raw_d1_t1t2 -> vrt_d1_t1t2 [label=""];
         raw_d1_t2t3 -> vrt_d1_t2t3 [label=""];

         vrt_d1_t1t2 -> S1_on_DEM_d1_t1t2;
         vrt_d1_t2t3 -> S1_on_DEM_d1_t2t3;
         raw_d1_t1t2 -> S1_on_DEM_d1_t1t2;
         raw_d1_t2t3 -> S1_on_DEM_d1_t2t3;

         gamma_area_d1_t1t2 -> o_lia_d1_t1;
         gamma_area_d1_t2t3 -> o_lia_d1_t2;

         o_gamma_area_d1_t1 -> nwb_gamma_area;
         o_gamma_area_d1_t2 -> nwb_gamma_area;

         nwb_gamma_area   -> mult_d1;
         nwb_gamma_area   -> mult_d2;
         nwb_gamma_area   -> mult_dn;
         nwb_d1_b0 -> mult_d1;
         nwb_d2_b0 -> mult_d2;
         nwb_dn_b0 -> mult_dn;

         mult_d1 -> nwb_d1;
         mult_d2 -> nwb_d2;
         mult_dn -> nwb_dn;
     }


.. _prepare_VRT_s1-proc:
.. index:: Agglomerate DEM

Agglomerate DEM files in a VRT that covers S1 footprint
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

:Inputs:      All DEM files that intersect an original :ref:`input S1 image <paths.s1_images>`
:Output:      A :ref:`VRT file <dem-vrt-files>`
:Function:    :func:`osgeo.gdal.BuildVRT`
:StepFactory: :class:`s1tiling.libs.otbwrappers.AgglomerateDEMOnS1`

All DEM files that intersect an original :ref:`input S1 image
<paths.s1_images>` are agglomerated in a :ref:`VRT file <dem-vrt-files>`.


.. _sardemproject_s1-proc:
.. index:: Project SAR coordinates onto DEM

Project SAR coordinates onto DEM
++++++++++++++++++++++++++++++++

:Inputs:         - An original :ref:`input S1 image <paths.s1_images>` (geometry)
                 - The associated :ref:`VRT file <dem-vrt-files>`
:Output:         A :ref:`SAR DEM projected file <S1_on_dem-files>`
:OTBApplication: :external:std:doc:`DiapOTB SARDEMProjection <Applications/app_SARDEMProjection>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.app_SARDEMProjectionImageEstimation`

This step projects the coordinates of original :ref:`input S1 image
<paths.s1_images>` in the geometry of the DEM VRT file.


.. _sargammaareaimageestimation-proc:
.. index:: Project GAMMA_ARE coordinates onto SAR

Project GAMMA_AREA coordinates onto SAR
++++++++++++++++++++++++++++++++

:Inputs:         - An original :ref:`input S1 image <paths.s1_images>` (geometry)
                 - The associated :ref:`VRT file <dem-vrt-files>`
                 - The associated :ref:`SAR DEM projected file <S1_on_dem-files>`
:Output:         A :ref:`GAMMA_AREA Cartesian coordinates file <gamma_area-files>`
:OTBApplication: :external:std:doc:`Our patched version of DiapOTB SARGammaAreaImageEstimation
                 <Applications/app_SARGammaAreaImageEstimation>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.SARGammaAreaImageEstimation`

This step estimates the GAMMA_AREA Cartesian coordinates on the ground in the geometry
of the original :ref:`input S1 image <paths.s1_images>`.


.. _ortho_gamma_area-proc:
.. index:: Orthorectification of GAMMA_AREA maps

Orthorectification of GAMMA_AREA maps
++++++++++++++++++++++++++++++

:Inputs:      A :ref:`Gamma Area Local Incidence Angle map GAMMA_AREA map <gamma_area-s1-files>` in the original S1 image geometry
:Output:      The associated :ref:`GAMMA_AREA map file(s) <gamma_area-s2-half-files>`
              orthorectified on the target S2 tile.
:OTBApplication: :external:std:doc:`Orthorectification
                 <Applications/app_OrthoRectification>`
:StepFactory: :class:`s1tiling.libs.otbwrappers.OrthoRectifyGAMMA_AREA`

This steps ortho-rectifies the GAMMA_AREA map image(s) in S1 geometry to S2 grid.

It uses the following parameters from the request configuration file:

- :ref:`[Processing].orthorectification_gridspacing
  <Processing.orthorectification_gridspacing>`
- :ref:`[Processing].orthorectification_interpolation_method
  <Processing.orthorectification_interpolation_method>`
- :ref:`[Paths].dem_dir <paths.dem_dir>`
- :ref:`[Paths].geoid_file <paths.geoid_file>`


.. _concat_gamma_area-proc:
.. index:: Concatenation of GAMMA_AREA maps

Concatenation of GAMMA_AREA maps
+++++++++++++++++++++++++

:Inputs:         A pair of :ref:`GAMMA_AREA map files <gamma_area-s2-half-files>` orthorectified on the target S2 tile.
:Output:         The :ref:`GAMMA_AREA map file(s) <gamma_area-files>` associated to the S2 grid
:OTBApplication: :external:std:doc:`Synthetize <Applications/app_Synthetize>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.ConcatGAMMA_AREA`

This step merges all the images of the orthorectified S1 GAMMA_AREA maps on a given S2
grid. As all orthorectified images are almost exclusive, they are concatenated
by taking the first non null pixel.


.. _gamma_area-data-caches:
.. index:: Data caches (GAMMA_AREA)

GAMMA_AREA specific data caches
------------------------

As with main dataflow, two kinds of data are cached, but only one is regularly
cleaned-up by S1 Tiling. The other kind is left along as the software cannot
really tell whether they could be reused later on or not.

.. important:: This means that you may have to regularly clean up this space.
