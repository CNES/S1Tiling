.. include:: <isoamsa.txt>

.. _dataflow-gammanaughtrtc:

.. index:: :math:`γ^0_{T}` RTC data flow

======================================================================
:math:`γ^0_{T}` RTC data flow
======================================================================

.. contents:: Contents:
   :local:
   :depth: 3

Two data flows are possible:

- with :ref:`S1GammaAreaMap` only γ area maps are produced,
- with :ref:`S1Processor` γ area maps are produced if not found, then
  :math:`γ^0_{T}` calibrated and orthorectified files are produced.

:math:`γ^0_{T}` RTC global processing
-------------------------------------

S1 Tiling processes by looping on all required S2 tiles within the time range.

For each S2 tile,

1. It :ref:`downloads <downloading_s1>` the necessary S1 images that intersect
   the S2 tile, within the specified time range, that are not already available
   in :ref:`input data cache <paths.s1_images>`
   (all scenarios)

2. Then, it makes sure the :ref:`associated γ area map <gamma_area_s2-files>`
   exists (all scenarios),

   0. It selects a pair of :ref:`input S1 images <paths.s1_images>` that
      intersect the S2 tile,
   1. For each :ref:`input S1 image <paths.s1_images>`

       1. It :ref:`prepares a VRT <prepare_VRT_s1-4rtc-proc>` of the DEM files
          that cover the image,
       2. It :ref:`projects <sardemproject_s1-4rtc-proc>` the coordinates of
          the input S1 image onto the geometry of the VRT,
       3. It :ref:`computes the γ area map <sargammaareaimageestimation-proc>`
          of each ground point,
       4. It :ref:`orthorectifies the γ area map <ortho_gamma_area-proc>` to
          the S2 tile

   2. It :ref:`concatenates <concat_gamma_area-proc>` both files into a single
      sine γ area map for the S2 tile.

3. Then, for each polarisation (S1Processor scenario only),

   1. It :ref:`calibrates with σ° LUT <calibration-proc>`, :ref:`cuts
      <cutting-proc>` and :ref:`orthorectifies <orthorectification>` all the S1
      images onto the S2 grid,
   2. It :ref:`superposes (concatenates) <concatenation-proc>` the
      orthorectified images into a single S2 tile,
   3. It :ref:`normalizes <apply_gamma_area-proc>` the σ° orthorectified image
      with the γ area map.


As with the main dataflow for all other calibrations (β°, γ°, or σ°), these
tasks are done :ref:`in parallel <parallelization>` in respect of all the
dependencies.


.. _rtc-processings:
.. _gamma_area-processings:

:math:`γ^0_{T}` RTC specific processings
----------------------------------------

.. graphviz::
    :name: graph_GAMMA_AREA_v1
    :caption: Tasks for processing 33NWC and 33NWB with GammaNaughtRTC calibration -- v1.2 workflow
    :alt: Complete task flow for processing 33NWC and 33NWB with GammaNaughtRTC calibration
    :align: center

     digraph "sphinx-ext-graphviz" {
         rankdir="LR";
         graph [fontname="Verdana", fontsize="12"];
         node [fontname="Verdana", fontsize="12", shape="note", target="_top", style=filled];
         edge [fontname="Sans", fontsize="9"];

         # ====[ γ area workflow
         vrt_d1_t1t2 [label="DEM VRT d1 t1-t2",         href="files.html#dem-vrt-files", fillcolor=palegoldenrod, group=rtc_t1];
         vrt_d1_t2t3 [label="DEM VRT d1 t2-t3",         href="files.html#dem-vrt-files", fillcolor=palegoldenrod, group=rtc_t2];

         S1_on_DEM_d1_t1t2 [label="S1 on DEM d1 t1-t2", href="files.html#s1-on-dem-files", fillcolor=palegoldenrod, group=rtc_t1];
         S1_on_DEM_d1_t2t3 [label="S1 on DEM d1 t2-t3", href="files.html#s1-on-dem-files", fillcolor=palegoldenrod, group=rtc_t2];

         # γ area on S1
         gamma_area_d1_t1t2 [label="γ AREA d1 t1-t2",   href="files.html#gamma-area-s1-files", fillcolor=palegoldenrod, group=rtc_t1];
         gamma_area_d1_t2t3 [label="γ AREA d1 t2-t3",   href="files.html#gamma-area-s1-files", fillcolor=palegoldenrod, group=rtc_t2];

         # γ area orthorectified on S2
         o_gamma_area_d1_t1 [label="γ AREA d1 t1 on 33NWB", href="files.html#gamma-area-s2-half-files", fillcolor=palegoldenrod, group=rtc_t1];
         o_gamma_area_d1_t2 [label="γ AREA d1 t2 on 33NWB", href="files.html#gamma-area-s2-half-files", fillcolor=palegoldenrod, group=rtc_t2];
         # γ area concatenated a selected (best coverage)
         nwb_gamma_area     [label="γ AREA on 33NWB",       href="files.html#gamma-area-s2-files", fillcolor=gold, group=rtc_t2];

         nwb_d1      [label="S2 γ° RTC 33NWB d1", href="files.html#full-s2-tiles", fillcolor=lightblue];
         nwb_d2      [label="S2 γ° RTC 33NWB d2", href="files.html#full-s2-tiles", fillcolor=lightblue];
         nwb_dn      [label="S2 γ° RTC 33NWB dn", href="files.html#full-s2-tiles", fillcolor=lightblue];

         mult_d1     [label="X", shape="circle"]
         mult_d2     [label="X", shape="circle"]
         mult_dn     [label="X", shape="circle"]

         raw_d1_t1t2 -> vrt_d1_t1t2 [label=""];
         raw_d1_t2t3 -> vrt_d1_t2t3 [label=""];

         vrt_d1_t1t2 -> S1_on_DEM_d1_t1t2;
         vrt_d1_t2t3 -> S1_on_DEM_d1_t2t3;
         raw_d1_t1t2 -> S1_on_DEM_d1_t1t2;
         raw_d1_t2t3 -> S1_on_DEM_d1_t2t3;

         vrt_d1_t1t2       -> gamma_area_d1_t1t2;
         vrt_d1_t2t3       -> gamma_area_d1_t2t3;
         raw_d1_t1t2       -> gamma_area_d1_t1t2;
         raw_d1_t2t3       -> gamma_area_d1_t2t3;
         S1_on_DEM_d1_t1t2 -> gamma_area_d1_t1t2;
         S1_on_DEM_d1_t2t3 -> gamma_area_d1_t2t3;

         gamma_area_d1_t1t2 -> o_gamma_area_d1_t1 [label="ortho"];
         gamma_area_d1_t2t3 -> o_gamma_area_d1_t2 [label="ortho"];

         o_gamma_area_d1_t1 -> nwb_gamma_area [label="concatenation"];
         o_gamma_area_d1_t2 -> nwb_gamma_area [label="concatenation"];

         # =====[ Inputs nodes
         raw_d1_t1t2 [label="Raw d1 t1-t2", href="configuration.html#paths-s1-images", shape="folder", fillcolor=green]
         raw_d1_t2t3 [label="Raw d1 t2-t3", href="configuration.html#paths-s1-images", shape="folder", fillcolor=green]

         raw_d2_t1t2 [label="Raw d2 t1'-t2'", href="configuration.html#paths-s1-images", shape="folder", fillcolor=green]
         raw_d2_t2t3 [label="Raw d2 t2'-t3'", href="configuration.html#paths-s1-images", shape="folder", fillcolor=green]

         raw_dn_t1t2 [label="Raw dn t1'-t2'", href="configuration.html#paths-s1-images", shape="folder", fillcolor=green]
         raw_dn_t2t3 [label="Raw dn t2'-t3'", href="configuration.html#paths-s1-images", shape="folder", fillcolor=green]

         { rank = same ;  raw_d1_t1t2 raw_d1_t2t3 raw_d2_t1t2 raw_d2_t2t3 raw_dn_t1t2 raw_dn_t2t3}

         # =====[ Classic workflow
         o_nwb_d1_t1 [label="Orthorectified σ° 33NWB d1 t1", href="files.html#orthorectified-files", fillcolor=lightyellow]
         o_nwb_d1_t2 [label="Orthorectified σ° 33NWB d1 t2", href="files.html#orthorectified-files", fillcolor=lightyellow]

         o_nwb_d2_t1 [label="Orthorectified σ° 33NWB d2 t'1", href="files.html#orthorectified-files", fillcolor=lightyellow]
         o_nwb_d2_t2 [label="Orthorectified σ° 33NWB d2 t'2", href="files.html#orthorectified-files", fillcolor=lightyellow]

         o_nwb_dn_t1 [label="Orthorectified σ° 33NWB dn t'1", href="files.html#orthorectified-files", fillcolor=lightyellow]
         o_nwb_dn_t2 [label="Orthorectified σ° 33NWB dn t'2", href="files.html#orthorectified-files", fillcolor=lightyellow]

         # Concatenated σ° calibrated + orthorectified nodes
         nwb_d1_b0 [label="S2 σ° 33NWB d1", href="files.html#full-s2-tiles", fillcolor=pink]
         nwb_d2_b0 [label="S2 σ° 33NWB d2", href="files.html#full-s2-tiles", fillcolor=pink]
         nwb_dn_b0 [label="S2 σ° 33NWB dn", href="files.html#full-s2-tiles", fillcolor=pink]

         # Classic workflow up to concatenated σ° calibrated + orthorectified nodes
         raw_d1_t1t2 -> o_nwb_d1_t1 [label="σ° cal | noise | cut | ortho"];
         raw_d1_t2t3 -> o_nwb_d1_t2 [label="σ° cal | noise | cut | ortho"];
         raw_d2_t1t2 -> o_nwb_d2_t1 [label="σ° cal | noise | cut | ortho"];
         raw_d2_t2t3 -> o_nwb_d2_t2 [label="σ° cal | noise | cut | ortho"];
         raw_dn_t1t2 -> o_nwb_dn_t1 [label="σ° cal | noise | cut | ortho"];
         raw_dn_t2t3 -> o_nwb_dn_t2 [label="σ° cal | noise | cut | ortho"];

         o_nwb_d1_t1 -> nwb_d1_b0 [label="concatenation"];
         o_nwb_d1_t2 -> nwb_d1_b0 [label="concatenation"];
         o_nwb_d2_t1 -> nwb_d2_b0 [label="concatenation"];
         o_nwb_d2_t2 -> nwb_d2_b0 [label="concatenation"];
         o_nwb_dn_t1 -> nwb_dn_b0 [label="concatenation"];
         o_nwb_dn_t2 -> nwb_dn_b0 [label="concatenation"];

         # ===================================

         nwb_gamma_area -> mult_d1;
         nwb_gamma_area -> mult_d2;
         nwb_gamma_area -> mult_dn;
         nwb_d1_b0      -> mult_d1;
         nwb_d2_b0      -> mult_d2;
         nwb_dn_b0      -> mult_dn;

         mult_d1 -> nwb_d1;
         mult_d2 -> nwb_d2;
         mult_dn -> nwb_dn;

         # =====[ Align
         {
             rank = same ;
             vrt_d1_t1t2 vrt_d1_t2t3 o_nwb_d1_t1 o_nwb_d1_t2 o_nwb_d2_t1 o_nwb_d2_t2 o_nwb_dn_t1 o_nwb_dn_t2
             edge[ style=invis];
             vrt_d1_t1t2 -> vrt_d1_t2t3 -> o_nwb_d1_t1 -> o_nwb_d1_t2 -> o_nwb_d2_t1 -> o_nwb_d2_t2 -> o_nwb_dn_t1 -> o_nwb_dn_t2
         }
         {
             edge[ style=invis];
             raw_d1_t1t2 -> vrt_d1_t1t2 -> gamma_area_d1_t1t2 -> o_gamma_area_d1_t1;
         }
     }


.. _prepare_VRT_s1-4rtc-proc:
.. index:: Agglomerate DEM

Agglomerate DEM files in a VRT that covers S1 footprint (RTC)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

:Input:       All DEM files that intersect an original :ref:`input S1 image <paths.s1_images>`
:Output:      A :ref:`VRT file <dem-vrt-files>`
:Function:    :func:`osgeo.gdal.BuildVRT`
:StepFactory: :class:`s1tiling.libs.otbwrappers.AgglomerateDEMOnS1`

All DEM files that intersect an original :ref:`input S1 image
<paths.s1_images>` are agglomerated in a :ref:`VRT file <dem-vrt-files>`.


.. _resample_DEM-proc:
.. index:: Resample DEM

Resample DEM (RTC)
++++++++++++++++++

:Inputs:      A :ref:`VRT file <dem-vrt-files>`
:Outputs:     Resampled DEM image
:OTBApplication: :external+OTB:std:doc:`Applications/app_RigidTransformResample`
:StepFactory: :class:`s1tiling.libs.otbwrappers.ResampleDEM`

The DEM from the VRT are resampled by the chosen resampling factors
(:ref:`resample_dem_factor_x <processing.resample_dem_factor_x>` and
:ref:`resample_dem_factor_y <processing.resample_dem_factor_y>`).


.. _sardemproject_s1-4rtc-proc:
.. index:: Project SAR coordinates onto DEM

Project SAR coordinates onto DEM
++++++++++++++++++++++++++++++++

:Inputs:         - An original :ref:`input S1 image <paths.s1_images>` (geometry)
                 - The associated :ref:`VRT file <dem-vrt-files>`, or a
                   resampled version.
:Output:         A :ref:`SAR DEM projected file <S1_on_dem-files>`
:OTBApplication: :external:std:doc:`Our patched version of DiapOTB
                 SARDEMProjection <Applications/app_SARDEMProjection>`

                 .. note::
                     Beware, this OTB application isn't distributed with OTB
                     yet. It has to be installed specifically on your machine.
                     It will be already installed in the :ref:`docker images
                     <docker>` though.
:StepFactory:    :class:`s1tiling.libs.otbwrappers.SARDEMProjectionImageEstimation`

This step projects the coordinates of original :ref:`input S1 image
<paths.s1_images>` in the geometry of the DEM VRT file.


.. _sargammaareaimageestimation-proc:
.. index:: Project γ area coordinates onto SAR

Project γ area coordinates onto SAR
+++++++++++++++++++++++++++++++++++

:Inputs:         - An original :ref:`input S1 image <paths.s1_images>` (geometry)
                 - The associated :ref:`VRT file <dem-vrt-files>`
                 - The associated :ref:`SAR DEM projected file <S1_on_dem-files>`
:Output:         A :ref:`γ area cartesian coordinates file <gamma_area_s2-files>`
:OTBApplication: `SARGammaAreaImageEstimation
                 <https://gitlab.orfeo-toolbox.org/s1-tiling/rtc_gamma0>`_

                 .. note::
                     Beware, this OTB application isn't distributed with OTB
                     yet. It has to be installed specifically on your machine.
                     It will be already installed in the :ref:`docker images
                     <docker>` though.
:StepFactory:    :class:`s1tiling.libs.otbwrappers.SARGammaAreaImageEstimation`

This step estimates the γ area coordinates on the ground in the geometry of the
original :ref:`input S1 image <paths.s1_images>`.


.. _ortho_gamma_area-proc:
.. index:: Orthorectification of γ area maps

Orthorectification of γ area maps
+++++++++++++++++++++++++++++++++

:Inputs:      A :ref:`γ area map <gamma_area-s1-files>` in the original S1 image geometry
:Output:      The associated :ref:`γ area map file(s) <gamma_area-s2-half-files>`
              orthorectified on the target S2 tile.
:OTBApplication: :external+OTB:std:doc:`Orthorectification
                 <Applications/app_OrthoRectification>`
:StepFactory: :class:`s1tiling.libs.otbwrappers.OrthoRectifyGAMMA_AREA`

This steps ortho-rectifies the γ area map image(s) in S1 geometry to S2 grid.

It uses the following parameters from the request configuration file:

- :ref:`[Processing].orthorectification_gridspacing
  <Processing.orthorectification_gridspacing>`
- :ref:`[Processing].orthorectification_interpolation_method
  <Processing.orthorectification_interpolation_method>`
- :ref:`[Paths].dem_dir <paths.dem_dir>`
- :ref:`[Paths].geoid_file <paths.geoid_file>`


.. _concat_gamma_area-proc:
.. index:: Concatenation of γ area maps

Concatenation of γ area maps
++++++++++++++++++++++++++++

:Inputs:         A pair of :ref:`γ area map files <gamma_area-s2-half-files>` orthorectified on the target S2 tile.
:Output:         The :ref:`γ area map file(s) <gamma_area_s2-files>` associated to the S2 grid
:OTBApplication: :external+OTB:std:doc:`Synthetize <Applications/app_Synthetize>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.ConcatenateGAMMA_AREA`

This step merges all the images of the orthorectified S1 γ area maps on a given
S2 grid. As all orthorectified images are almost exclusive, they are
concatenated by taking the first non-null pixel.


.. _apply_gamma_area-proc:
.. index:: Application of RTC maps

Application of γ area maps to σ° calibrated S2 images
+++++++++++++++++++++++++++++++++++++++++++++++++++++

:Inputs:         - The :ref:`γ area map file <gamma_area_s2-files>` associated to
                   the S2 grid
                 - A σ° calibrated, cut and orthorectified image on the S2 grid
:Output:         :ref:`final S2 tiles <full-S2-tiles>`, :math:`γ^0_{T}`
                 calibrated
:OTBApplication: `SARGammaAreaToGammaNaughtRTCImageEstimation
                 <https://gitlab.orfeo-toolbox.org/s1-tiling/rtc_gamma0>`_

                 .. note::
                     Beware, this OTB application isn't distributed with OTB
                     yet. It has to be installed specifically on your machine.
                     It will be already installed in the :ref:`docker images
                     <docker>` though.
:StepFactory:    :class:`s1tiling.libs.otbwrappers.ApplyGammaNaughtRTCCalibration`

This final step applies γ area map (in S2 grid geometry) to σ° calibrated files
orthorectified on the S2 grid.


.. _gamma_area-data-caches:
.. index:: Data caches (γ area)

γ area specific data caches
---------------------------

As with main dataflow, two kinds of data are cached, but only one is regularly
cleaned-up by S1 Tiling. The other kind is left along as the software cannot
really tell whether they could be reused later on or not.

.. important:: This means that you may have to regularly clean up this space.
