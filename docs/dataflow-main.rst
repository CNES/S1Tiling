.. include:: <isoamsa.txt>

.. _dataflow-main:

.. index:: S1Tiling data flow

======================================================================
S1Tiling data flow
======================================================================

.. contents:: Contents:
   :local:
   :depth: 3

Overall processing
------------------

S1 Tiling processes by looping on all required S2 tiles within the time range.

For each S2 tile,

1. It :ref:`downloads <downloading>` the necessary S1 images that intersect the
   S2 tile, within the specified time range, that are not already available in
   :ref:`input data cache <paths.s1_images>`

2. Then, for each polarisation,

   1. It :ref:`calibrates <calibration>`, :ref:`cuts <cutting>` and
      :ref:`orthorectifies <orthorectification>` all the S1 images onto the S2
      grid
   2. It :ref:`superposes (concatenates) <concatenation>` the orthorectified
      images into a single S2 tile.
   3. It :ref:`builds masks <mask-generation>`, :ref:`if required <Mask.generate_border_mask>`


.. _parallelization:

.. index:: parallelization

Parallelization
---------------

The actual sequencing is not linear. S1 Tiling starts by building a
dependencies graph expressed as a Dask Tasks graph. Dask framework then takes
care of distributing the computations.

In the following processing of 33NWB and 33NWC from 2020-01-01 to 2020-01-10,
only two S2 images are generated. It's done by processing in parallel (but in
any order compatible with the dependencies represented in the graph),

- the S1 image inputs (first column) by :ref:`calibrating <calibration>` and
  :ref:`cutting <cutting>` them to obtain...
- the :ref:`orthoready files <orthoready-files>` (second column), which are in
  turn :ref:`orthorectified <orthorectification>` to obtain...
- the :ref:`orthorectified files <orthorectified-files>` (third column), which
  are in turn :ref:`concatenated <concatenation>` to obtain...
- the :ref:`final S2 products <full-S2-tiles>` (fourth column),
- :ref:`border masks <mask-files>`  can in turn be :ref:`generated
  <mask-generation>` from them -- not represented on the graph.

.. graphviz::
    :name: graph_S1Processing
    :caption: Tasks for processing 33NWB and 33NWC
    :alt: Complete task flow for processing 33NWB and 33NWC
    :align: center

     digraph "sphinx-ext-graphviz" {
         rankdir="LR";
         graph [fontname="Verdana", fontsize="12"];
         node [fontname="Verdana", fontsize="12"];
         edge [fontname="Sans", fontsize="9"];

         raw_t1t2 [label="Raw t1-t2", target="_top", href="files.html#inputs", shape="folder", fillcolor=green, style=filled]
         raw_t2t3 [label="Raw t2-t3", target="_top", href="files.html#inputs", shape="folder", fillcolor=green, style=filled]
         raw_t3t4 [label="Raw t3-t4", target="_top", href="files.html#inputs", shape="folder", fillcolor=green, style=filled]

         or_t1t2 [label="OrthoReady t1-t2", target="_top", href="files.html#orthoready-files", shape="note", fillcolor=lightyellow, style=filled]
         or_t2t3 [label="OrthoReady t2-t3", target="_top", href="files.html#orthoready-files", shape="note", fillcolor=lightyellow, style=filled]
         or_t3t4 [label="OrthoReady t3-t4", target="_top", href="files.html#orthoready-files", shape="note", fillcolor=lightyellow, style=filled]

         o_nwc_t1 [label="Orthorectified 33NWC t1", target="_top", href="files.html#orthorectified-files", shape="note", fillcolor=lightyellow, style=filled]
         o_nwc_t2 [label="Orthorectified 33NWC t2", target="_top", href="files.html#orthorectified-files", shape="note", fillcolor=lightyellow, style=filled]
         o_nwb_t2 [label="Orthorectified 33NWB t2", target="_top", href="files.html#orthorectified-files", shape="note", fillcolor=lightyellow, style=filled]
         o_nwb_t3 [label="Orthorectified 33NWB t3", target="_top", href="files.html#orthorectified-files", shape="note", fillcolor=lightyellow, style=filled]

         nwb [label="S2 33NWB" shape="note", target="_top", href="files.html#full-S2-tiles", fillcolor=lightblue, style=filled]
         nwc [label="S2 33NWC" shape="note", target="_top", href="files.html#full-S2-tiles", fillcolor=lightblue, style=filled]

         raw_t1t2 -> or_t1t2 [label="calibration + cut"];
         raw_t2t3 -> or_t2t3 [label="calibration + cut"];
         raw_t3t4 -> or_t3t4 [label="calibration + cut"];

         or_t1t2 -> o_nwc_t1 [label="orthorectification"];
         or_t2t3 -> o_nwc_t2 [label="orthorectification"];
         or_t2t3 -> o_nwb_t2 [label="orthorectification"];
         or_t3t4 -> o_nwb_t3 [label="orthorectification"];

         o_nwc_t1 -> nwc [label="concatenation"];
         o_nwc_t2 -> nwc [label="concatenation"];
         o_nwb_t2 -> nwb [label="concatenation"];
         o_nwb_t3 -> nwb [label="concatenation"];
     }


.. _processings:

The processings
---------------

.. _downloading:
.. index:: downloading

Downloading of S1 products
++++++++++++++++++++++++++

The downloading of S1 products is optional and done only if
:ref:`[DataSource].download <DataSource.download>` option is set to ``True``.

S1 products are downloaded with `eodag <https://github.com/CS-SI/eodag>`_.
See :ref:`[DataSource].eodag_config <DataSource.eodag_config>`  regarding its
configuration.

Downloaded files are stored into the directory specified by
:ref:`[Paths].s1_images <Paths.s1_images>` option. If the directory doesn't
exist, it's created on the fly.


.. warning::

    In case a S1 product could not be downloaded in time (because it's OFFLINE,
    or because there is a network error, or a provider error...) for an
    intersecting and requested S2 tile, the associated S2 product won't be
    generated -- even if the other S1 product that intersects the S2 tile is
    present (correctly downloaded or on disk).

    .. note::

        There is no way to know that a S1 product is missing when working only
        from the disk (i.e. when :ref:`[DataSource].download
        <DataSource.download>` option is set to ``False``). In that case
        incomplete products may be generated. Their time stamp won't contain
        :samp:`txxxxxx` but the actual time at which the S1 product starts.

        Beware, in that case we may see a :samp:`txxxxxx` S2 product + a
        :samp:`t{{hhmmss}}` S2 product.

    .. note::

        While a S1 product may be impossible to download in time when
        processing a S2 tile, it may still be downloaded later on for a next
        tile.  Yet S1Tiling won't try to reprocess the first tile for which the
        product was initially missing.

        You'll have to run it again with the same parameters!




.. _calibration:
.. index:: SAR Calibration

SAR Calibration
+++++++++++++++

:Input:          An original :ref:`input S1 image <paths.s1_images>`
:Output:         None: chained in memory with :ref:`cutting <cutting>`
:OTBApplication: :std:doc:`OTB SARCalibration application
                 <Applications/app_SARCalibration>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.Calibrate`

This step applies σ°, β°, or γ° radiometric correction.
The type of calibration is controlled with :ref:`[Processing].calibration
<Processing.calibration>` option. It also permits to remove thermal noise
:ref:`if required <Processing.remove_thermal_noise>`.

.. note:: At the end of this step, no file is produced as calibration is piped
   in memory with :ref:`cutting <cutting>` to produce :ref:`orthorectification
   ready images <orthoready-files>`

.. note:: An extra artificial step is realized just after calibration to
   replace null values produced by denoising with a :ref:`minimal signal value
   <Processing.lower_signal_value>` > 0. This way, 0 can be used all along the
   way as the *no data* value.


.. _cutting:
.. index:: Margin cutting

Margins cutting
+++++++++++++++

:Input:          None: chained in memory from :ref:`SAR Calibration
                 <calibration>`
:Output:         - Either chained in memory with :ref:`orthorectification
                   <orthorectification>`
                 - or :ref:`orthorectification ready images <orthoready-files>`
:OTBApplication: :std:doc:`OTB ResetMargin application
                 <Applications/app_ResetMargin>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.CutBorders`

This step takes care of resetting margins content to 0 when too many no-data
are detected within the margin. This phenomenon happens on coasts. The margins
aren't cut out like what :std:doc:`ExtractROI <Applications/app_ExtractROI>`
would do but filled with 0's, which permits to keeps the initial geometry.

The implemented heuristic is to:

- always cut 1000 pixels on the sides (2 x 10 km) products before Sentinel-1
  IPF v2.90 (see `MPC-0243: Masking "No-value" Pixels on GRD Products generated
  by the Sentinel-1 ESA IPF
  <https://sentinels.copernicus.eu/documents/247904/2142675/Sentinel-1-masking-no-value-pixels-grd-products-note.pdf/32f11e6f-68b1-4f0a-869b-8d09f80e6788?t=1518545526000>`_),
- and 1600 pixels (16km) on the top (/resp on the bottom) of the image if more
  than 2000 NoData (NoData is assimilated with 0 here) pixels are detected on
  the 100th row from the top (/resp from the bottom).

.. note::
   The heuristic can be overridden thanks
   :ref:`[Processing].override_azimuth_cut_threshold_to
   <Processing.override_azimuth_cut_threshold_to>` option.

At the end of this step, :ref:`orthorectification ready images
<orthoready-files>` may be produced. It could be interresting to :ref:`cache
<data-caches>` these product as a same cut-and-calibrated S1 image can be
orthorectified into several S2 grids it intersects. The default processing of
these products in memory can be disabled by passing ``--cache-before-ortho`` to
program:`S1Processor`.


.. _orthorectification:
.. index:: Orthorectification

Orthorectification
++++++++++++++++++

:Input:          - Either chained in memory from :ref:`cutting <cutting>`
                 - or :ref:`orthorectification ready images <orthoready-files>`
:Output:         :ref:`orthorectified S1 images <orthorectified-files>`
:OTBApplication: :std:doc:`OTB OrthoRectification application
                 <Applications/app_OrthoRectification>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.OrthoRectify`

This steps ortho-rectifies the cut and calibrated (aka "orthoready") image in
S1 geometry to S2 grid.

It uses the following parameters from the request configuration file:

- :ref:`[Processing].orthorectification_gridspacing
  <Processing.orthorectification_gridspacing>`
- :ref:`[Processing].orthorectification_interpolation_method
  <Processing.orthorectification_interpolation_method>`
- :ref:`[Paths].srtm <paths.srtm>`
- :ref:`[Paths].geoid_file <paths.geoid_file>`


.. _concatenation:
.. index:: Concatenation

Concatenation
+++++++++++++

:Inputs:         One or two consecutive :ref:`orthorectified S1 images
                 <orthorectified-files>`
:Output:         The main product of S1 Tiling: the :ref:`final S2 tiles
                 <full-S2-tiles>`
:OTBApplication: :std:doc:`OTB Synthetize application
                 <Applications/app_Synthetize>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.Concatenate`

This step takes care of merging all the images of the orthorectified S1
products on a given S2 grid. As all orthorectified images are almost exclusive,
they are concatenated by taking the first non null pixel.

This step produces the main product of S1 Tiling: the :ref:`final S2 tiles
<full-S2-tiles>`.

.. list-table::
  :widths: auto
  :header-rows: 0
  :stub-columns: 0

  * - .. image:: _static/concat.jpeg
           :scale: 50%
           :alt:   Two orthorectified and exclusive S1 images
           :align: right

    - |Rarrtl|

    - .. image:: _static/s1a_33NWB_vh_DES_007_20200108txxxxxx.jpeg
           :scale: 50%
           :alt:   The orthorectified result
           :align: left


.. _mask-generation:
.. index:: Border mask generation

Border mask generation
++++++++++++++++++++++

:Inputs:          :ref:`final S2 tiles <full-S2-tiles>`
:Output:          :ref:`border masks <mask-files>`
:OTBApplications: - :std:doc:`OTB BandMath application <Applications/app_BandMath>`
                  - :std:doc:`OTB BinaryMorphologicalOperation application
                    <Applications/app_BinaryMorphologicalOperation>`
:StepFactories:   - :class:`s1tiling.libs.otbwrappers.BuildBorderMask`
                  - :class:`s1tiling.libs.otbwrappers.SmoothBorderMask`

If :ref:`requested <Mask.generate_border_mask>`, :ref:`border masks
<mask-files>` are generated.

The actual generation is done in two steps:

1. :std:doc:`OTB BandMath application <Applications/app_BandMath>` is used to
   generate border masks by saturating non-zero data to 1's.
2. :std:doc:`OTB BinaryMorphologicalOperation application
   <Applications/app_BinaryMorphologicalOperation>` is used to smooth border
   masks with a ball of 5x5 radius used for *opening*.

.. _data-caches:
.. index:: Data caches

Data caches
-----------

Two kinds of data are cached, but only one is regularly cleaned-up by S1
Tiling. The other kind is left along as the software cannot really tell whether
they could be reused later on or not.

.. important:: This means that you may have to regularly clean up this space.


.. _cache.S1:

Downloaded S1 files
+++++++++++++++++++

S1 files are downloaded in :ref:`[Paths].s1_images <Paths.s1_images>`.
directory. Whenever there are more than 1000 S1 products in that directory,
only the 1000 most recent are kept. The oldest ones are automatically removed.

.. _caches.tmp-orthoready:

OrthoReady S1 Files
+++++++++++++++++++

OrthoRectification is done on images cut, and calibrated. A same cut and
calibrated Sentinel-1 image can be orthorectified onto different Sentinel-2
tiles.

This means it could be interresting to cache these intermediary products as
files. Yet, this is not the default behaviour. Indeed at this time, S1-Tiling
cannot know when an "OrthoReady" file is no longer required, nor organize the
processing of S1 images to help deleting those temporary files as soon as
possible. In other words, it's up to you to clean these temporary files, and to
make sure to not request too many S2 tiles on long time ranges.

That's why the default behaviour is to process "OrthoReady" product in memory.
Also, this is not necessarily a big performance issue.
Indeed, given OTB internals, producing an orthorectified S1 image onto a S2
tile does not calibrate the whole S1 image, but only the minimal region
overlapped by the S2 tile.


.. note:: Unless you execute :program:`S1Processor` with
   ``--cache-before-ortho``, cutting, calibration and orthorectification are
   chained in memory.


.. warning:: When executed with ``--cache-before-ortho``, :ref:`Cut and
   calibrated (aka "OrthoReady") files <orthoready-files>` are stored in
   :ref:`%(tmp) <paths.tmp>`:samp:`/S1/` directory.
   Do not forget to regularly clean up this space.
