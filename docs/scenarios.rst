.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

Scenarios
---------

.. contents:: Contents:
   :local:
   :depth: 3


.. _scenario.S1Processor:

Orthorectify pairs of Sentinel-1 images on Sentinel-2 grid
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This is the main scenario where pairs of Sentinel-1 images are:

- calibrated according to β\ :sup:`0`, γ\ :sup:`0` or σ\ :sup:`0` calibration
- then orthorectified onto the Sentinel-2 grid,
- to be finally concatenated.

The unique elements in this scenario are:

- the :ref:`calibration option <Processing.calibration>` that must be
  either one of ``beta``, ``sigma`` or ``gamma``
- the main executable which is :ref:`S1Processor`.

All options go in a :ref:`request configuration file <request-config-file>`
(e.g. ``MyS1ToS2.cfg`` in ``workingdir``). Important options will be:

- the time range (:ref:`first_date <DataSource.first_date>` and
  :ref:`last_date <DataSource.last_date>`),
- the :ref:`Sentinel-2 tiles <DataSource.roi_by_tiles>`,
- the orthorectification options (in :ref:`[Processing] <Processing>`),
- the directories where images are downloaded, produced, etc.
- the download credentials for the chosen data provider -- see
  :ref:`eodag_config <DataSource.eodag_config>`.


Then running S1Tiling is as simple as:

.. code:: bash

        cd workingdir
        S1Processor MyS1ToS2.cfg

Eventually,

- The S1 products will be downloaded in :ref:`s1_images <paths.s1_images>`.
- The orthorectified tiles will be generated in :ref:`output <paths.output>`.
- Temporary files will be produced in :ref:`tmp <paths.tmp>`.

.. note:: S1 Tiling never cleans the :ref:`tmp directory <paths.tmp>` as its
   files are :ref:`cached <data-caches>` in between runs. This means you will
   have to watch this directory and eventually clean it.


.. _scenario.S1ProcessorLIA:

Orthorectify pairs of Sentinel-1 images on Sentinel-2 grid with σ\ :sup:`0`\ :sub:`RTC` NORMLIM calibration
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In this scenario, the calibration applied is the :math:`σ^0_{RTC}` NORMLIM
calibration described in [Small2011]_.

.. [Small2011] D. Small, "Flattening Gamma: Radiometric Terrain Correction for
   SAR Imagery," in IEEE Transactions on Geoscience and Remote Sensing, vol.
   49, no. 8, pp. 3081-3093, Aug. 2011, doi: 10.1109/TGRS.2011.2120616.

In S1Tiling, we have chosen to precompute Local Incidence Angle (LIA) maps on
Sentinel-2 grid. Given a series of Sentinel-1 images to orthorectify on a
Sentinel-2 grid, we select a pair of Sentinel-1 images to compute the LIA
in the geometry of these images. The LIA map is then projected, through
orthorectification, on a Sentinel-2 tile.

That map will then be used for all series of pairs of Sentinel-1 images that
intersect the associated S2 tile.

Regarding options, the only difference with previous scenario are:

- the :ref:`calibration option <Processing.calibration>` that needs to be
  ``normlim``,
- the :ref:`directory <Paths.eof_dir>` where EOF files will be searched for, or
  downloaded to.
- the :ref:`directory <Paths.lia>` where LIA maps will be searched for, or
  produced in.
- a single pair of :ref:`platform <DataSource.platform_list>` + :ref:`relative
  orbit <datasource.relative_orbit_list>` to which the Local Incidence Angles
  will be calculated,

S1Tiling will then automatically take care of:

- obtaining the precise orbit files (EOF), if none match the request
  parameters,
- producing, or using existing, maps of sin(LIA) for each Sentinel-2 tiles --
  given an orbit and its direction,
- producing intermediary products calibrated with β\ :sup:`0` LUT.

.. list-table::
  :widths: auto
  :header-rows: 0
  :stub-columns: 0

  * - .. figure:: _static/sin_LIA_s1a_33NWB_DES_007.jpeg
            :alt: sine(LIA)
            :scale: 50%

            Map of sine(LIA) on 33NWB descending orbit 007

    - .. carousel::
            :show_controls:
            :show_indicators:
            :show_fade:
            :show_shadows:
            :show_dark:
            :show_captions_below:
            :data-bs-interval: false
            :data-bs-pause: false

            .. figure:: _static/s1a_33NWB_vh_DES_007_20200108txxxxxx_beta.jpeg
                :alt: 33NWB β° calibrated
                :scale: 50%

                33NWB β° calibrated -- 20200108

            .. figure:: _static/s1a_33NWB_vh_DES_007_20200108txxxxxx_NormLim.jpeg
                :alt: 33NWB NORMLIM σ° RTC calibrated
                :scale: 50%

                33NWB NORMLIM σ° RTC calibrated -- 20200108

            .. figure:: _static/s1a_33NWB_vh_DES_007_20200108txxxxxx_Normlim_filtered_lee.jpeg
                :alt: 33NWB NORMLIM σ° RTC calibrated and filtered
                :scale: 50%

                33NWB σ° RTC calibrated and despeckled (Lee) -- 20200108



.. warning::
   If you wish to parallelize this scenario and dedicate a different cluster
   node to each date -- as recommended in “:ref:`scenario.parallelize_date`”
   scenario, you will **NEED** to produce all the LIA maps beforehand.
   Otherwise, a same file may be concurrently written to from different nodes,
   and it will likely end up corrupted.

.. note::
   This scenario requires `NORMLIM σ0
   <https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0>`_ binaries.
   At the moment, NORMLIM σ\ :sup:`0` binaries need to be compiled manually.
   Unless you use either S1Tiling docker images, or S1Tiling on CNES TREX
   cluster.

.. note::
   This scenario requires to configure either ``cop_dataspace`` data provider
   in :ref:`eodag configuration file <datasource.eodag_config>`, or to enter
   valid EarthData credentials in your :file:`~/.netrc` file (can be overridden
   with :envvar:`$NETRC`).

.. _scenario.S1LIAMap:

Pre-produce maps of Local Incidence Angles for σ\ :sup:`0`\ :sub:`RTC` NORMLIM calibration
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

While :ref:`S1Processor` is able to produce the necessary LIA maps on the
fly, it is not able to do so when parallelization is done manually over time
ranges -- as described in “:ref:`scenario.parallelize_date`” scenario.

A different program is provided to compute the LIA maps beforehand:
:ref:`S1LIAMap`. It takes the exact same parameter files as
:ref:`S1Processor`. A few options will be ignored though: calibration type,
masking… But the following (non-obvious) options are mandatory:

- :ref:`[DataSource].platform_list <datasource.platform_list>` -- but only a
  single value shall be used
- :ref:`[DataSource].relative_orbit_list <datasource.relative_orbit_list>` --
  but only a single value shall be used
- :ref:`[DataSource].first_date <datasource.first_date>` and
  :ref:`[DataSource].last_date <datasource.last_date>` if
  :ref:`[DataSource].download <datasource.download>` it ``True``.

.. code:: bash

        cd workingdir
        # Yes, the same file works!
        S1LIAMap MyS1ToS2.cfg


.. note::
   LIA maps are perfect products to be stored and reused.

.. note::
   This scenario requires `NORMLIM σ0
   <https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0>`_ binaries.
   At the moment, NORMLIM σ\ :sup:`0` binaries need to be compiled manually.
   Unless you use either S1Tiling docker images, or S1Tiling on CNES TREX
   cluster.

.. note::
   To run :ref:`S1LIAMap` from the official S1Tiling docker, use ``--lia``
   as the first parameter to the docker execution (just before the request
   configuration file and other S1LIAMap related parameters). See :ref:`Using
   S1LIAMap with a docker <docker.S1LIAMap>`.


.. _scenario.masks:

Generate masks on final products
++++++++++++++++++++++++++++++++

Pixel masks of valid data can be produced in all :ref:`S1Processor`
scenarios when the option :ref:`generate_border_mask
<Mask.generate_border_mask>` is ``True``.

.. _scenario.parallelize_date:

Process huge quantities of data
+++++++++++++++++++++++++++++++

This use case concerns people that:

- have a lot of images to process over many tiles and over a consequent
  time-range,
- and have access to computing resources like HPC clusters

In that case, S1Tiling will be much more efficient if the parallelization is
done time-wise. We recommend cutting the full time range in smaller subranges,
and to distribute each subrange (with all S2 tiles) to a different node -- with
jobarrays for instances.


.. warning::
   This scenario is not compatible with ``normlim`` calibration where the LIA
   maps would be computed on-the-fly. For ``normlim`` calibration, it's
   imperative to precompute (and store LIA maps) before going massively
   parallel.


.. _scenario.choose_dem:

Use any other set of DEM inputs
+++++++++++++++++++++++++++++++

By default, S1Tiling comes with a GPKG database that associates SRTM30
geometries to the SRTM tile filename.

In order to use other DEM inputs, we need:

1. DEM files stored in :ref:`[PATHS].dem_dir <paths.dem_dir>` directory.
   |br|
   The format of these DEM files needs to be supported by OTB/GDAL.

2. A DEM (GPKG) database that holds a key (or set of keys) that enable(s) to
   locate/name DEM files associated to a DEM geometry.
   |br|
   Set the :ref:`[PATHS].dem_database <paths.dem_database>` key accordingly.
   |br|
   For instance, `eotile <https://github.com/CS-SI/eotile>`_ provides a couple
   of DEM databases for various types of DEM files.

3. A naming scheme that will associate an identifier key from the :ref:`DEM
   database <paths.dem_database>` to a DEM filename (located in
   :ref:`[PATHS].dem_dir <paths.dem_dir>` directory).
   |br|
   Set the :ref:`[PATHS].dem_format <paths.dem_format>` key accordingly.
   |br|
   The default :file:`{{id}}.hgt` associates the ``id`` key to STRM 30 m DEM
   files.
   |br|
   Using `eotile <https://github.com/CS-SI/eotile>`_ :file:`DEM_Union.gpkg` as
   DEM database, we could instead use:

   - :file:`{{Product10}}.tif` for Copernicus 30 m DEM files, using
     ``Product10`` key from the GPKG file.
   - :file:`{{Product30}}.tif` for Copernicus 90 m DEM files, using
     ``Product30`` key from the GPKG file.

4. Make sure to use a Geoid file compatible with the chosen DEM. For instance
   S1Tiling is shipped with EGM96 Geoid with is compatible with SRTM.
   On the other hand, Copernicus DEM is related to EGM2008 (a.k.a. EGM08)

