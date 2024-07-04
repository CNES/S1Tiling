.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. _use:

.. index:: usage

======================================================================
Usage
======================================================================

.. contents:: Contents:
   :local:
   :depth: 3

Scenarios
---------

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
- the main executable which is :program:`S1Processor`.

All options go in a :ref:`request configuration file <request-config-file>`
(e.g.  ``MyS1ToS2.cfg`` in ``workingdir``). Important options will be:

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
- the :ref:`directory <Paths.lia>` where LIA maps will be searched for, or
  produced in.


S1Tiling will then automatically take care of:

- producing, or using existing, maps of sin(LIA) for each Sentinel-2 tiles --
  given an orbit and it direction,
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
   node to each date -- as recommended in ":ref:`scenario.parallelize_date`"
   scenario, you will **NEED** produce all the LIA maps beforehand.
   Otherwise a same file may be concurrently written to from different nodes,
   and it will likely end up corrupted.

.. note::
   This scenario requires `NORMLIM σ0
   <https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0>`_ binaries.
   At the moment, NORMLIM σ\ :sup:`0` binaries need to be compiled manually.
   Unless you use either S1Tiling docker images, or S1Tiling on CNES TREX
   cluster.


.. _scenario.S1LIAMap:

Preproduce maps of Local Incidence Angles for σ\ :sup:`0`\ :sub:`RTC` NORMLIM calibration
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

While :program:`S1Processor` is able to produce the necessary LIA maps on the
fly, it is not able to do so when parallelization is done manually over time
ranges -- as described in ":ref:`scenario.parallelize_date`" scenario.

A different program is provided to compute the LIA maps beforehand:
:program:`S1LIAMap`. It takes the exact same parameter files as
:program:`S1Processor`. A few options will be ignored though: calibration type,
masking....

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
   To run :program:`S1LIAMap` from the official S1Tiling docker, use ``--lia``
   as the first parameter to the docker execution (just before the the
   request configuration file and other S1LIAMap related parameters). See
   :ref:`Using S1LIAMap with a docker <docker.S1LIAMap>`.


.. _scenario.masks:

Generate masks on final products
++++++++++++++++++++++++++++++++

Pixel masks of valid data can be produced in all :program:`S1Processor`
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
done time-wise. We recommended to cut the full time range in smaller subranges,
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

By default S1Tiling comes with a GPKG database that associates SRTM30 geometries to the SRTM tile filename.

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
   The default :file:`{{id}}.hgt` associates the ``id`` key to STRM 30m DEM
   files.
   |br|
   Using `eotile <https://github.com/CS-SI/eotile>`_ :file:`DEM_Union.gpkg` as
   DEM database, we could instead use:

   - :file:`{{Product10}}.tif`  for Copernicus 30m DEM files, using
     ``Product10`` key from the GPKG file.
   - :file:`{{Product30}}.tif`  for Copernicus 90m DEM files, using
     ``Product30`` key from the GPKG file.

4. Make sure to use a Geoid file compatible with the chosen DEM. For instance
   S1Tiling is shipped with EGM96 Geoid with is compatible with SRTM.
   On the other hand, Copernicus DEM is related to EGM2008 (a.k.a EGM08)

.. _request-config-file:

.. index:: Request configuration file

Request Configuration file
--------------------------

The request configuration file passed to ``S1Processor`` is in ``.ini`` format.
It is expected to contain the following entries.

You can use this :download:`this template
<../s1tiling/resources/S1Processor.cfg>`, as a starting point.

.. _paths:

``[PATHS]`` section
+++++++++++++++++++

.. list-table::
  :widths: auto
  :header-rows: 1
  :stub-columns: 1

  * - Option
    - Description

      .. _paths.s1_images:
  * - ``s1_images``
    - Where S1 images are downloaded thanks to `EODAG
      <https://github.com/CS-SI/eodag>`_.
      |br|
      S1Tiling will automatically take care to keep at most 1000 products in
      that directory -- the 1000 last that have been downloaded.
      |br|
      This enables to cache downloaded S1 images in beteen runs.

      .. _paths.output:
  * - ``output``
    - Where products are generated.

      .. _paths.lia:
  * - ``lia``
    - Where Local Incidence Maps and sin(LIA) products are generated. Its
      default value is ``{output}/_LIA``.

      .. _paths.tmp:
  * - ``tmp``
    - Where :ref:`intermediary files <temporary-files>` are produced, and
      sometimes :ref:`cached <data-caches>` for longer periods.

      .. _paths.geoid_file:
  * - ``geoid_file``
    - Path to Geoid model. If left unspecified, it'll point automatically to
      the geoid resource shipped with S1 Tiling.

      .. _paths.dem_database:
  * - ``dem_database``
    - Path to DEM (``.gpkg``) database.
      |br|
      By default points to the internal :file:`shapefile/srtm_tiles.gpkg` file
      which knows the geometry of SRTM 30 DEM files.

      .. _paths.dem_dir:
  * - ``dem_dir``
    - Path to DEM files.

      .. _paths.dem_format:
  * - ``dem_format``
    - Filename format string to locate the DEM file associated to an
      *identifier* within the :ref:`[PATHS].dem_dir <paths.dem_dir>` directory.
      |br|
      By default associates the ``id`` key of tiles found in the :ref:`DEM
      database <paths.dem_database>` to :file:`{{id}}.hgt`. One may want to use
      the keys from `eotile <https://github.com/CS-SI/eotile>`_ DEM database
      like for instance :file:`{{Product10}}.tif` for Copernicus 30m DEM.

      .. _paths.srtm:
  * - ``srtm``
    - **(deprecated)** Use :ref:`[PATHS].dem_dir <paths.dem_dir>`. Path to SRTM files.

.. _DataSource:

``[DataSource]`` section
++++++++++++++++++++++++

.. list-table::
  :widths: auto
  :header-rows: 1
  :stub-columns: 1

  * - Option
    - Description

      .. _DataSource.download:
  * - ``download``
    - If ``True``, activates the downloading from specified data provider for
      the ROI, otherwise only local S1 images already in :ref:`s1_images
      <paths.s1_images>` will be processed.

      .. _DataSource.eodag_config:
  * - ``eodag_config``
    - Designates where the EODAG configuration file is expected to be found.
      |br|
      Default value: :file:`%(HOME)s/.config/eodag/eodag.yml`.

      From S1Tiling point of view, EODAG configuration file will list the
      authentification credentials for the know providers and their respective
      priorities.
      |br|
      See `EODAG § on Configure EODAG
      <https://eodag.readthedocs.io/en/latest/getting_started_guide/configure.html>`_

      For instance, given a PEPS account, :file:`$HOME/.config/eodag/eodag.yml` could
      contain

      .. code-block:: yaml

          peps:
              auth:
                  credentials:
                      username: THEUSERNAME
                      password: THEPASSWORD


      .. _DataSource.nb_parallel_downloads:
  * - ``nb_parallel_downloads``
    - Number of parallel downloads (+ unzip) of source products.

      .. warning::

          Don't abuse this setting as the data provider may not support too many
          parallel requests.


      .. _DataSource.roi_by_tiles:
  * - ``roi_by_tiles``
    - The Region of Interest (ROI) for downloading is specified in roi_by_tiles
      which will contain a list of MGRS tiles. If ``ALL`` is specified, the
      software will download all images needed for the processing (see
      :ref:`Processing`)

      .. code-block:: ini

          [DataSource]
          roi_by_tiles : 33NWB

      .. _DataSource.platform_list:
  * - ``platform_list``
    - Defines the list of platforms from where come the products to download
      and process.
      Valid values are ``S1A`` or ``S1B``.

      .. _DataSource.polarisation:
  * - ``polarisation``
    - Defines the polarisation mode of the products to download and process.
      Only six values are valid: ``HH-HV``, ``VV-VH``, ``VV``, ``VH``, ``HV``,
      and ``HH``.

      .. _DataSource.orbit_direction:
  * - ``orbit_direction``
    - Download only the products acquired in ascending (``ASC``) or in
      descending (``DES``) order.  By default (when left unspecified), no
      filter is applied.

      .. warning::
        Each relative orbit is exclusive to one orbit direction,
        :ref:`orbit_direction <DataSource.orbit_direction>` and
        :ref:`relative_orbit_list <DataSource.relative_orbit_list>` shall be
        considered as exclusive.

      .. _DataSource.relative_orbit_list:
  * - ``relative_orbit_list``
    - Download only the products from the specified relative orbits. By default
      (when left unspecified), no filter is applied.

      .. warning::
        Each relative orbit is exclusive to one orbit direction,
        :ref:`orbit_direction <DataSource.orbit_direction>` and
        :ref:`relative_orbit_list <DataSource.relative_orbit_list>` shall be
        considered as exclusive.

      .. _DataSource.first_date:
  * - ``first_date``
    - Initial date in ``YYYY-MM-DD`` format.

      .. _DataSource.last_date:
  * - ``last_date``
    - Final date in ``YYYY-MM-DD`` format.

      .. _DataSource.tile_to_product_overlap_ratio:
  * - ``tile_to_product_overlap_ratio``
    - Percentage of tile area to be covered for a single or a pair of
      Sentinel-1 products to be retained.

      The number is expected as an integer in the [1..100] range.

.. _Mask:

``[Mask]`` section
++++++++++++++++++

.. list-table::
  :widths: auto
  :header-rows: 1
  :stub-columns: 1

  * - Option
    - Description

      .. _Mask.generate_border_mask:
  * - ``generate_border_mask``
    - This option allows you to choose if you want to generate border masks of
      the S2 image file produced. Values are ``True`` or ``False``.


.. _Processing:

``[Processing]`` section
++++++++++++++++++++++++

.. list-table::
  :widths: auto
  :header-rows: 1
  :stub-columns: 1

  * - Option
    - Description

      .. _Processing.cache_dem_by:
  * - ``cache_dem_by``
    - Tells whether DEM and Geoid files are copied in a temporary directory, or
      if symbolic links are to be created.

      For performance reasons with OTB 7.X, it's better to regroup the minimal
      subset of the DEM files required for processing. Symbolic links work
      fine most of the time, however if the files are on a remote shared
      filesystem (GPFS, NAS...), performances will be degraded. In those cases,
      it's better to copy the required DEM files on a local filesystem.

      :ref:`Geoid file <paths.geoid_file>` will be also copied (or symlinked),
      but in :samp:`{{tmp}}/geoid/`. It won't be removed automatically.  You
      can also do it manually before running S1Tiling.

      Two values are supported for this option: ``copy`` and ``symlink``.
      (default: ``symlink``).

      .. _Processing.calibration:
  * - ``calibration``
    - Defines the calibration type: ``gamma``, ``beta``, ``sigma``, or
      ``normlim``.

      .. _Processing.remove_thermal_noise:
  * - ``remove_thermal_noise``
    - Activate the thermal noise removal in the images. Values are ``True`` or ``False``.

      .. important::

         This feature requires a version of OTB >= 7.4.0

      .. _Processing.lower_signal_value:
  * - ``lower_signal_value``
    - Noise removal may set some pixel values to 0.
      However, 0, is currently reserved by S1Tiling chain as a "nodata" value
      introduced by :ref:`Margin Cutting<cutting-proc>` and
      :ref:`Orthorectification <orthorectification-proc>`.

      This parameter defines which value to use instead of 0 when :ref:`noise is
      removed <Processing.remove_thermal_noise>`.  By default: 1e-7 will be
      used.

      .. _Processing.nodata:
  * - ``nodata.LIA``
    - Nodata value to use in :ref:`LIA files <lia-files>`

      .. _Processing.output_spatial_resolution:
  * - ``output_spatial_resolution``
    - Pixel size (in meters) of the output images

      .. _Processing.tiles_shapefile:
  * - ``tiles_shapefile``
    - Path and filename of the tile shape definition (ESRI Shapefile). If left
      unspecified, it'll point automatically to the `Features.shp` shapefile
      resource shipped with S1 Tiling.

      .. _Processing.orthorectification_gridspacing:
  * - ``orthorectification_gridspacing``
    - Grid spacing (in meters) for the interpolator in the orthorectification
      process for more information, please consult the `OTB OrthoRectification
      application
      <https://www.orfeo-toolbox.org/CookBook/Applications/app_OrthoRectification.html>`_.

      A nice value is 4 x output_spatial_resolution

      .. _Processing.orthorectification_interpolation_method:
  * - ``orthorectification_interpolation_method``
    - Interpolation method used in the orthorectification process
      for more information, please consult the `OTB OrthoRectification
      application
      <https://www.orfeo-toolbox.org/CookBook/Applications/app_OrthoRectification.html>`_.

      Default value is set to nearest neighbor interpolation (nn) to keep compatibilty with previous results
      By the way linear method could be more interesting.
      Note that the bco method is not currently supported

      .. _Processing.tiles:
  * - ``tiles``, ``tiles_list_in_file``
    - Tiles to be processed.
      The tiles can be given as a list:

      * ``tiles``: list of tiles (comma separated). Ex:

        .. code-block:: ini

            tiles: 33NWB,33NWC

      * tiles_list_in_file: tile list in a ASCII file. Ex:

        .. code-block:: ini

            tiles_list_in_file : ~/MyListOfTiles.txt

      .. _Processing.mode:
  * - ``mode``
    - Running mode:

      - ``Normal``: prints normal, warning and errors on screen
      - ``debug``: also prints debug messages, and forces
        ``$OTB_LOGGER_LEVEL=DEBUG``
      - ``logging``: saves logs to files


      Ex.:

      .. code-block:: ini

        mode : debug logging

      .. _Processing.nb_parallel_processes:
  * - ``nb_parallel_processes``
    - Number of processes to be running in :ref:`parallel <parallelization>`
      |br|
      This number defines the number of Dask Tasks (and indirectly of OTB
      applications) to be executed in parallel.

      .. note::
        For optimal performances, ``nb_parallel_processes*nb_otb_threads``
        should be <= to the number of cores on the machine.

      .. _Processing.ram_per_process:
  * - ``ram_per_process``
    - RAM allowed per OTB application pipeline, in MB.

      .. _Processing.nb_otb_threads:
  * - ``nb_otb_threads``
    - Numbers of threads used by each OTB application. |br|

      .. note::
        For optimal performances, ``nb_parallel_processes*nb_otb_threads``
        should be <= to the number of cores on the machine.

      .. _Processing.produce_lia_map:
  * - ``produce_lia_map``
    - When :ref:`LIA sine map <lia-files>` is produced, we may also desire the
      angle values in degrees (x100).

      Possible values are:

      :``True``:         Do generate the angle map in degrees x 100.
      :``False``:        Don't generate the angle map in degrees x 100.

      .. note::
        This option will be ignored when no LIA sine map is required. The LIA
        sine map is produced by :ref:`S1LIAMap program <scenario.S1LIAMap>` ,
        or when :ref:`calibration mode <Processing.calibration>` is
        ``"normlim"``.

      .. _Processing.dem_warp_resampling_method:
  * - ``dem_warp_resampling_method``
    - DEM files projected on S2 tiles are required to produce :ref:`LIA maps
      <lia-files>`.
      This parameters permits to select the resampling method that
      :external:std:doc:`gdalwarp <programs/gdalwarp>` will use.

      The possible values are: ``near``, ``bilinear``, ``cubic``,
      ``cubicspline``, ``lanczos``, ``average``, ``rms``, ``mode``, ``max``,
      ``min``, ``med``, ``q1``, ``q3`` and ``qum``.

      .. _Processing.override_azimuth_cut_threshold_to:
  * - ``override_azimuth_cut_threshold_to``
    - Permits to override the analysis on whether top/bottom lines shall be
      forced to 0 in :ref:`cutting step <cutting-proc>`. |br|

      Possible values are:

      :``True``:         Force cutting at the 1600th upper and the 1600th lower
                         lines.
      :``False``:        Force to keep every line.
      :not set/``None``: Default analysis heuristic is used.

      .. warning::
        This option is not meant to be used. It only makes sense in some very
        specific scenarios like tests.


      .. _Processing.fname_fmt:
  * - ``fname_fmt.*``
    - Set of filename format templates that permits to override the default
      filename formats used to generate filenames.

      The filename formats can be overridden for both intermediary and final
      products. Only the final products are documented here. Filename formats
      for intermediary products are best left alone.

      If you change any, make sure to not introduce ambiguity by removing a
      field that would be used to distinguish two unrelated products.

      Available fields come from :func:`internal metadata <s1tiling.libs.steps.StepFactory.complete_meta>`. The main
      ones of interest are:

      .. list-table::
        :widths: auto
        :header-rows: 1
        :stub-columns: 1

        * - Field
          - Content
          - Applies to geometry

        * - flying_unit_code
          - ``s1a``, ``s1b``
          - S1/S2
        * - tile_name
          - ex: ``33NWB``
          - S2

        * - polarisation
          - ``hh``, ``hv``, ``vh``, ``vv``
          - S1/S2

        * - orbit_direction
          - ``ASC``/``DES``
          - S1/S2

        * - orbit
          - 5-digits number that identifies the S1 orbit
          - S1/S2

        * - acquisition_time
          - the full timestamp (:samp:`{yymmdd}t{hhmmss}`)
          - S1/S2

        * - acquisition_day
          - only the day (:samp:`{yymmdd}txxxxxx`)
          - S1/S2

        * - acquisition_stamp
          - either the full timestamp (:samp:`{yymmdd}t{hhmmss}`), or the day
            (:samp:`{yymmdd}txxxxxx`)
          - S1/S2

        * - LIA_kind
          - ``LIA``/``sin_LIA``
          - S2

        * - basename
          - Filename of initial S1 image.
          - S1

        * - rootname
          - ``basename`` without the file extension.
          - S1

        * - calibration_type
          - ``beta``/``gamma``/``sigma``/``dn``/``Normlim``
          - S1/S2

        * - polarless_basename
          - Same as ``basename`` (with file extension), but without
            ``polarisation`` field. Used when the product only depends on the
            S1 image geometry and not its content.
          - S1

        * - polarless_rootname
          - Same as ``rootname`` (without file extension), but without
            ``polarisation`` field. Used when the product only depends on the
            S1 image geometry and not its content.
          - S1

      .. _Processing.fname_fmt.concatenation:
  * - ``fname_fmt.concatenation``
    - File format pattern for :ref:`concatenation products <full-S2-tiles>`,
      for β°, σ° and γ° calibrations.

      Default value: :samp:`{{flying_unit_code}}_{{tile_name}}_{{polarisation}}_{{orbit_direction}}_{{orbit}}_{{acquisition_stamp}}.tif`

      .. _Processing.fname_fmt.lia_corrected:
  * - ``fname_fmt.s2_lia_corrected``
    - File format pattern for :ref:`concatenation products <full-S2-tiles>`
      when NORMLIM calibrated.

      Default value: :samp:`{{flying_unit_code}}_{{tile_name}}_{{polarisation}}_{{orbit_direction}}_{{orbit}}_{{acquisition_stamp}}_NormLim.tif`

      .. _Processing.fname_fmt.lia_product:
  * - ``fname_fmt.lia_product``
    - File format pattern for LIA and sin(LIA) files

      Default value: :samp:`{{LIA_kind}}_{{flying_unit_code}}_{{tile_name}}_{{orbit_direction}}_{{orbit}}.tif`

      .. _Processing.fname_fmt.filtered:
  * - ``fname_fmt.filtered``
    - File format pattern for :ref:`filtered files <filtered-files>`

      Default value: :samp:`{{flying_unit_code}}_{{tile_name}}_{{polarisation}}_{{orbit_direction}}_{{orbit}}_{{acquisition_stamp}}_filtered.tif`
      for β°, σ° and γ° calibrations,

      Default value: :samp:`{{flying_unit_code}}_{{tile_name}}_{{polarisation}}_{{orbit_direction}}_{{orbit}}_{{acquisition_stamp}}_NormLim_filtered.tif` when NORMLIM calibrated.

      .. _Processing.dname_fmt:
  * - ``dname_fmt.*``
    - Set of directory format templates that permits to override the default
      directories where products are generated.

      The directory formats can only be overridden for final products.

      The only fields available are:

      .. list-table::
        :widths: auto
        :header-rows: 1
        :stub-columns: 1

        * - Field
          - Reference to

        * - :samp:`{{tile_name}}`
          - Name of the related tile.
        * - :samp:`{{out_dir}}`
          - :ref:`[PATHS].output <paths.output>`
        * - :samp:`{{tmp_dir}}`
          - :ref:`[PATHS].tmp <paths.tmp>`
        * - :samp:`{{lia_dir}}`
          - :ref:`[PATHS].lia <paths.lia>`

      .. list-table::
        :widths: auto
        :header-rows: 1
        :stub-columns: 1

        * - Products from
          - Option ``dname_fmt.?``
          - Default value

            .. _Processing.dname_fmt.tiled:
        * - :ref:`(β°/σ°/γ°/NORMLIM) Final tiled product <full-S2-tiles>`
          - ``.tiled``
          - :samp:`{{out_dir}}/{{tile_name}}`

            .. _Processing.dname_fmt.mask:
        * - :ref:`Masks <mask-files>`
          - ``.mask``
          - :samp:`{{out_dir}}/{{tile_name}}`

            .. _Processing.dname_fmt.lia_product:
        * - :ref:`degree(LIA) and sin(LIA) <lia-files>`
          - ``.lia_product``
          - :samp:`{{lia_dir}}`

            .. _Processing.dname_fmt.filtered:
        * - :ref:`Filtering <filtered-files>`
          - ``.filtered``
          - :samp:`{{out_dir}}/filtered/{{tile_name}}`

      .. _Processing.creation_options:
  * - ``creation_options.*``
    - Set of extra options to create certain products. Creation options take a
      first and optional pixel type (``uint8``, ``float64``...) and a list of
      `GDAL creation options
      <https://gdal.org/drivers/raster/gtiff.html#creation-options>`_.

      .. list-table::
        :widths: auto
        :header-rows: 1
        :stub-columns: 1

        * - Products from
          - Option ``creation_options.?``
          - Default value

            .. _Processing.creation_options.tiled:
        * - Orthorectification, :ref:`(β°/σ°/γ°/NORMLIM) Concatenation
            <full-S2-tiles>`...
          - ``.tiled``
          - ``COMPRESS=DEFLATE&gdal:co:PREDICTOR=3``

            .. _Processing.creation_options.filtered:
        * - :ref:`Filtering <filtered-files>`
          - ``.filtered``
          - ``COMPRESS=DEFLATE&gdal:co:PREDICTOR=3``

            .. _Processing.creation_options.mask:
        * - :ref:`Masks <mask-files>`
          - ``.mask``
          - ``uint8 COMPRESS=DEFLATE``

            .. _Processing.creation_options.lia_deg:
        * - :ref:`LIA (in degrees * 100) <lia-files>`
          - ``.lia_deg``
          - ``uint16 COMPRESS=DEFLATE&gdal``

            .. _Processing.creation_options.lia_sin:
        * - :ref:`sin(LIA) <lia-files>`
          - ``.lia_sin``
          - ``COMPRESS=DEFLATE&gdal:co:PREDICTOR=3``

.. _Filtering:

``[Filtering]`` section
+++++++++++++++++++++++

.. note:: Multitemporal filtering is not yet integrated in S1Tiling.


.. list-table::
  :widths: auto
  :header-rows: 1
  :stub-columns: 1

  * - Option
    - Description

      .. _Filtering.filter:
  * - ``filter``
    - If ``none`` or empty, then no filtering is done. Otherwise the following
      spatial speckling filter methods from :external:doc:`OTB Despeckle application
      <Applications/app_Despeckle>` are supported: ``Lee``, ``Frost``,
      ``Gammamap``, ``Kuan``.

      .. _Filtering.window_radius:
  * - ``window_radius``
    - Sets the window radius for the spatial filtering. |br|
      Take care that it is a radius, i.e. radius=1 means the filter does an 3x3
      pixels averaging.

      .. _Filtering.deramp:
  * - ``deramp``
    - Deramp factor -- for Frost filter only. |br|
      Factor use to control the exponential function used to weight effect of
      the distance between the central pixel and its neighborhood. Increasing
      the deramp parameter will lead to take more into account pixels farther
      from the center and therefore increase the smoothing effects.

      .. _Filtering.nblooks:
  * - ``nblooks``
    - Number of looks -- for all but Frost => Lee, Gammamap and Kuan

      .. _Filtering.keep_non_filtered_products:
  * - ``keep_non_filtered_products``
    - If not caring for non-filtered product (and if filter method is
      specified), then the orthorectified and concatenated products won't be
      considered as mandatory and they will not be kept at the end of the
      processing.
      This (exclusion) feature cannot be used alongside
      :ref:`[Mask].generate_border_mask <Mask.generate_border_mask>` (i.e.
      ``keep_non_filtered_products`` cannot be False if
      ``generate_border_mask`` is True)

      .. warning::
           Note: This feature is only supported after LIA calibration as of
           V1.0 of S1Tiling.  See Issue `#118
           <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/118>`_.


.. commented-out-to-be-implemented:
      .. _Filtering.reset_outcore:
  * - ``reset_outcore``
    - - If ``True``, the outcore of the multiImage filter is reset before
        filtering. It means that the outcore is recomputed from scratch with
        the new images only.
      - If ``False``, the outcore is updated with the new images. Then, the
        outcore integrates previous images and new images.


.. index:: Log configuration

Log configuration
-----------------
Default logging configuration is provided in ``S1Tiling`` installing directory.

It can be overridden by dropping a file similar to
:download:`../s1tiling/logging.conf.yaml` in the same directory as the one
where the :ref:`request configuration file <request-config-file>` is. The file
is expected to follow :py:mod:`logging configuration <logging.config>` file
syntax.

.. warning::
   This software expects the specification of:

   - ``s1tiling``, ``s1tiling.OTB`` :py:class:`loggers <logging.Logger>`;
   - and ``file`` and ``important`` :py:class:`handlers <logging.Handler>`.

When :ref:`mode <Processing.mode>` contains ``logging``, we make sure that
``file`` and ``important`` :py:class:`handlers <logging.Handler>` are added to
the handlers of ``root`` and ``distributed.worker`` :py:class:`loggers
<logging.Logger>`. Note that this is the default configuration.

When :ref:`mode <Processing.mode>` contains ``debug`` the ``DEBUG`` logging
level is forced into ``root`` logger, and ``$OTB_LOGGER_LEVEL`` environment
variable is set to ``DEBUG``.

.. _clusters:

.. index:: Clusters

Working on clusters
-------------------

.. todo::

  By default S1Tiling works on single machines. Internally it relies on
  :py:class:`distributed.LocalCluster` a small adaptation would be required to
  work on a multi-nodes cluster.

.. warning::

  When executing multiple instances of S1Tiling simultaneously, make sure to
  use different directories for:

  - logs -- running S1Tiling in different directories, like :file:`$TMPDIR/`
    on TREX, should be enough
  - storing :ref:`input files <paths.s1_images>`, like for instance
    :file:`$TMPDIR/data_raw/` on TREX for instance.

.. _exit_codes:

Process return code
-------------------

The following exit code are produced when :program:`S1Processor` returns:

.. list-table::
  :widths: auto
  :header-rows: 1
  :stub-columns: 1

  * - Exit code
    - Description

  * - 0
    - Execution successful
  * - 66
    - Some OTB tasks could not be executed properly. See the final report in
      the main log.
  * - 67
    - Downloading error. See the log produced.
  * - 68
    - When offline S1 data could not be retrieved before the configured
      timeout, the associated S2 products will not be generated and this exit
      code will be used. See the log produced.

      If more critical errors occur, this exit will be superceded.
  * - 69
    - .. todo::

        Output disk full
  * - 70
    - .. todo::

        Cache disk full (when using option ``--cache-before-ortho``)
  * - 71
    - An empty data safe has been found and needs to be removed so it can be
      fetched again. See the log produced.
  * - 72
    - Error detected in the configuration file. See the log produced.
  * - 73
    - While ``ALL`` Sentinel-2 tiles for which there exist an overlapping
      Sentinel-1 product have been :ref:`requested <DataSource.roi_by_tiles>`,
      no Sentinel-1 product has been found in the :ref:`requested time range
      <DataSource.first_date>`. See the log produced.
  * - 74
    - No Sentinel-1 product has been found that intersects the :ref:`requested
      Sentinel-2 tiles <DataSource.roi_by_tiles>` within the :ref:`requested
      time range <DataSource.first_date>`.

      If :ref:`downloading <DataSource.download>` has been disabled, S1
      products are searched in the :ref:`local input directory
      <paths.s1_images>`.  See the log produced.
  * - 75
    - Cannot find all the :ref:`DEM products <paths.dem_dir>` that cover the
      :ref:`requested Sentinel-2 tiles <DataSource.roi_by_tiles>`. See the log
      produced.
  * - 76
    - :ref:`Geoid file <paths.geoid_file>` is missing or the specified path is
      incorrect. See the log produced.
  * - 77
    - Some processing cannot be done because external applications cannot
      be executed. Likelly OTB and/or NORMLIM related applications aren't
      correctly installed.
      See the log produced.

  * - any other
    - Unknown error. It could be related to `Bash
      <https://www.redhat.com/sysadmin/exit-codes-demystified>`_ or to `Python
      <https://docs.python.org/3/library/os.html#os._exit>`_ reserved error
      codes.
