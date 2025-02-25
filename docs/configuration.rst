.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

======================================================================
Configuration
======================================================================

.. _request-config-file:

.. index:: Request configuration file

Request Configuration file
--------------------------

The request configuration file passed to :ref:`S1Processor` is in ``.ini``
format.  It is expected to contain the following entries.

.. note::
   :ref:`S1LIAMap` and :ref:`S1IAMap` work with a subset of the following
   configuration keys. Unsupported keys will simply be ignored.

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
      that directory -- the 1000 last products that have been downloaded.
      |br|
      This enables to cache downloaded S1 images in beteen runs.

      .. _paths.output:
  * - ``output``
    - Where products are generated.

      .. _paths.ia:
  * - ``ia``
    - Where (non Local) Incidence Maps and sin(IA) products are generated. Its
      default value is ``{output}/_IA``.

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

      .. _paths.eof_dir:
  * - ``eof_dir``
    - Where precise orbit orbit files (EOF) are expected to be found, or where
      they would be downloaded on the fly.
      Default value is ``{output}/_EOF``.

      See also :ref:`faq.eof`.


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

      For instance, given a PEPS account, :file:`$HOME/.config/eodag/eodag.yml`
      could contain

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

          Don't abuse this setting as the data provider may not support too
          many parallel requests.

      .. _DataSource.roi_by_tiles:
  * - ``roi_by_tiles``
    - The Region of Interest (ROI) for downloading is specified in
      ``roi_by_tiles`` which will contain a list of MGRS tile names. If ``ALL``
      is specified, the software will download all images needed for the
      processing (see :ref:`Processing`)

      .. code-block:: ini

          [DataSource]
          roi_by_tiles : 33NWB

      .. _DataSource.platform_list:
  * - ``platform_list``
    - Defines the list of platforms from where come the products to download
      and process.
      Valid values are ``S1A`` or ``S1B``.

      .. warning::
        A single value is expected in :ref:`NORMLIM scenarios <scenarios>`.

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
      .. warning::
        A single value is expected in :ref:`NORMLIM and Ellipsoid Incide Angle
        scenarios <scenarios>`.

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
      the S2 image files produced. Values are ``True`` or ``False``.


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

      For performance reasons with OTB, it's better to regroup the minimal
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
    - Activate the thermal noise removal in the images. Values are ``True`` or
      ``False``.

      .. _Processing.lower_signal_value:
  * - ``lower_signal_value``
    - Noise removal may set some pixel values to 0.
      However, 0, is currently reserved by S1Tiling chain as a "no-data" value
      introduced by :ref:`Margin Cutting<cutting-proc>` and
      :ref:`Orthorectification <orthorectification-proc>`.

      This parameter defines which value to use instead of 0 when :ref:`noise is
      removed <Processing.remove_thermal_noise>`.  By default: 1e-7 will be
      used.

      .. _Processing.nodata:
  * - ``nodata.IA``
    - No-data value to use in :ref:`IA files <ia-files>`

  * - ``nodata.LIA``
    - No-data value to use in :ref:`LIA files <lia-files>`

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
      process. For more information, please consult the `OTB OrthoRectification
      application
      <https://www.orfeo-toolbox.org/CookBook/Applications/app_OrthoRectification.html>`_.

      A nice value is ``4 x output_spatial_resolution``

      .. _Processing.orthorectification_interpolation_method:
  * - ``orthorectification_interpolation_method``
    - Interpolation method used in the orthorectification process.
      For more information, please consult the `OTB OrthoRectification
      application
      <https://www.orfeo-toolbox.org/CookBook/Applications/app_OrthoRectification.html>`_.

      Default value is set to nearest neighbor interpolation (``nn``) to keep
      compatibilty with previous results ; Linear method could be more
      interesting.
      Note that the bco method is not currently supported.

      .. _Processing.tiles:
  * - ``tiles``, ``tiles_list_in_file``
    - Tiles to be processed.
      The tiles can be given as a list:

      * ``tiles``: list of tiles (comma separated). Ex:

        .. code-block:: ini

            tiles: 33NWB, 33NWC

      * ``tiles_list_in_file``: tile list in a ASCII file. Ex:

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
        should be ≤ to the number of cores on the machine.

      .. _Processing.ram_per_process:
  * - ``ram_per_process``
    - RAM allowed per OTB application pipeline, in MB.

      .. _Processing.nb_otb_threads:
  * - ``nb_otb_threads``
    - Numbers of threads used by each OTB application. |br|

      .. note::
        For optimal performances, ``nb_parallel_processes*nb_otb_threads``
        should be ≤ to the number of cores on the machine.

      .. _Processing.produce_lia_map:
  * - ``produce_lia_map``
    - When :ref:`LIA sine map <lia-files>` are produced, we may also desire the
      angle values in degrees (x100).

      Possible values are:

      :``True``:         Do generate the angle map in degrees x 100.
      :``False``:        Don't generate the angle map in degrees x 100.

      .. note::
        This option will be ignored when no LIA sine map is required. The LIA
        sine map is produced by :ref:`S1LIAMap program <scenario.S1LIAMap>`, or
        when :ref:`calibration mode <Processing.calibration>` is ``"normlim"``.

      .. _Processing.ia_maps_to_produce:
  * - ``ia_maps_to_produce``
    - By default, :ref:`S1IAMap program <scenario.S1IAMap>` produce a map of
      the incidence angle to the WGS84 ellipsoid in degrees x 100. This option
      permis to select which of the 4 :ref:`IA maps <ia-files>` will be
      generated.

      :``deg``: map in degrees x 100
      :``cos``: cosine map
      :``sin``: sine map
      :``tan``: tangent map

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

        * - IA_kind
          - ``IA``/``cos_IA``/``sin_IA``/``tan_IA``
          - S2

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

      .. _Processing.fname_fmt.ia_product:
  * - ``fname_fmt.ia_product``
    - File format pattern for IA cos(IA), sin(IA) and tan(IA) files

      Default value: :samp:`{{IA_kind}}_{{flying_unit_code}}_{{tile_name}}_{{orbit}}.tif`

      .. _Processing.fname_fmt.lia_product:
  * - ``fname_fmt.lia_product``
    - File format pattern for LIA and sin(LIA) files

      Default value: :samp:`{{LIA_kind}}_{{flying_unit_code}}_{{tile_name}}_{{orbit}}.tif`

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
        * - :samp:`{{ia_dir}}`
          - :ref:`[PATHS].ia <paths.ia>`
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

            .. _Processing.dname_fmt.ia_product:
        * - :ref:`degree(IA), cos(IA), sin(IA) and tan(IA) <ia-files>`
          - ``.ia_product``
          - :samp:`{{ia_dir}}`

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

            .. _Processing.creation_options.ia_deg:
        * - :ref:`IA (in degrees * 100) <ia-files>`
          - ``.ia_deg``
          - ``uint16 COMPRESS=DEFLATE&gdal``

            .. _Procescosg.creation_options.ia_cos:
        * - :ref:`cos(IA) <ia-files>`
          - ``.ia_cos``
          - ``COMPRESS=DEFLATE&gdal:co:PREDICTOR=3``

            .. _Processing.creation_options.ia_sin:
        * - :ref:`sin(IA) <ia-files>`
          - ``.ia_sin``
          - ``COMPRESS=DEFLATE&gdal:co:PREDICTOR=3``

            .. _Procestang.creation_options.ia_tan:
        * - :ref:`tan(IA) <ia-files>`
          - ``.ia_tan``
          - ``COMPRESS=DEFLATE&gdal:co:PREDICTOR=3``

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
    - The following spatial speckling filter methods from :external:doc:`OTB
      Despeckle application <Applications/app_Despeckle>` are supported:
      ``Lee``, ``Frost``, ``Gammamap``, ``Kuan``. If ``none`` or empty, then
      no filtering is done.

      .. _Filtering.window_radius:
  * - ``window_radius``
    - Sets the window radius for the spatial filtering. |br|
      Be cautious: this does expect a radius, i.e. radius=1 means the filter
      does an 3x3 pixels averaging.

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

  By default, S1Tiling works on single machines. Internally it relies on
  :py:class:`distributed.LocalCluster` a small adaptation would be required to
  work on a multi-nodes cluster.

.. warning::

  When executing multiple instances of S1Tiling simultaneously, make sure to
  use different directories for:

  - logs -- running S1Tiling in different directories, like :file:`$TMPDIR/`
    on TREX, should be enough
  - storing :ref:`input files <paths.s1_images>`, like for instance
    :file:`$TMPDIR/data_raw/` on TREX for instance.

