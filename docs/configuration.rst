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
    - Input directory where **unzipped** Sentinel-1 input products are
      searched, and downloaded to thanks to `EODAG
      <https://github.com/CS-SI/eodag>`_.
      |br|
      S1Tiling will automatically take care to keep at most 1000 products in
      that directory -- the 1000 last products that have been downloaded.
      |br|
      This enables to cache downloaded S1 images in between runs.

      .. _paths.output:
  * - ``output``
    - Root output directory where products are generated.

      .. _paths.ia:
  * - ``ia``
    - Output directory for the generated (Ellipsoid) Incidence Maps and
      cos(IA)/sin(IA) products. Its default value is ``{output}/_IA``.

      .. _paths.lia:
  * - ``lia``
    - Output directory for the generated Local Incidence Maps and sin(LIA)
      products. Its default value is ``{output}/_LIA``.

      .. _paths.gamma_area:
  * - ``gamma_area``
    - Output directory for the generated γ Area products. Its default value is
      ``{output}/_GAMMA_AREA``.

      .. _paths.tmp:
  * - ``tmp``
    - Directory where :ref:`intermediary files <temporary-files>` are produced,
      and sometimes :ref:`cached <data-caches>` for longer periods.

      .. _paths.geoid_file:
  * - ``geoid_file``
    - Path to Geoid model. If left unspecified, it'll point automatically to
      the geoid resource shipped with S1 Tiling.

      .. warning:: Make sure to use an EGM2008 model for Copernicus DEM files.

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

      .. _paths.dem_info:
  * - ``dem_info``
    - DEM identifier to inject in the products GeoTIFF metadata under
      ``DEM_INFO`` key. If not defined, the last part (basename) of
      :ref:`[Paths].dem_dir <paths.dem_dir>` will be used.

      See § :ref:`scenario.choose_dem` entry for more detailled information.

      .. _paths.srtm:
  * - ``srtm``
    - **(deprecated)** Use :ref:`[PATHS].dem_dir <paths.dem_dir>`. Path to SRTM files.

      .. _paths.eof_dir:
  * - ``eof_dir``
    - Where precise orbit files (EOF) are expected to be found, or where they
      would be downloaded to on the fly.
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
    - When ``True``, enables the downloading:

      - of Sentinel-1 images, that intersects the ROI, from the specified data
        provider -- only local images already in :ref:`s1_images
        <paths.s1_images>` will be processed otherwise.
      - and of missing EOF products, in scenarios that require them (currently
        for the production of :ref:`LIA <scenario.S1LIAMap>` and :ref:`IA
        <scenario.S1IAMap>` maps).

      .. _DataSource.eodag_config:
  * - ``eodag_config``
    - Designates where the EODAG configuration file is expected to be found.
      |br|
      Default value is fetched in order: :samp:`${{EODAG_CFG_FILE}}` >
      :samp:`${{EODAG_CFG_DIR}}/eodag.yml` >
      :file:`%(HOME)s/.config/eodag/eodag.yml`.

      From S1Tiling point of view, EODAG configuration file will list the
      authentification credentials for the know providers and their respective
      priorities.
      |br|
      See :external+eodag:std:doc:`EODAG § on Configure EODAG
      <getting_started_guide/configure>`

      For instance, given a geodes account, :file:`$HOME/.config/eodag/eodag.yml`
      could contain

      .. code-block:: yaml

          geodes:
              auth:
                  credentials:
                      username: THEUSERNAME
                      password: THEPASSWORD

      .. _DataSource.nb_parallel_downloads:
  * - ``nb_parallel_downloads``
    - Number of parallel downloads (+ unzip) of source products.

      .. admonition:: deprecated

          This feature is currently disabled in current version of S1Tiling
          See `#190
          <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/190>`_.

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
    - Filter to restrict the list of Sentinel-1 platforms from where the
      products, to download and process, come from. |br|
      Valid values are comma separated lists of ``S1A``, ``S1B``, and ``S1C``.
      By default (when left unspecified), no filter is applied.

      .. warning::
        Only one single value is expected in :ref:`NORMLIM and Ellipsoid
        Incidence Angle scenarios <scenarios>`.

      .. _DataSource.polarisation:
  * - ``polarisation``
    - Filter on the polarisation mode of the Sentinel-1 products to download
      and process.  |br|
      Only six values are valid: ``HH-HV``, ``VV-VH``, ``VV``, ``VH``, ``HV``,
      and ``HH``.

      .. _DataSource.orbit_direction:
  * - ``orbit_direction``
    - Filter on the orbit direction of the Sentinel-1 products to download and
      process. |br|
      Only two values are valid: ``ASC`` (ascending mode order) and ``DSC``
      (descending mode order). By default (when left unspecified), no filter is
      applied.

      .. warning::
        Each relative orbit is exclusive to one orbit direction,
        :ref:`orbit_direction <DataSource.orbit_direction>` and
        :ref:`relative_orbit_list <DataSource.relative_orbit_list>` shall be
        considered as exclusive.

      .. _DataSource.relative_orbit_list:
  * - ``relative_orbit_list``
    - Filter to download and process only the Sentinel-1 products from the
      specified relative orbits. |br|
      Valid values are comma separated list of relative orbit numbers (∈
      [1..175]). By default (when left unspecified), no filter is applied.

      .. warning::
        Each relative orbit is exclusive to one orbit direction,
        :ref:`orbit_direction <DataSource.orbit_direction>` and
        :ref:`relative_orbit_list <DataSource.relative_orbit_list>` shall be
        considered as exclusive.
      .. warning::
        At least one value is expected in :ref:`NORMLIM and Ellipsoid Incidence
        Angle scenarios <scenarios>`.

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
    - Enable the generation of border masks for the S2-aligned image files
      produced. |br|
      Valid values are ``True`` or ``False``.


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
    - Defines the calibration type to apply: ``gamma``, ``beta``, ``sigma``,
      ``normlim`` or ``gamma_naught_rtc``.

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
  * - ``nodata.*``
    -
      .. list-table::
        :widths: auto
        :header-rows: 1
        :stub-columns: 1

        * - Option
          - Description

            .. _Processing.nodata.IA:
        * - ``nodata.IA``
          - No-data value to use in :ref:`IA files <ia-files>`

            .. _Processing.nodata.LIA:
        * - ``nodata.LIA``
          - No-data value to use in :ref:`LIA files <lia-files>`

            .. _Processing.nodata.RTC:
        * - ``nodata.RTC``
          - No-data value to use when applying :ref:`Gamma Area map
            <apply_gamma_area-proc>`

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
      process. For more information, please consult `OTB OrthoRectification
      application documentation
      <https://www.orfeo-toolbox.org/CookBook/Applications/app_OrthoRectification.html>`_.

      A nice value is ``4 x output_spatial_resolution``

      .. _Processing.orthorectification_interpolation_method:
  * - ``orthorectification_interpolation_method``
    - Interpolation method used in the orthorectification process.
      For more information, please consult `OTB OrthoRectification application
      documentation
      <https://www.orfeo-toolbox.org/CookBook/Applications/app_OrthoRectification.html>`_.

      Default value is set to nearest neighbor interpolation (``nn``) to keep
      compatibilty with previous results ; Linear method could be more
      interesting.
      Note that the ``bco`` method is not currently supported.

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
    - Number of processes to run in :ref:`parallel <parallelization>`
      |br|
      This number defines the number of Dask Tasks (and indirectly of OTB
      applications) that will be executed in parallel.

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
      permits to select which of the 4 :ref:`IA maps <ia-files>` will be
      generated.

      :``deg``: map in degrees x 100
      :``cos``: cosine map
      :``sin``: sine map
      :``tan``: tangent map

      .. _Processing.dem_warp_resampling_method:
  * - ``dem_warp_resampling_method``
    - DEM files projected on S2 tiles are required to produce :ref:`LIA maps
      <lia-files>` and  :ref:`GAMMA_AREA maps <gamma_area_s2-files>`.
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
    - Set of filename format templates that permit to override the default
      filename formats used to generate filenames.

      The filename formats can be overridden for both intermediary and final
      products. Only the final products are documented here. Filename formats
      for intermediary products are best left alone.

      If you change any, make sure to not introduce ambiguity by removing a
      field that would be used to distinguish two unrelated products.

      Available fields come from :func:`internal metadata
      <s1tiling.libs.steps.StepFactory.complete_meta>`. The main ones of
      interest are:

      .. list-table::
        :widths: auto
        :header-rows: 1
        :stub-columns: 1

        * - Field
          - Content
          - Applies to geometry

        * - flying_unit_code
          - ``s1a``, ``s1b``, ``s1c``
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
          - 3-digits, 0-padded, number that identifies the S1 product relative
            orbit
          - S1/S2

        * - absolute_orbit
          - 5-digits, 0-padded, number that identifies the S1 product absolute
            orbit
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

        * - acquisition_start
          - the full timestamp (:samp:`{yymmdd}t{hhmmss}`) of the first
            Sentinel-1 input image used in a :ref:`concatenation
            <concatenation-proc>`

            .. warning::

                In start-over situations this key, unlike
                :samp:`{{acquisition_day}}`, cannot permit to known whether an
                existing S2 product has been generated from a single, or from
                two, S1 input image(s).
                |br|
                While S1Tiling avoids generating a S2 output when a S1 input
                has been detected missing in on-line mode, it has no way of
                knowing in :ref:`offline mode <DataSource.download>`. In which
                case partial S2 products could be generated, but then with a
                name that'll make them impossible to distinguish from complete
                S2 products if :samp:`{{acquisition_start}}` is used.
                Starting S1Tiling again over already generated products, this
                time in on-line mode, will not update partial S2 products.

          - S2

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
          - :samp:`{{basename}}` without the file extension.
          - S1

        * - calibration_type
          - ``beta``/``gamma``/``sigma``/``dn``/``Normlim``/``gamma_naught_rtc``
          - S1/S2

        * - polarless_basename
          - Same as :samp:`{{basename}}` (with file extension), but without
            :samp:`{{polarisation}}` field. Used when the product only depends
            on the S1 image geometry and not its content.
          - S1

        * - polarless_rootname
          - Same as :samp:`{{rootname}}` (without file extension), but without
            :samp:`{{polarisation}}` field. Used when the product only depends
            on the S1 image geometry and not its content.
          - S1

        * - filter_method
          - When spatial filtering is activated: ``lee``, ``frost``,
            ``Gammamap``, or ``kuan``
          - S2


      .. note::

        :ref:`All Python standard format specifiers <formatspec>` plus extra
        conversion fields are supported:

        - ``!c`` will capitalize a field -- only the first letter will be in
          uppercase
        - ``!l`` will output the field in lowercase
        - ``!u`` will output the field in uppercase

        .. admonition:: example

            Theia filename format (ex.
            :file:`S1A_L1ORT_31TCH_VH_SIG_ASC_132_20250218t174708.tif`)
            would be expressed in the following way:

            .. code:: ini

                fname_fmt.concatenation : {flying_unit_code!u}_L1ORT_{tile_name}_{polarisation!u}_{calibration_type!u:.3}_{orbit_direction}_{orbit}_{acquisition_start}.tif
                fname_fmt.filtered      : {flying_unit_code!u}_L1ORT_{tile_name}_{polarisation!u}_{calibration_type!u:.3}_{orbit_direction}_{orbit}_{acquisition_start}_filtered_{filter_method!u:.3}.tif

      .. _Processing.fname_fmt.concatenation:
  * - ``fname_fmt.concatenation``
    - File format pattern for :ref:`concatenation products <full-S2-tiles>`,
      for β°, σ° and γ° calibrations.

      Default value: {fname_fmt_concatenation}

      .. _Processing.fname_fmt.lia_corrected:
  * - ``fname_fmt.s2_lia_corrected``
    - File format pattern for :ref:`concatenation products <full-S2-tiles>`
      when NORMLIM calibrated.

      Default value: {fname_fmt_lia_corrected}

      .. _Processing.fname_fmt.ia_product:
  * - ``fname_fmt.ia_product``
    - File format pattern for IA cos(IA), sin(IA) and tan(IA) files

      Default value: {fname_fmt_ia_product}

      .. _Processing.fname_fmt.lia_product:
  * - ``fname_fmt.lia_product``
    - File format pattern for LIA and sin(LIA) files

      Default value: {fname_fmt_lia_product}

      .. _Processing.fname_fmt.gamma_area_corrected:
  * - ``fname_fmt.s2_gamma_area_corrected``
    - File format pattern for :ref:`concatenation products <full-S2-tiles>`
      when GammaNaughtRTC calibrated.

      Default value: {fname_fmt_gamma_area_corrected}

      .. _Processing.fname_fmt.gamma_area_product:
  * - ``fname_fmt.gamma_area_product``
    - File format pattern for GAMMA_AREA files

      Default value: {fname_fmt_gamma_area}

      .. _Processing.fname_fmt.filtered:
  * - ``fname_fmt.filtered``
    - File format pattern for :ref:`filtered files <filtered-files>` in
      standard calibrations

      Default value:

      - {fname_fmt_filtered} for β°, σ° and γ° calibrations,

  * - ``fname_fmt.filtered_calib``
    - File format pattern for :ref:`filtered files <filtered-files>` in terrain
      corrected calibrations

      Default value:

      - {fname_fmt_filtered_calib}, IOW: …
      - {fname_fmt_filtered_lia} when NORMLIM calibrated.
      - {fname_fmt_filtered_rtc} when GammaNaughtRTC calibrated.

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
        * - :samp:`{{gamma_area_dir}}`
          - :ref:`[PATHS].gamma_area <paths.gamma_area>`

      .. list-table::
        :widths: auto
        :header-rows: 1
        :stub-columns: 1

        * - Products from
          - Option ``dname_fmt.?``
          - Default value

            .. _Processing.dname_fmt.tiled:
        * - :ref:`(β°/σ°/γ°/NORMLIM/γ°RTC) Final tiled product <full-S2-tiles>`
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

            .. _Processing.dname_fmt.gamma_area_product:
        * - :ref:`GAMMA_AREA <gamma_area_s2-files>`
          - ``.gamma_area_product``
          - :samp:`{{gamma_area_dir}}`

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

        * - Products from
          - Option ``creation_options.?``
          - Default value

            .. _Processing.creation_options.tiled:
        * - Orthorectification, :ref:`(β°/σ°/γ°/NORMLIM/γ°RTC) Concatenation
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
          - ``uint16 COMPRESS=DEFLATE``

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
          - ``uint16 COMPRESS=DEFLATE``

            .. _Processing.creation_options.lia_sin:
        * - :ref:`sin(LIA) <lia-files>`
          - ``.lia_sin``
          - ``COMPRESS=DEFLATE&gdal:co:PREDICTOR=3``

            .. _Processing.creation_options.s1_on_dem:
        * - :ref:`S1 image information projected on DEM <S1_on_dem-files>`
          - ``.s1_on_dem``
          - ``float32 COMPRESS=DEFLATE, BIGTIFF=YES, PREDICTOR=3, TILED=YES, BLOCKXSIZE=1024, BLOCKYSIZE=1024``

            .. warning::

              This default ``float32`` setting is insufficient for a good
              precision (ECEF coordinates < 1 meter). Yet it has been chosen
              for people working on machines that don't have more than 60 GB of
              memory. See the note in :ref:`γ area map procuction scenario
              <scenario.S1GammaAreaMap.ram-greedy>`.

            .. _Processing.creation_options.gamma_area:
        * - :ref:`GAMMA_AREA in meters square <gamma_area_s2-files>`
          - ``.gamma_area``
          - ``float32 COMPRESS=DEFLATE``

      .. _Processing.disable_streaming:
  * - ``disable_streaming.*``
    - Disables `OTB streaming in some applications
      <https://www.orfeo-toolbox.org/CookBook/C++/StreamingAndThreading.html>`_.

      .. list-table::
        :widths: auto
        :header-rows: 1
        :stub-columns: 1

        * - Option
          - Step
          - Default value
          - Remarks

            .. _Processing.disable_streaming.normals_on_s2:
        * - ``.normals_on_s2``
          - :class:`ComputeNormalsOnS2
            <s1tiling.libs.otbwrappers.ComputeNormalsOnS2>`
          - ``True``
          - Work around `OTB issue #2442
            <https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/issues/2442>`_.

            .. _Processing.disable_streaming.gamma_area:
        * - ``.gamma_area``
          - :class:`SARGammaAreaImageEstimation <s1tiling.libs.otbwrappers.SARGammaAreaImageEstimation>`
          - ``False``
          - Eliminates artefacts in between streaming tile when ``True``, which
            requires a lot of memory see :ref:`γ area production scenario
            <scenario.S1GammaAreaMap.ram-greedy>`.

            .. _Processing.disable_streaming.apply_gamma_area:
        * - ``.apply_gamma_area``
          - :class:`ApplyGammaNaughtRTCCalibration <s1tiling.libs.otbwrappers.ApplyGammaNaughtRTCCalibration>`
          - ``False``
          -

      .. _Processing.rtc:
  * - :math:`γ^0_{T}` RTC specific options
    - The following options are exclusively used for the :ref:`preparation
      <scenario.S1GammaAreaMap>` and the :ref:`application of γ area maps
      <scenario.S1ProcessorRTC>`.

      .. list-table::
        :widths: auto
        :header-rows: 1
        :stub-columns: 1

        * - Option
          - Purpose
          - Default value

            .. _Processing.use_resampled_dem:
        * - ``use_resampled_dem``
          - Tell to work on resampled DEM before computing γ area maps. The
            resampling factors are specified in :ref:`resample_dem_factor_x
            <processing.resample_dem_factor_x>` and :ref:`resample_dem_factor_y
            <processing.resample_dem_factor_y>`
          - ``True``

            .. _Processing.resample_dem_factor_x:
        * - ``resample_dem_factor_x``
          - Resampling factor on X axis used on DEM when
            :ref:`use_resampled_dem <processing.use_resampled_dem>` is set.
          - 2.0

            .. _Processing.resample_dem_factor_y:
        * - ``resample_dem_factor_y``
          - Resampling factor on Y axis used on DEM when
            :ref:`use_resampled_dem <processing.use_resampled_dem>` is set.
          - 2.0

            .. _Processing.distribute_area:
        * - ``distribute_area``
          - Distribute area on pixel's neighbours (corners) in output geometry.
            |br|
            Used in :class:`SARGammaAreaImageEstimation <s1tiling.libs.otbwrappers.SARGammaAreaImageEstimation>`
          - ``False``

            .. _Processing.inner_margin_ratio:
        * - ``inner_margin_ratio``
          - Ratio of largest direction for DEM 's tile margin for inner tiles.
            Set to 0 to ignore.
            |br|
            Used in :class:`SARGammaAreaImageEstimation <s1tiling.libs.otbwrappers.SARGammaAreaImageEstimation>`
          - 0.01

            .. _Processing.outer_margin_ratio:
        * - ``outer_margin_ratio``
          - Ratio of largest direction for DEM 's tile margin for outer tiles.
            Set to 0 to ignore.
            |br|
            Used in :class:`SARGammaAreaImageEstimation <s1tiling.libs.otbwrappers.SARGammaAreaImageEstimation>`
          - 0.04

            .. _Processing.min_gamma_area:
        * - ``min_gamma_area``
          - Minimum area before entering shadow. |br|
            Used in :class:`ApplyGammaNaughtRTCCalibration <s1tiling.libs.otbwrappers.ApplyGammaNaughtRTCCalibration>`
          - 1.0

            .. _Processing.calibration_factor:
        * - ``calibration_factor``
          - Scalar calibration factor value. |br|
            Used in :class:`ApplyGammaNaughtRTCCalibration <s1tiling.libs.otbwrappers.ApplyGammaNaughtRTCCalibration>`
          - 1.0


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
    - The following spatial speckling filter methods from
      :external+OTB:doc:`OTB Despeckle application
      <Applications/app_Despeckle>` are supported: ``Lee``, ``Frost``,
      ``Gammamap``, ``Kuan``. If ``none`` or empty, then no filtering is done.

      .. _Filtering.window_radius:
  * - ``window_radius``
    - Sets the window radius for the spatial filtering. |br|
      Be cautious: this does expect a radius, i.e. radius=1 means the filter
      does an 3x3 pixels averaging.

      .. _Filtering.deramp:
  * - ``deramp``
    - Deramp factor -- for Frost filter only. |br|
      Factor used to control the exponential function used to weight effect of
      the distance between the central pixel and its neighbourhood. Increasing
      the deramp parameter will lead to take into account more pixels farther
      from the centre and therefore increase the smoothing effects.

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

.. _Metadata:

``[Metadata]`` section
++++++++++++++++++++++

You can place in this section any extra ``key : value`` information that you
want written in the GeoTIFF metadata of S1Tiling products.

Example:

.. code:: ini

    [Metadata]
    # Extra geotiff metadata to write in products
    Contact : My Self <some.one@somewhe.re>
    Something Important: you need to known this!


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
