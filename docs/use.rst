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

Given a :ref:`request configuration file <request-config-file>` (e.g.
``MyS1ToS2.cfg`` in ``workingdir``), running S1Tiling is as simple as::

        cd workingdir
        S1Processor MyS1ToS2.cfg


Then

- The S1 products will be downloaded in :ref:`s1_images <paths.s1_images>`.
- The orthorectified tiles will be generated in :ref:`output <paths.output>`.
- Temporary files will be produced in :ref:`tmp <paths.tmp>`.

.. note:: S1 Tiling never cleans the :ref:`tmp directory <paths.tmp>` as its
   files are :ref:`cached <data-caches>` in between runs. This means you will
   have to watch this directory and eventually clean it.


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
    - Where S1 images are downloaded thanks to `eodag
      <https://github.com/CS-SI/eodag>`_.
      |br|
      S1Tiling will automatically take care to keep at most 1000 products in
      that directory -- the 1000 last that have been downloaded.
      |br|
      This enables to cache downloaded S1 images in beteen runs.

      .. _paths.output:
  * - ``output``
    - Where products are generated.

      .. _paths.tmp:
  * - ``tmp``
    - Where :ref:`intermediary files <temporary-files>` are produced, and
      sometimes :ref:`cached <data-caches>` for longer periods.

      .. _paths.geoid_file:
  * - ``geoid_file``
    - Path to Geoid model. If left unspecified, it'll point automatically to
      the geoid resource shipped with S1 Tiling.

      .. _paths.srtm:
  * - ``srtm``
    - Path to SRTM files.

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
    - Designates where the eodag configuration file is expected to be found.
      |br|
      Default value: :file:`%(HOME)s/.config/eodag/eodag.yml`.

      From S1Tiling point of view, eodag configuration file will list the
      authentification credentials for the know providers and their respective
      priorities.
      |br|
      See `eodag § on How to configure authentication for available providers
      <https://eodag.readthedocs.io/en/latest/intro.html#how-to-configure-authentication-for-available-providers>`_

      For instance, given a PEPS account, :file:`$HOME/.config/eodag/eodag.yml` could
      contain

      .. code-block:: yaml

          peps:
              auth:
                  credentials:
                      username: THEUSERNAME
                      password: THEPASSWORD


      .. _DataSource.nb_parallel_processes:
  * - ``nb_parallel_processes``
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

      .. _DataSource.polarisation:
  * - ``polarisation``
    - Defines the polarisation mode of the products to downloads.
      Only two values are valid: ``HH-HV`` and ``VV-VH``.

      .. _DataSource.first_date:
  * - ``first_date``
    - Initial date in ``YYYY-MM-DD`` format.

      .. _DataSource.last_date:
  * - ``last_date``
    - Final date in ``YYYY-MM-DD`` format.

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
      the S2 image file produced.


.. _Processing:

``[Processing]`` section
++++++++++++++++++++++++

.. list-table::
  :widths: auto
  :header-rows: 1
  :stub-columns: 1

  * - Option
    - Description

      .. _Processing.calibration:
  * - ``calibration``
    - Defines the calibration type: ``gamma`` or ``sigma``

      .. _Processing.remove_thermal_noise:
  * - ``remove_thermal_noise``
    - Shall the thermal noise be removed?

      .. important::

         This feature requires a version of OTB >= 7.4.0

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

      .. _Processing.tile_to_product_overlap_ratio:
  * - ``tile_to_product_overlap_ratio``
    - Percentage of tile area to be covered for a tile to be retained in
      ``ALL`` mode

      .. note::
        At this moment this field is ignored, but it's likely to be used in the
        future.

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

      .. _Processing.override_azimuth_cut_threshold_to:
  * - ``override_azimuth_cut_threshold_to``
    - Permits to override the analysis on whether top/bottom lines shall be
      forced to 0 in :ref:`cutting step <cutting>`. |br|

      Possible values are:

      :``True``:         Force cutting at the 1600th upper and the 1600th lower
                         lines.
      :``False``:        Force to keep every line.
      :not set/``None``: Default analysis heuristic is used.

      .. warning::
        This option is not meant to be used. It only makes sense in some very
        specific scenarios like tests.

.. _Filtering:

``[Filtering]`` section
+++++++++++++++++++++++

.. note:: The following options will eventually be used for the multitemporal
   filtering. They are not used by S1Tiling application.


.. list-table::
  :widths: auto
  :header-rows: 1
  :stub-columns: 1

  * - Option
    - Description

      .. _Filtering.filtering_activated:
  * - ``filtering_activated``
    - If ``True``, the multiImage filtering is activated after the tiling process

      .. _Filtering.reset_outcore:
  * - ``reset_outcore``
    - - If ``True``, the outcore of the multiImage filter is reset before
        filtering. It means that the outcore is recomputed from scratch with
        the new images only.
      - If ``False``, the outcore is updated with the new images. Then, the
        outcore integrates previous images and new images.

      .. _Filtering.window_radius:
  * - ``window_radius``
    - Sets the window radius for the spatial filtering. |br|
      Take care that it is a radius, i.e. radius=1 means the filter does an 3x3
      pixels averaging.


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
    on HAL, should be enough
  - storing :ref:`input files <paths.s1_images>`, like for instance
    :file:`$TMPDIR/data_raw/` on HAL for instance.

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
    - .. todo::

        Download incomplete (data not available online (`#71
        <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/71>`_)
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
    - Cannot find all the :ref:`SRTM products <paths.srtm>` that cover the
      :ref:`requested Sentinel-2 tiles <DataSource.roi_by_tiles>`. See the log
      produced.
  * - 76
    - :ref:`Geoid file <paths.geoid_file>` is missing or the specified path is
      incorrect. See the log produced.

  * - any other
    - Unknown error. It could be related to `Bash
      <https://www.redhat.com/sysadmin/exit-codes-demystified>`_ or to `Python
      <https://docs.python.org/3/library/os.html#os._exit>`_ reserved error
      codes.
