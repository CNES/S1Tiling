.. _exit_codes:

Process return code
-------------------

The following exit codes are produced when :ref:`S1Processor`, :ref:`S1LIAMap`
:ref:`S1IAMap`, or :ref:`S1GammaAreaMap` returns:

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

      If more critical errors occur, this exit code will be superseded.
  * - 69
    - .. todo::

        Output disk full
  * - 70
    - .. todo::

        Cache disk full (when using option :option:`--cache-before-ortho
        <S1Processor --cache-before-ortho>`)
  * - 71
    - An empty data safe has been found and needs to be removed, so it can be
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
    - :ref:`Geoid file <paths.geoid_file>` is missing, or the specified path is
      incorrect. See the log produced.
  * - 77
    - Some processing cannot be done because external applications cannot
      be executed. Likely OTB and/or NORMLIM and/or γ° RTC related applications
      aren't correctly installed.
      See the log produced.

  * - any other
    - Unknown error. It could be related to `Bash
      <https://www.redhat.com/sysadmin/exit-codes-demystified>`_ or to `Python
      <https://docs.python.org/3/library/os.html#os._exit>`_ reserved error
      codes.
