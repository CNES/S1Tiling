.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. _use:

======================================================================
Usage
======================================================================

Given a :ref:`request configuration <request-config-file>` (e.g.
``MyS1ToS2.cfg`` in ``workingdir``), running S1Tiling is as simple as::

        cd workingdir
        S1Processor MyS1ToS2.cfg

The S1 products will be downloaded in :ref:`S1Images <paths.S1Images>`.

Temporary files will be produced in :ref:`tmp <paths.tmp>`.

The orthorectified tiles will be generated in :ref:`Output <paths.Output>`.


.. _request-config-file:

Request Configuration file
--------------------------

The request configuration file passed to ``S1Processor`` is in ``.ini`` format.
It is expected to contain the following entries.

You can use this :download:`this template
<../s1tiling/resources/S1Processor.cfg>`, as a starting point.

.. _default:

``[DEFAULT]`` section
+++++++++++++++++++++

.. list-table::
  :widths: auto
  :header-rows: 1
  :stub-columns: 1

  * - Option
    - Description

      .. _default.region:
  * - ``region``
    -

      .. todo::

        This field is ignored. Remove it.

.. _paths:

``[PATHS]`` section
+++++++++++++++++++

.. list-table::
  :widths: auto
  :header-rows: 1
  :stub-columns: 1

  * - Option
    - Description

      .. _paths.S1Images:
  * - ``S1Images``
    - Where S1 images are downloaded thanks to `eodag
      <https://github.com/CS-SI/eodag>`_.
      |br|
      S1Tiling will automatically take care to keep at most 1000 products in
      that directory -- the 1000 last that have been downloaded.
      |br|
      This enables to cache downloaded S1 images in beteen runs.

      .. _paths.output:
  * - ``Output``
    - Where products are generated.

      .. _paths.tmp:
  * - ``tmp``
    - Where intermediary files are stored.
      |br|
      As a S1 image may need to be used to produce several S2 tiles, once
      orthorectified, S1 images are cached in the tmp directory.

      .. note:: It's up to you, end-user, to clean that directory regularly.

      .. _paths.GeoidFile:
  * - ``GeoidFile``
    - Path to Geoid model.

      .. _paths.SRTM:
  * - ``SRTM``
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

      .. _DataSource.Download:
  * - ``Download``
    - If ``True``, activates the downloading from specified data provider for
      the ROI, otherwise only local S1 images already in :ref:`S1Images
      <paths.S1Images>` will be processed.

      .. _DataSource.eodagConfig:
  * - ``eodagConfig``
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


      .. _DataSource.ROI_by_tiles:
  * - ``ROI_by_tiles``
    - The Region of Interest (ROI) for downloading is specified in ROI_by_tiles
      which will contain a list of MGRS tiles. If ``ALL`` is specified, the
      software will download all images needed for the processing (see
      :ref:`Processing`)

      .. code-block:: ini

          [DataSource]
          ROI_by_tiles : 33NWB

      .. _DataSource.Polarisation:
  * - ``Polarisation``
    - Defines the polarisation mode of the products to downloads.
      Only two values are valid: ``HH-HV`` and ``VV-VH``.

      .. _DataSource.first_date:
  * - ``first_date``
    - Initial date in ``YY-MM-DD`` format.

      .. _DataSource.last_date:
  * - ``last_date``
    - Final date in ``YY-MM-DD`` format.

.. _Mask:

``[Mask]`` section
++++++++++++++++++

.. list-table::
  :widths: auto
  :header-rows: 1
  :stub-columns: 1

  * - Option
    - Description

      .. _Mask.Generate_border_mask:
  * - ``Generate_border_mask``
    - This option allows you to choose if you want to generate border mask.

.. _Processing:

``[Processing]`` section
++++++++++++++++++++++++

.. list-table::
  :widths: auto
  :header-rows: 1
  :stub-columns: 1

  * - Option
    - Description

      .. _Processing.Calibration:
  * - ``Calibration``
    - Defines the type of calibration: ``gamma`` or ``sigma``

      .. _Processing.Remove_thermal_noise:
  * - ``Remove_thermal_noise``
    - Shall the thermal noise be removed?

      .. _Processing.OutputSpatialResolution:
  * - ``OutputSpatialResolution``
    - Pixel size (in meters) of the output images

      .. _Processing.TilesShapefile:
  * - ``TilesShapefile``
    - Path and filename of the tile shape definition (ESRI Shapefile)

      .. _Processing.SRTMShapefile:
  * - ``SRTMShapefile``
    - Path and filename of the SRTM shape definition (ESRI Shapefile)

      .. _Processing.Orthorectification_gridspacing:
  * - ``Orthorectification_gridspacing``
    - Grid spacing for the interpolator in the orthorectification process for
      more information, please consult the OTB orthorectification application.

      A nice value is 4 x OutputSpatialResolution

      .. _Processing.BorderThreshold:
  * - ``BorderThreshold``
    - Threshold on the image level to be considered as zeros

      .. _Processing.Tiles:
  * - ``Tiles``, ``TilesListInFile``
    - Tiles to be processed.
      The tiles can be given as a list:

      * ``Tiles``: list of tiles (comma separated). Ex:

        .. code-block:: ini

            Tiles: 33NWB,33NWC

      * TilesListInFile: tile list in a ASCII file. Ex:

        .. code-block:: ini

            TilesListInFile : ~/MyListOfTiles.txt

      .. _Processing.TileToProductOverlapRatio:
  * - ``TileToProductOverlapRatio``
    - Percentage of tile area to be covered for a tile to be retained in
      ``ALL`` mode

      .. todo::

        This field is ignored. Remove it.

      .. _Processing.Mode:
  * - ``Mode``
    - Running mode:

      - ``Normal``: prints normal, warning and errors on screen
      - ``debug``: also prints debug messages, and forces
        ``$OTB_LOGGER_LEVEL=DEBUG``
      - ``logging``: saves logs to files

      .. code-block:: ini

        Mode : debug logging

      .. _Processing.NbParallelProcesses:
  * - ``NbParallelProcesses``
    - Number of processes to be running in parallel |br|
      This number defines the number of S1 images to be processed in parallel.

      .. note:: Must be <= to the number of cores on the machine.

      .. _Processing.RAMPerProcess:
  * - ``RAMPerProcess``
    - RAM Allower per process in MB

      .. _Processing.OTBNbThreads:
  * - ``OTBNbThreads``
    - Numbers of threads used by each OTB application. |br|

      .. note::
        For an optimal performance, ``NbParallelProcesses*OTBNbThreads`` should
        be <= to the number of cores on the machine.


.. _Filtering:

``[Filtering]`` section
+++++++++++++++++++++++

.. todo:: Remove this section as it isn't used.


.. list-table::
  :widths: auto
  :header-rows: 1
  :stub-columns: 1

  * - Option
    - Description

      .. _Filtering.Filtering_activated:
  * - ``Filtering_activated``
    - If ``True``, the multiImage filtering is activated after the tiling process

      .. _Filtering.Reset_outcore:
  * - ``Reset_outcore``
    - - If ``True``, the outcore of the multiImage filter is reset before
        filtering. It means that the outcore is recomputed from scratch with
        the new images only.
      - If ``False``, the outcore is updated with the new images. Then, the
        outcore integrates previous images and new images.

      .. _Filtering.Window_radius:
  * - ``Window_radius``
    - Sets the window radius for the spatial filtering. |br|
      Take care that it is a radius, i.e. radius=1 means the filter does an 3x3
      pixels averaging.


Log configuration
-----------------
Default logging configuration is provided in ``S1Tiling`` installing directory.

It can be overridden by dropping a file similar to
:download:`../s1tiling/logging.conf.yaml` in the same directory as the one
where the :ref:`request-config-file` is. The file is expected to follow
:py:mod:`logging configuration <logging.config>` file syntax.

.. warning::
   This software expects the specification of:

   - ``s1tiling``, ``s1tiling.OTB`` :py:class:`loggers <logging.Logger>`;
   - and ``file`` and ``important`` :py:class:`handlers <logging.Handler>`.

When :ref:`Mode <Processing.Mode>` contains ``logging``, we make sure that
``file`` and ``important`` :py:class:`handlers <logging.Handler>` are added to
the handlers of ``root`` and ``distributed.worker`` :py:class:`loggers
<logging.Logger>`. Note that this is the default configuration.

When :ref:`Mode <Processing.Mode>` contains ``debug`` the ``DEBUG`` logging
level is forced into ``root`` logger, and ``$OTB_LOGGER_LEVEL`` environment
variable is set to ``DEBUG``.

.. _clusters:

Working on clusters
-------------------

.. todo::

  By default S1Tiling works on single machines. Internally it relies on
  :py:class:`distributed.LocalCluster` a small adaptation would be required to
  work on a multi-nodes cluster.
