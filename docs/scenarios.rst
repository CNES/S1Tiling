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

In this scenario pairs of Sentinel-1 images are:

- calibrated according to β\ :sup:`0`, γ\ :sup:`0` or σ\ :sup:`0` calibration
- then orthorectified onto MGRS Sentinel-2 grid,
- to be finally concatenated.

The unique elements in this scenario are:

- the :ref:`calibration option <Processing.calibration>` that must be
  either one of ``beta``, ``sigma`` or ``gamma``

  - see the dedicated scenario for :ref:`σ° NORMLIM calibration
    <scenario.S1ProcessorLIA>`
  - see the dedicated scenario for :ref:`γ° RTC calibration
    <scenario.S1ProcessorRTC>`
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
- The orthorectified images will be generated in :ref:`output <paths.output>`.
- Temporary files will be produced in :ref:`tmp <paths.tmp>`.

.. note:: S1 Tiling never cleans the :ref:`tmp directory <paths.tmp>` as its
   files are :ref:`cached <data-caches>` in between runs. This means you will
   have to watch this directory and eventually clean it.


.. _scenario.S1ProcessorLIA:

Orthorectify pairs of Sentinel-1 images on Sentinel-2 grid with σ\ :sup:`0`\ :sub:`T` NORMLIM calibration
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This scenario is a variation of the :ref:`previous one <scenario.S1Processor>`.
The difference lies in the calibration applied: it is the :math:`σ^0_{T}`
NORMLIM calibration described in [Small2011]_.

In S1Tiling, we have chosen to precompute Local Incidence Angle (LIA) maps on
MGRS Sentinel-2 grid. Given a precise orbit file, a relative orbit and a MGRS
tile, we directly compute the correction map on the selected Sentinel-2 tile.

That map will then be used for all series of pairs of Sentinel-1 images, of
compatible orbit, β° calibrated and projected to the associated S2 tile.

Regarding options, the only difference with previous scenario are:

- the :ref:`calibration option <Processing.calibration>` that needs to be
  ``normlim``,
- the :ref:`directory <Paths.eof_dir>` where EOF files will be searched for, or
  downloaded to.
- the :ref:`directory <Paths.lia>` where LIA maps will be searched for, or
  produced in.
- at least one :ref:`platform <DataSource.platform_list>` and one
  :ref:`relative orbit <datasource.relative_orbit_list>` are required, to which
  the Local Incidence Angles will be calculated,

S1Tiling will then automatically take care of:

- obtaining the precise orbit files (EOF), if none match the request
  parameters,
- producing, or using existing, :ref:`maps of sin(LIA) <lia-files>` for each
  MGRS Sentinel-2 tiles -- given an orbit,
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
                :alt: 33NWB NORMLIM σ° T calibrated
                :scale: 50%

                33NWB NORMLIM σ° T calibrated -- 20200108

            .. figure:: _static/s1a_33NWB_vh_DES_007_20200108txxxxxx_Normlim_filtered_lee.jpeg
                :alt: 33NWB NORMLIM σ° T calibrated and filtered
                :scale: 50%

                33NWB σ° T calibrated and despeckled (Lee) -- 20200108



.. warning::
   If you wish to parallelize this scenario and to dedicate a different cluster
   node to each date -- as recommended in “:ref:`scenario.parallelize_date`”
   scenario, you will **NEED** to produce all the LIA maps **beforehand**.
   Otherwise, a same file may be concurrently written to from different nodes,
   and it will likely end up corrupted.

.. note::
   This scenario requires `NORMLIM σ°
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

Pre-produce maps of Local Incidence Angles for σ\ :sup:`0`\ :sub:`T` NORMLIM calibration
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

While :ref:`S1Processor` is able to produce the necessary LIA maps on the
fly, it is not able to do so when parallelization is done manually over time
ranges -- as described in “:ref:`scenario.parallelize_date`” scenario.

A dedicated program is provided to compute the LIA maps beforehand:
:ref:`S1LIAMap`. It takes the exact same parameter files as
:ref:`S1Processor`. A few options will be ignored though: calibration type,
masking… But the following (non-obvious) options are mandatory:

- :ref:`[DataSource].platform_list <datasource.platform_list>` -- however only
  a single value will be used
- :ref:`[DataSource].relative_orbit_list <datasource.relative_orbit_list>`
- :ref:`[DataSource].first_date <datasource.first_date>` and
  :ref:`[DataSource].last_date <datasource.last_date>` if
  :ref:`[DataSource].download <datasource.download>` is ``True`` and EOF files
  are missing.

.. code:: bash

        cd workingdir
        # Yes, the same file works!
        S1LIAMap MyS1ToS2.cfg


.. note::
   LIA maps are perfect products to be stored and reused.

.. note::
   This scenario requires `NORMLIM σ°
   <https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0>`_ binaries.
   At the moment, NORMLIM σ\ :sup:`0` binaries need to be compiled manually.
   Unless you use either S1Tiling docker images, or S1Tiling on CNES TREX
   cluster.

.. note::
   To run :ref:`S1LIAMap` from the official S1Tiling docker, use ``--lia`` as
   the first parameter to the docker execution (just before the request
   configuration file and other S1LIAMap related parameters). See
   :ref:`docker.S1LIAMap`.


.. _scenario.S1IAMap:


Produce maps of Ellipsoid Incidence Angles
++++++++++++++++++++++++++++++++++++++++++

S1Tiling can produce :ref:`maps of cosine, sine and/or tangent of the incidence
angle over the WGS84 ellipsoid <ia-files>`, thanks to :ref:`S1IAMap program
<S1IAMap>`.
See :ref:`dataflow-eia` for more detailed information on the internal operation
sequencing.

The typical use case is the following:

1. :ref:`Sine and cosine maps <ia-files>` have been generated (with
   :ref:`S1IAMap`), and cached, for all MGRS Sentinel-2 tiles of interest.
2. Series of :ref:`calibrated and ortho-rectified Sentinel-1 data
   <full-S2-tiles>` have been generated for a given calibration (typically
   :ref:`σ° <processing.calibration>`), and possibly made available on data
   providers like `CNES's Geodes <https://geodes-portal.cnes.fr>`_.
3. You can obtain the same product in other calibrations very quickly by
   applying the corrective sine/cosine map on the Sentinel-2 tiles product.

Relevant parameters (step 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It takes a very similar parameter file as :ref:`S1Processor`.
Actually the same file can be used: only relevant parameters will be taken in
account:

- :ref:`[Paths].output <paths.output>` or :ref:`[Paths].ia <paths.ia>`
- :ref:`[Paths].eof_dir <paths.eof_dir>`
- :ref:`[Paths].tmp <paths.tmp>`
- :ref:`[DataSource].eodag_config <datasource.eodag_config>`
- :ref:`[Processing].tiles <processing.tiles>`
- :ref:`[DataSource].platform_list <datasource.platform_list>` -- but only a
  single value shall be used
- :ref:`[DataSource].relative_orbit_list <datasource.relative_orbit_list>`
- :ref:`[DataSource].first_date <datasource.first_date>` and
  :ref:`[DataSource].last_date <datasource.last_date>` if
  :ref:`[DataSource].download <datasource.download>` is ``True`` and EOF files
  are missing.
- :ref:`[Processing].output_spatial_resolution
  <processing.output_spatial_resolution>`,
- :ref:`[Processing].ia_maps_to_produce <processing.ia_maps_to_produce>`,
- plus filename name format options, parallelization options…

.. code:: bash

        cd workingdir
        # Yes, the same file works!
        S1IAMap MyS1ToS2.cfg

Apply IA maps to σ° calibrated S1Tiling products (step 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When an input product has been :ref:`σ° calibrated <processing.calibration>`,
products in other calibrations can be generated thanks to
:download:`apply-calibration-map.sh
<../s1tiling/resources/apply-calibration-map.sh>`. This is the recommended
approach.

To convert a σ° calibrated product into:

- a β° calibrated product, the σ° image needs to be divided by the :ref:`sine
  map <ia-files>`

  .. code:: bash

    # Either by hand, with OTB, which leaves incorrect CALIBRATION metadata
    otbcli_BandMath \
        -il  s1a_tile_polar_dir_087_time_sigma.tif sin_IA_s1a_tile_087.tif \
        -exp 'im1b1/im2b1' \
        -out s1a_tile_polar_dir_087_time_beta.tif
    # Fix the incorrect metadata
    gdal_edit.py -mo CALIBRATION=beta s1a_tile_polar_dir_087_time_beta.tif

    # ----------------------------------------------------------------------
    # Or, by hand, with gdal, but all metadata will be lost
    gdal_calc.py \
        -A    s1a_tile_polar_dir_087_time_sigma.tif \
        -B    sin_IA_s1a_tile_087.tif \
        --calc "A/B"
        --out s1a_tile_polar_dir_087_time_beta.tif

    # ----------------------------------------------------------------------
    # Or, wrapped for batch application, with OTB, correct metadata
    # RECOMMENDED approach
    apply-calibration-map.sh -c beta --dirmap path/to_sinIA_files path/to/S1Tiling/products

- a γ° calibrated product, the σ° image needs to be divided by the :ref:`cosine
  map <ia-files>`

  .. code:: bash

    # Either by hand, with OTB, which leaves incorrect CALIBRATION metadata
    otbcli_BandMath \
        -il  s1a_tile_polar_dir_087_time_sigma.tif cos_IA_s1a_tile_087.tif \
        -exp 'im1b1/im2b1' \
        -out s1a_tile_polar_dir_087_time_gamma.tif
    # Fix the incorrect metadata
    gdal_edit.py -mo CALIBRATION=gamma s1a_tile_polar_dir_087_time_beta.tif

    # ----------------------------------------------------------------------
    # Or, by hand, with gdal, but all metadata will be lost
    gdal_calc.py \
        -A    s1a_tile_polar_dir_087_time_sigma.tif \
        -B    cos_IA_s1a_tile_087.tif \
        --calc "A/B"
        --out s1a_tile_polar_dir_087_time_gamma.tif

    #-------------------- --------------------------------------------------
    # Or, wrapped for batch application, with OTB, correct metadata
    # RECOMMENDED approach
    apply-calibration-map.sh -c gamma --dirmap path/to_cosIA_files path/to/S1Tiling/products

Notes
^^^^^

.. note::
   Given the calibration is applied on the Sentinel-2 tile geometry, and not in
   the original Sentinel-1 image geometry, small precision differences may be
   observed between this approach and :ref:`the one where the desired
   calibration is applied at the beginning of the processing
   <scenario.S1Processor>`.

.. note::
   This scenario requires `NORMLIM σ°
   <https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0>`_ binaries.
   At the moment, NORMLIM σ\ :sup:`0` binaries need to be compiled manually.
   Unless you use either S1Tiling docker images, or S1Tiling on CNES TREX
   cluster.

.. note::
   To run :ref:`S1IAMap` from the official S1Tiling docker, use ``--ia`` as the
   first parameter to the docker execution (just before the request
   configuration file and other S1IAMap related parameters). See
   :ref:`docker.S1IAMap`.


.. _scenario.S1ProcessorRTC:

Orthorectify pairs of Sentinel-1 images on Sentinel-2 grid with γ\ :sup:`0`\ :sub:`T` calibration
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This scenario is a variation of the :ref:`previous orthorectification scenario
<scenario.S1Processor>`.
The difference lies in the calibration applied: it is the :math:`γ^0_{T}`
calibration described in [Small2011]_.

In S1Tiling, we have chosen to precompute Gamma Area maps on Sentinel-2 grid.

Given a series of Sentinel-1 images to orthorectify on a Sentinel-2 grid, we
select a pair of Sentinel-1 images to compute the associated Gamma Area maps in
the geometry of these images. The maps are then projected, through
orthorectification, on a Sentinel-2 tile, and eventually concatenated.

The resulting map will then be used for all series of orthorectified pairs of
Sentinel-1 images that intersect the associated S2 tile, on the same orbit.

S1Tiling will automatically take care of:

- producing, or using existing, :ref:`γ area maps <gamma_area_s2-files>` for
  each Sentinel-2 tiles -- given an orbit and its direction,
- producing intermediary products calibrated with σ\ :sup:`0` LUT.

Relevant parameters
^^^^^^^^^^^^^^^^^^^
Regarding options, the only difference with previous scenario are:

- the :ref:`calibration option <Processing.calibration>` that needs to be
  ``gamma_naught_rtc``,
- :ref:`[Paths].gamma_area <Paths.gamma_area>`, the directory where :ref:`γ
  area maps <gamma_area_s2-files>` will be searched for, or produced in.

Also, these specific options can be overridden:

- :ref:`[Processing].min_gamma_area <processing.min_gamma_area>`
- :ref:`[Processing].calibration_factor <processing.calibration_factor>`
- :ref:`[Processing].disable_streaming.gamma_area <processing.disable_streaming.apply_gamma_area>`
- :ref:`[Processing].resample_dem_factor_x <processing.resample_dem_factor_x>`
- :ref:`[Processing].resample_dem_factor_y <processing.resample_dem_factor_y>`

- :ref:`[Processing].distribute_area <processing.distribute_area>`
- :ref:`[Processing].disable_streaming.gamma_area <processing.disable_streaming.gamma_area>`
- :ref:`[Processing].inner_margin_ratio <processing.inner_margin_ratio>`
- :ref:`[Processing].outer_margin_ratio <processing.outer_margin_ratio>`

Notes
^^^^^
.. warning::
   If you wish to parallelize this scenario and dedicate a different cluster
   node to each date -- as recommended in “:ref:`scenario.parallelize_date`”
   scenario, you will **NEED** to produce all the γ areas maps beforehand.
   Otherwise, a same file may be concurrently written to from different nodes,
   and it will likely end up corrupted.

.. note::
   This scenario requires `GammaNaughtRTC
   <https://gitlab.orfeo-toolbox.org/s1-tiling/RTC_gamma0>`_ binaries.
   At the moment, γ\ :sup:`0`\ :sub:`T` binaries need to be compiled manually.
   Unless you use either S1Tiling docker images, or S1Tiling on CNES TREX
   cluster.

.. note::
   This scenario permits processing wide time ranges. Only one pair of input S1
   files will be used to generate the :ref:`gamma_area_s2-files` (for a given
   MGRS S2 tile + orbit), however all compatible pairs will be downloaded
   anyway as they will be orthorectified and calibrated.

   Also in this scenario, misleading warnings may be reported at the end of the
   execution: :ref:`S1Processor` may fail to download some `redundants` S1
   input files and yet the :ref:`gamma_area_s2-files` is properly generated.

   .. code::

       WARNING  - Execution report: 8 errors detected
       INFO     -  - Success: OUTPUT/31TCH/s1a_31TCH_vv_DES_110_20250205t060110_GammaNaughtRTC.tif
       INFO     -  - Success: OUTPUT/31TCH/s1a_31TCH_vv_DES_110_20250217t060109_GammaNaughtRTC.tif
       INFO     -  - Download failure: 's1a_31TCH_*_DES_110_20250301txxxxxx_GammaNaughtRTC.tif' cannot be produced because of the following issues with the inputs: [Failed to download EOProduct(id=S1A_IW_GRDH_1SDV_20250301T060109_20250301T060134_058107_072D11_BB75, provider=peps): None: None]
       INFO     -  - Download failure: 'GAMMA_AREA_s1a_31TCH_DES_110.tif' cannot be produced because of the following issues with the inputs: [Failed to download EOProduct(id=S1A_IW_GRDH_1SDV_20250301T060109_20250301T060134_058107_072D11_BB75, provider=peps): None: None]
       INFO     -  - Download failure: 's1a_31TCH_*_DES_110_20250313txxxxxx_GammaNaughtRTC.tif' cannot be produced because of the following issues with the inputs: [Failed to download EOProduct(id=S1A_IW_GRDH_1SDV_20250313T060109_20250313T060134_058282_07342C_D156, provider=peps): None: Max retries exceeded with url: /resto/collections/S1/86509853-0578-591f-984a-a2451a844a62/download?issuerId=peps (Caused by None)]
       INFO     -  - Download failure: 'GAMMA_AREA_s1a_31TCH_DES_110.tif' cannot be produced because of the following issues with the inputs: [Failed to download EOProduct(id=S1A_IW_GRDH_1SDV_20250313T060109_20250313T060134_058282_07342C_D156, provider=peps): None: Max retries exceeded with url: /resto/collections/S1/86509853-0578-591f-984a-a2451a844a62/download?issuerId=peps (Caused by None)]
       INFO     -  - Download failure: 's1a_31TCH_*_DES_110_20250325txxxxxx_GammaNaughtRTC.tif' cannot be produced because of the following issues with the inputs: [Failed to download EOProduct(id=S1A_IW_GRDH_1SDV_20250325T060109_20250325T060134_058457_073B0C_F2AE, provider=peps): None: Max retries exceeded with url: /resto/collections/S1/cc859ae9-aab1-5690-bc3a-856e592fb9a7/download?issuerId=peps (Caused by None)]
       INFO     -  - Download failure: 'GAMMA_AREA_s1a_31TCH_DES_110.tif' cannot be produced because of the following issues with the inputs: [Failed to download EOProduct(id=S1A_IW_GRDH_1SDV_20250325T060109_20250325T060134_058457_073B0C_F2AE, provider=peps): None: Max retries exceeded with url: /resto/collections/S1/cc859ae9-aab1-5690-bc3a-856e592fb9a7/download?issuerId=peps (Caused by None)]
       INFO     - Situation: 0 computations errors. 0 search failures. 4 download failures. 0 download timeouts


.. _scenario.S1GammaAreaMap:

Pre-produce Gamma Area maps for γ\ :sup:`0`\ :sub:`T` calibration
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

While :ref:`S1Processor` is able to produce the necessary :ref:`γ area maps
<gamma_area_s2-files>` on the fly, it is not able to do so when parallelization
is done manually over time ranges -- as described in
“:ref:`scenario.parallelize_date`” scenario.

A different program is provided to compute the Gamma Area maps beforehand:
:ref:`S1GammaAreaMap`. It takes the exact same parameter files as
:ref:`S1Processor`. A few options will be ignored though: calibration type,
masking… But the following (non-obvious) options are mandatory:

.. code:: bash

        cd workingdir
        # Yes, the same file works!
        S1GammaAreaMap MyS1ToS2.cfg


.. note::
   Gamma Area maps are perfect products to be stored and reused.

.. note::
   This scenario requires `GammaNaughtRTC
   <https://gitlab.orfeo-toolbox.org/s1-tiling/RTC_gamma0>`_ binaries.
   At the moment, γ\ :sup:`0`\ :sub:`T` binaries need to be compiled manually.
   Unless you use either S1Tiling docker images, or S1Tiling on CNES TREX
   cluster.

.. note::
   To run :ref:`S1GammaAreaMap` from the official S1Tiling docker, use
   ``--gamma_area`` as the first parameter to the docker execution (just before
   the request configuration file and other S1GammaAreaMap related parameters).
   See :ref:`Using S1GammaAreaMap with a docker <docker.S1GammaAreaMap>`.


.. _scenario.S1GammaAreaMap.ram-greedy:
.. warning::
   `SARGammaAreaImageEstimation application <https://gitlab.orfeo-toolbox.org/s1-tiling/RTC_gamma0>`_
   is really RAM greedy.
   In order to work on ground coordinates, with a precision under 1 meter, we
   need to project :ref:`Cartesian coordinates from S1 image onto DEM geometry
   <S1_on_dem-files>` in ``float64`` precision. These files weight around 60
   Giga Bytes, in memory, when :ref:`DEM are 2x,2x resampled
   <Processing.use_resampled_dem>`. The precision can be tuned with
   :ref:`[Processing].creation_options.s1_on_dem option
   <Processing.creation_options.s1_on_dem>`. This option will also permit to
   reduce the file footprint through compression.

   Finally, to guarantee no artefact happens between streaming tiles,
   `SARGammaAreaImageEstimation` needs to load the entirety of these files;
   in order words, :ref:`streaming would need to be disabled
   <Processing.disable_streaming.gamma_area>`.

   .. code:: ini

       [Processing]
       ram_per_process              = 70000
       disable_streaming.gamma_area = True
       creation_options.s1_on_dem   = float64 COMPRESS=DEFLATE, BIGTIFF=YES, PREDICTOR=3, TILED=YES, BLOCKXSIZE=1024, BLOCKYSIZE=1024


.. warning::
   Do not use a wide time range in this scenario. Indeed, all compatible pairs
   of S1 inputs will be downloaded, even if in the end only one pair will be
   used to produce the :ref:`gamma_area_s2-files`.

   Also in this scenario, misleading warnings may be reported at the end of the
   execution: :ref:`S1GammaAreaMap` may fail to download some `redundants`
   S1 input files and yet the :ref:`gamma_area_s2-files` is properly
   generated.

   .. code::

       WARNING  - Execution report: 4 errors detected
       INFO     -  - Success: OUTPUT/_GAMMA_AREA/GAMMA_AREA_s1a_31TCH_DES_110.tif
       INFO     -  - Download failure: 'GAMMA_AREA_s1a_31TCH_DES_110.tif' cannot be produced because of the following issues with the inputs: [Failed to download EOProduct(id=S1A_IW_GRDH_1SDV_20250301T060109_20250301T060134_058107_072D11_BB75, provider=peps): None: None]
       INFO     -  - Download failure: 'GAMMA_AREA_s1a_31TCH_DES_110.tif' cannot be produced because of the following issues with the inputs: [Failed to download EOProduct(id=S1A_IW_GRDH_1SDV_20250313T060109_20250313T060134_058282_07342C_D156, provider=peps): None: Max retries exceeded with url: /resto/collections/S1/86509853-0578-591f-984a-a2451a844a62/download?issuerId=peps (Caused by None)]
       INFO     -  - Download failure: 'GAMMA_AREA_s1a_31TCH_DES_110.tif' cannot be produced because of the following issues with the inputs: [Failed to download EOProduct(id=S1A_IW_GRDH_1SDV_20250325T060109_20250325T060134_058457_073B0C_F2AE, provider=peps): None: Max retries exceeded with url: /resto/collections/S1/cc859ae9-aab1-5690-bc3a-856e592fb9a7/download?issuerId=peps (Caused by None)]
       INFO     - Situation: 0 computations errors. 0 search failures. 4 download failures. 0 download timeouts


.. _scenario.masks:

Generate masks on final products
++++++++++++++++++++++++++++++++

:ref:`Pixel masks <mask-files>` of valid data can be produced in all
:ref:`S1Processor` scenarios when the option :ref:`generate_border_mask
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
   This scenario is not compatible with ``normlim`` and ``gamma_naught_rtc``
   calibrations where the LIA or γ Area maps would be computed on-the-fly. For
   these calibrations, it's imperative to precompute (and store the correction
   maps) before going massively parallel.


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
   The default :file:`{{id}}.hgt` associates the ``id`` key to STRM 30m DEM
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
