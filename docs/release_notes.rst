.. _release_notes:

Release notes
=============

Version 1.0.0
-------------

This version is a major improvement over v 0.3.x versions. A few breaking
changes have been made in parameters, internal API...

v1.0.0 Improvements
+++++++++++++++++++

- This new version can automatically :ref:`produce Local Incidence Angle Maps
  <scenario.S1LIAMap>` over requested S2 tiles thanks to :program:`S1LIAMap`,
  or :ref:`generate S2 products <scenario.S1ProcessorLIA>` calibrated with
  :math:`σ^0_{RTC}` NORMLIM calibration described in [Small2011]_.
  (`#56 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/56>`_,
  `#119 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/119>`_).

- Correctly update image metadata, and fill in particular the list of
  Sentinel-1 :samp:`INPUT_FILES` used to produce tiles, as well as the full
  list of :samp:`ACQUISITION_DATETIME_{{id}}`
  (`#25 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/25>`_).

- New filters can be used to select input products: :ref:`platform_list
  <DataSource.platform_list>`, :ref:`orbit_direction
  <DataSource.orbit_direction>`, :ref:`relative_orbit_list
  <DataSource.relative_orbit_list>` and :ref:`tile_to_product_overlap_ratio
  <DataSource.tile_to_product_overlap_ratio>`.
  (`#83 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/83>`_,
  `#110 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/110>`_,
  `#133 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/133>`_.

- Null values obtained after the optional *denoising* done during the
  :ref:`calibration <calibration>` wil be set to a :ref:`minimal signal value
  <Processing.lower_signal_value>` > 0. The objective is to keep 0 as the
  *nodata* value.
  (`#87 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/87>`_).

- Spatial Speckle Filtering is supported
  (`#116 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/116>`_).

v1.0.0 Bug fixed
++++++++++++++++

- Offline S1 products are now correctly detected and processed
  (`#71 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/71>`_,
  `#93 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/93>`_,
  `#108 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/108>`_):

    - their associated (and available) products won't be used to produce a S2
      product,
    - the final report will list the S1 products that could not be retrieved,
    - and the exit code :ref:`exits.OFFLINE_DATA (68) <exit_codes>` will be
      used.

- Logging will be done in ``DEBUG`` mode only if :ref:`required
  <Processing.mode>`. Logging code has also been simplified and cleaned.
  (`#132 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/132>`_)

v1.0.0 Breaking changes
+++++++++++++++++++++++

- :ref:`[DataSource].eodagConfig <DataSource.eodag_config>` has been renamed
  ``eodag_config``, to follow ``snake_case``. Old naming scheme is still
  supported, but deprecated.
  (`#129 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/129>`_).

Version 0.3.2
-------------

Improvements over version 0.3

v0.3.2 Improvements
+++++++++++++++++++

- Avoid downloading of already processed S1 images
  (`#107 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/107>`_).

Version 0.3.1
-------------

Bug fixes for version 0.3

v0.3.1 Bug fixed
++++++++++++++++

- Don't produce partial products when complete ones already exist for a given
  S2 tile at a requested time
  (`#104 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/104>`_).

Version 0.3.0
-------------

This version is a minor release with critical but non trivial fixes before
version 1.0.0

v0.3.0 Improvements
+++++++++++++++++++

- Don't remove timestamp from final products when no concatenation is done
  (`#69 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/69>`_).
  Up to now timetag was always changed to ``txxxxxx``
- Update to support noise removal which has been fixed in OTB 7.4.0. This
  processing is now disabled with prior versions of OTB
  (`#89 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/89>`_,
  `#95 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/95>`_).
- Dask ``cluster`` and ``client`` handles are always closed. This avoids memory
  leaks from other programs that wish to use S1Tiling as a library.
  (`!50 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/merge_requests/50>`_)
- Permit also to filter polarisation only on ``VV``, ``VH``, ``HV``, or ``HH``
  (`#92 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/92>`_).

v0.3.0 Optimizations
++++++++++++++++++++

- Downloading and unzipping of Sentinel-1 products is done in parallel
  (`!31 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/merge_requests/31>`_)

- Support copying or symlinking SRTM files into a local temporary directory.
  Previously, SRTM files were always symlinked.
  (`#94 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/94>`_).


Version 0.2
-----------

This version is a major release where the project architecture has been
completely changed to enable multiple improvements and optimizations.

v0.2 Improvements
+++++++++++++++++

- Provide the possibility to use linear interpolation for orthorectification step
- Support OTB 7.3
- Various return code after execution are now provided (`#72 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/72>`_)
- Improved architecture to help maintenance
- Project stability has been improved

    - Non-regression tests has been added
    - OTB applications write into temporary files that are renamed after
      completion

- Most temporary files are automatically removed

    - Files that are detected to be no longer required
      (`#38 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/38>`_)
    - SRTM symlinks
      (`#21 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/21>`_)
    - To ignore unrelated files

- Start-over on process interruption has been fixed
  (`#23 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/23>`_)

    - to not use incomplete files
    - to analyse start-over situation once
      (`#22 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/22>`_)

- Sentinel-1 products can be retrieved from many providers thanks to
  `eodag <https://github.com/CS-SI/eodag>`_
  (`#7 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/7>`_,
  `#12 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/12>`_)
- Syntax of :ref:`request configuration files <request-config-file>` been
  simplified
  (`#36 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/36>`_)
- Installation can be done with ``pip``
- Documentation has been written
- Improved logs
  (`#2 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/2>`_)

    - Multiple log files are produced.
    - They can be sent by mail (though configuration)
    - Log level are supported
    - A summary of the successfully of failed computations is provided.

v0.2 Bug fixed
++++++++++++++

- Fix thermnal noise usage ((`#84 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/84>`_)
- Fix pylint error ((`#82 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/82>`_)
- Improve the srtm tiles database to avoid to request srtm tile which don't exist ((`#81 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/81>`_)
- Work on the more complete product when there are multiple overlapping
  products (`#47
  <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/47>`_)
- Multiple errors related to temporary files have been fixed
  (`#6 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/6>`_...)
- List of requested S2 tiles syntax has been relaxed
  (https://github.com/CNES/S1Tiling/issues/2)

v0.2 Optimizations
++++++++++++++++++

- Disk usage has been minimized: most OTB applications are chained into memory
  (`#4 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/4>`_,
  `#10 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/10>`_,
  `#52 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/10>`_)

    - This reduces stress on IO that often are a bottleneck on clusters

- Dedicated and optimized OTB applications have been written for :ref:`cutting
  <cutting>`  and :ref:`calibration <calibration>`
- Execute only the processes that are needed to produce the requested products
- Parallelization is done with dask
  (`#11 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/11>`_)

    - This permits to parallelize computations of different types

- When there is only one file to concatenate, it's simply renamed
  (`#24 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/24>`_).

v0.2 Breaking changes
+++++++++++++++++++++

- Previous :ref:`configuration files <request-config-file>` will need to be
  updated:

    - ``snake_case`` is used for option names
    - a few options have changed (``[DataSource]`` section)

- No script is provided yet to run S1Tiling on several nodes

- Multitemporal speckle filtering has been removed from S1Tiling processing. Users have to apply their own speckle filtering, according their needs (for example with OTB applications OTBDespeckle or with remote modules OTBMultitempFilterOutcore and OTBMultitempFilterFiltering)

- The config key `srtm_shapefile` is no more available to users.
