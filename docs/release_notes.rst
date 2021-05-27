.. _release_notes:

Release notes
=============

Version 0.2
-----------

This version is a major release where the project architecture has been
completely changed to enable multiple improvements and optimizations.

Improvements
++++++++++++

- Support OTB 7.3
- Various return code after execution are now provided ((`#72 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/72>`_)
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

Bug fixed
+++++++++

- Fix pylint error ((`#82 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/82>`_)
- Improve the srtm tiles database to avoid to request srtm tile which don't exist ((`#81 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/81>`_)
- Work on the more complete product when there are multiple overlapping
  products (`#47
  <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/47>`_)
- Multiple errors related to temporary files have been fixed
  (`#6 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/6>`_...)
- List of requested S2 tiles syntax has been relaxed
  (https://github.com/CNES/S1Tiling/issues/2)

Optimizations
+++++++++++++

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

Breaking changes
++++++++++++++++

- Previous :ref:`configuration files <request-config-file>` will need to be
  updated:

    - ``snake_case`` is used for option names
    - a few options have changed (``[DataSource]`` section)

- No script is provided yet to run S1Tiling on several nodes

- Multitemporal speckle filtering has been removed from S1Tiling processing. Users have to apply their own speckle filtering, according their needs (for example with OTB applications OTBDespeckle or with remote modules OTBMultitempFilterOutcore and OTBMultitempFilterFiltering)

- The config key `srtm_shapefile` is no more available to users.