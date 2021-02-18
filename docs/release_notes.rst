.. _release_notes:

Release notes
=============

Version 0.2
-----------

This version is a major release where the project architecture has been
completely changed to enable multiple improvements and optimizations.

Improvements
++++++++++++

- Improved architecture to help maintenance
- Project stability has been improved

    - Non-regression tests has been added
    - OTB applications write into temporary files that are renamed after
      completion

- Most temporary files are automatically removed

    - Files that are detected to be no longer required
    - SRTM symlinks
    - To ignore unrelated files

- Start-over on process interruption has been fixed

    - to not use incomplete files
    - to analyse start-over situation once

- Sentinel-1 products can be retrieved from many providers thanks to
  `eodag <https://github.com/CS-SI/eodag>`_
- Installation can be done with ``pip``
- Documentation has been written
- Improved logs

    - Multiple log files are produced.
    - They can be sent by mail (though configuration)
    - Log level are supported
    - A summary of the successfully of failed computations is provided.

Bug fixed
+++++++++

- Work on the more complete product when there are multiple overlapping
  products
- Multiple errors related to temporary files have been fixed
- List of requested S2 tiles syntax has been relaxed
  (https://github.com/CNES/S1Tiling/issues/2)

Optimizations
+++++++++++++

- Disk usage has been minimized: most OTB applications are chained into memory

    - This reduces stress on IO that often are a bottleneck on clusters

- Dedicated and optimized OTB applications have been written for :ref:`cutting
  <cutting>`  and :ref:`calibration <calibration>`
- Execute only the processes that are needed to produce the requested products
- Parallelization is done with dask

    - This permits to parallelize computations of different types

- When there is only one file to concatenate, it's simply renamed.

Breaking changes
++++++++++++++++

- Previous :ref:`configuration files <request-config-file>` will need to be
  updated:

    - ``snake_case`` is used for option names
    - a few options have changed (``[DataSource]`` section)

- No script is provided yet to run S1Tiling on several nodes
