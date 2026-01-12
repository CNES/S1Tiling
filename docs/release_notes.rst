.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. _release_notes:

Release notes
=============

Version 1.3.0
-------------

The main features of this version are mainly related to performances
optimizations, and the quality of diagnosed errors.

v1.3.0 Improvements
+++++++++++++++++++

- Searching performances of the DEM tiles that intersect the requested S2 MGRS
  tiles has been greatly improved
  (`#201 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/201>`_).
- Project packaging is migrated to :file:`pyproject.toml`
  (`#198 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/198>`_).
- Make sure tiles are always processed in the same order
  (`#206 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/206>`_).
- Improve the error message when attempting to patch the no-data value of a
  read-only geoid file, when DEM and Geoid are symlinked
  (`#213 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/213>`_).
- S1 products are no longer scanned in LIA and IA scenarios
  (`#212 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/212>`_).

v1.3.0 Breaking changes
+++++++++++++++++++++++

- From now on, precise orbit files can only be fetched from ``cop_dataspace``
  and other providers supported by EODAG
  (`#197 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/197>`_).

v1.3.0 Bugs fixed
+++++++++++++++++

- PyPi wheel production has been restored
  (`#204 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/204>`_).
- Report (and don't crash) when no image with the expected polarities in found
  in time compatible S1 products
  (`#208 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/208>`_).
- Fix crash on time-out during download
  (`#214 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/214>`_).
- Bug in computing orbit direction
  (`LIA#25 <https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0/-/issues/25>`_).


Release notes
=============

Version 1.2.0
-------------

The main features of this version are:

- S1Tiling can now produce :ref:`Gamma Area Maps <scenario.S1GammaAreaMap>`
  over requested S2 MGRS tiles, and :ref:`generate S2 products
  <scenario.S1ProcessorRTC>` calibrated with the :math:`γ^0_{T}` calibration
  described in [Small2011]_.
- S1Tiling can generate :ref:`maps of incidence angles to the WGS84 ellipsoid
  <ia-files>`, over requested S2 MGRS tiles, from precise orbit files.
- The computation of :ref:`Local Incidence Angle maps <lia-files>` has evolved
  to use precise orbit files as well.
- Sentinel-1C is now supported.

v1.2.0 Breaking changes
+++++++++++++++++++++++

- Compatibility to OTB 7.x (and even 8.x) is no longer actively pursued.
  S1Tiling may work with older versions of OTB, but with no guarantees
  (`#164 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/164>`_).
- Compatibility to Python 3.8 is no longer actively pursued as Python 3.8 has
  reached its end-of-life in 2024.
  S1Tiling may work with older versions of Python, but with no guarantees
  (`#158 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/158>`_).
- The default Geoid file used is changed from :file:`egm96.grd` to
  :file:`egm96.gtx`
  (`#185 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/185>`_).
- The `platform unit code` has been removed from the various correction maps
  produced
  (`#193 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/193>`_).
  |br|
  If you want to use maps you have previously produced, you can change their
  :ref:`respective filename formats <Processing.fname_fmt>` to their previous
  values

  .. code:: ini

    fname_fmt.ia_product         : {IA_kind}_{flying_unit_code}_{tile_name}_{orbit}.tif
    fname_fmt.lia_product        : {LIA_kind}_{flying_unit_code}_{tile_name}_{orbit}.tif
    fname_fmt.gamma_area_product : GAMMA_AREA_{flying_unit_code}_{tile_name}_{orbit_direction}_{orbit}.tif

  Please note that various precision improvements have been made in
  S1Tiling 1.2.0: like for instance `#151
  <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/151>`_, and
  that newly produced maps should have a better quality.
- PyPi wheel production has been momentarily disabled
  (`#203 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/203>`_).

v1.2.0 Improvements
+++++++++++++++++++

- This new version can produce :ref:`Gamma Area Maps <scenario.S1GammaAreaMap>`
  over requested S2 tiles thanks to :ref:`S1GammaAreaMap`, or :ref:`generate S2
  products <scenario.S1ProcessorRTC>` calibrated with the :math:`γ^0_{T}`
  calibration described in [Small2011]_.
  (`#90 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/90>`_).
- Use precise orbit files, downloaded on-the-fly, to compute :ref:`Local
  Incidence Angle maps <lia-files>`
  (`#151 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/151>`_).
- Support eodag 3
  (`#170 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/170>`_,
  `#177 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/177>`_,
  `#178 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/178>`_).
  An indirect consequence is that products will be downloaded into
  :samp:`{{s1images}}/{{product_name}}/` instead of
  :samp:`{{s1images}}/{{product_name}}/{{product_name}}.SAFE/`. The old output
  directory structure is still supported for backward compatibility reasons.
- Generate :ref:`maps of incidence angles to the WGS84 ellipsoid <ia-files>`
  (`#161 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/161>`_).
- New GeoTIFF metadata are written in the images produced by S1Tiling
  (`#171 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/171>`_).

  - :ref:`DEM_INFO <paths.dem_info>` that will be set when relevant,
  - and any pairs of ``key=value`` that are specified in the :ref:`[Metadata]
    <metadata>` configuration section.
- Add support for Sentinel1-C launched end 2024
  (`#175 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/175>`_).
- File decoding and encoding is now done is parallel
  (`#184 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/184>`_).
- Undocumented creations options are available on :ref:`intermediary files
  <temporary-files>` investigated with to :option:`--debug-caches
  <S1Processor --debug-caches>` option.
- Files removal is now tested ([d3934aa]).
- Streaming is disabled with OTB 9.1.0 (and prior) when producing ground
  normals
  (`#181 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/181>`_).
- On construction, :class:`Configuration
  <s1tiling.libs.configuration.Configuration>` object can be injected domain
  specific checks. Introduced for
  (`#191 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/191>`_).
- S1Tiling kernel has been improved for later extraction and exploitation in
  other OTB based chains.
- Filename formats can now use new conversion fields to convert keys in
  uppercase or lowercase. A new time stamp is also defined to hold the first
  time stamp among the ones from the input S1 images
  (`#188 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/188>`_).
- Product downloading has been fixed to work with all data providers supported
  by EODAG v3.9.0+
  (`#168 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/168>`_).
- S1Tiling dockers will also be published on `DockerHub
  <https://hub.docker.com/r/cnes/s1tiling>`_
  (`#173 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/173>`_).


v1.2.0 Bugs fixed
+++++++++++++++++

- Improved analysis of the input Sentinel-1 files to download, depending on the
  ones already in the cache. It's meant to handle :math:`γ^0_{T}` related
  scenarios
  (`#113 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/113>`_).
  |br|
  Yet it's still impossible to not download every possible Sentinel-1 input in
  the :ref:`γ-area map production scneario <scenario.S1GammaAreaMap>`
- Parallel downloading has been inhibited to prevent subtle bugs when
  forwarding the errors detected.
  (`#190 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/190>`_).
- Timeout detection has been updated for the latest version of eodag.

Version 1.1.0
-------------

This version integrates 3 main improvements:

- it can support :ref:`DEM from any sources <scenario.choose_dem>` (Copernicus
  DEM, RGE Alti DEM…),
- it supports OTB 8 (and OTB 9) applications (while staying backward compatible
  with OTB 7.4.2),

v1.1.0 Improvements
+++++++++++++++++++

- Improve API (separate CLI from computing functions)
  (`#96 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/96>`_) --
  contributed by CS Group FRANCE.
- Support DEM files from other origins (Copernicus…). Their footprints,
  organization on disk… need to be deduced from a DEM database.
  (`#18 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/18>`_) --
  contributed by CS Group FRANCE.
- Add support for OTB 8 applications
  (`#105 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/105>`_)
  -- contributed by CS Group FRANCE.
- Add support for OTB 9 applications as well
  (`#152 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/152>`_).
- Support DEM databases in any spatial reference (they are not restricted to
  WGS84 any more)
  (`#149 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/149>`_).
- Change LIA workflow in order to minimize occurrences of artefacts in rugged
  areas, and to speed-up performances
  (`#149 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/149>`_).
- Product output directory can be configured through :ref:`dname_fmt.*
  <processing.dname_fmt>` options
  (`#148 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/148>`_).
- Product encoding/compression options can be configured through
  :ref:`creation_options.* <Processing.creation_options>` options
  (`#66 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/66>`_).
- GEOID file is also copied alongside DEM data when :ref:`[Processing].cache_dem_by
  <Processing.cache_dem_by>` option is on
  (`#123 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/123>`_).

v1.1.0 Bugs fixed
+++++++++++++++++

- Noise correction post-processing shall not transform wide no-data sides from
  Sentinel-1 IPF 2.90+ products into :ref:`minimal signal value
  <Processing.lower_signal_value>`
  (`#159 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/159>`_).

- Handling of `nodata` values has been improved
  (`#159 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/159>`_,
  `#160 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/160>`_).


v1.1.0 Breaking changes
+++++++++++++++++++++++

- ``ACQUISITION_DATETIME`` image metadata is now in UTC format (e.g.
  ``2020:01:08T04:41:50Z``). In previous versions it used to have the same
  format as ``TIFFTAG_DATETIME`` (i.e., ``2020:01:08 04:41:50``)


Version 1.0.0
-------------

This version is a major improvement over v 0.3.x versions. A few breaking
changes have been made in parameters, internal API…

v1.0.0 Improvements
+++++++++++++++++++

- This new version can automatically produce :ref:`produce Local Incidence
  Angle Maps <scenario.S1LIAMap>` over requested S2 MGRS tiles thanks to
  :ref:`S1LIAMap`, or :ref:`generate S2 products <scenario.S1ProcessorLIA>`
  calibrated with :math:`σ^0_{T}` NORMLIM calibration described in
  [Small2011]_.
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
  <DataSource.tile_to_product_overlap_ratio>`
  (`#83 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/83>`_,
  `#110 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/110>`_,
  `#133 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/133>`_).

- Null values obtained after the optional *denoising* done during the
  :ref:`calibration <calibration-proc>` will be set to a :ref:`minimal signal
  value <Processing.lower_signal_value>` > 0. The objective is to keep 0 as the
  *nodata* value.
  (`#87 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/87>`_).

- Spatial Speckle Filtering is supported
  (`#116 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/116>`_).

- Improve the reporting of search or download failures. Also give another
  chance to download products after download timeouts (in case other products
  have successfully been downloaded afterward the last timeout)
  (`!89 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/merge_requests/89>`_
  | `#139 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/139>`_)

- On search timeout, S1Tiling will insist a few times (5 by default, can be
  overridden through CLI option). This is meant as a workaround of `EODAG issue
  #908 <https://github.com/CS-SI/eodag/issues/908>`_.
  (`#140 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/140>`_).

v1.0.0 Bugs fixed
+++++++++++++++++

- Offline S1 products are now correctly detected and processed
  (`#71 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/71>`_,
  `#93 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/93>`_,
  `#108 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/108>`_):

    - their associated (and available) products won't be used to produce a S2
      product,
    - the final report will list the S1 products that could not be retrieved,
    - and the exit code :ref:`exits.OFFLINE_DATA (68) <exit_codes>` will be
      used.

- Discard download failure errors from previous tiles
  (`#139 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/139>`_)

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

v0.3.1 Bugs fixed
+++++++++++++++++

- Don't produce partial products when complete ones already exist for a given
  S2 tile at a requested time
  (`#104 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/104>`_).

Version 0.3.0
-------------

This version is a minor release with critical but non-trivial fixes before
version 1.0.0

v0.3.0 Improvements
+++++++++++++++++++

- Don't remove timestamp from final products when no concatenation is done
  (`#69 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/69>`_).
  Up to now time-tag was always changed to ``txxxxxx``
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
    - A summary of the successful or failed computations is provided.

v0.2 Bugs fixed
+++++++++++++++

- Fix thermal noise usage (`#84 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/84>`_)
- Fix pylint error (`#82 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/82>`_)
- Improve the SRTM tiles database to avoid to request SRTM tile which don't exist (`#81 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/81>`_)
- Work on the more complete product when there are multiple overlapping
  products (`#47
  <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/47>`_)
- Multiple errors related to temporary files have been fixed
  (`#6 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/6>`_)
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
  <cutting-proc>` and :ref:`calibration <calibration-proc>`
- Execute only the processes that are needed to produce the requested products
- Parallelization is done with Dask
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

- Multitemporal speckle filtering has been removed from S1Tiling processing.
  Users have to apply their own speckle filtering, according their needs (for
  example with OTB applications OTBDespeckle or with remote modules
  OTBMultitempFilterOutcore and OTBMultitempFilterFiltering)

- The config key `srtm_shapefile` is no more available to users.
