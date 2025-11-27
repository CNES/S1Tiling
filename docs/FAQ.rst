.. _FAQ:

.. index:: FAQ

Frequently Asked Questions
==========================

.. contents:: Contents:
   :local:
   :depth: 2

Q: How can I fix "`proj_create_from_database: ellipsoid not found"` messages?
-----------------------------------------------------------------------------

A: Just ignore the *error*. As far as we know, it has no incidence.

This message is produced by earlier versions of GDAL (used by OTB 7.4) on
Sentinel-1 products with an “unnamed ellipsoid”. If you execute ``gdalinfo`` on
these Sentinel-1 products you will also observe the *error*, independently of
S1Tiling or OTB.

Example:

.. code::

    $> gdalinfo s1a-iw-grd-vv-20200108t044150-20200108t044215-030704-038506-001.tiff
    ERROR 1: PROJ: proj_create_from_database: ellipsoid not found
    proj_create_from_database: ellipsoid not found
    Driver: GTiff/GeoTIFF
    Files: s1a-iw-grd-vv-20200108t044150-20200108t044215-030704-038506-001.tiff
    Size is 25345, 16817
    GCP Projection =
    GEOGCRS["WGS 84",
        DATUM["World Geodetic System 1984",
            ELLIPSOID["unnamed",6378137,298.25722356049,
    ...

It's likely related to `GDAL issue #2321
<https://github.com/OSGeo/gdal/issues/2321>`_, and tracked in `S1Tiling issue
#46 <https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/46>`_.

Q: Why do I get a log error when running several jobs?
------------------------------------------------------

A: When running S1Tiling in several jobs that can be executed simultaneously,
**DO NOT** execute S1Tiling in the same directory. Also, **DO NOT** use the
same global directory to generate temporary files. Indeed, parallel instances
of S1Tiling will write in the same files and corrupt them. Make sure to work in
different spaces.

In other words,

- execute S1Tiling from, for instance,
  :file:`${{PBS_O_WORKDIR}}/${{PBS_JOBID}}`, -- indeed, unlike :file:`${{TMPDIR}}/`,
  :file:`${{PBS_O_WORKDIR}}/${{PBS_JOBID}}` will persist job execution on PBS.
- and set :ref:`[PATHS].tmp <paths.tmp>` to
  :file:`${{TMPDIR}}/whatever-${{PBS_JOBID}}`.

This Q/A is tracked in `S1Tiling issue #70
<https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/70>`_.

Q: How can I overcome timeouts when searching for online products?
------------------------------------------------------------------

Some data providers like Geodes may fail to obtain in time the list of products
matching our criteria.

Since `EODAG <https://github.com/CS-SI/eodag>`_ v2.11.0, we can override the
default timeout value thanks to:

- the :envvar:`$EODAG__{PROVIDER}__SEARCH__TIMEOUT` environment variable,
- or the configuration option :samp:`{{provider}}.search.timeout`.

In case you have to cope with an earlier version of EODAG, you can still run
:ref:`S1Processor` with :option:`--nb_max_search_retries <S1Processor
--nb_max_search_retries>`.


.. _FAQ.EOF:

Q: How can I configure precise orbit files retrieval?
-----------------------------------------------------

EOF files will be downloaded either:

* on Copernicus Dataspace. In that case, add your ``cop_dataspace``
  credentials in :ref:`eodag configuration file <datasource.eodag_config>`.

  .. note::
      If your account is configured for `Two Factor Authentivcation` (2FA), then
      you need to
      set :envvar:`$EODAG__COP_DATASPACE__AUTH__CREDENTIALS__TOTP` and quickly
      run :ref:`LIA map production scenario <scenario.s1liamap>` while your
      `One Time Password` is still valid (< 30sec)

* or on EarthData. In that case add your Earthdata credentials in your
  :file:`~/.netrc` file (default location can be overridden with
  :envvar:`$NETRC`). e.g.

  .. code::

        machine urs.earthdata.nasa.gov
          login your.login
          password YoURpAssWoRd

If credentials are provided for both data providers, they will be interrogated
in order: Copernicus Dataspace first, then EarthData if no connexion could be
established to the former.

Q: Can I concatenate inputs that have different obit numbers?
-------------------------------------------------------------

**TL; DR**: No

**Long Answer**: This is not currently possible.

Internally S1Tiling relies on exact relative orbit numbers to determine that
two input Sentinel-1 products may contribute to a same output product.
Unfortunately, around the ANX the orbit number changes. It may not even be
exactly around the ANX but a little after.

This Q/A is tracked in `S1Tiling issue #189
<https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/189>`_.

.. _FAQ.nb_rtc_bands:

Q: Why the number of bands in γ-area maps varies?
-------------------------------------------------

Depending on how a :ref:`γ-area map <gamma_area_s2-files>` has been produced,
it may contain one band (γ-area) or two bands (γ-area + *nodata* mask).
It depends on whether one or two Sentinel-1 products have been used to produce
the map. When two are used, the second band is lost during :ref:`concatenation
<concat_gamma_area-proc>`.

This issue is of no consequence as very small areas are handled as *nodata*,
by final :ref:`normalization step <apply_gamma_area-proc>`.

Q: How can I ask another question?
----------------------------------

You can contact us and ask any question related to S1Tiling on `S1Tiling
discourse forum <https://forum.orfeo-toolbox.org/c/otb-chains/s1-tiling/11>`_.

If you think your issue is a bug, please follow the :ref:`procedure described
in the contributing guide <reporting bugs>`.
