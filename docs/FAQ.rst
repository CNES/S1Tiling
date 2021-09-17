.. _FAQ:

.. index:: FAQ

Frequently Asked Questions
==========================

.. contents:: Contents:
   :local:
   :depth: 2

Q: How can I fix "`proj_create_from_database: ellipsoid not found"` messages?
-----------------------------------------------------------------------------

A: Just ignore the *error*. As far as we known, it has no incidence.

This message is produced by current version of GDAL (used by OTB 7.3) on
Sentinel-1 products with an "unnamed ellipsoid". If you execute ``gdalinfo`` on
these Sentinel-1 products you will also observe the *error*, independently of
S1Tiling.

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

- execute for instance S1Tiling from, for instance,
  :file:`${PBS_O_WORKDIR}/${PBS_JOBID}`, -- unlike :file:`${TMPDIR}/`, on PBS
  :file:`${PBS_O_WORKDIR}/${PBS_JOBID}` will persist job execution.
- and set :ref:`[PATHS].tmp <paths.tmp>` to :file:`${TMPDIR}/whatever`.

This Q/A is tracked in `S1Tiling issue #70
<https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues/70>`_.

Q: How can I ask another question?
----------------------------------

You can contact us and ask any question related to S1Tiling on `S1Tiling
discourse forum <https://forum.orfeo-toolbox.org/c/otb-chains/s1-tiling/11>`_.

If you think your issue is a bug, please follow the :ref:`procedure described
in the contributing guide <reporting bugs>`.
