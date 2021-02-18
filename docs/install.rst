.. _install:

.. index:: installation

Installation
============

.. contents:: Contents:
   :local:
   :depth: 3

OTB & GDAL dependency
---------------------

S1 Tiling depends on `OTB 7.2+ <https://www.orfeo-toolbox.org/CookBook-7.2/>`_.
First install OTB on your platform. See the `related documentation
<https://www.orfeo-toolbox.org/CookBook-7.2/Installation.html>`_ to install OTB
on your system..

Then, you'll also need a version of GDAL which is compatible with your OTB
version.

- In case you're using OTB binary distribution, you'll need to **patch** the
  files provided.

  - For that purpose you can **drop** this simplified and generic version of
    :download:`gdal-config <../s1tiling/resources/gdal-config>` into the
    ``bin/`` directory where you've extracted OTB. This will permit :samp:`pip
    install gdal=={vernum}` to work correctly.
  - You'll also have to **patch** ``otbenv.profile`` to **insert** OTB ``lib/``
    directory at the start of :envvar:`$LD_LIBRARY_PATH`. This will permit
    ``python -c 'from osgeo import gdal'`` to work correctly.

- In case you've compiled OTB from sources, you shouldn't have this kind of
  troubles.

- On clusters where OTB has been compiled from sources, you can simply load the
  associated module:

        .. code-block:: bash

            # Example, on HAL:
            module load otb/7.2-Python3.7.2

.. note::
   We haven't tested yet with packages distributed for Linux OSes. It's likely
   you'll need to inject in your ``$PATH`` a version of :download:`gdal-config
   <../s1tiling/resources/gdal-config>` tuned to return GDAL configuration
   information.

Possible conflicts on Python version
++++++++++++++++++++++++++++++++++++

`eodag <https://github.com/CS-SI/eodag>`_ requires ``xarray`` which in turn
requires at least Python 3.6 while default OTB 7.2 binaries are built with
Python 3.5.  This means you'll likely need to recompile OTB Python bindings as
described in:
https://www.orfeo-toolbox.org/CookBook/Installation.html#recompiling-python-bindings


.. code-block:: bash

    cd OTB-7.2.0-Linux64
    source otbenv.profile
    # Load module on HAL
    module load gcc
    ctest3 -S share/otb/swig/build_wrapping.cmake -VV

Conflicts between rasterio default wheel and OTB binaries
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

We have found a compatibility issue between OTB and default rasterio
packaging. The kind that produces:

.. code-block:: none

    Unable to open EPSG support file gcs.csv

The problem comes from:

- OTB binaries that come with GDAL 3.1 and that set :envvar:`$GDAL_DATA` to
  the valid path in OTB binaries,
- and GDAL 2.5+ that no longer ships :file:`gcs.csv`,
- and rasterio (required by eodag) wheel that is statically built with gdal
  2.4.4 `(at this time: November 2020)
  <https://github.com/rasterio/rasterio-wheels/blob/master/env_vars.sh#L11>`_
- and GDAL 2.4.4 that requires :file:`gcs.csv` in :envvar:`$GDAL_DATA`

Either we can globally change :envvar:`$GDAL_DATA` to rasterio's one (which
requires an extra step, and which may introduce other problems), or we can
force rasterio to depend on GDAL library shipped with OTB. As a consequence ...

.. important::

    We recommend to use :program:`pip` with ``--no-binary rasterio``
    parameter.


S1 Tiling installation
----------------------

Then you can install S1 Tiling thanks to `pip`.

.. code-block:: bash

    # First go into a virtual environment (optional)
    # a- It could be a python virtual environment
    python -m venv myS1TilingEnv
    cd myS1TilingEnv
    source bin/activate
    # b- or a conda virtual environment
    conda create -n myS1TilingEnv python==3.7.2
    conda activate myS1TilingEnv

    # Then, upgrade pip and setuptools in your virtual environment
    python -m pip install --upgrade pip
    python -m pip install --upgrade setuptools

    # Finally, install S1 Tiling
    python -m pip install --use-feature=2020-resolver s1tiling --no-binary rasterio

    # Or, developper-version if you plan to work on S1 Tiling source code
    mkdir whatever && cd whatever
    git clone git@gitlab.orfeo-toolbox.org:s1-tiling/s1tiling.git
    cd s1tiling
    python -m pip install -r requirements-dev.txt --use-feature=2020-resolver

.. note::

    We have noted that the new ``--use-feature=2020-resolver`` helps resolve
    ``click`` version that eodag also uses.

.. note::

    The :file:`requirements*.txt` files already force rasterio wheel to be
    ignored.

