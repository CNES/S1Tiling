.. _install:

.. index:: installation

Installation
============

.. contents:: Contents:
   :local:
   :depth: 3

Manual installation with pip
----------------------------

OTB & GDAL dependency
+++++++++++++++++++++

S1 Tiling depends on OTB 7.2+, but we recommend the latest, `OTB 7.4
<https://www.orfeo-toolbox.org/CookBook-7.4/>`_ at the time.
First install OTB on your platform. See the `related documentation
<https://www.orfeo-toolbox.org/CookBook-7.4/Installation.html>`_ to install OTB
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
    ``python3 -c 'from osgeo import gdal'`` to work correctly.

        .. code-block:: bash

            # For instance, type this, once!
            echo 'LD_LIBRARY_PATH="${CMAKE_PREFIX_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' >> otbenv.profile


- In case you've compiled OTB from sources, you shouldn't have this kind of
  troubles.

- On clusters where OTB has been compiled from sources, you can simply load the
  associated module:

        .. code-block:: bash

            # Example, on HAL:
            module load otb/7.4-Python3.7.2

.. note::
   We haven't tested yet with packages distributed for Linux OSes. It's likely
   you'll need to inject in your ``$PATH`` a version of :download:`gdal-config
   <../s1tiling/resources/gdal-config>` tuned to return GDAL configuration
   information.

Possible conflicts on Python version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`eodag <https://github.com/CS-SI/eodag>`_ requires ``xarray`` which in turn
requires at least Python 3.6 while default OTB 7.4 binaries are built with
Python 3.5.  This means you'll likely need to recompile OTB Python bindings as
described in:
https://www.orfeo-toolbox.org/CookBook/Installation.html#recompiling-python-bindings


.. code-block:: bash

    cd OTB-7.4.0-Linux64
    source otbenv.profile
    # Load module on HAL
    module load gcc
    ctest3 -S share/otb/swig/build_wrapping.cmake -VV

Conflicts between rasterio default wheel and OTB binaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
   **TL;DR** In the case you install **other programs alongside S1Tiling** in
   the same environment, then use :program:`pip` with ``--no-binary rasterio``
   parameter.

   The current version of S1Tiling doesn't depend on any package that requires
   ``rasterio``, and thus ``pip install s1tiling`` is enough.


The following paragraph applies **only** in case you install other Python
programs alongside S1Tiling in the same environment.

We had found a compatibility issue between OTB and default rasterio packaging.
The kind that produces:

.. code-block:: none

    Unable to open EPSG support file gcs.csv

The problem came from:

- OTB binaries that come with GDAL 3.1 and that set :envvar:`$GDAL_DATA` to
  the valid path in OTB binaries,
- and GDAL 2.5+ that no longer ships :file:`gcs.csv`,
- and GDAL 2.4.4 that requires :file:`gcs.csv` in :envvar:`$GDAL_DATA`
- and rasterio (used to be required by eodag 1.x) wheel that was statically
  built with gdal 2.4.4

Either we could have globally changed :envvar:`$GDAL_DATA` to rasterio's one
(which requires an extra step, and which may introduce other problems), or we
could have forced rasterio to depend on GDAL library shipped with OTB.

Since December 15th 2020 `rasterio wheel
<https://github.com/rasterio/rasterio-wheels/blob/master/env_vars.sh#L11>`_
depends on GDAL 3.2, while OTB binaries depend on GDAL 3.1. We are not sure
there aren't any compatibility issues between both versions.

As a consequence,
if you are in this situation where you need S1Tiling, or may be just OTB, plus
any other package that relies on rasterio, then we highly recommend to use
:program:`pip` with ``--no-binary rasterio`` parameter to force OTB version of
GDAL and rasterio version of GDAL to be identical.


S1 Tiling installation
++++++++++++++++++++++

Then you can install S1 Tiling thanks to `pip`.

.. code-block:: bash

    # First go into a virtual environment (optional)
    # a- It could be a python virtual environment
    python3 -m venv myS1TilingEnv
    cd myS1TilingEnv
    source bin/activate
    # b- or a conda virtual environment
    conda create -n myS1TilingEnv python==3.7.2
    conda activate myS1TilingEnv

    # Then, upgrade pip and setuptools in your virtual environment
    python -m pip install --upgrade pip
    python -m pip install --upgrade setuptools

    # Finally, install S1 Tiling
    #   Note: older versions of pip used to require --use-feature=2020-resolver
    #   to install S1Tiling to resolve `click` version that `eodag` also uses.
    python -m pip install s1tiling

    # Or, developper-version if you plan to work on S1 Tiling source code
    mkdir whatever && cd whatever
    git clone git@gitlab.orfeo-toolbox.org:s1-tiling/s1tiling.git
    cd s1tiling
    python -m pip install -r requirements-dev.txt

.. note::

    The :file:`requirements*.txt` files already force rasterio wheel to be
    ignored.

Extra packages
++++++++++++++

You may want to install extra packages like `bokeh
<https://pypi.org/project/bokeh/>`_ to monitor the execution of the multiple
processing by Dask.


Using S1Tiling with a docker
----------------------------

As the installation of S1Tiling could be tedious, versions ready to be used are
provided as Ubuntu 18.04 dockers.

You can browse the full list of available dockers in `S1Tiling registry
<https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/container_registry>`_.
Their naming scheme is
:samp:`registry.orfeo-toolbox.org/s1-tiling/s1tiling:{{version}}-ubuntu-otb7.4.0`,
with the version being either ``develop``, ``latest`` or the version number of
a recent release.

The docker, containing the version of S1Tiling of which you're reading the
documentation (i.e. version :samp:`{VERSION}`), could be fetched with:

.. code-block:: bash

    docker pull registry.orfeo-toolbox.org/s1-tiling/s1tiling:{VERSION}-ubuntu-otb7.4.0

or even directly used with


.. code-block:: bash

    docker run                            \
        -v /localpath/to/MNT:/MNT         \
        -v "$(pwd)":/data                 \
        -v $HOME/.config/eodag:/eo_config \
        --rm -it registry.orfeo-toolbox.org/s1-tiling/s1tiling:{VERSION}-ubuntu-otb7.4.0 \
        /data/MyS1ToS2.cfg

.. note::

    This examle considers:

    - SRTM's are available on local host through :file:`/localpath/to/MNT/` and
      they will be mounted into the docker as :file:`/MNT/`.
    - Logs and output files will be produced in current working directory (i.e.
      :file:`$(pwd)`) which will be mounted as :file:`data/`.
    - EODAG configuration file to be in :file:`$HOME/.config/eodag` which will
      be mounted as :file:`/eo_config/`.
    - A :ref:`configuration file <request-config-file>` named
      :file:`MyS1ToS2.cfg` is present in current working directory.
    - And it relates to the volumes mounted in the docker in the following way:

        .. code-block:: ini

            [Paths]
            output : /data/data_out
            srtm : /MNT/SRTM_30_hgt
            ...
            [DataSource]
            eodagConfig : /eo_config/eodag.yml
            ...

