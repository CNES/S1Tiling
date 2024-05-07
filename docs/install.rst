.. _install:

.. index:: installation

Installation
============

.. contents:: Contents:
   :local:
   :depth: 3

Manual installation with pip
----------------------------

S1Tiling is a Linux Python software which is based on Python packages but also to C++ software OTB and GDAL.
We recommend to use a dedicated Python virtual environement and a dedicated OTB 9.0.0 binary installation to install S1Tiling.
If you want use the OTB 7.4.2 version please consider the S1Tiling previous version installation instructions.

Please find below a step by step installation:

.. code-block:: bash

    # First create a virtual environment and use it
    python3 -m venv venv-s1tiling
    source venv-s1tiling/bin/activate

    # Upgrade pip and setuptools in your virtual environment
    pip install --upgrade pip
    pip install --upgrade setuptools

    # Install and configure OTB (included embedded GDAL) for S1Tiling
    curl https://www.orfeo-toolbox.org/packages/archives/OTB/OTB-9.0.0-Linux.tar.gz -o ./OTB-9.0.0-Linux.tar.gz
    tar xf OTB-9.0.0-Linux.tar.gz --one-top-level=./venv-s1tiling/otb-9.0.0
    curl https://s1-tiling.pages.orfeo-toolbox.org/s1tiling/1.1.0rc1/_downloads/6b5223a542baf214a8a6820bf4e786cf/gdal-config -o venv-s1tiling/otb-9.0.0/bin/gdal-config
    # https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/raw/1.1.0rc1/s1tiling/resources/gdal-config?ref_type=tags&inline=false
    echo -e '\nLD_LIBRARY_PATH="${CMAKE_PREFIX_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' >> venv-s1tiling/otb-9.0.0/otbenv.profile
    source venv-s1tiling/otb-9.0.0/otbenv.profile

    # Install S1Tiling
    pip install S1Tiling==1.1.0rc1

On CNES cluster where OTB has been compiled from sources, you can simply load the
  associated module:

        .. code-block:: bash

            # Example, on TREX:
            module load otb/9.0.0-python3.8

        .. note::

            The installation script which is used on CNES clusters would be a
            good starting point. See: :download:`install-CNES.sh
            <../s1tiling/resources/install-CNES.sh>`

.. note::
   We haven't tested yet with packages distributed for Linux OSes. It's likely
   you'll need to inject in your ``$PATH`` a version of :download:`gdal-config
   <../s1tiling/resources/gdal-config>` tuned to return GDAL configuration
   information.

Installation scripts
++++++++++++++++++++

A couple of installation scripts used internally are provided.

CNES clusters installation script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:download:`install-CNES.sh <../s1tiling/resources/install-CNES.sh>` takes care
of installating S1Tiling on CNES HPC clusters.

.. list-table::
  :widths: auto
  :header-rows: 1
  :stub-columns: 0

  * - Requirements
    - It...

  * -
        - OTB installed from sources as a `Lmod
          <https://lmod.readthedocs.io/en/latest/?badge=latest>`_ module.
    -
        - Installs S1Tiling in a dedicated space on the clusters,
        - Defines a Python virtual environment where S1Tiling will reside,
        - Automatically generates a S1Tiling module file.

Linux machines installation script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:download:`install-rcbin.sh <../s1tiling/resources/install-rcbin.sh>` takes
care of installating S1Tiling on Linux machines

.. list-table::
  :widths: auto
  :header-rows: 1
  :stub-columns: 0

  * - Requirements
    - It...

  * -
        - An un-extracted OTB binary release,
        - Python 3.8+,
        - A directory where S1Tiling has been cloned,
        - Conda.

    -
        - Creates a conda environment for the selected python version (3.8 by
          default with OTB 7.x, 3.11 w/ OTB 8.x, and 3.12 w/ OTB 9.x),
        - Extracts the OTB binary release in the directory where the
          ``OTB-M.m.p-Linux64.run`` file is,
        - Patches ``UseOTB.cmake`` if need be (in case of C++ ABI mismatch in
          7.4.2 OTB release),
        - Patches :file:`otbenv.profile`,
        - Regenerates Python bindings for OTB,
        - Installs GDAL python bindings from sources (to match GDAL version
          shipped by OTB binaries),
        - Install S1Tiling from its source directory,
        - And automatically generates a S1Tiling module file named:
          ``s1tiling/otb{Mmp}-py{Mm}`` (Major/minor/patch).

          .. note::
            You can source :file:`otbenv.profile` and activate the conda
            environement manually if you don't use `Lmod
            <https://lmod.readthedocs.io/en/latest/?badge=latest>`_.

         .. note::
            You will still need to install `LIA extra applications
            <https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0>`_ in
            order to :ref:`produce LIA maps <scenario.s1liamap>`, or to apply
            :ref:`σ° NORMLIM calibration <scenario.s1processorlia>`.

Extra packages
++++++++++++++

You may want to install extra packages like `bokeh
<https://pypi.org/project/bokeh/>`_ to monitor the execution of the multiple
processing by Dask.


.. _docker:

Using S1Tiling with a docker
----------------------------

As the installation of S1Tiling could be tedious, versions ready to be used are
provided as Ubuntu dockers.

You can browse the full list of available dockers in `S1Tiling registry
<https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/container_registry>`_.
Their naming scheme is
:samp:`registry.orfeo-toolbox.org/s1-tiling/s1tiling:{{version}}-ubuntu-otb{{otbversion}}`,
with the version being either ``develop``, ``latest`` or the version number of
a recent release.

The docker, containing the version of S1Tiling of which you're reading the
documentation (i.e. version :samp:`{VERSION}`), could be fetched with:

.. code-block:: bash

    docker pull registry.orfeo-toolbox.org/s1-tiling/s1tiling:{VERSION}-ubuntu-otb9.0.0
    # or
    docker pull registry.orfeo-toolbox.org/s1-tiling/s1tiling:{VERSION}-ubuntu-otb7.4.2

or even directly used with

.. code-block:: bash

    docker run                            \
        -v /localpath/to/MNT:/MNT         \
        -v "$(pwd)":/data                 \
        -v $HOME/.config/eodag:/eo_config \
        --rm -it registry.orfeo-toolbox.org/s1-tiling/s1tiling:{VERSION}-ubuntu-otb9.0.0 \
        /data/MyS1ToS2.cfg

.. note::

    This example considers:

    - DEM's are available on local host through :file:`/localpath/to/MNT/` and
      they will be mounted into the docker as :file:`/MNT/`.
    - Logs and output files will be produced in current working directory (i.e.
      :file:`$(pwd)`) which will be mounted as :file:`data/`.
    - EODAG configuration file to be in :file:`$HOME/.config/eodag` which will
      be mounted as :file:`/eo_config/`.
    - A :ref:`configuration file <request-config-file>` named
      :file:`MyS1ToS2.cfg` is present in current working directory, which is
      seen from docker perspective as in :file:`data/` directory.
    - And it relates to the volumes mounted in the docker in the following way:

        .. code-block:: ini

            [Paths]
            output : /data/data_out
            dem_dir : /MNT/SRTM_30_hgt
            ...
            [DataSource]
            eodag_config : /eo_config/eodag.yml
            ...

.. _docker.S1LIAMap:

Using S1LIAMap with a docker
++++++++++++++++++++++++++++

It's also possible to run :program:`S1LIAMap` in the docker -- see :ref:`LIA
Map production scenario <scenario.S1LIAMap>`. In order to do that, pass
``--lia`` as the first parameter to the docker *entry point*.

In other word, run the docker with something like the following

.. code-block:: bash

    docker run                            \
        -v /localpath/to/MNT:/MNT         \
        -v "$(pwd)":/data                 \
        -v $HOME/.config/eodag:/eo_config \
        --rm -it registry.orfeo-toolbox.org/s1-tiling/s1tiling:{VERSION}-ubuntu-otb7.4.2 \
        --lia                             \
        /data/MyS1ToS2.cfg

The only difference with the *normal case* example: there is a ``--lia``
parameter in the penultimate line.
