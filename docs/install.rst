.. _install:

.. index:: installation

Installation
============

.. contents:: Contents:
   :local:
   :depth: 3

Manual installation with pip
----------------------------


From OTB binaries
+++++++++++++++++

S1Tiling is a Linux Python software which is based on Python packages but also
on C++ softwares OTB and GDAL.

We recommend to use a dedicated Python virtual environment and a dedicated OTB
{REF_OTB_VERSION} binary installation to install S1Tiling.
If you want use the OTB 7.4.2 version please consider the installation
instructions from previous S1Tiling version.

  .. note:: OTB 9+ binaries aren't compatible with older distributions of Linux like for instance Ubuntu 18.04.

Please find below a step by step installation:

.. code-block:: bash

    # First create a virtual environment and use it
    # | We won't document other approaches like conda (that enables selecting
    # | any version of Python), nor poetry, uv...
    python3 -m venv venv-s1tiling
    source venv-s1tiling/bin/activate

    # Upgrade pip and setuptools in your virtual environment
    pip install --upgrade pip
    pip install --upgrade setuptools

    # Make sure numpy is properly installed before updating GDAL python bindings
    pip install numpy

    # Install and configure OTB (included embedded GDAL) for S1Tiling
    # | Actually, since OTB 9, only the following packages are required:
    # | - OTB-{REF_OTB_VERSION}-Linux-FeaturesExtraction.tar.gz
    # | - OTB-{REF_OTB_VERSION}-Linux-Sar.tar.gz
    # | - OTB-{REF_OTB_VERSION}-Linux-Dependencies.tar.gz
    # | - OTB-{REF_OTB_VERSION}-Linux-Core.tar.gz
    # | But, OTB-{REF_OTB_VERSION}-Linux.tar.gz provides all OTB applications
    curl https://www.orfeo-toolbox.org/packages/archives/OTB/OTB-{REF_OTB_VERSION}-Linux.tar.gz -o ./OTB-{REF_OTB_VERSION}-Linux.tar.gz
    tar xf OTB-{REF_OTB_VERSION}-Linux.tar.gz --one-top-level=./venv-s1tiling/otb-{REF_OTB_VERSION}

    # Patch gdal-config with a generic and relocatable version
    curl https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/raw/{VERSION}/s1tiling/resources/gdal-config?ref_type={VERSION_TYPE}&inline=false -o venv-s1tiling/otb-{REF_OTB_VERSION}/bin/gdal-config
    echo -e '\nLD_LIBRARY_PATH="${CMAKE_PREFIX_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' >> venv-s1tiling/otb-{REF_OTB_VERSION}/otbenv.profile
    source venv-s1tiling/otb-{REF_OTB_VERSION}/otbenv.profile

    # Note that extra OTB applications for NORMLIM calibration support aren't
    # installed with this simplified procedure.
    # At the moment, you'll need to compile them manually from sources, or to
    # use S1Tiling docker distributions.

    # Install S1Tiling
    pip install S1Tiling=={VERSION}

.. note::
   We haven't tested yet with packages distributed for Linux OSes. It's likely
   you'll need to inject in your ``$PATH`` a version of :download:`gdal-config
   <../s1tiling/resources/gdal-config>` tuned to return GDAL configuration
   information.

On HPC clusters
+++++++++++++++

The procedure previously described stays valid. Yet you may already have
pre-installed modules for Python, GDAL, OTB...

As an inspiration, we provide the installation script used on CNES HPC
clusters. It may be a good starting point. See
:ref:`CNES installation script <install_cnes>` below.

   .. note::
      On CNES cluster where OTB has been compiled from sources, you can simply
      load the associated module:

      .. code-block:: bash

        # Example, on TREX:
        module load otb/9.0.0-python3.8


Installation scripts
++++++++++++++++++++

A couple of installation scripts used internally are provided.

.. _install_cnes:

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

    docker pull registry.orfeo-toolbox.org/s1-tiling/s1tiling:{VERSION}-ubuntu-otb{REF_OTB_VERSION}
    # or
    docker pull registry.orfeo-toolbox.org/s1-tiling/s1tiling:{VERSION}-ubuntu-otb7.4.1

or even directly used with

.. code-block:: bash

    docker run                            \
        -v /localpath/to/MNT:/MNT         \
        -v "$(pwd)":/data                 \
        -v $HOME/.config/eodag:/eo_config \
        --rm -it registry.orfeo-toolbox.org/s1-tiling/s1tiling:{VERSION}-ubuntu-otb{REF_OTB_VERSION} \
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
