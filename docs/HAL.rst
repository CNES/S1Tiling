.. _HAL:

.. index:: HAL

Some specificities about HAL cluster
====================================

.. contents:: Contents:
   :local:
   :depth: 3

Installation on HAL
-------------------

There are mainly two x two ways to install S1Tiling on HAL.

If one wants to install S1Tiling from sources instead of pipy, it could be done
from the following context. Then, in later steps, use ``"${S1TILING_SRC_DIR}"``
instead of ``s1tiling`` as ``pip`` parameter.

.. code:: bash

    # Proposed directories where it could be installed
    TST_DIR=/work/scratch/${USER}/S1Tiling/install
    S1TILING_ROOT_DIR=/work/scratch/${USER}/S1Tiling/
    S1TILING_SOURCES=sources
    S1TILING_SRC_DIR=${S1TILING_ROOT_DIR}/${S1TILING_SOURCES}

    cd "${S1TILING_ROOT_DIR}"
    git clone git@gitlab.orfeo-toolbox.org:s1-tiling/s1tiling.git ${S1TILING_SOURCES}

...from available OTB module (and w/ pip)
+++++++++++++++++++++++++++++++++++++++++++

.. code:: bash

    ml otb/7.4-python3.7.2

    # Create a pip virtual environment
    python -m venv install_with_otb_module

    # Configure the environment with:
    source install_with_otb_module/bin/activate
    # - an up-to-date pip
    python -m pip install --upgrade pip
    # - an up-to-date setuptools
    python -m pip install --upgrade setuptools

    # Finally, install S1Tiling from sources
    mkdir /work/scratch/${USER}/tmp
    TMPDIR=/work/scratch/${USER}/tmp/ python -m pip install s1tiling

    deactivate
    ml purge

To use it

.. code:: bash

    ml purge
    ml otb/7.4-python3.7.2
    source install_with_otb_module/bin/activate

    S1Processor requestfile.cfg

    deactivate
    ml purge

...from available OTB module (and w/ conda)
+++++++++++++++++++++++++++++++++++++++++++

.. code:: bash

    ml otb/7.4-python3.7.2

    # Create a conda environment
    ml conda
    conda create --prefix ./conda_install_with_otb_distrib python==3.7.2

    # Configure the environment with:
    conda activate "${TST_DIR}/conda_install_with_otb_distrib"
    # - an up-to-date pip
    python -m pip install --upgrade pip
    # - an up-to-date setuptools
    python -m pip install --upgrade setuptools

    # Finally, install S1Tiling from sources
    mkdir /work/scratch/${USER}/tmp
    TMPDIR=/work/scratch/${USER}/tmp/ python -m pip install s1tiling

    conda deactivate
    ml purge

To use it

.. code:: bash

    ml purge
    ml conda
    ml otb/7.4-python3.7.2
    conda activate "${TST_DIR}/conda_install_with_otb_distrib"

    S1Processor requestfile.cfg

    conda deactivate
    ml purge


...from released OTB binaries...
++++++++++++++++++++++++++++++++

Given :file:`otbenv.profile` cannot be unloaded, prefer the above methods based
on OTB module.

First let's start by installing OTB binaries somewhere in your personnal (or
project) environment.

.. code:: bash

    # Start from a clean environment
    ml purge
    cd "${TST_DIR}"
    # Install OTB binaries
    wget https://www.orfeo-toolbox.org/packages/OTB-7.4.0-Linux64.run
    bash OTB-7.4.0-Linux64.run

    # Patches gdal-config
    cp "${S1TILING_SRC_DIR}/s1tiling/resources/gdal-config" OTB-7.4.0-Linux64/bin/
    # Patches LD_LIBRARY_PATH
    echo "export LD_LIBRARY_PATH=\"$(readlink -f OTB-7.4.0-Linux64/lib)\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"" >> OTB-7.4.0-Linux64/otbenv.profile

.. note::

   :file:`gdal-config`  is either available from the sources
   (``${S1TILING_SRC_DIR}/s1tiling/resources/gdal-config``) or to download
   from :download:`here: gdal-config <../s1tiling/resources/gdal-config>`.

...and with conda
~~~~~~~~~~~~~~~~~

Given the OTB binaries installed, we still need to update the Python bindings
for the chosen version of Python.

.. code:: bash

    # Create a conda environment
    ml conda
    conda create --prefix ./conda_install_with_otb_distrib python==3.7.2

    # Configure the environment with:
    conda activate "${TST_DIR}/conda_install_with_otb_distrib"
    # - an up-to-date pip
    python -m pip install --upgrade pip
    # - an up-to-date setuptools
    python -m pip install --upgrade setuptools
    # - numpy in order to compile OTB python bindinds for Python 3.7.2
    pip install numpy

    # - load OTB binaries
    source OTB-7.4.0-Linux64/otbenv.profile
    # load cmake and gcc to compile the binding
    ml cmake gcc
    # And update the bindings
    (cd OTB-7.4.0-Linux64/ && ctest -S share/otb/swig/build_wrapping.cmake -VV)
    ml unload cmake gcc

    # Finally, install S1Tiling from sources
    mkdir /work/scratch/${USER}/tmp
    TMPDIR=/work/scratch/${USER}/tmp/ python -m pip install s1tiling

    conda deactivate
    ml purge


To use it

.. code:: bash

    ml purge
    ml conda
    conda activate "${TST_DIR}/conda_install_with_otb_distrib"
    source "${TST_DIR}/OTB-7.4.0-Linux64/otbenv.profile"

    S1Processor requestfile.cfg

    conda deactivate
    ml purge

...and with pip
~~~~~~~~~~~~~~~~~

Given the OTB binaries installed, we still need to update the Python bindings
for the chosen version of Python.

.. code:: bash

    # Create a pip virtual environment
    ml python
    python -m venv install_with_otb_binaries

    # Configure the environment with:
    source install_with_otb_binaries/bin/activate
    # - an up-to-date pip
    python -m pip install --upgrade pip
    # - an up-to-date setuptools
    python -m pip install --upgrade setuptools
    # - numpy in order to compile OTB python bindinds for Python
    pip install numpy

    # - load OTB binaries
    source OTB-7.4.0-Linux64/otbenv.profile
    # load cmake and gcc to compile the binding
    ml cmake gcc
    # And update the bindings
    (cd OTB-7.4.0-Linux64/ && ctest -S share/otb/swig/build_wrapping.cmake -VV)
    ml unload cmake gcc

    # Finally, install S1Tiling from sources
    mkdir /work/scratch/${USER}/tmp
    TMPDIR=/work/scratch/${USER}/tmp/ python -m pip install s1tiling

    deactivate
    ml purge

To use it

.. code:: bash

    ml purge
    source install_with_otb_binaries/bin/activate
    source "${TST_DIR}/OTB-7.4.0-Linux64/otbenv.profile"

    S1Processor requestfile.cfg

    deactivate
    ml purge

Executing S1 Tiling as a job
----------------------------

The theory
++++++++++

A few options deserve our attention when running S1 Tiling as a job on a
cluster like HAL.

.. list-table::
  :widths: auto
  :header-rows: 1
  :stub-columns: 1

  * - Option
    - Need to know

  * - :ref:`[PATHS].tmp <paths.tmp>`
    - Temporary files shall not be generated on the GPFS, instead, they are
      best generated locally in :file:`$TMPDIR`. Set this option to
      :file:`%(TMPDIR)s/s1tiling` for instance.

      .. code:: ini

          [PATHS]
          tmp : %(TMPDIR)s/s1tiling


      .. warning::

         Of course, we shall not use :file:`$TMPDIR` when running S1 Tiling on
         ``visu`` nodes. Actually, we should **not** use S1 Tiling for
         intensive computation on nodes not dedicated to computations.

  * - :ref:`[PATHS].srtm <paths.srtm>`
    - SRTM files are stored in
      :file:`/work/datalake/static_aux/MNT/SRTM_30_hgt`.

      .. code:: ini

          [PATHS]
          srtm : /work/datalake/static_aux/MNT/SRTM_30_hgt

  * - :ref:`[Processing].nb_otb_threads <Processing.nb_otb_threads>`
    - This is the number of threads that will be used by each OTB application
      pipeline.

  * - :ref:`[Processing].nb_parallel_processes <Processing.nb_parallel_processes>`
    - This is the number of OTB application pipelines that will be executed in
      parallel.

  * - :ref:`[Processing].ram_per_process <Processing.ram_per_process>`
    - RAM allowed per OTB application pipeline, in MB.

  * - PBS resources
    - - At this time, S1 Tiling does not support multiple and related jobs. We
        can have multiple jobs but they should use different working spaces and
        so on. This means ``select`` value shall be one.

      - The number of CPUs should be equal to the number of threads * the
        number of parallel processes -- and it shall not be less than the
        product of these two options.

      - The required memory shall be greater that the number of parallel
        processes per the RAM allowed to each OTB pipeline.

      This means, that for

      .. code:: ini

          # The request file
          [Processing]
          nb_parallel_processes: 10
          nb_otb_threads: 2
          ram_per_process: 4096


      Then the job request shall contain at least

      .. code:: bash

        #PBS -l select=1:ncpus=20:mem=40gb
        # always 1 for select
        # cpu = 2 * 10 => 20
        # mem = 10 * 4096 => 40gb

TL;DR: here is an example
+++++++++++++++++++++++++

PBS job file
~~~~~~~~~~~~

.. code:: bash

    #!/bin/bash
    #PBS -N job-s1tiling
    #PBS -l select=1:ncpus=20:mem=40gb
    #PBS -l walltime=1:00:00

    # NB: Using 5Gb per cpu

    # The number of allocated CPUs is in the select parameter let's extract it
    # automatically
    NCPUS=$(qstat -f "${PBS_JOBID}" | awk '/resources_used.ncpus/{print $3}')
    # Let's use 2 threads in each OTB application pipeline
    export NB_OTB_THREADS=2
    # Let's deduce the number of OTB application pipelines to run in parallel
    export NB_OTB_PIPELINES=$(($NCPUS / $NB_OTB_THREADS))
    # These two variables have been exported to be automatically used from the
    # S1 tiling request file.

    # Let's suppose we have a S1Tiling module -- which will be the case
    # eventually. See the previous sections in the meantime.
    ml s1tiling

    mkdir -p "${PBS_O_WORKDIR}/${PBS_JOBID}"
    cd "${PBS_O_WORKDIR}/${PBS_JOBID}"
    S1Processor S1Processor.cfg || {
        echo "Echec de l'exécution de programme" >&2
        exit 2
    }


S1 Tiling request file: :file:`S1Processor.cfg`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ini

      [PATHS]
      tmp : %(TMPDIR)s/s1tiling
      srtm : /work/datalake/static_aux/MNT/SRTM_30_hgt
      ...

      [Processing]
      # Let's use the exported environment variables thanks to "%()s" syntax
      nb_parallel_processes: %(NB_OTB_PIPELINES)s
      nb_otb_threads: %(NB_OTB_THREADS)s
      ram_per_process: 4096
      ...
