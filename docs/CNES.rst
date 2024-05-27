.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. _CNES:

.. index:: HAL
.. index:: TREX

Some specificities about CNES clusters
======================================

.. contents:: Contents:
   :local:
   :depth: 3

Using S1Tiling Lmod module on HAL/TREX
--------------------------------------

S1Tiling is already installed on HAL and TREX (since June 2023). It's available
through `Lmod <https://lmod.readthedocs.io/en/latest/?badge=latest>`_; see also
HAL/TREX user guides.

.. code:: bash

    # Use the lastest version
    ml s1tiling

    # Check versions available
    ml av version

    # Activate a specific version
    ml s1tiling/1.0.0-otb7.4.2
    # Or...
    ml s1tiling/1.1.0rc1-otb9.0.0


.. note::

    Only S1Tiling 1.0 will ever be installed with a dependency to OTB 7.4.2
    (and Python 3.8.4 and g++ 8.2). |br|
    S1Tiling 1.1.0 will be installed once with a dependency to OTB 7.4.2, and
    once with a dependency to OTB 9.

Installation on HAL/TREX
------------------------

You may prefer to install S1Tiling yourself. In that case, there are mainly two
X two ways to install S1Tiling on CNES clusters.

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

    ml otb/7.4.2-python3.8.4-gcc8.2

    # Create a pip virtual environment
    python -m venv install_with_otb_module

    # Configure the environment with:
    source install_with_otb_module/bin/activate
    # - an up-to-date pip
    python -m pip install --upgrade pip
    # - an up-to-date setuptools==57.5.0
    python -m pip install --upgrade setuptools==57.5.0

    # Finally, install S1Tiling from sources
    mkdir /work/scratch/${USER}/tmp
    TMPDIR=/work/scratch/${USER}/tmp/ python -m pip install s1tiling

    deactivate
    ml purge

To use it

.. code:: bash

    ml purge
    ml otb/7.4.2-python3.8.4-gcc8.2
    source install_with_otb_module/bin/activate

    S1Processor requestfile.cfg

    deactivate
    ml purge


.. note::

    This is the approach that has been chosen by the installation script we use
    internally. See: :download:`install-CNES.sh
    <../s1tiling/resources/install-CNES.sh>`

    Prefer the next approach based on conda if you wish to use a different
    version of Python.

...from available OTB module (and w/ conda)
+++++++++++++++++++++++++++++++++++++++++++

.. note::
   This approach permits to select a different version of Python, but it will
   be a bit more complex to correctly adjust the desired version of gdal python
   bindings to be exactly the same as the one used to generate OTB module. This
   isn't demonstrated here.

.. code:: bash

    ml otb/7.4.2-python3.8.4-gcc8.2

    # Create a conda environment
    ml conda
    conda create --prefix ./conda_install_with_otb_distrib python==3.8.4

    # Configure the environment with:
    conda activate "${TST_DIR}/conda_install_with_otb_distrib"
    # - an up-to-date pip
    python -m pip install --upgrade pip
    # - an up-to-date setuptools==57.5.0
    python -m pip install --upgrade setuptools==57.5.0

    # Finally, install S1Tiling from sources
    mkdir /work/scratch/${USER}/tmp
    TMPDIR=/work/scratch/${USER}/tmp/ python -m pip install s1tiling

    conda deactivate
    ml purge

To use it

.. code:: bash

    ml purge
    ml conda
    ml otb/7.4.2-python3.8.4-gcc8.2
    conda activate "${TST_DIR}/conda_install_with_otb_distrib"

    S1Processor requestfile.cfg

    conda deactivate
    ml purge


...from released OTB binaries...
++++++++++++++++++++++++++++++++

Given :file:`otbenv.profile` cannot be unloaded, prefer the above methods based
on OTB module.

First let's start by installing OTB binaries somewhere in your personal (or
project) environment.

.. code:: bash

    # Start from a clean environment
    ml purge
    cd "${TST_DIR}"
    # Install OTB binaries
    wget https://www.orfeo-toolbox.org/packages/OTB-7.4.2-Linux64.run
    bash OTB-7.4.2-Linux64.run

    # Patches gdal-config
    cp "${S1TILING_SRC_DIR}/s1tiling/resources/gdal-config" OTB-7.4.2-Linux64/bin/
    # Patches LD_LIBRARY_PATH
    echo "export LD_LIBRARY_PATH=\"$(readlink -f OTB-7.4.2-Linux64/lib)\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"" >> OTB-7.4.2-Linux64/otbenv.profile

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
    conda create --prefix ./conda_install_with_otb_distrib python==3.8.4

    # Configure the environment with:
    conda activate "${TST_DIR}/conda_install_with_otb_distrib"
    # - an up-to-date pip
    python -m pip install --upgrade pip
    # - an up-to-date setuptools==57.5.0
    python -m pip install --upgrade setuptools==57.5.0
    # - numpy in order to compile OTB python bindinds for Python 3.8.4
    pip install numpy
    # - gdal python bindinds shall be compatible with libgdal.so shipped w/ OTB binaries
    pip --no-cache-dir install "gdal==$(gdal-config --version)" --no-binary :all:

    # - load OTB binaries
    source OTB-7.4.2-Linux64/otbenv.profile
    # load cmake and gcc to compile the binding
    ml cmake gcc
    # And update the bindings
    (cd OTB-7.4.2-Linux64/ && ctest -S share/otb/swig/build_wrapping.cmake -VV)
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
    source "${TST_DIR}/OTB-7.4.2-Linux64/otbenv.profile"

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
    # - an up-to-date setuptools==57.5.0
    python -m pip install --upgrade setuptools==57.5.0
    # - numpy in order to compile OTB python bindinds for Python
    pip install numpy
    # - gdal python bindinds shall be compatible with libgdal.so shipped w/ OTB binaries
    pip --no-cache-dir install "gdal==$(gdal-config --version)" --no-binary :all:

    # - load OTB binaries
    source OTB-7.4.2-Linux64/otbenv.profile
    # load cmake and gcc to compile the binding
    ml cmake gcc
    # And update the bindings
    (cd OTB-7.4.2-Linux64/ && ctest -S share/otb/swig/build_wrapping.cmake -VV)
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
    source "${TST_DIR}/OTB-7.4.2-Linux64/otbenv.profile"

    S1Processor requestfile.cfg

    deactivate
    ml purge

Executing S1 Tiling as a job
----------------------------

The theory
++++++++++

A few options deserve our attention when running S1 Tiling as a job on a
cluster like HAL or TREX.

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

  * - :ref:`[PATHS].dem_dir <paths.dem_dir>`
    - Original DEM files are stored in
      :file:`/work/datalake/static_aux/MNT/SRTM_30_hgt`.

      .. code:: ini

          [PATHS]
          dem_dir : /work/datalake/static_aux/MNT/SRTM_30_hgt

  * - :ref:`[Processing].cache_dem_by <Processing.cache_dem_by>`
    - DEM files, and GEOID file, should be **copied** locally on
      :ref:`[PATHS].tmp <paths.tmp>` instead of being symlinked over the GPFS.

      .. code:: ini

          [Processing]
          cache_dem_by : copy

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
        so on. This means PBS ``select`` value shall be one.

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

SLRUM job file (TREX)
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    #!/bin/bash
    #SBATCH --account=...
    #SBATCH --partition=cpu2022   # jobs < 72h
    #SBATCH --qos=...
    #SBATCH -N 1                  # number of nodes (or --nodes=1)
    #SBATCH -n 1                  # number of tasks (or --ntasks=1)
    #SBATCH --cpus-per-task=20    # number of cpus par task
    #SBATCH --mem=160G            # memory per core
    #SBATCH --time=00:59:00       # Wall Time 59mn
    #SBATCH -J job-s1tiling

    # The number of allocated CPUs
    NCPUS=${SLURM_CPUS_PER_TASK}
    # Let's use 2 threads in each OTB application pipeline
    export NB_OTB_THREADS=2
    # Let's deduce the number of OTB application pipelines to run in parallel
    export NB_OTB_PIPELINES=$(($NCPUS / $NB_OTB_THREADS))
    # These two variables have been exported to be automatically used from the
    # S1tiling request file.

    # Let's use an existing S1Tiling module
    ml s1tiling/1.1.0rc1-otb9.0.0

    # Expecting S1Processor.cfg in ${SLURM_SUBMIT_DIR}, the logs will be
    # produced in a subdirectory named after the the JOB ID.
    WORK_DIR="${SLURM_SUBMIT_DIR}/${SLURM_JOB_ID}"
    mkdir -p "${WORK_DIR}"
    cd "${WORK_DIR}"
    S1Processor --cache-before-ortho ../S1Processor.cfg || {
        code=$?
        echo "Echec de l'exécution de programme" >&2
        exit ${code}
    }

PBS job file (HAL)
~~~~~~~~~~~~~~~~~~

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
    # S1tiling request file.

    # Let's use an existing S1Tiling module
    ml s1tiling/1.1.0rc1-otb9.0.0

    # Expecting S1Processor.cfg in ${PBS_O_WORKDIR}, the logs will be
    # produced in a subdirectory named after the the JOB ID.
    WORK_DIR="${PBS_O_WORKDIR}/${PBS_JOBID}"
    mkdir -p "${WORK_DIR}"
    cd "${WORK_DIR}"
    S1Processor --cache-before-ortho ../S1Processor.cfg || {
        code=$?
        echo "Echec de l'exécution de programme" >&2
        exit ${code}
    }


S1 Tiling request file: :file:`S1Processor.cfg`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ini

      [PATHS]
      tmp : %(TMPDIR)s/s1tiling
      dem_dir : /work/datalake/static_aux/MNT/SRTM_30_hgt
      ...

      [Processing]
      cache_dem_by: copy
      # Let's use the exported environment variables thanks to "%()s" syntax
      nb_parallel_processes: %(NB_OTB_PIPELINES)s
      nb_otb_threads: %(NB_OTB_THREADS)s
      ram_per_process: 4096
      ...
