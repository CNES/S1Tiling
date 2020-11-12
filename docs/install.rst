.. _install:

Installation
============

OTB & GDAL dependency
---------------------

S1 Tiling depends on `OTB 7.2+ <https://www.orfeo-toolbox.org/CookBook-7.2/>`_.
First install OTB on your platform. See the `related documentation
<https://www.orfeo-toolbox.org/CookBook-7.2/Installation.html>`_ to install OTB
on your system..

Then, you'll also need a version of GDAL which is compatible with your OTB
version.

- In case you're using OTB binary distribution, you'll need to patch the files
  provided.

  - For that purpose you can drop this simplified and generic version of
    :download:`gdal-config <../s1tiling/resources/gdal-config>` into the
    ``bin/`` directory where you've extracted OTB. This will permit :samp:`pip
    install gdal=={vernum}` to work correctly.
  - You'll also have to patch ``otbenv.profile`` to insert OTB ``lib/``
    directory at the start of ``$LD_LIBRARY_PATH``. This will permit ``python -c
    'from osgeo import gdal'`` to work correctly.

- In case you've compiled OTB from sources, you shouldn't have this kind of
  troubles.

.. note::
   We haven't tested yet with packages distributed for Linux OSes. It's likely
   you'll need to inject in your ``$PATH`` a version of :download:`gdal-config
   <../s1tiling/resources/gdal-config>` tuned to return GDAL configuration
   information.

Possible conflicts
++++++++++++++++++

`eodag <https://github.com/CS-SI/eodag>`_ requires ``xarray`` which in turn
requires Python 3.6 while default OTB package is built with Python 3.5. This
means you'll likely need to recompile OTB Python bindings as described in:
https://www.orfeo-toolbox.org/CookBook/Installation.html#recompiling-python-bindings


.. code-block:: bash

    cd OTB-7.2.0-Linux64
    source otbenv.profile
    ctest -S share/otb/swig/build_wrapping.cmake -VV


S1 Tiling installation
----------------------

Then you can install S1 Tiling thanks to `pip`.

.. code-block:: bash

    # First go into a virtual environment (optional)
    python -m venv myS1TilingEnv
    cd myS1TilingEnv
    source bin/activate

    # Then, upgrade pip and setuptools in your virtual environment
    python -m pip install --upgrade pip
    python -m pip install --upgrade setuptools

    # Finally, install S1 Tiling
    python -m pip install --use-feature=2020-resolver s1tiling

    # Or, developper-version if you plan to work on S1 Tiling source code
    mkdir whatever && cd whatever
    git clone git@gitlab.orfeo-toolbox.org:s1-tiling/s1tiling.git
    cd s1tiling
    python -m pip install -r requirements-dev.txt --use-feature=2020-resolver


.. note::

    We have noted that the new ``--use-feature=2020-resolver`` helps resolve
    ``click`` version that eodag also uses.
