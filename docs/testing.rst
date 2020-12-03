.. _testing:

.. index:: testing

======================================================================
Testing
======================================================================

.. contents:: Contents:
   :local:
   :depth: 3

Contributors will want to test their changes against a baseline to ensure no
regression appear.

S1-Tiling tests are not part of an integrated continous workflow. They are
meant to be run on an on-demand basis.

At this moment we only have a single end-to-end test on S2 33NWB tile on S1
images acquired in January 2020.

.. _baseline:

The baseline
------------

There are two ways to obtain the baseline:

- Either we have given you an authentication token to the S3 server where we
  have stored the current baseline.

  In that case, thanks to `MinIO client
  <https://docs.min.io/docs/minio-client-quickstart-guide.html>`_, you can:

  - first and once: register your machine to the S3 server we use

     .. code:: bash

         mc config host add minio-otb https://s3.orfeo-toolbox.org/ <access-key> <secret-key> --api S3v4

  - then retrieve the baseline data thanks to

     .. code:: bash

         mc cp --recursive minio-otb/s1-tiling/baseline /some/local/path

  Instead of ``mc``, you can also use ``rclone`` -- which is for instance
  already installed on HAL.

- Or you'll need to first establish the baseline from a version of S1Tiliing
  known to work correctly, before introducing any change.

  Organise the :ref:`S1 images <paths.s1_images>` downloaded into a directory
  named :file:`inputs` and the results into a directory named :file:`expected`.


.. _pytest:

Running the tests
-----------------

S1 Tiling tests depend on pytest. You can see all the supported options with:

.. code:: bash

    pytest --help

In particular they depend on the following options:

.. option:: --baselinedir=BASELINEDIR

   Directory where the baseline is.

.. option:: --outputdir=OUTPUTDIR

   Directory where the S2 products will be generated.

   .. warning::

       Don't forget to clean it eventually.

.. option:: --tmpdir=TMPDIR

   Directory where the temporary files will be generated.

   .. warning::

       Don't forget to clean it eventually.


.. option:: --srtmdir=SRTMDIR

   Directory where SRTM files are -- default value: :envvar:`$SRTM_DIR`

.. option:: --download

   Download the input files with eodag instead of using the compressed ones
   from the baseline. If true, raw S1 products will be downloaded into
   :file:`{tmpdir}/inputs`.

