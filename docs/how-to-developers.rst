.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. _howto_dev:

.. index:: Developer documentation (how to)

======================================================================
How To's...
======================================================================

How to add a new processing?
----------------------------

This is done by deriving :class:`StepFactory
<s1tiling.libs.otbpipeline.StepFactory>`. You'll find many examples in the
default :ref:`step factories <Existing Processings>`.

The important points are to decide:

- Where should the step happen in the sequence of pipelines? |br|
  In all cases, don't forget to add it in a pipeline registered in the sequence
  of pipelines.
- Shall its result be considered as a public product, or an intermediary step?
  |br|
  A public product is expected to be always produced. It shall then conclude a
  :ref:`pipeline <Pipelines>`. Also the pipeline shall be registered with
  ``product_required=True`` in that case.

- What would be the name of the result files? |br|
  Override :func:`build_step_output_filename()
  <s1tiling.libs.otbpipeline.StepFactory.build_step_output_filename>` with the
  answer.

  .. note::

      Even if there is no OTB application behind the step, this method needs to
      forward the filename of the input as done in
      :func:`AnalyseBorders.build_step_output_filename()
      <s1tiling.libs.otbwrappers.AnalyseBorders.build_step_output_filename>`.

- Which configuration options would be needed? |br|
  Copy them from the constructor that will be passed the
  :class:`s1tiling.libs.configuration.Configure` object.
- What meta information should be filled-in? |br|
  This should be done in :func:`complete_meta()
  <s1tiling.libs.otbpipeline.StepFactory.complete_meta>`. |br|
  Meta information can be used:

  - immediately by other methods like :func:`parameters()
    <s1tiling.libs.otbpipeline.StepFactory.parameters>`,
  - or by later steps in the pipeline.
- If there is an OTB application behind the step -- which should be the case
  for most processing steps.

In case the step relates to an OTB application:

- What would be the name of the temporary files while they are being produced? |br|
  Return the information from :func:`build_step_output_tmp_filename()
  <s1tiling.libs.otbpipeline.StepFactory.build_step_output_tmp_filename>`,
- Where the product should be produced? |br|
  Return the information from :func:`output_directory()
  <s1tiling.libs.otbpipeline.StepFactory.output_directory>` -- this is
  typically used from :func:`build_step_output_filename()
  <s1tiling.libs.otbpipeline.StepFactory.build_step_output_filename>`.
- What parameters shall be sent to the OTB application? |br|
  Return the information from :func:`parameters()
  <s1tiling.libs.otbpipeline.StepFactory.parameters>`.
- What are the parameters expected by the OTB application from the images that
  could be passed in-memory? |br|
  The default are ``"in"`` and ``"out"`` but could be overridden in the
  constructor of the new step factory through the parameters ``param_in`` and
  ``param_out``. See for instance
  :func:`s1tiling.libs.otbwrappers.OrthoRectify.__init__` implementation.
- What is the OTB application? |br|
  Its name is expected to be passed to the constructor of the parent class,
  from the constructor of the new class.

Technically all other methods from :class:`StepFactory
<s1tiling.libs.otbpipeline.StepFactory>` could be overridden. For instance,
:func:`create_step() <s1tiling.libs.otbpipeline.StepFactory.create_step>` could
be overridden to change the type of :ref:`Steps` instantiated.

Release a new version
---------------------

Here is a short list of the actions to do for each new release.

1. Update the :ref:`release notes <release_notes>`

2. Make sure :file:`__meta__.py` version matches the name of the version to be
   released.
   Don't forget the `rcX` suffix if need be.

  Version format is expected to follow the following convention:
  ``M.m(.p)(rcX)`` See
  https://packaging.python.org/guides/distributing-packages-using-setuptools/#standards-compliance-for-interoperability

  Let's extract version number into a variable to simplify following steps

  .. code:: bash

      version="$(awk '/version/ {print $3}' s1tiling/__meta__.py | xargs )"
      echo "version: ${version}"

3. Handle all the issues associated for the related milestone.

4. Push ``develop`` branch.

   .. code::

       git checkout develop && git push

4. Merge ``develop`` branch into ``master``

   .. code::

       git checkout master && git merge develop

5. Push ``master`` branch.

   .. code::

       git checkout master && git push


6 Create a git tag matching the version number

   .. code::

       git tag -a "${version}"
       # And fill in version information

7. Push the tag

   .. code::

       git push --tags


   .. note::

       From there on, the CI will automatically take care of registering the
       source distribution (only; and not the wheel!) on pypi as if we had
       manually ran

       .. code::

           # Prepare the packets for pipy
           python3 setup.py sdist bdist_wheel

           # Push to pipy
           python3 -m twine upload --repository pypi dist/S1Tiling-${version}*


8. Update :file:`__meta__.py` version to the next expected version.
    Do not use the `rcX` suffix for the moment.
