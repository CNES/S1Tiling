.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. _howto_dev:

.. index:: Developer documentation (how to)

======================================================================
How To's…
======================================================================

How to add a new processing?
----------------------------

This is done by deriving from :class:`StepFactory
<s1tiling.libs.steps.StepFactory>`, or from :class:`OTBStepFactory
<s1tiling.libs.steps.OTBStepFactory>`. You'll find many examples in the
default :ref:`step factories <Existing Processings>`.

The important points are to decide:

- Where should the step happen in the sequence of pipelines? |br|
  In all cases, don't forget to add it in a pipeline registered in the sequence
  of pipelines.
- Shall its result be considered as a public product, or an intermediary step?
  |br|
  A public product is expected to be always produced. It shall then conclude a
  :ref:`pipeline <Pipelines>`. Also, the pipeline shall be registered with
  ``product_required=True`` in that case.

- What would be the name of the result files? |br|
  Override :func:`build_step_output_filename()
  <s1tiling.libs.steps.StepFactory.build_step_output_filename>` with the
  answer.

  .. note::

      Even if there is no OTB application behind the step, this method needs to
      forward the filename of the input as done in
      :func:`AnalyseBorders.build_step_output_filename()
      <s1tiling.libs.otbwrappers.AnalyseBorders.build_step_output_filename>`.

- Which configuration options would be needed? |br|
  Copy them from the constructor that will be passed the
  :class:`s1tiling.libs.configuration.Configuration` object.
- What meta information should be filled-in? |br|
  This should be done in :func:`complete_meta()
  <s1tiling.libs.steps.StepFactory.complete_meta>`. |br|
  Meta information can be used:

  - immediately by other methods like :func:`parameters()
    <s1tiling.libs.steps._FileProducingStepFactory.parameters>`,
  - or by later steps in the pipeline.
- If there is an OTB application behind the step -- which should be the case
  for most processing steps.

In case the step relates to an OTB application:

- What parameters shall be sent to the OTB application? |br|
  Return the information from :func:`parameters()
  <s1tiling.libs.steps._FileProducingStepFactory.parameters>`.
- What are the parameters expected by the OTB application from the images that
  could be passed in-memory? |br|
  The default are ``"in"`` and ``"out"`` but could be overridden in the
  constructor of the new step factory through the parameters ``param_in`` and
  ``param_out``. See for instance
  :func:`s1tiling.libs.otbwrappers.OrthoRectify.__init__` implementation.
- What is the OTB application? |br|
  Its name is expected to be passed to the constructor of the parent class,
  from the constructor of the new class.

.. note::

    Most of the time, inheriting of :class:`OTBStepFactory
    <s1tiling.libs.steps.OTBStepFactory>` is the best choice. Still, it's
    possible to take over and to manually answer the following questions:

    - What would be the name of the temporary files while they are being produced? |br|
      Return the information from :func:`build_step_output_tmp_filename()
      <s1tiling.libs.steps.StepFactory.build_step_output_tmp_filename>`,
    - Where the product should be produced? |br|
      Return the information from :func:`output_directory()
      <s1tiling.libs.steps._FileProducingStepFactory.output_directory>` -- this is
      typically used from :func:`build_step_output_filename()
      <s1tiling.libs.steps.StepFactory.build_step_output_filename>`.

Technically all other methods from :class:`StepFactory
<s1tiling.libs.steps.StepFactory>` could be overridden. For instance,
:func:`create_step() <s1tiling.libs.steps.StepFactory.create_step>` could
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

3. Make sure version number is up-to-date in :file:`CITATION.cff` and
   :file:`.zenodo.json` files.

4. Handle all the issues associated for the related milestone, and close it.

5. Push ``develop`` branch.

   .. code::

       git checkout develop && git push

6. Wait for its pipeline to succeed. Go back and fix what needs fixeing
   otherwise.

7. Merge ``develop`` branch into ``master``

   .. code::

       git checkout master && git pull && git merge develop

8. Push ``master`` branch.

   .. code::

       git push

9 Create a git tag matching the version number

   .. code::

       git tag -a "${version}"
       # And fill in version information

10. Push the tag

   .. code::

       git push --tags


   .. note::

       From there on, the CI will automatically take care of registering the
       source distribution (only; and not the wheel!) on PyPi as if we had
       manually run

       .. code::

           # Prepare the packets for PyPi
           python3 setup.py sdist

           # Push to PyPi
           python3 -m twine upload --repository pypi dist/S1Tiling-${version}*

11. For major and minor versions, create a branch named after this version. It
   will help to track issues patching of that version independently of work
   done on the next version, tracked in ``develop``.

   .. code::

       git checkout -b release-${version}
       git push --set-upstream origin release-${version}

12. Go to `github mirror <https://github.com/CNES/S1Tiling>`_, once the
   repository has been mirrored, to create a new release. This will
   automatically generate a new DOI on zenodo.

13. Go to the new `zenodo release <https://doi.org/10.5281/zenodo.17237358>`_
    and update project metadata that cannot be set in :file:`CITATION.cff`
    file.

14. Eventually, update :file:`__meta__.py` version to the next expected
    version. Do not use the `rcX` suffix for the moment.

15. Announce the new release to the World.
