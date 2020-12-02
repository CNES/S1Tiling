.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. _developers:

.. index:: Developer documentation

======================================================================
Design notes
======================================================================

.. contents:: Contents:
   :local:
   :depth: 3


.. _dev_pipeline:

.. index:: Pipelines

Pipelines
---------
Internally S1 Tiling defines a series of pipelines. Actually, it distinguishes
**pipeline descriptions** from actual pipelines. The actual pipelines are
generated from their description and input files, and handled internally; they
won't be described.

Each pipeline corresponds to a series of :ref:`processings <processing classes>`.
The intended and original design is to have a direct match: one processing ==
one OTB application, and to permit to chain OTB applications in memory through
OTB Python bindings.

Actually, a processing doesn't always turn into the execution of an OTB
application, sometimes we need to do other computations.

When we need to have files produced at some point, we end a pipeline, the next
one can take over from that point.

At this moment, the following sequence of pipelines is defined:

.. code:: python

    pipelines = PipelineDescriptionSequence(config)
    pipelines.register_pipeline([AnalyseBorders, Calibrate, CutBorders], 'PrepareForOrtho', product_required=False)
    pipelines.register_pipeline([OrthoRectify],                          'OrthoRectify',    product_required=False)
    pipelines.register_pipeline([Concatenate],                                              product_required=True)
    if config.mask_cond:
        pipelines.register_pipeline([BuildBorderMask, SmoothBorderMask], 'GenerateMask',    product_required=True)


For instance, to minimize disk usage, we could chain in-memory
orthorectification directly after the border cutting by removing the second
pipeline, and by registering the following step into the first pipeline
instead:

.. code:: python

    pipelines.register_pipeline([AnalyseBorders, Calibrate, CutBorders, OrthoRectify],
                                'OrthoRectify', product_required=False)

Dask: tasks
-----------

Given :ref:`pipeline descriptions <dev_pipeline>`, a requested S2 tile and its
intersecing S1 images, S1 Tiling builds a set of dependant :std:doc:`Dask tasks
<graphs>`. Each task corresponds to an actual pipeline which will transform a
given image into another named image product.

.. _dev_processings:

Processing Classes
------------------

Again the processing classes are split in two families:

- the factories: :class:`StepFactory <s1tiling.libs.otbpipeline.StepFactory>`
- the instances: :class:`Step <s1tiling.libs.otbpipeline.Step>`

Step Factories
++++++++++++++

.. autoclass:: s1tiling.libs.otbpipeline.StepFactory
   :members:
   :show-inheritance:
   :undoc-members:


Steps
+++++

Step types are usually instanciated automatically.

- :class:`FirstStep <s1tiling.libs.otbpipeline.FirstStep>` is instanciated
  automatically by the program from existing files (downloaded, or produced by
  a pipeline earlier in the sequence of pipelines)
- :class:`MergeStep <s1tiling.libs.otbpipeline.MergeStep>` is also instanciated
  automatically as an alternative to :class:`FirstStep
  <s1tiling.libs.otbpipeline.FirstStep>` in the case of steps that expect
  several input files. This is for instance the case of :class:`Concatenate
  <s1tiling.libs.otbwrappers.Concatenate>` inputs. A step is recognized to
  await several inputs when the dependency analysis phase found several
  possible inputs that lead to a product.
- :class:`Step <s1tiling.libs.otbpipeline.Step>` is the main class for steps
  that execute an OTB application.
- :class:`AbstractStep <s1tiling.libs.otbpipeline.AbstractStep>` is the root
  class of steps hierarchy. It still get instanciated automatically for steps
  not related to an OTB application.

``AbstractStep``
~~~~~~~~~~~~~~~~

.. autoclass:: s1tiling.libs.otbpipeline.AbstractStep
   :members:
   :show-inheritance:

``FirstStep``
~~~~~~~~~~~~~

.. autoclass:: s1tiling.libs.otbpipeline.FirstStep
   :members:
   :show-inheritance:

``Step``
~~~~~~~~

.. autoclass:: s1tiling.libs.otbpipeline.Step
   :members:
   :show-inheritance:

``MergeStep``
~~~~~~~~~~~~~

.. autoclass:: s1tiling.libs.otbpipeline.MergeStep
   :members:
   :show-inheritance:


Existing processings
++++++++++++++++++++

The :ref:`domain processings <processings>` are defined through
:class:`StepFactory` subclasses, which in turn will instanciate domain unaware
subclasses of :class:`AbstractStep` for the actual processing.

``AnalyseBorders``
~~~~~~~~~~~~~~~~~~

.. autoclass:: s1tiling.libs.otbwrappers.AnalyseBorders
   :members:
   :show-inheritance:

``Calibrate``
~~~~~~~~~~~~~

.. autoclass:: s1tiling.libs.otbwrappers.Calibrate
   :members:
   :show-inheritance:

``CutBorders``
~~~~~~~~~~~~~~

.. autoclass:: s1tiling.libs.otbwrappers.CutBorders
   :members:
   :show-inheritance:

``OrthoRectify``
~~~~~~~~~~~~~~~~

.. autoclass:: s1tiling.libs.otbwrappers.OrthoRectify
   :members:
   :show-inheritance:

``Concatenate``
~~~~~~~~~~~~~~~

.. autoclass:: s1tiling.libs.otbwrappers.Concatenate
   :members:
   :show-inheritance:

``BuildBorderMask``
~~~~~~~~~~~~~~~~~~~

.. autoclass:: s1tiling.libs.otbwrappers.BuildBorderMask
   :members:
   :show-inheritance:

``SmoothBorderMask``
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: s1tiling.libs.otbwrappers.SmoothBorderMask
   :members:
   :show-inheritance:


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
  typicallly used from :func:`build_step_output_filename()
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
<s1tiling.libs.otbpipeline.StepFactory>` could be overriden. For instance,
:func:`create_step() <s1tiling.libs.otbpipeline.StepFactory.create_step>` could
be overridden to change the type of :ref:`Steps` instanciated.
