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
intersecting S1 images, S1 Tiling builds a set of dependant :std:doc:`Dask tasks
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

Step types are usually instantiated automatically.

- :class:`FirstStep <s1tiling.libs.otbpipeline.FirstStep>` is instantiated
  automatically by the program from existing files (downloaded, or produced by
  a pipeline earlier in the sequence of pipelines)
- :class:`MergeStep <s1tiling.libs.otbpipeline.MergeStep>` is also instantiated
  automatically as an alternative to :class:`FirstStep
  <s1tiling.libs.otbpipeline.FirstStep>` in the case of steps that expect
  several input files. This is for instance the case of :class:`Concatenate
  <s1tiling.libs.otbwrappers.Concatenate>` inputs. A step is recognized to
  await several inputs when the dependency analysis phase found several
  possible inputs that lead to a product.
- :class:`Step <s1tiling.libs.otbpipeline.Step>` is the main class for steps
  that execute an OTB application.
- :class:`AbstractStep <s1tiling.libs.otbpipeline.AbstractStep>` is the root
  class of steps hierarchy. It still get instantiated automatically for steps
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
:class:`StepFactory` subclasses, which in turn will instantiate domain unaware
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
