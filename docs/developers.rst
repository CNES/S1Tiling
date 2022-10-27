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
   :depth: 4


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
one(s) can take over from that point.

Simple pipelines
++++++++++++++++

In simple cases, we can chain the output of an in-memory pipeline of OTB
applications into the next pipeline.

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

Complex pipelines
+++++++++++++++++

In more complex cases, the product of a pipeline will be used as input of
several other pipelines. Also a pipelines can have several inputs coming from
different other pipelines.

To do so, we name each pipeline, so we can use that name as input of other
pipelines.

For instance, LIA producing pipelines are described this way

.. code:: python

    pipelines = PipelineDescriptionSequence(config, dryrun=dryrun)
    dem = pipelines.register_pipeline([AgglomerateDEM],
        'AgglomerateDEM',
        inputs={'insar': 'basename'})
    demproj = pipelines.register_pipeline([ExtractSentinel1Metadata, SARDEMProjection],
        'SARDEMProjection',
        is_name_incremental=True,
        inputs={'insar': 'basename', 'indem': dem})
    xyz = pipelines.register_pipeline([SARCartesianMeanEstimation],
        'SARCartesianMeanEstimation',
        inputs={'insar': 'basename', 'indem': dem, 'indemproj': demproj})
    lia = pipelines.register_pipeline([ComputeNormals, ComputeLIA],
        'Normals|LIA',
        is_name_incremental=True,
        inputs={'xyz': xyz})

    # "inputs" parameter doesn't need to be specified in all the following
    # pipeline declarations but we still use it for clarity!
    ortho  = pipelines.register_pipeline([filter_LIA('LIA'), OrthoRectifyLIA],
        'OrthoLIA',
        inputs={'in': lia},
        is_name_incremental=True)
    concat = pipelines.register_pipeline([ConcatenateLIA],
        'ConcatLIA',
        inputs={'in': ortho})
    select = pipelines.register_pipeline([SelectBestCoverage],
        'SelectLIA',
        product_required=True,
        inputs={'in': concat})
    ortho_sin  = pipelines.register_pipeline([filter_LIA('sin_LIA'), OrthoRectifyLIA],
        'OrthoSinLIA',
        inputs={'in': lia},
        is_name_incremental=True)
    concat_sin = pipelines.register_pipeline([ConcatenateLIA],
        'ConcatSinLIA',
        inputs={'in': ortho_sin})
    select_sin = pipelines.register_pipeline([SelectBestCoverage],
        'SelectSinLIA',
        product_required=True,
        inputs={'in': concat_sin})


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

.. inheritance-diagram:: s1tiling.libs.otbpipeline.OTBStepFactory s1tiling.libs.otbpipeline.ExecutableStepFactory s1tiling.libs.otbpipeline._FileProducingStepFactory s1tiling.libs.otbpipeline.Store
   :parts: 1
   :top-classes: s1tiling.libs.otbpipeline.StepFactory
   :private-bases:

``StepFactory``
~~~~~~~~~~~~~~~

.. autoclass:: s1tiling.libs.otbpipeline.StepFactory
   :members:
   :show-inheritance:
   :undoc-members:

   .. automethod:: _update_filename_meta_pre_hook
   .. automethod:: _update_filename_meta_post_hook

``_FileProducingStepFactory``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: s1tiling.libs.otbpipeline._FileProducingStepFactory
   :members:
   :show-inheritance:
   :undoc-members:

   .. automethod:: __init__

``OTBStepFactory``
~~~~~~~~~~~~~~~~~~

.. autoclass:: s1tiling.libs.otbpipeline.OTBStepFactory
   :members:
   :show-inheritance:
   :undoc-members:

   .. automethod:: __init__

``ExecutableStepFactory``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: s1tiling.libs.otbpipeline.ExecutableStepFactory
   :members:
   :show-inheritance:
   :undoc-members:

``Store``
~~~~~~~~~

.. autoclass:: s1tiling.libs.otbpipeline.Store
   :members:
   :show-inheritance:
   :undoc-members:


Steps
+++++

.. inheritance-diagram:: s1tiling.libs.otbpipeline.Step s1tiling.libs.otbpipeline.FirstStep s1tiling.libs.otbpipeline.ExecutableStep s1tiling.libs.otbpipeline.MergeStep s1tiling.libs.otbpipeline.StoreStep s1tiling.libs.otbpipeline._StepWithOTBApplication
   :parts: 1
   :top-classes: s1tiling.libs.otbpipeline.AbstractStep
   :private-bases:

Step types are usually instantiated automatically.

- :class:`FirstStep <s1tiling.libs.otbpipeline.FirstStep>` is instantiated
  automatically by the program from existing files (downloaded, or produced by
  a pipeline earlier in the sequence of pipelines)
- :class:`MergeStep <s1tiling.libs.otbpipeline.MergeStep>` is also instantiated
  automatically as an alternative to :class:`FirstStep
  <s1tiling.libs.otbpipeline.FirstStep>` in the case of steps that expect
  several input files of the same type. This is for instance the case of
  :class:`Concatenate <s1tiling.libs.otbwrappers.Concatenate>` inputs. A step
  is recognized to await several inputs when the dependency analysis phase
  found several possible inputs that lead to a product.
- :class:`Step <s1tiling.libs.otbpipeline.Step>` is the main class for steps
  that execute an OTB application.
- :class:`ExecutableStep <s1tiling.libs.otbpipeline.ExecutableStep>` is the
  main class for steps that execute an external application.
- :class:`AbstractStep <s1tiling.libs.otbpipeline.AbstractStep>` is the root
  class of steps hierarchy. It still get instantiated automatically for steps
  not related to any kind of application.

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

``_StepWithOTBApplication``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: s1tiling.libs.otbpipeline._StepWithOTBApplication
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

``ExecutableStep``
~~~~~~~~~~~~~~~~~~

.. autoclass:: s1tiling.libs.otbpipeline.ExecutableStep
   :members:
   :show-inheritance:


Existing processings
++++++++++++++++++++

The :ref:`domain processings <processings>` are defined through
:class:`StepFactory` subclasses, which in turn will instantiate domain unaware
subclasses of :class:`AbstractStep` for the actual processing.

Main processings
~~~~~~~~~~~~~~~~

``ExtractSentinel1Metadata``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbwrappers.ExtractSentinel1Metadata
   :members:
   :show-inheritance:

``AnalyseBorders``
^^^^^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbwrappers.AnalyseBorders
   :members:
   :show-inheritance:

``Calibrate``
^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbwrappers.Calibrate
   :members:
   :show-inheritance:

``CutBorders``
^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbwrappers.CutBorders
   :members:
   :show-inheritance:

``OrthoRectify``
^^^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbwrappers.OrthoRectify
   :members:
   :show-inheritance:

``Concatenate``
^^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbwrappers.Concatenate
   :members:
   :show-inheritance:

``BuildBorderMask``
^^^^^^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbwrappers.BuildBorderMask
   :members:
   :show-inheritance:

``SmoothBorderMask``
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbwrappers.SmoothBorderMask
   :members:
   :show-inheritance:

``SpatialDespeckle``
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbwrappers.SpatialDespeckle
   :members:
   :show-inheritance:

Processings for advanced calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These processings permit to produce Local Incidence Angles Maps for
σ\ :sub:`0`\ :sup:`NORMLIM` calibration.

``AgglomerateDEM``
^^^^^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbwrappers.AgglomerateDEM
   :members:
   :show-inheritance:

``SARDEMProjection``
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbwrappers.SARDEMProjection
   :members:
   :show-inheritance:

``SARCartesianMeanEstimation``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbwrappers.SARCartesianMeanEstimation
   :members:
   :show-inheritance:

``ComputeNormals``
^^^^^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbwrappers.ComputeNormals
   :members:
   :show-inheritance:

``ComputeLIA``
^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbwrappers.ComputeLIA
   :members:
   :show-inheritance:

``filter_LIA()``
^^^^^^^^^^^^^^^^

.. autofunction:: s1tiling.libs.otbwrappers.filter_LIA

.. autoclass:: s1tiling.libs.otbwrappers._FilterStepFactory
   :members:
   :show-inheritance:

``OrthoRectifyLIA``
^^^^^^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbwrappers.OrthoRectifyLIA
   :members:
   :show-inheritance:

``ConcatenateLIA``
^^^^^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbwrappers.ConcatenateLIA
   :members:
   :show-inheritance:

``SelectBestCoverage``
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbwrappers.SelectBestCoverage
   :members:
   :show-inheritance:

``ApplyLIACalibration``
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbwrappers.ApplyLIACalibration
   :members:
   :show-inheritance:


Filename generation
+++++++++++++++++++

At each step, product filenames are automatically generated by
:func:`StepFactory.update_filename_meta <s1tiling.libs.otbpipeline.StepFactory.update_filename_meta>` function.
This function is first used to generate the task execution graph. (It's still
used a second time, live, but
this should change eventually)

The exact filename generation is handled by
:func:`StepFactory.build_step_output_filename <s1tiling.libs.otbpipeline.StepFactory.build_step_output_filename>` and
:func:`StepFactory.build_step_output_tmp_filename <s1tiling.libs.otbpipeline.StepFactory.build_step_output_tmp_filename>`
functions to define the final filename and the working filename (used when the
associated product is being computed).

In some very specific cases, where no product is generated, these functions
need to be overridden. Otherwise, a default behaviour is proposed in
:class:`_FileProducingStepFactory <s1tiling.libs.otbpipeline._FileProducingStepFactory>` constructor.
It is done through the parameters:

- ``gen_tmp_dir``: that defines where temporary files are produced.
- ``gen_output_dir``: that defines where final files are produced. When this
  parameter is left unspecified, the final product is considered to be a
  :ref:`intermediary files <temporary-files>` and it will be stored in the
  temporary directory. The distinction is useful for final and required
  products.
- ``gen_output_filename``: that defines the naming policy for both temporary
  and final filenames.

.. important::

    As the filenames are used to define the task execution graph, it's
    important that every possible product (and associated production task) can
    be uniquely identified without any risk of ambiguity. Failure to comply
    will destabilise the data flows.

    If for some reason you need to define a complex data flow where an output
    can be used several times as input in different Steps, or where a Step has
    several inputs of same or different kinds, or where several products are
    concurrent and only one would be selected, please check all
    :class:`StepFactories <s1tiling.libs.otbpipeline.StepFactory>` related to
    :ref:`LIA dataflow <dataflow-lia>`.

Available naming policies
~~~~~~~~~~~~~~~~~~~~~~~~~

.. inheritance-diagram:: s1tiling.libs.otbpipeline.ReplaceOutputFilenameGenerator s1tiling.libs.otbpipeline.TemplateOutputFilenameGenerator s1tiling.libs.otbpipeline.OutputFilenameGeneratorList
   :parts: 1
   :top-classes: s1tiling.libs.otbpipeline.OutputFilenameGenerator
   :private-bases:

Three filename generators are available by default. They apply a transformation
on the ``basename`` meta information.

``ReplaceOutputFilenameGenerator``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbpipeline.ReplaceOutputFilenameGenerator

``TemplateOutputFilenameGenerator``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbpipeline.TemplateOutputFilenameGenerator

Most filename format templates can be fine tuned to end-user ideal filenames.
While the filenames used for intermediary products may be changed, it's not
recommended for data flow stability.
See :ref:`[Processing].fname_fmt.* <Processing.fname_fmt>` for the short list
of filenames meants to be adapted.

``OutputFilenameGeneratorList``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: s1tiling.libs.otbpipeline.OutputFilenameGeneratorList

Hooks
~~~~~

:func:`StepFactory._update_filename_meta_pre_hook <s1tiling.libs.otbpipeline.StepFactory._update_filename_meta_pre_hook>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes it's necessary to analyse the input files, and/or their names before
being able to build the output filename(s). This is meant to be done by
overriding
:func:`StepFactory._update_filename_meta_pre_hook <s1tiling.libs.otbpipeline.StepFactory._update_filename_meta_pre_hook>`
method.  Lightweight analysing is meant to be done here, and its result can
then be stored into ``meta`` dictionary, and returned.

It's typically used alongside
:class:`TemplateOutputFilenameGenerator <s1tiling.libs.otbpipeline.TemplateOutputFilenameGenerator>`.

:func:`StepFactory._update_filename_meta_post_hook <s1tiling.libs.otbpipeline.StepFactory._update_filename_meta_post_hook>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`StepFactory.update_filename_meta <s1tiling.libs.otbpipeline.StepFactory.update_filename_meta>`
provides various values to metadata. This hooks permits to override the values
associated to task names, product existence tests, and so on.
