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
generated from their description and input files, and they are handled
internally; they won't be described here.

Each pipeline corresponds to a series of :ref:`processings <processing classes>`.
The intended and original design is to have a direct match: one processing ==
one OTB application, and to permit chaining OTB applications in memory through
OTB Python bindings.

However, a processing doesn't always turn into the execution of an OTB
application, sometimes we need to do other computations like calling a python
function or executing an external program. Other times, we just need to do some
analysis that will be reused later on in the pipeline.

When we need to have files produced at some point, we end a pipeline, the next
one(s) can take over from that point.

.. autosummary::
   :toctree: api

   s1tiling.libs.otbpipeline.PipelineDescriptionSequence
   s1tiling.libs.otbpipeline.FirstStepFactory

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
several other pipelines. Also, a pipeline can have several inputs coming from
different other pipelines.

To do so, we name each pipeline, so we can use that name as input of other
pipelines.

For instance, LIA producing pipelines are described this way

.. code:: python

    pipelines = PipelineDescriptionSequence(config, dryrun=dryrun)
    dem = pipelines.register_pipeline([AgglomerateDEMOnS1],
        'AgglomerateDEMOnS1',
        inputs={'insar': 'basename'})
    demproj = pipelines.register_pipeline([ExtractSentinel1Metadata, SARDEMProjection],
        'SARDEMProjection',
        is_name_incremental=True,
        inputs={'insar': 'basename', 'indem': dem})
    xyz = pipelines.register_pipeline([SARCartesianMeanEstimation],
        'SARCartesianMeanEstimation',
        inputs={'insar': 'basename', 'indem': dem, 'indemproj': demproj})
    lia = pipelines.register_pipeline([ComputeNormals, ComputeLIAOnS1],
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


.. _dev_pipeline_inputs:

Pipeline inputs
+++++++++++++++

In order to build the `Direct Acyclic Graph (DAG)` of tasks, that will be
executed through the pipelines described, we need to inject inputs.

Pipeline inputs need to be registered explicitly. This is done through
``FirstStepFactories`` passed to
:func:`PipelineDescriptionSequence.register_inputs
<s1tiling.libs.otbpipeline.PipelineDescriptionSequence.register_inputs>`.
Each :class:`FirstStepFactory <s1tiling.libs.otbpipeline.FirstStepFactory>`
takes care of returning a list of :class:`FirstSteps
<s1tiling.libs.steps.FirstStep>`. These ``FirstSteps`` are expected to hold
metadata that will be used to generate the DAG of tasks. They may also obtain
related products on-the-fly. For instance:
:func:`s1_raster_first_inputs_factory` and :func:`eof_first_inputs_factory`
first check which products are already on disk before trying to download the
missing ones.

e.g.:

.. code:: python

   pipelines.register_inputs('basename', s1_raster_first_inputs_factory)
   pipelines.register_inputs('basename', tilename_first_inputs_factory)
   pipelines.register_inputs('basename', eof_first_inputs_factory)

As the ``PipelineDescriptionSequence`` tries to be as independent of the actual
domain as possible, it doesn't know which information is expected by all the
registered ``FirstStepFactories``. By default,
:class:`Configuration <s1tiling.libs.configuration.Configuration>` information
is passed. But some other information needs to be declared in one or several
calls to
:func:`PipelineDescriptionSequence.register_extra_parameters_for_input_factories
<s1tiling.libs.otbpipeline.PipelineDescriptionSequence.register_extra_parameters_for_input_factories>`.

e.g.:

.. code:: python

    pipelines.register_extra_parameters_for_input_factories(
        tile_name=tilename,               # Used by all
    )

    pipelines.register_extra_parameters_for_input_factories(
        dag=dag,                          # Used by eof_first_inputs_factory
        s1_file_manager=s1_file_manager,  # Used by s1_raster_first_inputs_factory
        dryrun=dryrun,                    # Used by all
    )

.. note:: In simplified developer jardon, we use `Factory Method` design
   pattern to inverse dependencies.

Dask: tasks
-----------

Given :ref:`pipeline descriptions <dev_pipeline>`, a requested S2 tile and its
intersecting S1 images, S1 Tiling builds a set of dependant
:external:doc:`Dask tasks <graphs>`. Each task corresponds to an actual
pipeline which will transform a given image into another named image product.

.. _dev_processings:

Processing Classes
------------------

Again the processing classes are split in two families:

- the factories: :class:`StepFactory <s1tiling.libs.steps.StepFactory>`
- the instances: :class:`Step <s1tiling.libs.steps.Step>`

Step Factories
++++++++++++++

Step factories are the main entry point to add new processings. They are meant
to inherit from either one of :class:`OTBStepFactory
<s1tiling.libs.steps.OTBStepFactory>`, :class:`AnyProducerStepFactory
<s1tiling.libs.steps.AnyProducerStepFactory>`, or :class:`ExecutableStepFactory
<s1tiling.libs.steps.ExecutableStepFactory>`.

They describe processings, and they are used to instantiate the actual
:ref:`step <Steps>` that do the processing.

.. inheritance-diagram:: s1tiling.libs.steps.OTBStepFactory s1tiling.libs.steps.ExecutableStepFactory s1tiling.libs.steps.AnyProducerStepFactory s1tiling.libs.steps._FileProducingStepFactory s1tiling.libs.steps.Store
   :parts: 1
   :top-classes: s1tiling.libs.steps.StepFactory
   :private-bases:


.. autosummary::
   :toctree: api

   s1tiling.libs.steps.StepFactory
   s1tiling.libs.steps._FileProducingStepFactory
   s1tiling.libs.steps.OTBStepFactory
   s1tiling.libs.steps.AnyProducerStepFactory
   s1tiling.libs.steps.ExecutableStepFactory
   s1tiling.libs.steps.Store

Steps
+++++

Step types are usually instantiated automatically. They are documented for
convenience, but they are not expected to be extended.

- :class:`FirstStep <s1tiling.libs.steps.FirstStep>` is instantiated
  automatically by the program from existing files (downloaded, or produced by
  a pipeline earlier in the sequence of pipelines)
- :class:`MergeStep <s1tiling.libs.steps.MergeStep>` is also instantiated
  automatically as an alternative to :class:`FirstStep
  <s1tiling.libs.steps.FirstStep>` in the case of steps that expect
  several input files of the same type. This is for instance the case of
  :class:`Concatenate <s1tiling.libs.otbwrappers.Concatenate>` inputs. A step
  is recognized to await several inputs when the dependency analysis phase
  found several possible inputs that lead to a product.
- :class:`Step <s1tiling.libs.steps.Step>` is the main class for steps
  that execute an OTB application.
- :class:`AnyProducerStep <s1tiling.libs.steps.AnyProducerStep>` is the
  main class for steps that execute a Python function.
- :class:`ExecutableStep <s1tiling.libs.steps.ExecutableStep>` is the
  main class for steps that execute an external application.
- :class:`AbstractStep <s1tiling.libs.steps.AbstractStep>` is the root
  class of steps hierarchy. It still gets instantiated automatically for steps
  not related to any kind of application.

.. inheritance-diagram:: s1tiling.libs.steps.Step s1tiling.libs.steps.FirstStep s1tiling.libs.steps.ExecutableStep s1tiling.libs.steps.AnyProducerStep s1tiling.libs.steps.MergeStep s1tiling.libs.steps.StoreStep s1tiling.libs.steps._ProducerStep s1tiling.libs.steps._OTBStep s1tiling.libs.steps.SkippedStep
   :parts: 1
   :top-classes: s1tiling.libs.steps.AbstractStep
   :private-bases:

.. autosummary::
   :toctree: api

   s1tiling.libs.steps.AbstractStep
   s1tiling.libs.steps.FirstStep
   s1tiling.libs.steps.MergeStep
   s1tiling.libs.steps._ProducerStep
   s1tiling.libs.steps._OTBStep
   s1tiling.libs.steps.Step
   s1tiling.libs.steps.SkippedStep
   s1tiling.libs.steps.AnyProducerStep
   s1tiling.libs.steps.ExecutableStep
   s1tiling.libs.steps.StoreStep

Existing processings
++++++++++++++++++++

The :ref:`domain processings <processings>` are defined through
:class:`StepFactory <s1tiling.libs.steps.StepFactory>` subclasses, which in
turn will instantiate domain unaware subclasses of :class:`AbstractStep
<s1tiling.libs.steps.AbstractStep>` for the actual processing.

Common internal processings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

   s1tiling.libs.otbwrappers._OrthoRectifierFactory
   s1tiling.libs.otbwrappers._ConcatenatorFactory

Main processings
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

   s1tiling.libs.otbwrappers.ExtractSentinel1Metadata
   s1tiling.libs.otbwrappers.AnalyseBorders
   s1tiling.libs.otbwrappers.Calibrate
   s1tiling.libs.otbwrappers.CutBorders
   s1tiling.libs.otbwrappers.OrthoRectify
   s1tiling.libs.otbwrappers.Concatenate
   s1tiling.libs.otbwrappers.BuildBorderMask
   s1tiling.libs.otbwrappers.SmoothBorderMask
   s1tiling.libs.otbwrappers.SpatialDespeckle

Processings for advanced calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`β^0_{E}`, :math:`γ^0_{E}` correction LUTs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: api

   s1tiling.libs.otbwrappers.ComputeGroundAndSatPositionsOnEllipsoid
   s1tiling.libs.otbwrappers.ComputeEllipsoidNormalsOnS2
   s1tiling.libs.otbwrappers.ComputeIAOnS2

:math:`σ^0_{T}` NORMLIM calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These processings permit producing Local Incidence Angles Maps for
:math:`σ^0_{T}` NORMLIM calibration.

.. autosummary::
   :toctree: api

   s1tiling.libs.otbwrappers.AgglomerateDEMOnS2
   s1tiling.libs.otbwrappers.ProjectDEMToS2Tile
   s1tiling.libs.otbwrappers.ProjectGeoidToS2Tile
   s1tiling.libs.otbwrappers.SumAllHeights
   s1tiling.libs.otbwrappers.ComputeGroundAndSatPositionsOnDEMFromEOF
   s1tiling.libs.otbwrappers.ComputeNormalsOnS2
   s1tiling.libs.otbwrappers.ComputeLIAOnS2
   s1tiling.libs.otbwrappers.filter_LIA
   s1tiling.libs.otbwrappers.ApplyLIACalibration

:math:`γ^0_{T}` RTC calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These processings permit producing γ area Maps for :math:`γ^0_{T}` RTC
calibration.

.. autosummary::
   :toctree: api

   s1tiling.libs.otbwrappers.AgglomerateDEMOnS1
   s1tiling.libs.otbwrappers.ProjectGeoidToDEM
   s1tiling.libs.otbwrappers.NaNifyNoData
   s1tiling.libs.otbwrappers.ResampleDEM
   s1tiling.libs.otbwrappers.SARDEMProjectionImageEstimation
   s1tiling.libs.otbwrappers.SARGammaAreaImageEstimation
   s1tiling.libs.otbwrappers.OrthoRectifyGAMMA_AREA
   s1tiling.libs.otbwrappers.ConcatenateGAMMA_AREA
   s1tiling.libs.otbwrappers.SelectGammaNaughtAreaBestCoverage
   s1tiling.libs.otbwrappers.ApplyGammaNaughtRTCCalibration


Deprecated processings for advanced calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following processings used to be used in v1.0 of S1Tiling, along some of
the previous ones. Starting from v1.1, they are deprecated.

.. autosummary::
   :toctree: api

   s1tiling.libs.otbwrappers.SARDEMProjection
   s1tiling.libs.otbwrappers.SARCartesianMeanEstimation
   s1tiling.libs.otbwrappers.OrthoRectifyLIA
   s1tiling.libs.otbwrappers.ComputeNormalsOnS1
   s1tiling.libs.otbwrappers.ComputeLIAOnS1
   s1tiling.libs.otbwrappers.ConcatenateLIA
   s1tiling.libs.otbwrappers.SelectBestCoverage
   s1tiling.libs.otbwrappers.ComputeGroundAndSatPositionsOnDEM


Filename generation
+++++++++++++++++++

At each step, product filenames are automatically generated by
:func:`StepFactory.update_filename_meta
<s1tiling.libs.steps.StepFactory.update_filename_meta>` function.
This function is first used to generate the task execution graph. (It's still
used a second time, live, but this should change eventually)

The exact filename generation is handled by
:func:`StepFactory.build_step_output_filename <s1tiling.libs.steps.StepFactory.build_step_output_filename>` and
:func:`StepFactory.build_step_output_tmp_filename <s1tiling.libs.steps.StepFactory.build_step_output_tmp_filename>`
functions to define the final filename and the working filename (used when the
associated product is being computed).

In some very specific cases, where no product is generated, these functions
need to be overridden. Otherwise, a default behaviour is proposed in
:class:`_FileProducingStepFactory <s1tiling.libs.steps._FileProducingStepFactory>` constructor.
It is done through the parameters:

- ``gen_tmp_dir``: that defines where temporary files are produced.
- ``gen_output_dir``: that defines where final files are produced. When this
  parameter is left unspecified, the final product is considered to be a
  :ref:`intermediary files <temporary-files>`, and it will be stored in the
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
    :class:`StepFactories <s1tiling.libs.steps.StepFactory>` related to
    :ref:`LIA dataflow <dataflow-lia>`.

Available naming policies
~~~~~~~~~~~~~~~~~~~~~~~~~

.. inheritance-diagram:: s1tiling.libs.file_naming.ReplaceOutputFilenameGenerator s1tiling.libs.file_naming.TemplateOutputFilenameGenerator s1tiling.libs.file_naming.OutputFilenameGeneratorList
   :parts: 1
   :top-classes: s1tiling.libs.file_naming.OutputFilenameGenerator
   :private-bases:

Three filename generators are available by default. They apply a transformation
on the ``basename`` meta information.

.. autosummary::
   :toctree: api

   s1tiling.libs.file_naming.ReplaceOutputFilenameGenerator
   s1tiling.libs.file_naming.TemplateOutputFilenameGenerator
   s1tiling.libs.file_naming.OutputFilenameGeneratorList


Hooks
~~~~~

:func:`StepFactory._update_filename_meta_pre_hook <s1tiling.libs.steps.StepFactory._update_filename_meta_pre_hook>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes it's necessary to analyse the input files, and/or their names before
being able to build the output filename(s). This is meant to be done by
overriding
:func:`StepFactory._update_filename_meta_pre_hook <s1tiling.libs.steps.StepFactory._update_filename_meta_pre_hook>`
method.  Lightweight analysing is meant to be done here, and its result can
then be stored into ``meta`` dictionary, and returned.

It's typically used alongside
:class:`TemplateOutputFilenameGenerator <s1tiling.libs.steps.TemplateOutputFilenameGenerator>`.

:func:`StepFactory._update_filename_meta_post_hook <s1tiling.libs.steps.StepFactory._update_filename_meta_post_hook>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`StepFactory.update_filename_meta <s1tiling.libs.steps.StepFactory.update_filename_meta>`
provides various values to metadata. This hook permits to override the values
associated to task names, product existence tests, and so on.
