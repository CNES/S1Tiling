#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2021 (c) CESBIO. All rights reserved.
#
#   This file is part of S1Tiling project
#       https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# =========================================================================
#
# Authors: Thierry KOLECK (CNES)
#          Luc HERMITTE (CS Group)
# =========================================================================

"""
This module provides pipeline for chaining OTB applications, and a pool to execute them.
"""

import os
import shutil
from pathlib import Path
import re
import copy
from abc import ABC, abstractmethod
import logging
import logging.handlers
import multiprocessing
import subprocess

# memory leaks
from distributed import get_worker
import objgraph
from pympler import tracker # , muppy

import otbApplication as otb
from . import Utils

logger = logging.getLogger('s1tiling')

re_tiff = re.compile(r'\.tiff?$')


def otb_version():
    """
    Returns the current version on OTB (through a call to ResetMargin -version)
    The result is cached
    """
    if not hasattr(otb_version, "_version"):
        try:
            r = subprocess.run(['otbcli_ResetMargin', '-version'], stdout=subprocess.PIPE , stderr=subprocess.STDOUT )
            version = r.stdout.decode('utf-8').strip('\n')
            version = re.search(r'\d+(\.\d+)+$', version)[0]
            logger.info("OTB version detected on the system is %s", version)
            otb_version._version = version
        except Exception as ex:  # pylint: disable=broad-except
            logger.exception(ex)
            raise RuntimeError("Cannot determine current OTB version")
    return otb_version._version


def as_app_shell_param(param):
    """
    Internal function used to stringigy value to appear like a a parameter for a program
    launched through shell.

    foo     -> 'foo'
    42      -> 42
    [a, 42] -> 'a' 42
    """
    if   isinstance(param, list):
        return ' '.join(as_app_shell_param(e) for e in param)
    elif isinstance(param, int):
        return param
    else:
        return "'%s'" % (param,)


def in_filename(meta):
    """
    Helper accessor to access the input filename of a `Step`.
    """
    assert 'in_filename' in meta
    return meta['in_filename']


def out_filename(meta):
    """
    Helper accessor to access the ouput filename of a `Step`.
    """
    return meta.get('out_filename')


def get_task_name(meta):
    """
    Helper accessor to the task name related to a `Step`.

    By default, the task name is stored in `out_filename` key.
    In the case of reducing :class:`MergeStep`, a dedicated name shall be
    provided. See :class:`Concatenate`

    Important, task names shall be unique and attributed to a single step.
    """
    if 'task_name' in meta:
        return meta['task_name']
    else:
        return out_filename(meta)


def out_extended_filename_complement(meta):
    """
    Helper accessor to the extended filename to use to produce the image.
    """
    return meta.get('out_extended_filename_complement', '')


def product_exists(meta):
    """
    Helper accessor that teels whether the product described by the metadata
    already exists.
    """
    if 'does_product_exist' in meta:
        return meta['does_product_exist']()
    else:
        return os.path.isfile(out_filename(meta))


def update_out_filename(updated_meta, with_meta):
    """
    Helper function to update the `out_filename` from metadata.
    Meant to be used metadata associated to products made of several inputs
    like Concatenate.
    """
    if 'update_out_filename' in updated_meta:
        updated_meta['update_out_filename'](updated_meta, with_meta)


def files_exist(files):
    """
    Checks whether a single file, or all files from a list, exist.
    """
    if isinstance(files, str):
        return os.path.isfile(files)
    else:
        for file in files:
            if not os.path.isfile(file):
                return False
        return True


def execute(params, dryrun):
    """
    Helper function to execute any external command.

    And log its execution, measure the time it takes.
    """
    msg = ' '.join([str(p) for p in params])
    logging.info('$> '+msg)
    if not dryrun:
        with ExecutionTimer(msg, True) as t:
            subprocess.run(args=params, check=True)


class AbstractStep:
    """
    Internal root class for all actual `Step` s.

    There are four kinds of steps:

    - :class:`FirstStep` that contains information about input files
    - :class:`Step` that registers an otbapplication binding
    - :class:`StoreStep` that momentarilly disconnect on-memory pipeline to force storing of
      the resulting file.
    - :class:`ExecutableStep` executes external applications

    The step will contain information like the current input file, the current output
    file...
    """
    def __init__(self, *unused_argv, **kwargs):
        """
        constructor
        """
        meta = kwargs
        if 'basename' not in meta:
            logger.critical('no "basename" in meta == %s', meta)
        assert 'basename' in meta
        # Clear basename from any noise
        self._meta = meta

    @property
    def is_first_step(self):
        """
        Tells whether this step is the first of a pipeline.
        """
        return True

    @property
    def meta(self):
        """
        Step meta data property.
        """
        return self._meta

    @property
    def basename(self):
        """
        Basename property will be used to generate all future output filenames.
        """
        return self._meta['basename']

    @property
    def out_filename(self):
        """
        Property that returns the name of the file produced by the current step.
        """
        assert 'out_filename' in self._meta
        return self._meta['out_filename']

    @property
    def shall_store(self):
        """
        No step required its result to be stored on disk and to break in_memory
        connection by default.
        However, the artificial Step produced by :class:`Store` factory will force the
        result of the `previous` app to be stored on disk.
        """
        return False

    def release_app(self):
        """
        Makes sure that steps with applications are releasing the application
        """
        pass


class ExecutableStep(AbstractStep):
    """
    Generic step for calling any external application.
    """
    def __init__(self, exename, *argv, **kwargs):
        """
        constructor
        """
        super().__init__(None, *argv, **kwargs)
        self._exename = exename

    def execute_and_write_output(self):  # pylint: disable=no-self-use
        dryrun = self.meta.get('dryrun', False)
        logger.debug("ExecutableStep: %s (%s)", self, self.meta)
        execute([self.exename]+ self.parameters(meta), dryrun)
        if 'post' in self.meta and not dryrun:
            for hook in self.meta['post']:
                hook(self.meta)
        self.meta['pipe'] = [self.out_filename]


class _StepWithOTBApplication(AbstractStep):
    """
    Internal intermediary type for `Step` that have an application object.
    Not meant to be used directly.

    Parent type for:
    - :class:`Step`  that will own the application
    - and :class:`StoreStep` that will just reference the application from the previous step
    """
    def __init__(self, app, *argv, **kwargs):
        """
        constructor
        """
        # logger.debug("Create Step(%s, %s)", app, meta)
        super().__init__(*argv, **kwargs)
        self._app = app
        self._out = None  # shall be overriden in child classes.

    def __del__(self):
        """
        Makes sure the otb app is released
        """
        if self._app:
            self.release_app()

    def release_app(self):
        # Only `Step` will delete, here we just reset the reference
        self._app = None

    @property
    def app(self):
        """
        OTB Application property.
        """
        return self._app

    @property
    def is_first_step(self):
        return self._app is None

    @property
    def param_out(self):
        """
        Name of the "out" parameter used by the OTB Application.
        Default is likely to be "out", whie some applications use "io.out".
        """
        return self._out


class Step(_StepWithOTBApplication):
    """
    Interal specialized `Step` that holds a binding to an OTB Application.

    The application binding is expected to be built by a dedicated :class:`StepFactory` and
    passed to the constructor.
    """
    def __init__(self, app, *argv, **kwargs):
        """
        constructor
        """
        # logger.debug("Create Step(%s, %s)", app, meta)
        super().__init__(app, *argv, **kwargs)
        self._out = kwargs.get('param_out', 'out')

    def release_app(self):
        del self._app
        super().release_app()  # resets self._app to None

    def execute_and_write_output(self):  # pylint: disable=no-self-use
        """
        Method to call on the last step of a pipeline.
        """
        raise TypeError("A normal Step is not meant to be the last step of a pipeline!!!")


class StepFactory(ABC):
    """
    Abstract factory for `Step`.

    Meant to be inherited for each possible OTB application used in a pipeline.
    """
    def __init__(self, name, *unused_argv, **kwargs):
        assert name
        self._name    = name
        # logger.debug("new StepFactory(%s)", name)

    @property
    def name(self):
        """
        Step Name property.
        """
        assert isinstance(self._name, str)
        return self._name

    @abstractmethod
    def parameters(self, meta):
        """
        Method to access the parameters to inject into the OTB application associated to the
        current step.
        This method will be specialized in child classes.
        """
        pass

    @abstractmethod
    def build_step_output_filename(self, meta):
        """
        Filename of the step output.

        See also :func:`build_step_output_tmp_filename()` regarding the actual processing.
        """
        pass

    @abstractmethod
    def build_step_output_tmp_filename(self, meta):
        """
        Returns temporary filename to use in output of the current OTB Application.

        When an OTB application is harshly interrupted (crash or user
        interruption), it leaves behind an incomplete (and thus invalid) file.
        In order to ignore those files when a pipeline is restarted, an
        temporary filename is used by the OTB application.
        Once the application exits with success, the file will be renamed into
        :func:`build_step_output_filename()`, and possibly moved into
        :func:`output_directory()` if this is a final product.
        """
        pass

    @abstractmethod
    def output_directory(self, meta):
        """
        Output directory for the step product.
        """
        pass

    def set_output_pixel_type(self, app, meta):
        """
        Permits to have steps force the output pixel data.
        Does nothing by default.
        Override this method to change the output pixel type.
        """
        pass

    def update_filename_meta(self, meta):  # to be overridden
        """
        Duplicates, completes, and return, the `meta` dictionary with specific
        information for the current factory regarding tasks analysis.

        This method is used:

        - while analysing the dependencies to build the task graph -- in this
          use case the relevant information are the file names and paths.
        - and indirectly before instanciating a new :class:`Step`

        Other metadata not filled here:

        - :func:`get_task_name` which is deduced from `out_filename`  by default
        - :func:`out_extended_filename_complement`
        """
        meta = meta.copy()
        meta['in_filename']        = out_filename(meta)
        meta['out_filename']       = self.build_step_output_filename(meta)
        meta['out_tmp_filename']   = self.build_step_output_tmp_filename(meta)
        meta['pipe']               = meta.get('pipe', []) + [self.__class__.__name__]
        meta['does_product_exist'] = lambda : os.path.isfile(out_filename(meta))
        meta.pop('task_name', None)
        meta.pop('task_basename', None)
        meta.pop('update_out_filename', None)
        return meta

    def complete_meta(self, meta):  # to be overridden
        """
        Duplicates, completes, and return, the `meta` dictionary with specific
        information for the current factory regarding :class:`Step` instanciation.
        """
        meta.pop('out_extended_filename_complement', None)
        return self.update_filename_meta(meta)

    def create_step(self, input: AbstractStep, in_memory: bool, unused_previous_steps):
        """
        Instanciates the step related to the current :class:`StepFactory`,
        that consumes results from the previous `input` step.

        1. This methods starts by updating metadata information through
        :func:`complete_meta()` on the `input` metadata.

        2. in case the new step isn't related to an OTB application,
        nothing specific is done, we'll just return an :class:`AbstractStep`

        Note: While `previous_steps` is ignored in this specialization, it's
        used in :func:`Store.create_step()` where it's eventually used to
        release all OTB Application objects.
        """
        # TODO: distinguish step description & step
        assert issubclass(type(input), AbstractStep)
        meta = self.complete_meta(input.meta)

        # Return previous app?
        return AbstractStep(**meta)


class Outcome:
    """
    Kind of monad à la C++ ``std::expected<>``, ``boost::Outcome``.

    It stores tasks results which could be:
    - either the filename of task product,
    - or the error message that leads to the task failure.
    """
    def __init__(self, value_or_error):
        """
        constructor
        """
        self.__value_or_error    = value_or_error
        self.__is_error          = issubclass(type(value_or_error), BaseException)
        self.__related_filenames = []

    def __bool__(self):
        return not self.__is_error

    def add_related_filename(self, filename):
        """
        Register a filename related to the result.
        """
        self.__related_filenames.append(filename)
        return self

    def __repr__(self):
        if self.__is_error:
            msg = 'Failed to produce %s' % (self.__related_filenames[-1])
            if len(self.__related_filenames) > 1:
                errored_files = ', '.join(self.__related_filenames[:-1])
                msg += ' because %s could not be produced: ' % (errored_files, )
            else:
                msg += ': '
            msg += '%s' % (self.__value_or_error, )
            return msg
        else:
            return 'Success: %s' % (self.__value_or_error)


class Pipeline:
    """
    Pipeline of OTB applications.

    It's instanciated as a list of :class:`AbstractStep` s.
    :func:`Step.execute_and_write_output()` will be executed on the last step
    of the pipeline.

    Internal class only meant to be used by  :class:`Pool`.
    """
    # Should we inherit from contextlib.ExitStack?
    def __init__(self, do_measure, in_memory, do_watch_ram, name=None, output=None):
        self.__pipeline     = []
        self.__do_measure   = do_measure
        self.__in_memory    = in_memory
        self.__do_watch_ram = do_watch_ram
        self.__name         = name
        self.__output       = output
        self.__inputs       = []

    def __repr__(self):
        return self.name

    def set_inputs(self, inputs: list):
        """
        Set the input(s) of the instanciated pipeline.
        The `inputs` is parameter expected to be a list of {'key': [metas...]} that'll
        get tranformed into a dictionary of {'key': :class:`AbstractStep`}.

        Some :class:`AbstractStep` will actually be :class:`MergeStep` instances.
        """
        logger.debug("Pipeline(%s).set_inputs(%s)", self.__name, inputs)
        # all_keys = set().union(*(input.keys() for input in inputs))
        all_keys = inputs.keys()
        for key in all_keys:
            # inputs_associated_to_key = [input[key] for input in inputs if key in input]
            inputs_associated_to_key = inputs[key]
            if len(inputs_associated_to_key) == 1:
                self.__inputs.append({key: FirstStep(**inputs_associated_to_key[0])})
            else:
                self.__inputs.append({key: MergeStep(inputs_associated_to_key)})

    @property
    def _input_filenames(self):
        """
        Property _input_filenames
        """
        return [input[k].out_filename for input in self.__inputs for k in input]

    @property
    def name(self):
        """
        Name of the pipeline.
        It's either user registered or automatically generated from the
        registered :class:`StepFactory` s.
        """
        appname = (self.__name or '|'.join(crt.appname for crt in self.__pipeline))
        return '%s -> %s from %s' % (appname, self.__output, self._input_filenames)

    @property
    def output(self):
        """
        Expected pipeline final output file.
        """
        return self.__output

    @property
    def shall_watch_ram(self):
        """
        Tells whether objects in RAM shall be watched for memory leaks.
        """
        return self.__do_watch_ram

    def push(self, otbstep: StepFactory):
        """
        Registers a StepFactory into the pipeline.
        """
        assert isinstance(otbstep, StepFactory)
        self.__pipeline.append(otbstep)

    def do_execute(self):
        """
        Execute the pipeline.

        1. Makes sure the inputs exist -- unless in dry-run mode
        2. Incrementaly create the steps of the pipeline.
        3. Return the resulting output filename, or the caught errors.
        """
        assert self.__input
        TODO()
        if not files_exist(self.__input.out_filename) and not self.__input.meta.get('dryrun', False):
            msg = "Cannot execute %s as input %s doesn't exist" % (self, self.__input.out_filename)
            logger.warning(msg)
            return Outcome(RuntimeError(msg))
        # print("LOG:", os.environ['OTB_LOGGER_LEVEL'])
        assert self.__pipeline  # shall not be empty!
        steps = [self.__input]
        for crt in self.__pipeline:
            step = crt.create_step(steps[-1], self.__in_memory, steps)
            steps.append(step)

        res = steps[-1].out_filename
        assert res == self.output
        steps = None
        return Outcome(res)


# TODO: try to make it static...
def execute4dask(pipeline, *args, **unused_kwargs):
    """
    Internal worker function used by Dask to execute a pipeline.

    Returns the product filename(s) or the caught error in case of failure.
    """
    logger.debug('Parameters for %s: %s', pipeline, args)
    watch_ram = pipeline.shall_watch_ram
    if watch_ram:
        objgraph.show_growth(limit=5)
    try:
        assert len(args) == 1
        for arg in args[0]:
            # logger.info('ARG: %s (%s)', arg, type(arg))
            if isinstance(arg, Outcome) and not arg:
                logger.warning('Abort execution of %s. Error: %s', pipeline, arg)
                return copy.deepcopy(arg).add_related_filename(pipeline.output)
        # Any exceptions leaking to Dask Scheduler would end the execution of the scheduler.
        # That's why errors need to be caught and transformed here.
        logger.info('Execute %s', pipeline)
        res = pipeline.do_execute().add_related_filename(pipeline.output)
        pipeline = None

        if watch_ram:
            objgraph.show_growth()

            # all_objects = muppy.get_objects()
            # sum1 = summary.summarize(all_objects)
            # summary.print_(sum1)
            w = get_worker()
            if not hasattr(w, 'tracker'):
                w.tr = tracker.SummaryTracker()
            w.tr.print_diff()
        return res
    except Exception as ex:  # pylint: disable=broad-except  # Use in nominal code
    # except RuntimeError as ex:  # pylint: disable=broad-except  # Use when debugging...
        logger.exception('Execution of %s failed', pipeline)
        logger.debug('Parameters for %s were: %s', pipeline, args)
        return Outcome(ex).add_related_filename(pipeline.output)


class PipelineDescription:
    """
    Pipeline description:
    - stores the various factory steps that describe a pipeline,
    - can tell the expected product name given an input.
    - tells whether its product is required
    """
    def __init__(self, factory_steps, name=None, product_required=False, is_name_incremental=False, inputs=None):
        """
        constructor
        """
        assert factory_steps  # shall not be None or empty
        self.__factory_steps       = factory_steps
        self.__is_name_incremental = is_name_incremental
        self.__is_product_required = product_required
        if name:
            self.__name = name
        else:
            self.__name = '|'.join([step.name for step in self.__factory_steps])
        self.__inputs              = inputs
        # logger.debug("New pipeline: %s; required: %s, incremental: %s", '|'.join([step.name for step in self.__factory_steps]), self.__is_product_required, self.__is_name_incremental)

    def expected(self, input_meta):
        """
        Returns the expected name of the product of this pipeline
        """
        assert self.__factory_steps  # shall not be None or empty
        if self.__is_name_incremental:
            res = input_meta
            for step in self.__factory_steps:
                res = step.update_filename_meta(res)
        else:
            res = self.__factory_steps[-1].update_filename_meta(input_meta)
        logger.debug("expected: %s(%s) -> %s", self.__name, input_meta['out_filename'], out_filename(res))
        return res

    @property
    def inputs(self):
        """
        Property inputs
        """
        return self.__inputs

    @property
    def sources(self):
        """
        Property sources
        """
        # logger.debug("SOURCES(%s) = %s", self.name, self.__inputs)
        res = [(val if isinstance(val, str) else val.name) for (key,val) in self.__inputs.items()]
        return res

    @property
    def name(self):
        """
        Descriptive name of the pipeline specification.
        """
        assert isinstance(self.__name, str)
        return self.__name

    @property
    def product_is_required(self):
        """
        Tells whether the product if this pipeline is required.
        """
        return self.__is_product_required

    def instanciate(self, file, do_measure, in_memory, do_watch_ram):
        """
        Instanciates the pipeline specified.

        Note: It systematically registers a :class:`Store` step at the end
        if any :class:`StepFactory` is actually an :class:`OTBStepFactory`
        """
        pipeline = Pipeline(do_measure, in_memory, do_watch_ram, self.name, file)
        need_OTB_store = False
        for factory_step in self.__factory_steps + []:
            pipeline.push(factory_step)
            need_OTB_store = need_OTB_store or isinstance(factory_step, OTBStepFactory)  # TODO: use a dedicated function
        if need_OTB_store:
            pipeline.push(Store('noappname'))
        return pipeline


def to_dask_key(pathname):
    """
    Generate a simplified graph key name from a full pathname.
    - Strip directory name
    - Replace '-' with '_' as Dask has a special interpretation for '-' in key names.
    """
    # return Path(pathname).stem.replace('-', '_')
    return pathname.replace('-', '_')


def register_task(tasks, key, value):
    """
    Register a task named `key` in the right format.
    """
    tasks[key] = value


def generate_first_steps_from_manifests(raster_list, tile_name, dryrun):
    """
    Flatten all rasters from the manifest as a list of :class:`FirstStep`
    """
    inputs = []
    # Log commented and kept for filling in unit tests
    # logger.debug('Generate first steps from: %s', raster_list)
    for raster, tile_origin in raster_list:
        manifest = raster.get_manifest()
        for image in raster.get_images_list():
            start = FirstStep(
                    tile_name=tile_name,
                    tile_origin=tile_origin,
                    manifest=manifest,
                    basename=image,
                    dryrun=dryrun)
            inputs.append(start.meta)
    return inputs


class InputInfo:
    def __init__(self, input_meta):
        """
        constructor
        """
        self.__basename = input_meta['basename']
        self.__tasks    = {task_name(input_meta): input_meta}

    @property
    def basename(self):
        """
        Property basename
        """
        return self.__basename

    @property
    def tasks(self):
        """
        Property tasks
        """
        return self.__tasks


class TaskInputInfo:
    def __init__(self, pipeline):
        """
        constructor
        """
        self.__pipeline    = pipeline
        self._inputs       = {}   # map<source, meta / meta list>
        self._dependencies = []  # task names

    def add_input(self, origin, input_meta):
        if origin not in self._inputs:
            self._inputs[origin] = [input_meta]
        else:
            self._inputs[origin].append(input_meta)

    @property
    def pipeline(self):
        """
        Property pipeline
        """
        return self.__pipeline

    @property
    def inputs(self):
        """
        Property inputs
        """
        return self._inputs

    @property
    def dependencies(self):
        """
        Property dependencies
        """
        return self.__dependencies

    @property
    def input_task_names(self):
        """
        Property task names
        """
        logger.debug('input_task_names(%s) --> %s', self.pipeline.name, self.inputs)
        # TODO: use input_metas?
        tns = [get_task_name(meta) for inputs in self.inputs.values() for meta in inputs]
        return tns

    @property
    def input_metas(self):
        """
        Property input_metas
        """
        metas = [meta for inputs in self.inputs.values() for meta in inputs]
        return metas

class PipelineDescriptionSequence:
    """
    List of :class:`PipelineDescription` objects
    """
    def __init__(self, cfg):
        """
        constructor
        """
        assert cfg
        self.__cfg       = cfg
        self.__pipelines = []

    def register_pipeline(self, factory_steps, *args, **kwargs):
        """
        Register a pipeline description from:

        Params:
            :factory_steps:       List of non-instanciated :class:`StepFactory` classes
            :name:                Optional name for the pipeline
            :product_required:    Tells whether the pipeline product is expected as a
                                  final product
            :is_name_incremental: Tells whether `expected` filename needs evaluations of
                                  each intermediary steps of whether it can be directly
                                  deduced from the last step.
        """
        steps = [FS(self.__cfg) for FS in factory_steps]
        pipeline = PipelineDescription(steps, *args, **kwargs)
        self.__pipelines.append(pipeline)
        return pipeline

    def _build_dependencies(self, tile_name, raster_list, dryrun):
        """
        Runs the inputs through all pipeline descriptions to build the full list
        of intermediary and final products and what they require to be built.
        """
        first_inputs = generate_first_steps_from_manifests(
                tile_name=tile_name,
                raster_list=raster_list,
                dryrun=dryrun)

        pipelines_outputs = {'basename': first_inputs}  # TODO: find the right name _0/__/_firststeps/...?
        logger.debug('FIRST: %s', pipelines_outputs['basename'])

        required = {}     # (first batch) Final products identified as _needed to be produced_
        previous = {}     # Graph of deps: for a product tells how it's produced (pipeline + inputs)
        task_names_to_output_files_table = {}
        # +-> TODO: cache previous in order to remember which files already exists or not
        #     the difficult part is to flag as "generation successful" or not
        for pipeline in self.__pipelines:
            logger.debug('#############################################################################')
            logger.debug('Analysing |%s| dependencies', pipeline.name)
            logger.debug('Sources --> %s', pipeline.sources)
            outputs = []

            for origin, sources in pipeline.inputs.items():
                source_name = sources if isinstance(sources, str) else sources.name
                logger.debug('Checking sources from %s origin: %s', origin, source_name)
                # res = [(val if isinstance(val, str) else val.name) for (key,val) in self.__inputs.items()]
                inputs = [output for output in pipelines_outputs[source_name]]

                # Locate all inputs for the current pipeline
                # -> Select all inputs for pipeline sources from pipelines_outputs
                # inputs = [output for source in pipeline.sources for output in pipelines_outputs[source]]
                logger.debug('===========================================================================')
                logger.debug('FROM all inputs as %s: %s', origin, inputs)
                # TODO: + for origin in pipeline.sources
                for input in inputs:  # inputs are meta
                    logger.debug('----------------------------------------------------------------------')
                    logger.debug('* GIVEN "%s": %s', origin, input)
                    expected = pipeline.expected(input)
                    expected_taskname = get_task_name(expected)
                    logger.debug('  %s <-- from input: %s', expected_taskname, input)
                    logger.debug('  --> %s', expected)
                    # We cannot analyse early whether a task product is already
                    # there as some product have names that depend on all inputs
                    # (see Concatenate).
                    # This is why the full dependencies tree is produced at this
                    # time. Unrequired parts will be trimmed in the next task
                    # producing step.
                    if expected_taskname not in previous:
                        outputs.append(expected)
                        # previous[expected_taskname] = {'pipeline': pipeline, 'inputs': [input]}
                        # previous[expected_taskname] = {'pipeline': pipeline, 'inputs': [InputInfo(input)]}
                        previous[expected_taskname] = TaskInputInfo(pipeline=pipeline)
                        previous[expected_taskname].add_input(origin, input)
                        logger.debug('This is a new product: %s, with a source from %s', expected_taskname, origin)
                    elif get_task_name(input) not in previous[expected_taskname].input_task_names:
                        # previous[expected_taskname]['inputs'].append(input)
                        previous[expected_taskname].add_input(origin, input)
                        logger.debug('The %s task depends on one more input, updating its metadata to reflect the situation. Updating %s ...', expected_taskname, expected)
                        update_out_filename(expected, previous[expected_taskname])
                        logger.debug('...to (%s)', expected)
                        already_registered_next_input = [ni for ni in outputs if get_task_name(ni) == expected_taskname]
                        assert len(already_registered_next_input) == 1
                        update_out_filename(already_registered_next_input[0], previous[expected_taskname])
                    if pipeline.product_is_required:
                        required[expected_taskname] = expected
                    task_names_to_output_files_table[expected_taskname] = out_filename(expected)

            pipelines_outputs[pipeline.name] = outputs

        logger.debug('#############################################################################')
        logger.debug('#############################################################################')
        logger.debug('#############################################################################')
        required_task_names = set()
        for name, meta in required.items():
            logger.debug("check task_name: %s", name)
            if product_exists(meta):
                logger.debug("Ignoring %s as the product already exist", name)
                previous[name] = False  # for the next log
            else:
                required_task_names.add(name)

        logger.debug("Dependencies found:")
        for task_name, prev in previous.items():
            if prev:
                # logger.debug('- %s requires %s on %s', task_name, prev.pipeline.name, [m['out_filename'] for m in prev.inputs])
                logger.debug('- %s requires %s on %s', task_name, prev.pipeline.name, prev.inputs)
            else:
                logger.debug('- %s already exists, no need to produce it', task_name)
        return required_task_names, previous, task_names_to_output_files_table

    def _build_tasks_from_dependencies(self,
            required, previous, task_names_to_output_files_table,
            debug_otb, do_watch_ram):  # pylint: disable=no-self-use
        """
        Generates the actual list of tasks for :func:`dask.client.get()`.

        In case debug_otb is true, instead of a dictionary of tasks, an ordered list of tasks is
        returned in order to process sequentially each pipeline.

        `previous` is lade of:
        - "pipeline": reference to the :class:`PipelineDescription`
        - "inputs": list of the inputs (metadata)
        """
        tasks = {}
        logger.debug('Building all tasks')
        while required:
            new_required = set()
            for task_name in required:
                assert previous[task_name]
                base_task_name = to_dask_key(task_name)
                task_inputs    = previous[task_name].inputs
                pipeline_descr = previous[task_name].pipeline
                input_task_keys = [to_dask_key(tn) for tn in previous[task_name].input_task_names]
                assert list(input_task_keys)
                logger.debug('%s(%s) --> %s', task_name, list(input_task_keys), task_inputs)
                # TODO: check whether the pipeline shall be instanciated w/
                # file_name of task_name
                output_filename = task_names_to_output_files_table[task_name]
                pipeline_instance = pipeline_descr.instanciate(output_filename, True, True, do_watch_ram)
                pipeline_instance.set_inputs(task_inputs)
                logger.debug('~~> TASKS[%s] += %s(keys=%s)', base_task_name, pipeline_descr.name, list(input_task_keys))
                register_task(tasks, base_task_name, (execute4dask, pipeline_instance, input_task_keys))

                for t in previous[task_name].input_metas:  # check whether the inputs need to be produced as well
                    tn = get_task_name(t)
                    logger.debug('processing task %s', t)
                    if not product_exists(t):
                        logger.info('  => Need to register production of %s (for %s)', tn, pipeline_descr.name)
                        new_required.add(tn)
                    else:
                        logger.info('  => Starting %s from existing %s', pipeline_descr.name, tn)
                        register_task(tasks, to_dask_key(tn), FirstStep(**t))
            required = new_required
        return tasks

    def generate_tasks(self, tile_name, raster_list, debug_otb=False, dryrun=False, do_watch_ram=False):
        """
        Generate the minimal list of tasks that can be passed to Dask

        Params:
            :tile_name:   Name of the current S2 tile
            :raster_list: List of rasters that intersect the tile.
        TODO: Move into another dedicated class instead of PipelineDescriptionSequence
        """
        required, previous, task_names_to_output_files_table = self._build_dependencies(
                tile_name=tile_name,
                raster_list=raster_list,
                dryrun=dryrun)

        # Generate the actual list of tasks
        final_products = [to_dask_key(p) for p in required]
        tasks = self._build_tasks_from_dependencies(
                required=required,
                previous=previous,
                task_names_to_output_files_table=task_names_to_output_files_table,
                debug_otb=debug_otb,
                do_watch_ram=do_watch_ram)

        for final_product in final_products:
            assert debug_otb or final_product in tasks.keys()
        return tasks, final_products


# ======================================================================
# Some specific steps
class FirstStep(AbstractStep):
    """
    First Step:
    - no application executed
    """
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)
        if 'out_filename' not in self._meta:
            # If not set through the parameters, set it from the basename + out dir
            self._meta['out_filename'] = self._meta['basename']
        _, basename = os.path.split(self._meta['basename'])
        self._meta['basename'] = basename
        self._meta['pipe'] = [self._meta['out_filename']]

    def __str__(self):
        return 'FirstStep%s' % (self._meta,)

    def __repr__(self):
        return 'FirstStep%s' % (self._meta,)


class MergeStep(AbstractStep):
    """
    Kind of FirstStep that merges the result of one or several other steps.
    Used in entry of :class:`Concatenate`

    - no application executed
    """
    def __init__(self, steps, *argv, **kwargs):
        meta = {**(steps[0]._meta), **kwargs}  # kwargs override step0.meta
        super().__init__(*argv, **meta)
        self.__steps = steps
        self._meta['out_filename'] = [out_filename(s._meta) for s in steps]

    def __str__(self):
        return 'MergeStep%s' % (self.__steps,)

    def __repr__(self):
        return 'MergeStep%s' % (self.__steps,)


class StoreStep(_StepWithOTBApplication):
    """
    Artificial Step that takes cares of executing the last OTB application in the
    pipeline.
    """
    def __init__(self, previous: Step):
        assert not previous.is_first_step
        super().__init__(previous._app, *[], **previous.meta)
        self._out = previous.param_out

    @property
    def tmp_filename(self):
        """
        Property that returns the name of the file produced by the current step while
        the OTB application is running.
        Eventually, it'll get renamed into `self.out_filename` if the application
        succeeds.
        """
        assert 'out_tmp_filename' in self._meta
        return self._meta['out_tmp_filename']

    @property
    def shall_store(self):
        return True

    def execute_and_write_output(self):
        """
        Specializes :func:`execute_and_write_output()` to actually execute the
        OTB pipeline.
        """
        assert self._app
        do_measure = True  # TODO
        # logger.debug('meta pipe: %s', self.meta['pipe'])
        pipeline_name = '%s > %s' % (' | '.join(str(e) for e in self.meta['pipe']), self.out_filename)
        if os.path.isfile(self.out_filename):
            # This is a dirty failsafe, instead of analysing at the last
            # moment, it's be better to have a clear idea of all dependencies
            # and of what needs to be done.
            logger.info('%s already exists. Aborting << %s >>', self.out_filename, pipeline_name)
            return
        with Utils.ExecutionTimer('-> pipe << ' + pipeline_name + ' >>', do_measure):
            if not self.meta.get('dryrun', False):
                # TODO: catch execute failure, and report it!
                # logger.info("START %s", pipeline_name)
                with Utils.RedirectStdToLogger(logging.getLogger('s1tiling.OTB')):
                    # For OTB application execution, redirect stdout/stderr
                    # messages to s1tiling.OTB
                    self._app.SetParameterString(
                            self.param_out,
                            self.tmp_filename + out_extended_filename_complement(self.meta))
                    self._app.ExecuteAndWriteOutput()
                commit_otb_application(self.tmp_filename, self.out_filename)
        if 'post' in self.meta and not self.meta.get('dryrun', False):
            for hook in self.meta['post']:
                hook(self.meta)
        self.meta['pipe'] = [self.out_filename]


def commit_otb_application(tmp_filename, out_fn):
    """
    Concluding step that validates the execution of a successful OTB application.
    - Rename the tmp image into its final name
    - Rename the associated geom file (if any as well)
    """
    res = shutil.move(tmp_filename, out_fn)
    logger.debug('Renaming: %s <- mv %s %s', res, tmp_filename, out_fn)
    re_tiff = re.compile(r'\.tiff?$')
    tmp_geom = re.sub(re_tiff, '.geom', tmp_filename)
    if os.path.isfile(tmp_geom):
        out_geom = re.sub(re_tiff, '.geom', out_fn)
        res = shutil.move(tmp_geom, out_geom)
        logger.debug('Renaming: %s <- mv %s %s', res, tmp_geom, out_geom)
    assert not os.path.isfile(tmp_filename)


class _FileProducingStepFactory(StepFactory):
    """
    Abstract class that factorizes filename transformations and parameter
    handling for Steps that produce files, either with OTB or through external
    calls.

    :func:`create_step`  is kind of _abstract_ at this point.
    """
    def __init__(self, cfg,
            gen_tmp_dir, gen_output_dir, gen_output_filename,
            *argv, **kwargs):
        """
        Constructor

        See :func:`output_directory`, :func:`tmp_directory`,
        :func:`build_step_output_filename` and
        :func:`build_step_output_tmp_filename` for the usage of ``gen_tmp_dir``,
        ``gen_output_dir`` and ``gen_output_filename``.
        """
        super().__init__(*argv, **kwargs)
        is_a_final_step = gen_output_dir and gen_output_dir != gen_tmp_dir
        # logger.debug("%s -> final: %s <== gen_tmp=%s    gen_out=%s", self.name, is_a_final_step, gen_tmp_dir, gen_output_dir)

        self.__gen_tmp_dir         = gen_tmp_dir
        self.__gen_output_dir      = gen_output_dir if gen_output_dir else gen_tmp_dir
        self.__gen_output_filename = gen_output_filename
        self.__ram_per_process     = cfg.ram_per_process
        self.__tmpdir              = cfg.tmpdir
        self.__outdir              = cfg.output_preprocess if is_a_final_step else cfg.tmpdir
        logger.debug("new _FileProducingStepFactory(%s) -> TMPDIR=%s  OUT=%s", self.name, self.__tmpdir, self.__outdir)

    def output_directory(self, meta):
        """
        Accessor to where output files will be stored in case their production
        is required (i.e. not in-memory processing)

        This property is built from ``gen_output_dir`` construction parameter.
        Typical values for the parameter are:

        - ``os.path.join(cfg.output_preprocess, '{tile_name}'),`` where ``tile_name``
          is looked into ``meta`` parameter
        - ``None``, in that case the result will be the same as :func:`tmp_directory`.
          This case will make sense for steps that don't produce required products
        """
        return self.__gen_output_dir.format(**meta)

    def _get_nominal_output_basename(self, meta):
        """
        Returns the pathless basename of the produced file (internal).
        """
        if isinstance(self.__gen_output_filename, str):
            rootname = os.path.splitext(meta['basename'])[0]
            filename = self.__gen_output_filename.format(**meta, rootname=rootname)
        else:
            filename = meta['basename']
            if self.__gen_output_filename:
                filename = filename.replace(*self.__gen_output_filename)
        return filename

    def build_step_output_filename(self, meta):
        """
        Returns the names of typical result files in case their production
        is required (i.e. not in-memory processing).

        This specialization uses ``gen_output_filename`` list construction
        parameter as parameters for :func:`str.replace` function applied
        on ``meta['basename']``
        """
        filename = self._get_nominal_output_basename(meta)
        return os.path.join(self.output_directory(meta), filename)

    def tmp_directory(self, meta):
        """
        Directory used to store temporary files before they are renamed into
        their final version.

        This property is built from ``gen_tmp_dir`` construction parameter.
        Typical values for the parameter are:

        - ``os.path.join(cfg.tmpdir, 'S1')``
        - ``os.path.join(cfg.tmpdir, 'S2', '{tile_name}')`` where ``tile_name``
          is looked into ``meta`` parameter
        """
        return self.__gen_tmp_dir.format(**meta)

    def build_step_output_tmp_filename(self, meta):
        """
        This specialization of :func:`StepFactory.build_step_output_tmp_filename`
        will automatically insert ``.tmp`` before the filename extension.
        """
        filename = self._get_nominal_output_basename(meta)
        return os.path.join(self.tmp_directory(meta), re.sub(re_tiff, r'.tmp\g<0>', filename))

    @property
    def ram_per_process(self):
        """
        Property ram_per_process
        """
        return self.__ram_per_process


class OTBStepFactory(_FileProducingStepFactory):
    """
    Abstract StepFactory for all OTB Applications.

    This step aims at factoring recurring definitions.
    """
    def __init__(self, cfg,
            appname,
            gen_tmp_dir, gen_output_dir, gen_output_filename,
            *argv, **kwargs):
        """
        Constructor

        See :func:`output_directory`, :func:`tmp_directory`,
        :func:`build_step_output_filename` and
        :func:`build_step_output_tmp_filename` for the usage of ``gen_tmp_dir``,
        ``gen_output_dir`` and ``gen_output_filename``.
        """
        super().__init__(cfg, gen_tmp_dir, gen_output_dir, gen_output_filename, *argv, **kwargs)
        is_a_final_step = gen_output_dir and gen_output_dir != gen_tmp_dir
        # logger.debug("%s -> final: %s <== gen_tmp=%s    gen_out=%s", self.name, is_a_final_step, gen_tmp_dir, gen_output_dir)

        self._in                   = kwargs.get('param_in',  'in')
        self._out                  = kwargs.get('param_out', 'out')
        self._appname              = appname
        logger.debug("new OTBStepFactory(%s) -> app=%s", self.name, appname)

    @property
    def appname(self):
        """
        OTB Application property.
        """
        return self._appname

    @property
    def param_in(self):
        """
        Name of the "in" parameter used by the OTB Application.
        Default is likely to be "in", whie some applications use "io.in", often "il" for list of
        files...
        """
        return self._in

    @property
    def param_out(self):
        """
        Name of the "out" parameter used by the OTB Application.
        Default is likely to be "out", whie some applications use "io.out".
        """
        return self._out

    def create_step(self, input: AbstractStep, in_memory: bool, unused_previous_steps):
        """
        Instanciates the step related to the current :class:`StepFactory`,
        that consumes results from the previous `input` step.

        1. This methods starts by updating metadata information through
        :func:`complete_meta()` on the `input` metadata.

        2. Then, steps that wrap an OTB application will instanciate this
        application object, and:

           - either pipe the new application to the one from the `input` step
             if it wasn't a first step
           - or fill in the "in" parameter of the application with the
             :func:`out_filename` of the `input` step.

        2-bis. in case the new step isn't related to an OTB application,
        nothing specific is done, we'll just return an :class:`AbstractStep`

        Note: While `previous_steps` is ignored in this specialization, it's
        used in :func:`Store.create_step()` where it's eventually used to
        release all OTB Application objects.
        """
        # TODO: distinguish step description & step
        assert issubclass(type(input), AbstractStep)
        meta = self.complete_meta(input.meta)
        assert self.appname

        # Otherwise: step with an OTB application...
        if meta.get('dryrun', False):
            logger.warning('DRY RUN mode: ignore step and OTB Application creation')
            lg_from = input.out_filename if input.is_first_step else 'app'
            logger.debug('Register app: %s (from %s) %s', self.appname, lg_from, ' '.join('-%s %s' % (k, as_app_shell_param(v)) for k, v in self.parameters(meta).items()))
            meta['param_out'] = self.param_out
            return Step('FAKEAPP', **meta)
        with Utils.RedirectStdToLogger(logging.getLogger('s1tiling.OTB')):
            # For OTB application execution, redirect stdout/stderr messages to s1tiling.OTB
            app = otb.Registry.CreateApplication(self.appname)
            if not app:
                raise RuntimeError("Cannot create OTB application '" + self.appname + "'")
            parameters = self.parameters(meta)
            if input.is_first_step:
                if not files_exist(input.out_filename):
                    logger.critical("Cannot create OTB pipeline starting with %s as some input files don't exist (%s)", self.appname, input.out_filename)
                    raise RuntimeError("Cannot create OTB pipeline starting with %s as some input files don't exist (%s)" % (self.appname, input.out_filename))
                # parameters[self.param_in] = input.out_filename
                lg_from = input.out_filename
            else:
                app.ConnectImage(self.param_in, input.app, input.param_out)
                this_step_is_in_memory = in_memory and not input.shall_store
                # logger.debug("Chaining %s in memory: %s", self.appname, this_step_is_in_memory)
                app.PropagateConnectMode(this_step_is_in_memory)
                if this_step_is_in_memory:
                    # When this is not a store step, we need to clear the input parameters
                    # from its list, otherwise some OTB applications may comply
                    del parameters[self.param_in]
                lg_from = 'app'

            self.set_output_pixel_type(app, meta)
            logger.debug('Register app: %s (from %s) %s -%s %s',
                    self.appname, lg_from,
                    ' '.join('-%s %s' % (k, as_app_shell_param(v)) for k, v in parameters.items()),
                    self.param_out, as_app_shell_param(meta.get('out_filename', '???')))
            try:
                app.SetParameters(parameters)
            except Exception:
                logger.exception("Cannot set parameters to %s (from %s) %s" % (self.appname, lg_from, ' '.join('-%s %s' % (k, as_app_shell_param(v)) for k, v in parameters.items())))
                raise

        meta['param_out'] = self.param_out
        return Step(app, **meta)


class ExecutableStepFactory(_FileProducingStepFactory):
    # TODO: Factorize out _FileProducingStepFactory
    # - directory
    # - temp name VS final name
    """
    Abstract StepFactory for executing any external program.

    This step aims at factoring recurring definitions.
    """
    def __init__(self, cfg,
            exename,
            gen_tmp_dir, gen_output_dir, gen_output_filename,
            *argv, **kwargs):
        """
        Constructor

        See :func:`output_directory`, :func:`tmp_directory`,
        :func:`build_step_output_filename` and
        :func:`build_step_output_tmp_filename` for the usage of ``gen_tmp_dir``,
        ``gen_output_dir`` and ``gen_output_filename``.
        """
        super().__init__(cfg, gen_tmp_dir, gen_output_dir, gen_output_filename, *argv, **kwargs)
        self._exename              = exename
        logger.debug("new ExecutableStepFactory(%s) -> exe=%s", self.name, exename)

    def create_step(self, input: AbstractStep, in_memory: bool, previous_steps):
        logger.debug("Directly execute %s step", self.name)
        assert issubclass(type(input), AbstractStep)
        meta = self.complete_meta(input.meta)
        res = ExecutableStep(self._exename, **meta)
        return res


class Store(StepFactory):
    """
    Factory for Artificial Step that forces the result of the previous app
    to be stored on disk by breaking in-memory connection.

    While it could be used manually, it's meant to be automatically append
    at the end of a pipeline if any step is actually related to OTB.
    """
    def __init__(self, appname, *argv, **kwargs):
        super().__init__('(StoreOnFile)', "(StoreOnFile)", *argv, **kwargs)

    def create_step(self, input: Step, in_memory: bool, previous_steps):
        """
        Specializes :func:`create_step()` to trigger
        :func:`execute_and_write_output()` on the last step that relates to an
        OTB Application.
        """
        if input.is_first_step:
            assert False  # Should no longer happen!
            # Special case of by-passed inputs
            meta = input.meta.copy()
            return AbstractStep(**meta)

        res = StoreStep(input)
        try:
            res.execute_and_write_output()
        finally:
            # logger.debug("Collecting memory!")
            # Collect memory now!
            res.release_app()
            for step in previous_steps:
                step.release_app()
        return res

    # abstract methods...
    def parameters(self, meta):
        raise TypeError("No way to ask for the parameters from a StoreFactory")

    def output_directory(self, meta):
        raise TypeError("No way to ask for output dir of a StoreFactory")

    def build_step_output_filename(self, meta):
        raise TypeError("No way to ask for the output filename of a StoreFactory")

    def build_step_output_tmp_filename(self, meta):
        raise TypeError("No way to ask for the output temporary filename of a StoreFactory")


# ======================================================================
# Multi processing related (old) code
def mp_worker_config(queue):
    """
    Worker configuration function called by Pool().

    It takes care of initializing the queue handler in the subprocess.

    Params:
        :queue: multiprocessing.Queue used for passing logging messages from worker to main
            process.
    """
    qh = logging.handlers.QueueHandler(queue)
    logger = logging.getLogger()
    logger.addHandler(qh)


# TODO: try to make it static...
def execute4mp(pipeline):
    """
    Internal worker function used by multiprocess to execute a pipeline.
    """
    return pipeline.do_execute()


class PoolOfOTBExecutions:
    """
    Internal multiprocess Pool of OTB pipelines.
    """
    def __init__(self,
            title,
            do_measure,
            nb_procs, nb_threads,
            log_queue, log_queue_listener,
            debug_otb):
        """
        constructor
        """
        self.__pool = []
        self.__title              = title
        self.__do_measure         = do_measure
        self.__nb_procs           = nb_procs
        self.__nb_threads         = nb_threads
        self.__log_queue          = log_queue
        self.__log_queue_listener = log_queue_listener
        self.__debug_otb          = debug_otb

    def new_pipeline(self, **kwargs):
        """
        Register a new pipeline.
        """
        in_memory    = kwargs.get('in_memory', True)
        do_watch_ram = kwargs.get('do_watch_ram', False)
        pipeline = Pipeline(self.__do_measure, in_memory, do_watch_ram)
        self.__pool.append(pipeline)
        return pipeline

    def process(self):
        """
        Executes all the pipelines in parallel.
        """
        nb_cmd = len(self.__pool)

        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(self.__nb_threads)
        os.environ['OTB_LOGGER_LEVEL'] = 'DEBUG'
        if self.__debug_otb:  # debug OTB applications with gdb => do not spawn process!
            execute4mp(self.__pool[0])
        else:
            with multiprocessing.Pool(self.__nb_procs, mp_worker_config, [self.__log_queue]) as pool:
                self.__log_queue_listener.start()
                for count, result in enumerate(pool.imap_unordered(execute4mp, self.__pool), 1):
                    logger.info("%s correctly finished", result)
                    logger.info(' --> %s... %s%%', self.__title, count * 100. / nb_cmd)

                pool.close()
                pool.join()
                self.__log_queue_listener.stop()


class Processing:
    """
    Entry point for executing multiple instance of the same pipeline of
    different inputs.

    1. The object is initialized with a log queue and its listener
    2. The pipeline is registered with a list of :class`StepFactory` s
    3. The processing is done on a list of :class:`FirstStep` s
    """
    def __init__(self, cfg, debug_otb):
        self.__log_queue          = cfg.log_queue
        self.__log_queue_listener = cfg.log_queue_listener
        self.__cfg                = cfg
        self.__factory_steps      = []
        self.__debug_otb          = debug_otb

    def register_pipeline(self, factory_steps):
        """
        Register a list of :class:`StepFactory` s that describes a pipeline.
        """
        # Automatically append the final storing step
        self.__factory_steps = factory_steps + [Store]

    def process(self, startpoints):
        """
        Defines pipelines from the registered steps. Each pipeline is instanciated with a
        startpoint. Then they registered into the PoolOfOTBExecutions.
        The pool is finally executed.
        """
        assert self.__factory_steps
        pool = PoolOfOTBExecutions("testpool", True,
                self.__cfg.nb_procs, self.__cfg.OTBThreads,
                self.__log_queue, self.__log_queue_listener, debug_otb=self.__debug_otb)
        for startpoint in startpoints:
            logger.info("register processing of %s", startpoint.basename)
            pipeline = pool.new_pipeline(in_memory=True)
            pipeline.set_input(startpoint)
            for factory in self.__factory_steps:
                pipeline.push(factory(self.__cfg))

        logger.debug('Launch pipelines')
        pool.process()
