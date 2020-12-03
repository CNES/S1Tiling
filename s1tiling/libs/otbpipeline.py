#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2020 (c) CESBIO. All rights reserved.
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
import otbApplication as otb
from . import Utils

logger = logging.getLogger('s1tiling')


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


def out_extended_filename_complement(meta):
    """
    Helper accessor to the extended filename to use to produce the image.
    """
    return meta.get('out_extended_filename_complement', '')


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


class AbstractStep:
    """
    Internal root class for all actual `Step` s.

    There are three kinds of steps:

    - :class:`FirstStep` that contains information about input files
    - :class:`Step` that registers an otbapplication binding
    - :class:`StoreStep` that momentarilly disconnect on-memory pipeline to force storing of
      the resulting file.

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
    def __init__(self, appname, name, *unused_argv, **kwargs):
        self._in      = kwargs.get('param_in',  'in')
        self._out     = kwargs.get('param_out', 'out')
        self._appname = appname
        assert name
        self._name    = name
        logger.debug("new StepFactory(%s) -> app=%s", name, appname)

    @property
    def appname(self):
        """
        OTB Application property.
        """
        return self._appname

    @property
    def name(self):
        """
        Step Name property.
        """
        assert isinstance(self._name, str)
        return self._name

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

    def complete_meta(self, meta):  # to be overridden
        """
        Duplicates, completes, and return, the `meta` dictionary with specific
        information for the current factory.

        This method is used:

        - while analysing the dependencies to build the task graph -- in this
          use case the relevant information are the file names and paths.
        - before instanciating a new :class:`Step`

        Other metadata not filled here:

        - :func:`out_extended_filename_complement`
        """
        meta = meta.copy()
        meta['in_filename']      = out_filename(meta)
        meta['out_filename']     = self.build_step_output_filename(meta)
        meta['out_tmp_filename'] = self.build_step_output_tmp_filename(meta)
        meta['pipe']             = meta.get('pipe', []) + [self.__class__.__name__]
        return meta

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
        if not self.appname:
            # Return previous app?
            return AbstractStep(**meta)

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
            logger.debug('Register app: %s (from %s) %s', self.appname, lg_from, ' '.join('-%s %s' % (k, as_app_shell_param(v)) for k, v in parameters.items()))
            try:
                app.SetParameters(parameters)
            except Exception:
                logger.exception("Cannot set parameters to %s (from %s) %s" % (self.appname, lg_from, ' '.join('-%s %s' % (k, as_app_shell_param(v)) for k, v in parameters.items())))
                raise

        meta['param_out'] = self.param_out
        return Step(app, **meta)


class Outcome:
    """
    Kind of monad à la C++ `std::expected<>`, `boost::Outcome`.

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
        self.__related_filenames += [filename]
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
    def __init__(self, do_measure, in_memory, name=None, output=None):
        self.__pipeline   = []
        self.__do_measure = do_measure
        self.__in_memory  = in_memory
        self.__name       = name
        self.__output     = output
        self.__input      = None

    def __repr__(self):
        return self.name

    def set_input(self, input: AbstractStep):
        """
        Set the input of the instanciated pipeline.
        The `input` is expected to be either an :class:`AbstractStep` or a
        list of :class:`AbstractStep` in which case it'll be internally
        registered as a :class:`MergeStep`.
        """
        assert isinstance(input, (list, AbstractStep))
        if isinstance(input, list):
            if len(input) == 1:
                self.__input = input[0]
            else:
                assert input
                self.__input = MergeStep(input)
        else:
            self.__input = input

    @property
    def name(self):
        """
        Name of the pipeline.
        It's either user registered or automatically generated from the
        registered :class:`StepFactory` s.
        """
        appname = (self.__name or '|'.join(crt.appname for crt in self.__pipeline))
        return '%s -> %s from %s' % (appname, self.__output, self.__input.out_filename)

    @property
    def output(self):
        """
        Expected pipeline final output file.
        """
        return self.__output

    def push(self, otbstep: StepFactory):
        """
        Registers a StepFactory into the pipeline.
        """
        assert isinstance(otbstep, StepFactory)
        self.__pipeline += [otbstep]

    def do_execute(self):
        """
        Execute the pipeline.

        1. Makes sure the inputs exist -- unless in dry-run mode
        2. Incrementaly create the steps of the pipeline.
        3. Return the resulting output filename, or the caught errors.
        """
        assert self.__input
        if not files_exist(self.__input.out_filename) and not self.__input.meta.get('dryrun', False):
            msg = "Cannot execute %s as %s doesn't exist" % (self, self.__input.out_filename)
            logger.warning(msg)
            return Outcome(RuntimeError(msg))
        # print("LOG:", os.environ['OTB_LOGGER_LEVEL'])
        assert self.__pipeline  # shall not be empty!
        steps = [self.__input]
        for crt in self.__pipeline:
            step = crt.create_step(steps[-1], self.__in_memory, steps)
            steps += [step]

        return Outcome(steps[-1].out_filename)


# TODO: try to make it static...
def execute4dask(pipeline, *args, **unused_kwargs):
    """
    Internal worker function used by Dask to execute a pipeline.

    Returns the product filename(s) or the caught error in case of failure.
    """
    logger.debug('Parameters for %s: %s', pipeline, args)
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
        return pipeline.do_execute().add_related_filename(pipeline.output)
    except Exception as ex:  # pylint: disable=broad-except
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
    def __init__(self, factory_steps, name=None, product_required=False, is_name_incremental=False):
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

    def expected(self, input_meta):
        """
        Returns the expected name of the product of this pipeline
        """
        assert self.__factory_steps  # shall not be None or empty
        if self.__is_name_incremental:
            res = input_meta
            for step in self.__factory_steps:
                res = step.complete_meta(res)
        else:
            res = self.__factory_steps[-1].complete_meta(input_meta)
        # out_pathname = self.__factory_steps[-1].build_step_output_filename(res)
        out_pathname = out_filename(res)
        # logger.debug('%s / %s', dir, out_filename(res))
        # out_pathname = os.path.join(dir, out_filename(res))
        res['out_pathname'] = out_pathname
        logger.debug("%s(%s) -> %s", self.__name, input_meta['out_filename'], out_pathname)
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

    def instanciate(self, file, do_measure, in_memory):
        """
        Instanciates the pipeline specified.

        Note: It systematically register a :class:`Store` step at the end.
        """
        pipeline = Pipeline(do_measure, in_memory, self.name, file)
        for factory_step in self.__factory_steps + [Store('noappname')]:
            pipeline.push(factory_step)
        return pipeline


def to_dask_key(pathname):
    """
    Generate a simplified graph key name from a full pathname.
    - Strip directory name
    - Replace '-' with '_' as Dask has a special interpretation for '-' in key names.
    """
    return Path(pathname).stem.replace('-', '_')


def generate_first_steps_from_manifest(raster_list, tile_name, dryrun):
    """
    Flatten all rasters from the manifest as a list of :class:`FirstStep`
    """
    inputs = []
    for raster, tile_origin in raster_list:
        manifest = raster.get_manifest()
        for image in raster.get_images_list():
            start = FirstStep(
                    tile_name=tile_name,
                    tile_origin=tile_origin,
                    manifest=manifest,
                    basename=image,
                    dryrun=dryrun)
            inputs += [start.meta]
    return inputs


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

    def _build_dependencies(self, tile_name, raster_list, dryrun):
        """
        Runs the inputs through all pipeline descriptions to build the full list
        of intermediary and final products and what they required to be built.
        """
        inputs = generate_first_steps_from_manifest(
                tile_name=tile_name,
                raster_list=raster_list,
                dryrun=dryrun)

        required = set()  # (first batch) Final products identified as _needed to be produced_
        previous = {}     # Graph of deps: for a product tells how it's produced (pipeline + inputs)
        # +-> TODO: cache previous in order to remember which files already exist or not
        #     the difficult part is to flag as "generation successful" of not
        for pipeline in self.__pipelines:
            logger.debug('Analysing |%s| dependencies', pipeline.name)
            next_inputs = []
            for input in inputs:
                expected = pipeline.expected(input)
                next_inputs += [expected]
                expected_pathname = expected['out_pathname']
                if os.path.isfile(expected_pathname):
                    previous[expected_pathname] = False  # File exists
                else:
                    if expected_pathname not in previous:
                        previous[expected_pathname] = {'pipeline': pipeline, 'inputs': [input]}
                    elif not input['out_filename'] in (m['out_filename'] for m in previous[expected_pathname]['inputs']):
                        previous[expected_pathname]['inputs'].append(input)
                    if pipeline.product_is_required:
                        required.add(expected_pathname)
            inputs = next_inputs

        logger.debug("Dependencies found:")
        for path, prev in previous.items():
            if prev:
                logger.debug('- %s may require %s on %s', path, prev['pipeline'].name, [m['out_filename'] for m in prev['inputs']])
            else:
                logger.debug('- %s already exists, no need to produce it', path)
        return required, previous

    def _build_tasks_from_dependencies(self, required, previous, debug_otb):  # pylint: disable=no-self-use
        """
        Generates the actual list of tasks for :func:`dask.client.get()`.

        In case debug_otb is true, instead of a dictionary of tasks, an ordered list of tasks is
        returned in order to process sequentially each pipeline.
        """
        tasks = {} if not debug_otb else []
        while required:
            new_required = set()
            for file in required:
                assert previous[file]
                base_file = to_dask_key(file)
                task_inputs = previous[file]['inputs']
                # logger.debug('%s --> %s', file, task_inputs)
                input_files = [to_dask_key(m['out_filename']) for m in task_inputs]
                pipeline_instance = previous[file]['pipeline'].instanciate(file, True, True)
                pipeline_instance.set_input([FirstStep(**m) for m in task_inputs])
                if debug_otb:
                    tasks.append([base_file, (execute4dask, pipeline_instance, input_files)])
                else:
                    tasks[base_file] = (execute4dask, pipeline_instance, input_files)
                logger.debug('TASKS[%s] += %s(%s)', base_file, previous[file]['pipeline'].name, input_files)

                for t in task_inputs:  # check whether the inputs need to be produced as well
                    if not os.path.isfile(t['out_filename']):
                        logger.debug('Need to register %s', t['out_filename'])
                        new_required.add(t['out_filename'])
                    elif debug_otb:
                        tasks.append([to_dask_key(t['out_filename']), FirstStep(**t)])
                    else:
                        tasks[to_dask_key(t['out_filename'])] = FirstStep(**t)
            required = new_required
        return tasks

    def generate_tasks(self, tile_name, raster_list, debug_otb=False, dryrun=False):
        """
        Generate the minimal list of tasks that can be passed to Dask

        Params:
            :tile_name:   Name of the current S2 tile
            :raster_list: List of rasters that intersect the tile.
        TODO: Move into another dedicated class instead of PipelineDescriptionSequence
        """
        required, previous = self._build_dependencies(
                tile_name=tile_name,
                raster_list=raster_list,
                dryrun=dryrun)

        # Generate the actual list of tasks
        final_products = [to_dask_key(p) for p in required]
        tasks = self._build_tasks_from_dependencies(
                required=required,
                previous=previous,
                debug_otb=debug_otb)

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
        with Utils.RedirectStdToLogger(logging.getLogger('s1tiling.OTB')):
            # For OTB application execution, redirect stdout/stderr messages to s1tiling.OTB
            with Utils.ExecutionTimer('-> pipe << ' + pipeline_name + ' >>', do_measure):
                if not self.meta.get('dryrun', False):
                    # TODO: catch execute failure, and report it!
                    # logger.info("START %s", pipeline_name)
                    with Utils.RedirectStdToLogger(logging.getLogger('s1tiling.OTB')):
                        # For OTB application execution, redirect stdout/stderr messages to
                        # s1tiling.OTB
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


class Store(StepFactory):
    """
    Factory for Artificial Step that forces the result of the previous app
    to be stored on disk by breaking in-memory connection.
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
        in_memory = kwargs.get('in_memory', True)
        pipeline = Pipeline(self.__do_measure, in_memory)
        self.__pool += [pipeline]
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
