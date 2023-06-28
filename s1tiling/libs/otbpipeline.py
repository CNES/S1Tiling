#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2023 (c) CNES. All rights reserved.
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

# from __future__ import annotations  # Require Python 3.7+...

"""
This module provides pipeline for chaining OTB applications, and a pool to execute them.
"""

import os
import sys
import shutil
import re
import datetime
import copy
from abc import ABC, abstractmethod
from itertools import filterfalse
import logging
import logging.handlers
import multiprocessing
import subprocess
from pathlib import Path

# memory leaks
from distributed import get_worker
import objgraph
from pympler import tracker # , muppy

from osgeo import gdal
import otbApplication as otb
from . import Utils
from . import exits
from .outcome import Outcome

logger = logging.getLogger('s1tiling.pipeline')

re_tiff    = re.compile(r'\.tiff?$')
re_any_ext = re.compile(r'\.[^.]+$')  # Match any kind of file extension


def otb_version():
    """
    Returns the current version on OTB (through a call to ResetMargin -version)
    The result is cached
    """
    if not hasattr(otb_version, "_version"):
        try:
            r = subprocess.run(['otbcli_ResetMargin', '-version'], stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT)
            version = r.stdout.decode('utf-8').strip('\n')
            version = re.search(r'\d+(\.\d+)+$', version)[0]
            logger.info("OTB version detected on the system is %s", version)
            otb_version._version = version
        except Exception as ex:  # pylint: disable=broad-except
            logger.exception(ex)
            raise RuntimeError("Cannot determine current OTB version")
    return otb_version._version


def as_list(param):
    """
    Make sure ``param`` is either a list or encapsulated in a list.
    """
    if isinstance(param, list):
        return param
    else:
        return [param]


class OutputFilenameGenerator(ABC):
    """
    Abstract class for generating filenames.
    Several policies are supported as of now:
    - return the input string (default implementation)
    - replace a text with another one
    - {template} strings
    - list of any of the other two
    """
    def generate(self, basename, keys):  # pylint: disable=unused-argument,no-self-use
        """
        Default implementation does nothing.
        """
        return basename


class ReplaceOutputFilenameGenerator(OutputFilenameGenerator):
    """
    Given a pair ``[text_to_search, text_to_replace_with]``,
    replace the exact matching text with new text in ``basename`` metadata.
    """
    def __init__(self, before_afters):
        assert isinstance(before_afters, list)
        self.__before_afters = before_afters

    def generate(self, basename, keys):
        filename = basename.replace(*self.__before_afters)
        return filename


class CannotGenerateFilename(KeyError):
    """
    Exception used to filter out cases where a meta cannot serve as a direct
    input of a :class:`StepFactory`.
    """
    pass


class TemplateOutputFilenameGenerator(OutputFilenameGenerator):
    """
    Given a template: ``"text{key1}_{another_key}_.."``,
    inject the metadata instead of the template keys.
    """
    def __init__(self, template):
        assert isinstance(template, str)
        self.__template = template

    def generate(self, basename, keys):
        try:
            rootname = os.path.splitext(basename)[0]
            filename = self.__template.format(**keys, rootname=rootname)
            return filename
        except KeyError as e:
            raise CannotGenerateFilename(f'Impossible to generate a filename matching {self.__template} from {keys}') from e


class OutputFilenameGeneratorList(OutputFilenameGenerator):
    """
    Some steps produce several products.
    This specialization permits to generate several output filenames.

    It's constructed from other filename generators.
    """
    def __init__(self, generators):
        assert isinstance(generators, list)
        self.__generators = generators

    def generate(self, basename, keys):
        filenames = [generator.generate(basename, keys) for generator in self.__generators]
        return filenames


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
        return f"'{param}'"


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


def tmp_filename(meta):
    """
    Helper accessor to access the temporary ouput filename of a `Step`.
    """
    assert 'out_tmp_filename' in meta
    return meta.get('out_tmp_filename')


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


def _fetch_input_data(key, inputs):
    """
    Helper function that extract the meta data associated to a key from a
    multiple-inputs list of inputs.
    """
    keys = set().union(*(input.keys() for input in inputs))
    assert key in keys, f"Cannot find input '{key}' among {keys}"
    return [input[key] for input in inputs if key in input.keys()][0]


def manifest_to_product_name(manifest):
    """
    Helper function that returns the product name (SAFE directory without the
    ``.SAFE`` extension) from the full path to the :file:`manifest.safe` file.
    """
    fullpath = Path(manifest)
    return fullpath.parent.stem


def product_exists(meta):
    """
    Helper accessor that tells whether the product described by the metadata
    already exists.
    """
    if 'does_product_exist' in meta:
        return meta['does_product_exist']()
    else:
        return os.path.isfile(out_filename(meta))


def _is_compatible(output_meta, input_meta):
    """
    Tells whether ``input_meta`` is a valid input for ``output_meta``

    Uses the optional meta information ``is_compatible`` from ``output_meta``
    to tell whether they are compatible.

    This will be uses for situations where an input file will be used as input
    for several different new files. Typical example: all final normlimed
    outputs on a S2 tile will rely on the same map of sin(LIA). As such,
    the usual __input -> expected output__ approach cannot work.
    """
    if 'is_compatible' in output_meta:
        return output_meta['is_compatible'](input_meta)
    else:
        return False


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


def is_running_dry(meta):
    """
    Helper function to test whether metadata has ``dryrun`` property set to True.
    """
    return meta.get('dryrun', False)


def is_debugging_caches(meta):
    """
    Helper function to test whether metadata has ``debug_caches`` property set to True.
    """
    return meta.get('debug_caches', False)


def execute(params, dryrun):
    """
    Helper function to execute any external command.

    And log its execution, measure the time it takes.
    """
    msg = ' '.join([str(p) for p in params])
    logging.info('$> '+msg)
    if not dryrun:
        with Utils.ExecutionTimer(msg, True):
            subprocess.run(args=params, check=True)


class AbstractStep:
    """
    Internal root class for all actual `Step` s.

    There are several kinds of steps:

    - :class:`FirstStep` that contains information about input files
    - :class:`Step` that registers an otbapplication binding
    - :class:`StoreStep` that momentarilly disconnect on-memory pipeline to
      force storing of the resulting file.
    - :class:`ExecutableStep` that executes external applications
    - :class:`MergeStep` that operates a rendez-vous between several steps
      producing files of a same kind.

    The step will contain information like the current input file, the current
    output file...
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
        No OTB related step requires its result to be stored on disk and to
        break in_memory connection by default.

        However, the artificial Step produced by :class:`Store` factory will
        force the result of the `previous` application(s) to be stored on disk.
        """
        return False

    def release_app(self):
        """
        Makes sure that steps with applications are releasing the application
        """
        pass

    def clean_cache(self):
        """
        Takes care or removing intermediary files once we know they are no
        longer required like the orthorectified subtiles once the
        concatenation has been done.
        """
        if 'files_to_remove' in self.meta :
            files = self.meta['files_to_remove']
            # All possible geom files that may exist
            geoms = [re.sub(re_tiff, '.geom', fn) for fn in files]
            # All geaoms that do actually exist
            geoms = [fn for fn in geoms if os.path.isfile(fn)]
            files = files + geoms
            if is_debugging_caches(self.meta):
                logger.debug('NOT cleaning intermediary files: %s (cache debugging mode!)', files)
            else:
                logger.debug('Cleaning intermediary files: %s', files)
                if not is_running_dry(self.meta):
                    Utils.remove_files(files)
            self.meta.pop('files_to_remove', None)


    def _write_image_metadata(self):
        """
        Update Image metadata (with GDAL API).
        Fetch the new content in ``meta['image_metadata']``
        """
        img_meta = self.meta.get('image_metadata', {})
        # fullpath = out_filename(self.meta)
        fullpath = self.tmp_filename
        if not img_meta:
            logger.debug('No metadata to update in %s', fullpath)
            return
        if is_running_dry(self.meta):
            logger.debug("Don't set metadata in %s (dry-run)", fullpath)
            return
        def do_write(fullpath, img_meta):
            logger.debug('Set metadata in %s', fullpath)
            dst = gdal.Open(fullpath, gdal.GA_Update)
            assert dst

            for (kw, val) in img_meta.items():
                assert isinstance(val, str), f'GDAL metadata shall be strings. "{kw}" is a {val.__class__.__name__} (="{val}")'
                logger.debug(' - %s -> %s', kw, val)
                dst.SetMetadataItem(kw, val)
            dst.FlushCache()  # We really need to be sure it has been flushed now, if not closed
            del dst
            logger.debug('Metadata Set! (%s)', fullpath)
        if isinstance(fullpath, list):
            # Case of applications that produce several files like ComputeLIA
            for fp in fullpath:
                # TODO: how to specialize DESCRIPTION for each output image
                do_write(fp, img_meta)
        else:
            do_write(fullpath, img_meta)


class ExecutableStep(AbstractStep):
    """
    Generic step for calling any external application.
    """
    def __init__(self, exename, *argv, **kwargs):
        """
        Constructor.
        """
        super().__init__(None, *argv, **kwargs)
        self._exename = exename
        # logger.debug('ExecutableStep %s constructed', self._exename)

    @property
    def tmp_filename(self):
        """
        Property that returns the name of the file produced by the current step while
        the external application is running.
        Eventually, it'll get renamed into :func:`AbstractStep.out_filename` if
        the application succeeds.
        """
        return tmp_filename(self.meta)

    def execute_and_write_output(self, parameters):  # pylint: disable=no-self-use
        """
        Actually execute the external program.
        While the program runs, a temporary filename will be used as output.
        On successful execution, the output will be renamed to match its
        expected final name.
        """
        dryrun = is_running_dry(self.meta)
        logger.debug("ExecutableStep: %s (%s)", self, self.meta)
        execute([self._exename]+ parameters, dryrun)
        if not dryrun:
            self._write_image_metadata()
            commit_execution(self.tmp_filename, self.out_filename)
        if 'post' in self.meta and not dryrun:
            for hook in self.meta['post']:
                hook(self.meta)
        self.clean_cache()
        self.meta['pipe'] = [self.out_filename]


class _StepWithOTBApplication(AbstractStep):
    """
    Internal intermediary type for `Steps` that have an application object.
    Not meant to be used directly.

    Parent type for:

    - :class:`Step`  that will own the application
    - and :class:`StoreStep` that will just reference the application from the previous step
    """
    def __init__(self, app, *argv, **kwargs):
        """
        Constructor.
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
        Default is likely to be "out", while some applications use "io.out".
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


def _check_input_step_type(inputs):
    """
    Internal helper function that checks :func:`StepFactory.create_step()`
    ``inputs`` parameters is of the expected type, i.e.:
    list of dictionaries {'key': :class:`AbstractStep`}
    """
    assert all(issubclass(type(inp), dict) for inp in inputs), f"Inputs not of expected type: {inputs}"
    assert all(issubclass(type(step), AbstractStep) for inp in inputs for _, step in inp.items()), f"Inputs not of expected type: {inputs}"


class StepFactory(ABC):
    """
    Abstract factory for :class:`Step`

    Meant to be inherited for each possible OTB application or external
    application used in a pipeline.

    Sometimes we may also want to add some artificial steps that analyse
    products, filenames..., or step that help filter products for following
    pipelines.

    See: `Existing processings`_
    """
    def __init__(self, name, *unused_argv, **kwargs):  # pylint: disable=unused-argument
        assert name
        self._name               = name
        self.__image_description = kwargs.get('image_description', None)
        # logger.debug("new StepFactory(%s)", name)

    @property
    def name(self):
        """
        Step Name property.
        """
        assert isinstance(self._name, str)
        return self._name

    @property
    def image_description(self):
        """
        Property image_description, used to fill ``TIFFTAG_IMAGEDESCRIPTION``
        """
        return self.__image_description

    def check_requirements(self):
        """
        Abstract method used to test whether a :class:`StepFactory` has all
        its external requirements fulfilled. For instance,
        :class:`OTBStepFactory`'s will check their related OTB application can be executed.

        :return: ``None`` if  requirements are fulfilled.
        :return: A message indicating what is missing otherwise, and some
                 context how to fix it.
        """
        return None

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
        Returns a filename to a temporary file to use in output of the current application.

        When an OTB (/External) application is harshly interrupted (crash or
        user interruption), it leaves behind an incomplete (and thus invalid)
        file.
        In order to ignore those files when a pipeline is restarted, an
        temporary filename is used by the application.
        Once the application exits with success, the file will be renamed into
        :func:`build_step_output_filename()`, and possibly moved into
        :func:`_FileProducingStepFactory.output_directory()` if this is a final product.
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

        It's possible to inject some other metadata (that could be used from
        :func:`_get_canonical_input()` for instance) thanks to
        :func:`_update_filename_meta_pre_hook()`.
        """
        meta = meta.copy()
        self._update_filename_meta_pre_hook(meta)
        meta['in_filename']        = out_filename(meta)
        meta['out_filename']       = self.build_step_output_filename(meta)
        meta['out_tmp_filename']   = self.build_step_output_tmp_filename(meta)
        meta['pipe']               = meta.get('pipe', []) + [self.__class__.__name__]
        def check_product(meta):
            filename        = out_filename(meta)
            exist_file_name = os.path.isfile(filename)
            logger.debug('Checking %s product: %s => %s',
                    self.__class__.__name__,
                    filename, '∃' if exist_file_name else '∅')
            return exist_file_name
        meta['does_product_exist'] = lambda : check_product(meta)
        # meta['does_product_exist'] = lambda : os.path.isfile(out_filename(meta))
        meta.pop('task_name',           None)
        meta.pop('task_basename',       None)
        meta.pop('update_out_filename', None)
        meta.pop('is_compatible',       None)
        # for k in list(meta.keys()):  # Remove all entries associated to reduce_* keys
        #     if k.startswith('reduce_'):
        #         del meta[k]
        self._update_filename_meta_post_hook(meta)
        return meta

    def _update_filename_meta_pre_hook(self, meta):  # to be overridden  # pylint: disable=no-self-use
        """
        Hook meant to be overridden to complete product metadata before
        they are used to produce filenames or tasknames.

        Called from :func:`update_filename_meta()`
        """
        return meta

    def _update_filename_meta_post_hook(self, meta):  # to be overridden  # pylint: disable=no-self-use
        """
        Hook meant to be overridden to fix product metadata by
        overriding their default definition.

        Called from :func:`update_filename_meta()`
        """
        return meta

    def complete_meta(self, meta, all_inputs):  # to be overridden  # pylint: disable=unused-argument
        """
        Duplicates, completes, and return, the `meta` dictionary with specific
        information for the current factory regarding :class:`Step` instanciation.
        """
        meta.pop('out_extended_filename_complement', None)
        return self.update_filename_meta(meta)

    def update_image_metadata(self, meta, all_inputs):  # pylint: disable=unused-argument
        """
        Root implementation of :func:`update_image_metadata` that shall be
        specialized in every file producing Step Factory.
        """
        if 'image_metadata' not in meta:
            meta['image_metadata'] = {}
        imd = meta['image_metadata']
        imd['TIFFTAG_DATETIME'] = str(datetime.datetime.now().strftime('%Y:%m:%d %H:%M:%S'))
        if self.image_description:
            imd['TIFFTAG_IMAGEDESCRIPTION'] = self.image_description.format(
                    **meta,
                    flying_unit_code_short=meta['flying_unit_code'][1:].upper())

    def _get_inputs(self, previous_steps):  # pylint: disable=unused-argument,no-self-use
        """
        Extract the last inputs to use at the current level from all previous
        products seen in the pipeline.

        This method will need to be overridden in classes like
        :class:`ComputeLIA` in order to fetch N-1 "xyz" input.

        Postcondition:
            :``_check_input_step_type(result)`` is True
        """
        # By default, simply return the last step information
        assert len(previous_steps) > 0
        inputs = previous_steps[-1]
        _check_input_step_type(inputs)
        return inputs

    def _get_canonical_input(self, inputs):
        """
        Helper function to retrieve the canonical input associated to a list of inputs.
        By default, if there is only one input, this will be the one returned.
        Steps will multiple inputs will need to override this method.

        Precondition:
            :``_check_input_step_type(result)`` is True
        """
        _check_input_step_type(inputs)
        if len(inputs) == 1:
            return list(inputs[0].values())[0]
        else:
            # If this error is raised, this means the current step has several
            # inputs, we need to tell explicitely how the "main" input is found.
            keys = set().union(*(input.keys() for input in inputs))
            raise TypeError(f"No way to handle a multiple-inputs ({keys}) step from StepFactory: {self.__class__.__name__}")

    def create_step(self, in_memory: bool, previous_steps):  # pylint: disable=unused-argument
        """
        Instanciates the step related to the current :class:`StepFactory`,
        that consumes results from the previous `input` steps.

        1. This methods starts by updating metadata information through
        :func:`complete_meta()` on the ``input`` metadatas.

        2. in case the new step isn't related to an OTB application,
        nothing specific is done, we'll just return an :class:`AbstractStep`

        Note: While `previous_steps` is ignored in this specialization, it's
        used in :func:`Store.create_step()` where it's eventually used to
        release all OTB Application objects.
        """
        inputs = self._get_inputs(previous_steps)
        inp    = self._get_canonical_input(inputs)
        meta   = self.complete_meta(inp.meta, inputs)
        self.update_image_metadata(meta, inputs) # Needs to be done after complete_meta!

        # Return previous app?
        return AbstractStep(**meta)


class Pipeline:
    """
    Pipeline of OTB applications.

    It's instanciated as a list of :class:`AbstractStep` s.
    :func:`Step.execute_and_write_output()` will be executed on the last step
    of the pipeline.

    Internal class only meant to be used by :class:`PipelineDescriptionSequence`.
    """
    # Should we inherit from contextlib.ExitStack?
    def __init__(self, do_measure, in_memory, do_watch_ram, name=None, dryrun=False, output=None):
        self.__pipeline     = []
        # self.__do_measure   = do_measure
        self.__in_memory    = in_memory
        self.__do_watch_ram = do_watch_ram
        self.__dryrun       = dryrun
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
    def appname(self):
        """
        Name of the pipeline application(s).
        """
        appname = (self.__name or '|'.join(crt.appname for crt in self.__pipeline))
        return appname

    @property
    def name(self):
        """
        Name of the pipeline.
        It's either user registered or automatically generated from the
        registered :class:`StepFactory` s.
        """
        return f'{self.appname} -> {self.__output} from {self._input_filenames}'

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

    def check_requirements(self):
        """
        Check all the :class:`StepFactory`'s registered in the pipeline can be
        exexuted.

        :return: ``None`` if requirements are fulfilled.
        :return: A message indicating what is missing otherwise, and some
                 context how to fix it.
        """
        sing_plur = {True: 'are', False: 'is'}
        reqs = list(filter(None, (sf.check_requirements() for sf in self.__pipeline)))
        missing_reqs = [rq for rq, _ in reqs]
        contexts = set(ctx for _, ctx in reqs)
        if reqs:
            return f"{' and '.join(missing_reqs)} {sing_plur[len(missing_reqs) > 1]} required.", contexts
        else:
            return None

    def do_execute(self):
        """
        Execute the pipeline.

        1. Makes sure the inputs exist -- unless in dry-run mode
        2. Incrementaly create the steps of the pipeline.
        3. Return the resulting output filename, or the caught errors.
        """
        assert self.__inputs
        logger.info("INPUTS: %s", self.__inputs)
        tested_files = list(Utils.flatten_stringlist(
            [v.out_filename for inp in self.__inputs for _,v in inp.items()]))
        logger.info("Testing whether input files exist: %s", tested_files)
        missing_inputs = list(filterfalse(files_exist, tested_files))
        if len(missing_inputs) > 0 and not self.__dryrun:
            msg = f"Cannot execute {self} as the following input(s) {missing_inputs} do(es)n't exist"
            logger.warning(msg)
            return Outcome(RuntimeError(msg))
        # logger.debug("LOG OTB: %s", os.environ.get('OTB_LOGGER_LEVEL'))
        assert self.__pipeline  # shall not be empty!
        steps = [self.__inputs]
        for crt in self.__pipeline:
            step = crt.create_step(self.__in_memory, steps)
            if step:  # a StepFactory may return no step so it can be skipped
                steps.append([{'__last': step}])

        assert len(steps[-1]) == 1
        res = steps[-1][0]['__last'].out_filename
        assert res == self.output, \
                f"Step output {self.output} doesn't match expected output {res}.\nThis is likely happenning because pipeline name generation isn't incremental."
        steps = None
        # logger.debug('Pipeline "%s" terminated -> %s', self, res)
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
                logger.warning('Cancel execution of %s because an error has occured upstream on a dependent input file: %s', pipeline, arg)
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
        logger.debug('(ERROR) %s has been executed with the following parameters: %s', pipeline, args)
        return Outcome(ex).add_related_filename(pipeline.output).set_pipeline_name(pipeline.appname)


class PipelineDescription:
    """
    Pipeline description:
    - stores the various factory steps that describe a pipeline,
    - can tell the expected product name given an input.
    - tells whether its product is required
    """
    def __init__(self, factory_steps, dryrun, name=None, product_required=False,
            is_name_incremental=False, inputs=None):
        """
        constructor
        """
        assert factory_steps  # shall not be None or empty
        self.__factory_steps       = factory_steps
        self.__is_name_incremental = is_name_incremental
        self.__is_product_required = product_required
        self.__dryrun              = dryrun
        if name:
            self.__name = name
        else:
            self.__name = '|'.join([step.name for step in self.__factory_steps])
        assert inputs
        self.__inputs              = inputs
        # logger.debug("New pipeline: %s; required: %s, incremental: %s", '|'.join([step.name for step in self.__factory_steps]), self.__is_product_required, self.__is_name_incremental)

    def expected(self, input_meta):
        """
        Returns the expected name of the product(s) of this pipeline
        """
        assert self.__factory_steps  # shall not be None or empty
        try:
            if self.__is_name_incremental:
                res = input_meta
                for step in self.__factory_steps:
                    res = step.update_filename_meta(res)
            else:
                res = self.__factory_steps[-1].update_filename_meta(input_meta)
            logger.debug("    expected: %s(%s) -> %s", self.__name, input_meta['out_filename'], out_filename(res))
            return res
        except CannotGenerateFilename as e:  # pylint: disable=broad-except
            # logger.exception('expected(%s) rejected because', input_meta)
            logger.debug('%s => rejecting expected(%s)', e, input_meta)
            return None

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

        Returns:
            A :class:`Pipeline` instance
        """
        pipeline = Pipeline(do_measure, in_memory, do_watch_ram, self.name, self.__dryrun, file)
        need_OTB_store = False
        for factory_step in self.__factory_steps + []:
            pipeline.push(factory_step)
            need_OTB_store = need_OTB_store or isinstance(factory_step, OTBStepFactory)  # TODO: use a dedicated function
        if need_OTB_store:
            pipeline.push(Store('noappname'))
        return pipeline


    def __repr__(self):
        res = f'PipelineDescription: {self.name} ## Sources: {self.sources}'
        return res


def to_dask_key(pathname):
    """
    Generate a simplified graph key name from a full pathname.
    - Strip directory name
    - Replace '-' with '_' as Dask has a special interpretation for '-' in key names.
    """
    return pathname.replace('-', '_')


def register_task(tasks, key, value):
    """
    Register a task named `key` in the right format.
    """
    tasks[key] = value


def generate_first_steps_from_manifests(raster_list, tile_name, dryrun, debug_caches):
    """
    Flatten all rasters from the manifest as a list of :class:`FirstStep`
    """
    inputs = []
    # Log commented and kept for filling in unit tests
    # logger.debug('Generate first steps from: %s', raster_list)
    # for raster, tile_origin in raster_list:
    for raster_info in raster_list:
        raster = raster_info['raster']  # Actually a S1DateAcquisition object...

        manifest = raster.get_manifest()
        for image in raster.get_images_list():
            start = FirstStep(
                    tile_name=tile_name,
                    tile_origin=raster_info['tile_origin'],
                    tile_coverage=raster_info['tile_coverage'],
                    manifest=manifest,
                    basename=image,
                    dryrun=dryrun,
                    debug_caches=debug_caches)
            inputs.append(start.meta)
    return inputs


class TaskInputInfo:
    """
    Abstraction of the input(s) information associated to a particular task.

    Used to merge, or to stack, information about inputs.
    """
    def __init__(self, pipeline):
        """
        constructor
        """
        self.__pipeline     = pipeline
        self._inputs        = {}  # map<source, meta / meta list>

    def add_input(self, origin, input_meta, destination_meta):
        """
        Register a new input to the current task.

        Parameters:
            :origin:           Name of the source type the new input comes from
            :input_meta:       Meta information associated to the new input (could be a list)
            :destination_meta: Meta information associated to the current task

        Several situations are possible:

        - No input has been registered yet => simply register it
        - If current task has a "reduce_inputs_{origin}" key in its meta
          information, => use that function to filter which input is actually
          kept.
          This scenario is usefull in case several sets of inputs permit to
          obtain a same product (e.g. when we don't actually need the data, but
          only the geometry, etc).
        - Otherwise, stack the new input with the previous ones.
        """
        if origin not in self._inputs:
            logger.debug('    add_input[%s # %s]: first time <<-- %s', origin, out_filename(destination_meta), out_filename(input_meta))
            self._inputs[origin] = [input_meta]
            return True
        logger.debug('add_input[%s # %s]: not empty <<-- %s', origin, out_filename(destination_meta), out_filename(input_meta))
        logger.debug('check %s in %s', f'reduce_inputs_{origin}', destination_meta.keys())
        if f'reduce_inputs_{origin}' in destination_meta.keys():
            # logger.debug('add_input[%s]: self.__inputs[%s]= %s <--- %s', origin, origin, self._inputs[origin], destination_meta[f'reduce_inputs_{origin}'](self._inputs[origin] + [input_meta]))
            self._inputs[origin] = destination_meta[f'reduce_inputs_{origin}'](
                    self._inputs[origin] + [input_meta])
            return False
        else:
            self._inputs[origin].append(input_meta)
            return True

    @property
    def pipeline(self):
        """
        Property pipeline
        """
        return self.__pipeline

    @property
    def inputs(self):
        """
        Inputs associated to the task.

        It's organized as a dictionary that associates a source type to a meta or a list of meta
        information.
        """
        return self._inputs

    @property
    def input_task_names(self):
        """
        List of input tasks the current task depends on.
        """
        tns = [get_task_name(meta) for meta in self.input_metas]
        logger.debug('input_task_names(%s) --> %s', self.pipeline.name, tns)
        return tns

    @property
    def input_metas(self):
        """
        List of input meta informations the current task depends on.
        """
        metas = [meta for inputs in self.inputs.values() for meta in inputs]
        return metas

    def __repr__(self):
        res = 'TaskInputInfo:\n- inputs:\n'
        for k, inps in self.inputs.items():
            res += f'  - "{k}":\n'
            for val in inps:
                res += f'    - {val}\n'
        res += f'- pipeline: {self.pipeline}\n'
        return res


def _update_out_filename(updated_meta, with_meta):
    """
    Helper function to update the `out_filename` from metadata.
    Meant to be used metadata associated to products made of several inputs
    like Concatenate.
    """
    if 'update_out_filename' in updated_meta:
        updated_meta['update_out_filename'](updated_meta, with_meta)


def _register_new_input_and_update_out_filename(
        tasks: list,  #: list[TaskInputInfo],  # Require Python 3.7+ ?
        origin, input_meta, new_task_meta, outputs
        ):
    """
    Helper function to register a new input to a :class:`TaskInputInfo` and
    update the current task output filename if required.
    """
    task_name = get_task_name(new_task_meta)
    if isinstance(task_name, list):
        # TODO: correctly handle the case a task produce several filenames
        task_name = task_name[0]
    task_inputs = tasks[task_name]
    if task_inputs.add_input(origin, input_meta, new_task_meta):
        logger.debug('    The %s task depends on one more input, updating its metadata to reflect the situation.\nUpdating %s ...', task_name, new_task_meta)
        _update_out_filename(new_task_meta, task_inputs)
        logger.debug('    ...to (%s)', new_task_meta)
        already_registered_next_input = [ni for ni in outputs if get_task_name(ni) == task_name]
        assert len(already_registered_next_input) == 1, \
                f'{len(already_registered_next_input)} != 1 => {already_registered_next_input}'
        _update_out_filename(already_registered_next_input[0], task_inputs)
        # Can't we simply override the already_registered_next_input with expected fields?
        already_registered_next_input[0].update(new_task_meta)
    else:
        logger.debug('    The %s task depends on one more input, but only one will be kept.\n    %s has been updated.', task_name, new_task_meta)


class PipelineDescriptionSequence:
    """
    This class is the main entry point to describe pipelines.

    Internally, it can be seen as a list of :class:`PipelineDescription` objects.
    """
    def __init__(self, cfg, dryrun, debug_caches):
        """
        constructor
        """
        assert cfg
        self.__cfg          = cfg
        self.__pipelines    = []
        self.__dryrun       = dryrun
        self.__debug_caches = debug_caches

    def register_pipeline(self, factory_steps, *args, **kwargs):
        """
        Register a pipeline description from:

        Parameters:
            :factory_steps:       List of non-instanciated :class:`StepFactory` classes
            :name:                Optional name for the pipeline
            :product_required:    Tells whether the pipeline product is expected as a
                                  final product
            :is_name_incremental: Tells whether `expected` filename needs evaluations of
                                  each intermediary steps of whether it can be directly
                                  deduced from the last step.
        """
        steps = [FS(self.__cfg) for FS in factory_steps]
        assert 'dryrun' not in kwargs
        if 'inputs' not in kwargs:
            # Register the last pipeline as 'in' if nothing is specified
            kwargs['inputs'] = {'in' : self.__pipelines[-1] if self.__pipelines else 'basename'}
        pipeline = PipelineDescription(steps, self.__dryrun, *args, **kwargs)
        logger.debug('Register pipeline %s as %s', pipeline.name, [fs.__name__ for fs in factory_steps])
        self.__pipelines.append(pipeline)
        return pipeline

    def _build_dependencies(self, tile_name, raster_list):
        """
        Runs the inputs through all pipeline descriptions to build the full list
        of intermediary and final products and what they require to be built.
        """
        first_inputs = generate_first_steps_from_manifests(
                tile_name=tile_name,
                raster_list=raster_list,
                dryrun=self.__dryrun,
                debug_caches=self.__debug_caches)

        pipelines_outputs = {'basename': first_inputs}  # TODO: find the right name _0/__/_firststeps/...?
        logger.debug('FIRST: %s', pipelines_outputs['basename'])

        required = {}  # (first batch) Final products identified as _needed to be produced_
        previous = {}  # Graph of deps: for a product tells how it's produced (pipeline + inputs): Map<TaskInputInfo>
        task_names_to_output_files_table = {}
        # +-> TODO: cache previous in order to remember which files already exists or not
        #     the difficult part is to flag as "generation successful" or not
        for pipeline in self.__pipelines:
            logger.debug('#############################################################################')
            logger.debug('#############################################################################')
            logger.debug('Analysing |%s| dependencies', pipeline.name)
            logger.debug('Sources --> %s', pipeline.sources)
            outputs = []

            dropped_inputs = {}
            for origin, sources in pipeline.inputs.items():
                source_name = sources if isinstance(sources, str) else sources.name
                logger.debug('===========================================================================')
                logger.debug('* Checking sources from "%s" origin: %s', origin, source_name)
                # Locate all inputs for the current pipeline
                # -> Select all inputs for pipeline sources from pipelines_outputs
                inputs = [output for output in pipelines_outputs[source_name]]

                logger.debug('  FROM all %s inputs as "%s": %s', len(inputs), origin, [out_filename(i) for i in inputs])
                dropped = []
                for input in inputs:  # inputs are meta
                    logger.debug('  ----------------------------------------------------------------------')
                    logger.debug('  - GIVEN "%s" "%s": %s', origin, out_filename(input), input)
                    expected = pipeline.expected(input)
                    if not expected:
                        logger.debug("    No '%s' product can be generated from '%s' input '%s' ==> Ignore for now",
                                pipeline.name, origin, out_filename(input))
                        dropped.append(input)  # remember that source/input will be used differently
                        continue
                    expected_taskname = get_task_name(expected)
                    logger.debug('    %s <-- from input: %s', expected_taskname, out_filename(input))
                    logger.debug('    --> "%s": %s', out_filename(expected), expected)
                    # TODO: Correctly handle the case where a task produce
                    # several filenames. In that case we shall have only one
                    # task, but possibly, several following tasks may depend on
                    # the current task.
                    # For the moment, just keep the first, and use product
                    # selection pattern as in filter_LIA().
                    if isinstance(expected_taskname, list):
                        expected_taskname = expected_taskname[0] # TODO: see comment above

                    # We cannot analyse early whether a task product is already
                    # there as some product have names that depend on all
                    # inputs (see Concatenate).
                    # This is why the full dependencies tree is produced at
                    # this time. Unrequired parts will be trimmed in the next
                    # task producing step.
                    if expected_taskname not in previous:
                        outputs.append(expected)
                        previous[expected_taskname] = TaskInputInfo(pipeline=pipeline)
                        previous[expected_taskname].add_input(origin, input, expected)
                        logger.debug('    This is a new product: %s, with a source from "%s"', expected_taskname, origin)
                    elif get_task_name(input) not in previous[expected_taskname].input_task_names:
                        _register_new_input_and_update_out_filename(
                                tasks=previous,
                                origin=origin,
                                input_meta=input,
                                new_task_meta=expected,
                                outputs=outputs)
                    if pipeline.product_is_required:
                        # assert (expected_taskname not in required) or (required[expected_taskname] == expected)
                        required[expected_taskname] = expected
                    task_names_to_output_files_table[expected_taskname] = out_filename(expected)
                # endfor input in inputs:  # inputs are meta
                if dropped:
                    dropped_inputs[origin] = dropped
            # endfor origin, sources in pipeline.inputs.items():

            # For all new outputs, check which dropped inputs would be compatible
            logger.debug('* Checking dropped inputs: %s', dropped_inputs.keys())
            for output in outputs:
                for origin, inputs in dropped_inputs.items():
                    for input in inputs:
                        logger.debug("  - Is '%s' a '%s' input for '%s' ?", out_filename(input), origin, out_filename(output))
                        if _is_compatible(output, input):
                            logger.debug('    => YES')
                            _register_new_input_and_update_out_filename(
                                    tasks=previous,
                                    origin=origin,
                                    input_meta=input,
                                    new_task_meta=output,
                                    outputs=outputs)
                        else:
                            logger.debug('  => NO')

            pipelines_outputs[pipeline.name] = outputs

        logger.debug('#############################################################################')
        logger.debug('#############################################################################')
        required_task_names = set()
        for name, meta in required.items():
            logger.debug("check task_name: %s", name)
            if product_exists(meta):
                logger.debug("Ignoring %s as the product already exists", name)
                previous[name] = False  # for the next log
            else:
                required_task_names.add(name)

        logger.debug("Dependencies found:")
        for task_name, prev in previous.items():
            if prev:
                logger.debug('- %s requires %s on %s', task_name, prev.pipeline.name, [out_filename(i) for i in prev.input_metas])
            else:
                logger.debug('- %s already exists, no need to produce it', task_name)
        return required_task_names, previous, task_names_to_output_files_table

    def _build_tasks_from_dependencies(self,
            required, previous, task_names_to_output_files_table,
            do_watch_ram):  # pylint: disable=no-self-use
        """
        Generates the actual list of tasks for :func:`dask.client.get()`.

        `previous` is made of:
        - "pipeline": reference to the :class:`PipelineDescription`
        - "inputs": list of the inputs (metadata)
        """
        tasks = {}
        logger.debug('#############################################################################')
        logger.debug('#############################################################################')
        logger.debug('Building all tasks')
        while required:
            new_required = set()
            for task_name in required:
                assert previous[task_name]
                base_task_name = to_dask_key(task_name)
                task_inputs    = previous[task_name].inputs
                pipeline_descr = previous[task_name].pipeline
                first = lambda files : files[0] if isinstance(files, list) else  files
                input_task_keys = [to_dask_key(first(tn))
                        for tn in previous[task_name].input_task_names]
                assert list(input_task_keys)
                logger.debug('* %s(%s) --> %s', task_name, list(input_task_keys), task_inputs)
                output_filename = task_names_to_output_files_table[task_name]
                pipeline_instance = pipeline_descr.instanciate(
                        output_filename, True, True, do_watch_ram)
                pipeline_instance.set_inputs(task_inputs)
                logger.debug('  ~~> TASKS[%s] += %s(keys=%s)', base_task_name, pipeline_descr.name, list(input_task_keys))
                register_task(
                        tasks, base_task_name, (execute4dask, pipeline_instance, input_task_keys))

                for t in previous[task_name].input_metas:  # TODO: check whether the inputs need to be produced as well
                    tn = first(get_task_name(t))
                    logger.debug('  Processing task %s: %s', tn, t)
                    if not product_exists(t):
                        logger.info('    => Need to register production of %s (for %s)', tn, pipeline_descr.name)
                        new_required.add(tn)
                    else:
                        logger.info('    => Starting %s from existing %s', pipeline_descr.name, tn)
                        register_task(tasks, to_dask_key(tn), FirstStep(**t))
            required = new_required
        return tasks

    def _check_static_task_requirements(self, tasks):
        """
        Check all tasks have their requirement fulfilled for being generated.
        Typically that the related applications are installed and can be
        executed.

        If any requirement is missing, the execution is stopped.
        :todo: throw an exception instead of existing the process. See #96
        """
        logger.debug('#############################################################################')
        logger.debug('#############################################################################')
        logger.debug('Checking tasks static dependencies')
        missing_apps = {}
        contexts = set()
        for key, task in tasks.items():
            if isinstance(task, tuple):
                assert isinstance(task[1], Pipeline)
                req_ctx = task[1].check_requirements()
                if req_ctx:
                    req, ctx = req_ctx
                    if req not in missing_apps:
                        missing_apps[req] = []
                    missing_apps[req].append(key)
                    contexts.update(ctx)
            else:
                assert isinstance(task, FirstStep)
        if missing_apps:
            logger.error('Cannot execute S1Tiling because of the following reason(s):')
            for req, task_keys in missing_apps.items():
                logger.error("- %s for %s", req, task_keys)
            for ctx in contexts:
                logger.error(" -> %s", ctx)
            sys.exit(exits.MISSING_APP)
        else:
            logger.debug('All required applications are correctly available')


    def generate_tasks(self, tile_name, raster_list, do_watch_ram=False):
        """
        Generate the minimal list of tasks that can be passed to Dask

        Parameters:
            :tile_name:   Name of the current S2 tile
            :raster_list: List of rasters that intersect the tile.

        TODO: Move into another dedicated class instead of PipelineDescriptionSequence
        """
        required, previous, task_names_to_output_files_table = self._build_dependencies(
                tile_name=tile_name,
                raster_list=raster_list)

        # Generate the actual list of tasks
        final_products = [to_dask_key(p) for p in required]
        tasks = self._build_tasks_from_dependencies(
                required=required,
                previous=previous,
                task_names_to_output_files_table=task_names_to_output_files_table,
                do_watch_ram=do_watch_ram)
        self._check_static_task_requirements(tasks)

        for final_product in final_products:
            assert final_product in tasks
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
        return f'FirstStep{self._meta}'

    def __repr__(self):
        return f'FirstStep{self._meta}'

    @property
    def input_metas(self):
        """
        Specific to :class:`MergeStep` and :class:`FirstStep`: returns the
        metas from the inputs as a list.
        """
        return [self._meta]


class MergeStep(AbstractStep):
    """
    Kind of FirstStep that merges the result of one or several other steps
    of same kind.

    Used in input of :class:`Concatenate`

    - no application executed
    """
    def __init__(self, input_steps_metas, *argv, **kwargs):
        # meta = {**(input_steps_metas[0]._meta), **kwargs}  # kwargs override step0.meta
        meta = {**(input_steps_metas[0]), **kwargs}  # kwargs override step0.meta
        super().__init__(*argv, **meta)
        self.__input_steps_metas = input_steps_metas
        self._meta['out_filename'] = [out_filename(s) for s in input_steps_metas]

    def __str__(self):
        return f'MergeStep{self.__input_steps_metas}'

    def __repr__(self):
        return f'MergeStep{self.__input_steps_metas}'

    @property
    def input_metas(self):
        """
        Specific to :class:`MergeStep` and :class:`FirstStep`: returns the
        metas from the inputs as a list.
        """
        return self.__input_steps_metas


class StoreStep(_StepWithOTBApplication):
    """
    Artificial Step that takes care of executing the last OTB application in the
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
        return tmp_filename(self.meta)

    @property
    def shall_store(self):
        return True

    def set_out_parameters(self):
        """
        Takes care of setting all output parameters.
        """
        p_out = as_list(self.param_out)
        files = as_list(self.tmp_filename)
        assert self._app
        for po, tmp in zip(p_out, files):
            assert isinstance(po, str), f"String expected for param_out={po}"
            assert isinstance(tmp, str), f"String expected for output tmp filename={tmp}"
            self._app.SetParameterString(po, tmp + out_extended_filename_complement(self.meta))

    def execute_and_write_output(self):
        """
        Specializes :func:`execute_and_write_output()` to actually execute the
        OTB pipeline.
        """
        assert self._app
        do_measure = True  # TODO
        # logger.debug('meta pipe: %s', self.meta['pipe'])
        pipeline_name = '%s > %s' % (' | '.join(str(e) for e in self.meta['pipe']), self.out_filename)
        if files_exist(self.out_filename):
            # This is a dirty failsafe, instead of analysing at the last
            # moment, it's be better to have a clear idea of all dependencies
            # and of what needs to be done.
            logger.info('%s already exists. Aborting << %s >>', self.out_filename, pipeline_name)
            return
        with Utils.ExecutionTimer('-> pipe << ' + pipeline_name + ' >>', do_measure):
            if not is_running_dry(self.meta):
                # TODO: catch execute failure, and report it!
                # logger.info("START %s", pipeline_name)
                with Utils.RedirectStdToLogger(logging.getLogger('s1tiling.OTB')):
                    # For OTB application execution, redirect stdout/stderr
                    # messages to s1tiling.OTB
                    self.set_out_parameters()
                    self._app.ExecuteAndWriteOutput()
                self._write_image_metadata()
                commit_execution(self.tmp_filename, self.out_filename)
        if 'post' in self.meta and not is_running_dry(self.meta):
            for hook in self.meta['post']:
                # Note: we can't extract and pass meta-data around from this hook
                # Indeed the hook is executed at Store Factory level, while metadata
                # are passed around between around Factories and Steps.
                logger.debug("Execute post-hook for %s", self.out_filename)
                hook(self.meta, self.app)
        self.clean_cache()
        self.meta['pipe'] = [self.out_filename]


def commit_execution(tmp_fn, out_fn):
    """
    Concluding step that validates the successful execution of an application,
    whether it's an OTB application or an external executable.

    - Rename the tmp image into its final name
    - Rename the associated geom file (if any as well)
    """
    assert type(tmp_fn) == type(out_fn)
    if isinstance(out_fn, list):
        for t, o in zip(tmp_fn, out_fn):
            commit_execution(t, o)
        return
    logger.debug('Renaming: mv %s %s', tmp_fn, out_fn)
    shutil.move(tmp_fn, out_fn)
    tmp_geom = re.sub(re_tiff, '.geom', tmp_fn)
    if os.path.isfile(tmp_geom):
        out_geom = re.sub(re_tiff, '.geom', out_fn)
        logger.debug('Renaming: mv %s %s', tmp_geom, out_geom)
        shutil.move(tmp_geom, out_geom)
    logger.debug('-> %s renamed as %s', tmp_fn, out_fn)
    assert not os.path.isfile(tmp_fn)
    assert os.path.isfile(out_fn)


class _FileProducingStepFactory(StepFactory):
    """
    Abstract class that factorizes filename transformations and parameter
    handling for Steps that produce files, either with OTB or through external
    calls.

    :func:`create_step` is kind of *abstract* at this point.
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

    def parameters(self, meta):
        """
        Most steps that produce files will expect parameters.

        Warning: In :class:`ExecutableStepFactory`, parameters that designate
        output filenames are expected to use :func:`tmp_filename` and not
        :func:`out_filename`. Indeed products are meant to be first produced
        with temporary names before being renamed with their final names, once
        the operation producing them has succeeded.

        Note: This method is kind-of abstract --
        :class:`SelectBestCoverage <s1tiling.libs.otbwrappers.SelectBestCoverage>` is a
        :class:`_FileProducingStepFactory` but, it doesn't actualy consume parameters.
        """
        raise TypeError(f"An {self.__class__.__name__} step don't produce anything!")

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
        return str(self.__gen_output_dir).format(**meta)

    def _get_nominal_output_basename(self, meta):
        """
        Returns the pathless basename of the produced file (internal).
        """

        return self.__gen_output_filename.generate(meta['basename'], meta)

    def build_step_output_filename(self, meta):
        """
        Returns the names of typical result files in case their production
        is required (i.e. not in-memory processing).

        This specialization uses ``gen_output_filename`` naming policy
        parameter to build the output filename. See the `Available naming
        policies`_.
        """
        filename = self._get_nominal_output_basename(meta)
        in_dir = lambda fn : os.path.join(self.output_directory(meta), fn)
        if isinstance(filename, str):
            return in_dir(filename)
        else:
            return [in_dir(fn) for fn in filename]

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
        add_tmp = lambda fn : os.path.join(
                self.tmp_directory(meta), re.sub(re_any_ext, r'.tmp\g<0>', fn))
        if isinstance(filename, str):
            return add_tmp(filename)
        else:
            return [add_tmp(fn) for fn in filename]

    @property
    def ram_per_process(self):
        """
        Property ram_per_process
        """
        return self.__ram_per_process


class OTBStepFactory(_FileProducingStepFactory):
    """
    Abstract StepFactory for all OTB Applications.

    All step factories that wrap OTB applications are meant to inherit from
    :class:`OTBStepFactory`.
    """
    def __init__(self, cfg,
            appname,
            gen_tmp_dir, gen_output_dir, gen_output_filename,
            *argv, **kwargs):
        """
        Constructor.

        See:
            :func:`_FileProducingStepFactory.__init__`

        Parameters:
            :param_in:  Flag used by the default OTB application for the input file (default: "in")
            :param_out: Flag used by the default OTB application for the ouput file (default: "out")
        """
        super().__init__(cfg, gen_tmp_dir, gen_output_dir, gen_output_filename, *argv, **kwargs)
        # is_a_final_step = gen_output_dir and gen_output_dir != gen_tmp_dir
        # logger.debug("%s -> final: %s <== gen_tmp=%s    gen_out=%s", self.name, is_a_final_step, gen_tmp_dir, gen_output_dir)

        self._in                   = kwargs.get('param_in',  'in')
        self._out                  = kwargs.get('param_out', 'out')
        # param_in is only used in connected mode. As such a string is expected.
        assert self.param_in  is None or isinstance(self.param_in, str), f"String expected for {appname} param_in={self.param_in}"
        # param_out is always used.
        assert isinstance(self.param_out, (str, list)), f"String or list expected for {appname} param_out={self.param_out}"
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
        Default is likely to be "in", while some applications use "io.in", often "il" for list of
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

    def set_output_pixel_type(self, app, meta):
        """
        Permits to have steps force the output pixel data.
        Does nothing by default.
        Override this method to change the output pixel type.
        """
        pass

    def create_step(self, in_memory: bool, previous_steps):
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

        Note: it's possible to override this method to return no step
        (``None``). In that case, no OTB Application would be registered in
        the actual :class:`Pipeline`.
        """
        inputs = self._get_inputs(previous_steps)
        inp    = self._get_canonical_input(inputs)
        meta   = self.complete_meta(inp.meta, inputs)
        self.update_image_metadata(meta, inputs) # Needs to be done after complete_meta!
        assert self.appname

        # Otherwise: step with an OTB application...
        if is_running_dry(meta):
            logger.warning('DRY RUN mode: ignore step and OTB Application creation')
            lg_from = inp.out_filename if inp.is_first_step else 'app'
            logger.debug('Register app: %s (from %s) %s', self.appname, lg_from, ' '.join('-%s %s' % (k, as_app_shell_param(v)) for k, v in self.parameters(meta).items()))
            meta['param_out'] = self.param_out
            return Step('FAKEAPP', **meta)
        with Utils.RedirectStdToLogger(logging.getLogger('s1tiling.OTB')):
            # For OTB application execution, redirect stdout/stderr messages to s1tiling.OTB
            app = otb.Registry.CreateApplication(self.appname)
            if not app:
                raise RuntimeError("Cannot create OTB application '" + self.appname + "'")
            parameters = self.parameters(meta)
            if inp.is_first_step:
                if not files_exist(inp.out_filename):
                    logger.critical("Cannot create OTB pipeline starting with %s as some input files don't exist (%s)", self.appname, inp.out_filename)
                    raise RuntimeError(f"Cannot create OTB pipeline starting with {self.appname} as some input files don't exist ({inp.out_filename})")
                # parameters[self.param_in] = inp.out_filename
                lg_from = inp.out_filename
            else:
                assert isinstance(self.param_in, str), f"String expected for {self.param_in}"
                assert isinstance(inp.param_out, str), f"String expected for {self.param_out}"
                app.ConnectImage(self.param_in, inp.app, inp.param_out)
                this_step_is_in_memory = in_memory and not inp.shall_store
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
                logger.exception("Cannot set parameters to %s (from %s) %s", self.appname, lg_from, ' '.join('-%s %s' % (k, as_app_shell_param(v)) for k, v in parameters.items()))
                raise

        meta['param_out'] = self.param_out
        return Step(app, **meta)

    def check_requirements(self):
        """
        This specialization of :func:`check_requirements` checks whether the
        related OTB application can correctly be executed from S1Tiling.

        :return: A pair of the message indicating what is required, and some
                 context how to fix it -- by default: install OTB!
        :return: ``None`` otherwise.
        """
        app = otb.Registry.CreateApplication(self.appname)
        if not app:
            return f"{self.appname}", self.requirement_context()
        else:
            app = None
            return None

    def requirement_context(self):
        """
        Return the requirement context that permits to fix missing requirements.
        By default, OTB applications requires... OTB!
        """
        return "Please install OTB."


class ExecutableStepFactory(_FileProducingStepFactory):
    """
    Abstract StepFactory for executing any external program.

    All step factories that wrap OTB applications are meant to inherit from
    :class:`ExecutableStepFactory`.
    """
    def __init__(self, cfg,
            exename,
            gen_tmp_dir, gen_output_dir, gen_output_filename,
            *argv, **kwargs):
        """
        Constructor

        See:
            :func:`_FileProducingStepFactory.__init__`
        """
        super().__init__(cfg, gen_tmp_dir, gen_output_dir, gen_output_filename, *argv, **kwargs)
        self._exename              = exename
        logger.debug("new ExecutableStepFactory(%s) -> exe=%s", self.name, exename)

    def create_step(self, in_memory: bool, previous_steps):
        """
        This Step creation method does more than just creating the step.
        It also executes immediately the external process.
        """
        logger.debug("Directly execute %s step", self.name)
        inputs = self._get_inputs(previous_steps)
        inp    = self._get_canonical_input(inputs)
        meta   = self.complete_meta(inp.meta, inputs)
        self.update_image_metadata(meta, inputs) # Needs to be done after complete_meta!
        res    = ExecutableStep(self._exename, **meta)
        parameters = self.parameters(meta)
        res.execute_and_write_output(parameters)
        return res


class Store(StepFactory):
    """
    Factory for Artificial Step that forces the result of the previous app
    sequence to be stored on disk by breaking in-memory connection.

    While it could be used manually, it's meant to be automatically appended
    at the end of a pipeline if any step is actually related to OTB.
    """
    def __init__(self, appname, *argv, **kwargs):  # pylint: disable=unused-argument
        super().__init__('(StoreOnFile)', "(StoreOnFile)", *argv, **kwargs)
        # logger.debug('Creating Store Factory: %s', appname)

    def create_step(self, in_memory: bool, previous_steps):
        """
        Specializes :func:`StepFactory.create_step` to trigger
        :func:`StoreStep.execute_and_write_output` on the last step that
        relates to an OTB Application.
        """
        inputs = self._get_inputs(previous_steps)
        inp    = self._get_canonical_input(inputs)
        if inp.is_first_step:
            # assert False  # Should no longer happen, yet it does w/ ConcatLIA...
            # Special case of by-passed inputs
            meta = inp.meta.copy()
            return AbstractStep(**meta)

        # logger.debug('Creating StoreStep from %s', inp)
        res = StoreStep(inp)
        try:
            res.execute_and_write_output()
        finally:
            # logger.debug("Collecting memory!")
            # Collect memory now!
            res.release_app()
            for inps in previous_steps:
                for inp in inps:
                    for _, step in inp.items():
                        step.release_app()
        return res

    # abstract methods...

    def build_step_output_filename(self, meta):
        raise TypeError("No way to ask for the output filename of a Store Factory")

    def build_step_output_tmp_filename(self, meta):
        raise TypeError("No way to ask for the output temporary filename of a Store Factory")


# ======================================================================
# Multi processing related (old) code
def mp_worker_config(queue):
    """
    Worker configuration function called by Pool().

    It takes care of initializing the queue handler in the subprocess.

    Parameters:
        :queue: multiprocessing.Queue used for passing logging messages from worker to main
            process.
    """
    qh = logging.handlers.QueueHandler(queue)
    global logger
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
            pipeline.set_inputs(startpoint)
            for factory in self.__factory_steps:
                pipeline.push(factory(self.__cfg))

        logger.debug('Launch pipelines')
        pool.process()
