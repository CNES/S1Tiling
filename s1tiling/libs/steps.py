#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2025 (c) CNES.
#   Copyright 2022-2024 (c) CS GROUP France.
#
#   This file is part of S1Tiling project
#       https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
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
This module defines roots steps upon which all are defined
"""

import os
import shutil
import re
import datetime
from abc import ABC, abstractmethod
import fnmatch
import logging
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, NoReturn, Optional, Set, Tuple, Union

from osgeo import gdal
import otbApplication as otb


from .              import Utils
from .configuration import FileProducingConfiguration
from .file_naming   import OutputFilenameGenerator
from .meta          import (
    Meta,
    check_several_products,
    is_debugging_caches,
    is_running_dry,
    output_parameter,
    tmp_filename,
    out_filename,
    out_extended_filename_complement,
)
from .otbtools      import otb_version
from .utils.timer   import ExecutionTimer
from .utils.path    import files_exist

from ..__meta__     import __version__

logger = logging.getLogger('s1tiling.rootsteps')

re_tiff    = re.compile(r'\.tiff?$')
re_any_ext = re.compile(r'\.[^.]+$')  # Match any kind of file extension

InputList     = List[Dict[str, "AbstractStep"]]
OTBParameters = Dict[str, Union[str, int, float, bool, List[str]]]
ExeParameters = List[str]


# Disable the log warning about exception and GDAL.
gdal.UseExceptions()


def ram(r) -> Union[int, str]:
    """
    The expected type for the RAM parameter in OTB application changes between OTB 7.x and OTB 8.0.
    This function provides an abstraction that takes care of the exact type expected.
    """
    if otb_version() >= '8.0.0':
        assert isinstance(r, int)
        return r
    else:
        return str(r)


def as_list(param) -> List:
    """
    Make sure ``param`` is either a list or encapsulated in a list.
    """
    if isinstance(param, list):
        return param
    else:
        return [param]


def as_app_shell_param(param) -> str:
    """
    Internal function used to stringigy value to appear like a a parameter for a program launched
    through shell.

    foo     -> 'foo'
    42      -> 42
    [a, 42] -> 'a' 42

    :todo: Deprecate, use f"{param!r}" instead
    """
    return f"{param!r}"


def manifest_to_product_name(manifest: str) -> str:
    """
    Helper function that returns the product name (SAFE directory without the ``.SAFE`` extension)
    from the full path to the :file:`manifest.safe` file.

    Works with eodag v2 returned paths:

    >>> manifest1 = 'data_raw2/S1A_IW_GRDH_1SDV_20201228T060102_20201228T060127_035882_0433B6_25C7/S1A_IW_GRDH_1SDV_20201228T060102_20201228T060127_035882_0433B6_25C7.SAFE/manifest.safe'
    >>> manifest_to_product_name(manifest1)
    'S1A_IW_GRDH_1SDV_20201228T060102_20201228T060127_035882_0433B6_25C7'

    And the paths returned by eodag v3:

    >>> manifest2 = 'data_raw2/S1A_IW_GRDH_1SDV_20201228T060102_20201228T060127_035882_0433B6_25C7/manifest.safe'
    >>> manifest_to_product_name(manifest2)
    'S1A_IW_GRDH_1SDV_20201228T060102_20201228T060127_035882_0433B6_25C7'
    """
    fullpath = Path(manifest)
    return fullpath.parent.stem


def commit_execution(tmp_fn, out_fn) -> None:
    """
    Concluding step that validates the successful execution of an application, whether it's an OTB
    application or an external executable.

    - Rename the tmp image into its final name
    - Rename the associated geom file (if any as well)
    """
    assert type(tmp_fn) is type(out_fn), f"{tmp_fn=!r}  <--> {out_fn=!r}"

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


def execute(params: List[str], dryrun: bool) -> None:
    """
    Helper function to execute any external command.

    And log its execution, measure the time it takes.
    """
    msg = ' '.join([f"{p!r}" for p in params])
    logging.info(f'$> {msg}')
    if not dryrun:
        with ExecutionTimer(msg, True):
            subprocess.run(args=params, check=True)


class AbstractStep:
    """
    Internal root class for all actual `steps`.

    There are several kinds of steps:

    - :class:`FirstStep` that contains information about input files
    - :class:`Step` that registers an otbapplication binding
    - :class:`StoreStep` that momentarilly disconnect on-memory pipeline to force storing of the
      resulting file.
    - :class:`AnyProducerStep` that executes Python functions
    - :class:`ExecutableStep` that executes external applications
    - :class:`MergeStep` that operates a rendez-vous between several steps producing files of a same
      kind.

    The step will contain information like the current input file, the current output file… and
    variation points starting in ``_do_something()`` to specialize by overriding them in child
    classes.
    """

    def __init__(self, *unused_argv, **kwargs) -> None:
        meta = kwargs
        if 'basename' not in meta:
            logger.critical('no "basename" in meta == %s', meta)
        assert 'basename' in meta
        # Clear basename from any noise
        self._meta = meta

    @property
    def is_first_step(self) -> bool:
        """Tells whether this step is the first of a pipeline."""
        return True

    @property
    def meta(self) -> Meta:
        """Step meta data property."""
        return self._meta

    @property
    def basename(self) -> str:
        """Basename property will be used to generate all future output filenames."""
        return self._meta['basename']

    @property
    def out_filename(self) -> str:
        """Property that returns the name of the file produced by the current step."""
        assert 'out_filename' in self._meta
        return self._meta['out_filename']

    @property
    def shall_store(self) -> bool:
        """
        No OTB related step requires its result to be stored on disk and to break in_memory
        connection by default.

        However, the artificial Step produced by :class:`Store` factory will force the result of the
        `previous` application(s) to be stored on disk.
        """
        return False

    def release_app(self) -> None:
        """
        Makes sure that steps with applications are releasing the application (no-op for this class)
        """
        pass


class _ProducerStep(AbstractStep):
    """
    Root class for all Steps that produce files
    """

    @property
    def tmp_filename(self) -> str:
        """
        Property that returns the name of the file produced by the current step while the OTB
        application, or the executable, or even the gdal function is running.
        Eventually, it'll get renamed into `self.out_filename` if the application succeeds.
        """
        return tmp_filename(self.meta)

    @property
    def output_parameter(self):
        """
        Property that returns the output parameter to use as output of the application.
        See: :func:`output_parameter`
        """
        return output_parameter(self.meta)

    @property
    def pipeline_name(self):
        """Generate a name for the associated pipeline"""
        pipe = ' | '.join(str(e) for e in self.meta['pipe'])
        return f'{pipe} > {out_filename!r}'

    def execute_and_write_output(self, parameters, execution_parameters: Dict) -> None:
        """
        Actually produce the expected output. The how is still a variation point that'll get decided
        in :func:`_do_execute` specializations.

        While the output is produced, a temporary filename will be used as output.
        On successful execution, the output will be renamed to match its expected final name.
        """
        dryrun = is_running_dry(execution_parameters)
        logger.debug("_ProducerStep: %s (%s)", self.__class__.__name__, self.meta)
        do_measure = True  # TODO
        pipeline_name = self.pipeline_name
        if files_exist(self.out_filename):
            # This is a dirty failsafe, instead of analysing at the last moment, it's be better to
            # have a clear idea of all dependencies and of what needs to be done.
            logger.info('%s already exists. Aborting << %s >>', self.out_filename, pipeline_name)
            return
        with ExecutionTimer(f'-> pipe << {pipeline_name} >>', do_measure, logging.DEBUG):
            self._do_execute(parameters, dryrun)
            self._write_image_metadata(dryrun)
            if not dryrun:
                # TODO: catch execute failure, and report it!
                # logger.info("START %s", pipeline_name)
                commit_execution(self.tmp_filename, self.out_filename)
        if 'post' in self.meta and not dryrun:
            for hook in self.meta['post']:
                # Note: we can't extract and pass meta-data around from this hook
                # Indeed the hook is executed at Store Factory level, while metadata are passed
                # around between around Factories and Steps.
                logger.debug("Execute post-hook for %s", self.out_filename)
                self._do_call_hook(hook)
        self._clean_cache(dryrun, is_debugging_caches(execution_parameters))
        self.meta['pipe'] = [self.out_filename]

    @abstractmethod
    def _do_execute(self, parameters, dryrun: bool) -> None:
        """
        Variation point that takes care of the actual production.

        :meta public:
        """
        pass

    def _do_call_hook(self, hook: Callable) -> None:
        """
        Variation point that takes care to execute hooks.

        :meta public:
        """
        hook(self.meta)

    def _clean_cache(self, dryrun: bool, debug_caches: bool) -> None:
        """
        Takes care or removing intermediary files once we know they are no longer required like the
        orthorectified subtiles once the concatenation has been done.
        """
        if 'files_to_remove' in self.meta :
            files = self.meta['files_to_remove']
            # All possible geom files that may exist
            geoms = [re.sub(re_tiff, '.geom', fn) for fn in files if '.tif' in files]
            # All geoms that do actually exist
            geoms = [fn for fn in geoms if os.path.isfile(fn)]
            files = files + geoms
            if debug_caches:
                logger.debug('NOT cleaning intermediary files: %s (cache debugging mode!)', files)
            else:
                logger.debug('%sCleaning intermediary files: %s used for  %s', "(FAKE) " if dryrun else "", files, self.out_filename)
                if not dryrun:
                    Utils.remove_files(files)
            self.meta.pop('files_to_remove', None)

    def _write_image_metadata(self, dryrun: bool) -> None:
        """
        Update Image metadata (with GDAL API).
        Fetch the new content in ``meta['image_metadata']``

        .. precondition:: Call from non dryrun mode only
        """
        img_meta = self.meta.get('image_metadata', {})
        # fullpath = out_filename(self.meta)
        fullpath = self.tmp_filename
        if not img_meta:
            logger.debug('No metadata to update in %s', fullpath)
            return

        def do_log(fullpath, img_meta: Dict[str, Union[str, List[str]]], idx: int = -1) -> None:  # pragma: no cover
            logger.debug('(dryrun) Set metadata in %s', fullpath)
            for kw, val in img_meta.items():
                if isinstance(val, list):
                    assert 0 <= idx < len(val)
                    val = val[idx]
                logger.debug('(dryrun)  - %s -> %s', kw, val)
            logger.debug('(dryrun) Metadata Set! (%s)', fullpath)

        def do_write(fullpath, img_meta: Dict[str, Union[str, List[str]]], idx: int = -1) -> None:
            logger.debug('Set metadata in %s', fullpath)
            if not img_meta:
                return  # Nothing to update

            dst = gdal.Open(fullpath, gdal.GA_Update)
            assert dst
            all_metadata = dst.GetMetadata()

            def set_or_del(key: str, val: str):
                if val:
                    all_metadata[key] = val
                else:
                    all_metadata.pop(key, None)

            for kw, val in img_meta.items():
                logger.debug(' - %s -> %s', kw, val)
                if kw.endswith('*'):
                    assert isinstance(val, str), f'GDAL metadata shall be strings. "{kw}" is a {val.__class__.__name__} (="{val}")'
                    if not val:  # Expected scenario: we clear the keys.*
                        all_metadata = {m: all_metadata[m] for m in all_metadata if not fnmatch.fnmatch(m, kw)}
                    else:        # Unlikely scenario: new & same value for all
                        updated_kws = {m: val for m in all_metadata if fnmatch.fnmatch(m, kw)}
                        all_metadata.update(updated_kws)
                else:
                    if isinstance(val, list):
                        assert 0 <= idx < len(val)
                        val = val[idx]
                    assert isinstance(val, str), f'GDAL metadata shall be strings. "{kw}" is a {val.__class__.__name__} (="{val}")'
                    set_or_del(kw, val)

            dst.SetMetadata(all_metadata)
            dst.FlushCache()  # We really need to be sure it has been flushed now, if not closed
            del dst
            logger.debug('Metadata Set! (%s)', fullpath)

        do_apply = do_log if dryrun else do_write
        if isinstance(fullpath, list):
            # Case of applications that produce several files like ComputeLIA
            for idx, fp in enumerate(fullpath):
                # TODO: how to specialize DESCRIPTION for each output image
                do_apply(fp, img_meta, idx)
        else:
            do_apply(fullpath, img_meta)


class AnyProducerStep(_ProducerStep):
    """
    Generic step for running any Python code that produce files.

    Implicitly created by :class:`AnyProducerStepFactory`.
    """

    def __init__(self, action: Callable, *argv, **kwargs) -> None:
        super().__init__(None, *argv, **kwargs)
        self._action = action
        # logger.debug('AnyProducerStep %s constructed', self._exename)

    def _do_execute(self, parameters, dryrun: bool) -> None:
        """
        Takes care of executing the action stored as a function to call.

        :meta public:
        """
        self._action(parameters, dryrun)


class ExecutableStep(_ProducerStep):
    """
    Generic step for calling any external application.

    Implicitly created by :class:`ExecutableStepFactory`.
    """

    def __init__(self, exename: str, *argv, **kwargs) -> None:
        super().__init__(None, *argv, **kwargs)
        self._exename = exename
        # logger.debug('ExecutableStep %s constructed', self._exename)

    def _do_execute(self, parameters, dryrun: bool) -> None:
        """
        Takes care of executing the external program.

        :meta public:
        """
        execute([self._exename] + parameters, dryrun)


class _OTBStep(AbstractStep):
    """
    Step that have a reference to an OTB application.
    It could be an actual :class:`Step` holding an OTB application, or a :class:`SkippedStep` that
    forwards the OTB application from its previous step in the pipeline.

    **Note**: Both child classes are virtually the same. Yet, different types are used in order to
    really distinguish what is registered and executed.
    """

    def __init__(self, app, *argv, **kwargs) -> None:
        # logger.debug("Create Step(%s, %s)", app, meta)
        super().__init__(app, *argv, **kwargs)
        self._app = app
        self._out = kwargs.get('param_out', 'out')

    def release_app(self) -> None:
        """
        Makes sure that steps with applications are releasing the application!
        """
        self._app = None

    @property
    def app(self):
        """OTB Application property."""
        return self._app

    @property
    def is_first_step(self) -> bool:
        # TODO: does it make sense for an OTB step to have no application associated???
        return self._app is None

    @property
    def param_out(self) -> Optional[str]:
        """
        Name of the "out" parameter used by the OTB Application.
        Default is likely to be "out", while some applications use "io.out".
        """
        return self._out


class Step(_OTBStep):
    """
    Internal specialized `Step` that holds a binding to an OTB Application.

    The application binding is expected to be built by a dedicated :class:`StepFactory` and passed
    to the constructor.
    """

    # parent __init__ is perfect.

    def __del__(self) -> None:
        """
        Makes sure the otb app is released
        """
        if self._app:
            self.release_app()


class SkippedStep(_OTBStep):
    """
    Kind of OTB Step that forwards the OTB application of the previous step in the
    pipeline.
    """

    def __init__(self, app, *argv, **kwargs) -> None:
        assert app, "SkippedStep needs a valid OTB application to forward from a previous Step"
        super().__init__(app, *argv, **kwargs)


def _check_input_step_type(inputs: InputList) -> None:
    """
    Internal helper function that checks :func:`StepFactory.create_step()` ``inputs`` parameters is
    of the expected type, i.e.: list of dictionaries {'key': :class:`AbstractStep`}
    """
    assert isinstance(inputs, list)
    assert all(issubclass(type(inp), dict) for inp in inputs), f"Inputs not of expected type: {inputs}"
    assert all(issubclass(type(step), AbstractStep) for inp in inputs for _, step in inp.items()), f"Inputs not of expected type: {inputs}"


class StepFactory(ABC):
    """
    Abstract factory for :class:`AbstractStep`

    Meant to be inherited for each possible OTB application, external application... used in a
    pipeline.

    Sometimes we may also want to add some artificial steps that analyse products, filenames..., or
    step that help filter products for following pipelines.

    When steps are analysed, their *output filename(s)* are deduced. This information is stored in
    the *meta* dictionary under the key ``out_filename`` (and it's meant to be extracted through
    :func:`out_filename`). It can be a single filename or a list of filenames. Internally it will be
    used to :meth:`commit_execution` -- i.e. to rename tempory files with their final exact
    filenames. ``out_filename`` computation is supposed to be automatically done by the
    :class:`OutputFilenameGenerator` passed to the constructor of some step factories.

    Also step results need to be precisely identified. This identifier is extracted with
    :func:`get_task_name`. By default its value is the same as ``out_filename``. In some cases, we
    need to override this *task name*. This is meant to be done in
    :meth:`_update_filename_meta_post_hook` exclusively. A typical use case is when a steps
    produced several files. It's better in that case to have a single *task name*.

    At last, sometimes an (OTB) application produces several files, but it only takes a single ouput
    parameter which acts as a kind of filename pattern/format. For these situations we need an
    *output parameter* which is not the list of ``out_filename``s. This can be done through the
    *metadata* key ``output_parameter`` this is retrieved by :func:`output_parameter` helper
    function -- if no ``output_parameter`` information is set, this accessor function falls back to
    ``out_filename`` value. This *metadata* is also meant to be set exclusively in
    :meth:`_update_filename_meta_post_hook`.

    See: :ref:`Existing processings`
    """

    def __init__(self, name: str, *unused_argv, **kwargs) -> None:
        assert isinstance(name, str), f"{self.__class__.__name__} name is a {name.__class__.__name__}, not a string -> {name!r}"
        self._name               = name
        self.__image_description = kwargs.get('image_description', '')
        self.__extra_metadata    = kwargs.get('extra_metadata', {})
        # logger.debug("new StepFactory(%s)", name)

    @property
    def name(self) -> str:
        """Step Name property."""
        assert isinstance(self._name, str), f"Step name is a {self._name.__class__.__name__}, not a string -> {self._name!r}"
        return self._name

    @property
    def image_description(self) -> Union[str, List[str]]:
        """
        Property image_description, used to fill ``TIFFTAG_IMAGEDESCRIPTION``
        """
        return self.__image_description

    def check_requirements(self) -> Optional[Tuple[str, str]]:
        """
        Abstract method used to test whether a :class:`StepFactory` has all its external
        requirements fulfilled. For instance, :class:`OTBStepFactory`'s will check their related OTB
        application can be executed.

        :return: ``None`` if requirements are fulfilled.
        :return: A message indicating what is missing otherwise, and some context how to fix it.
        """
        return None

    @abstractmethod
    def build_step_output_filename(self, meta: Meta) -> Union[str, List[str]]:
        """
        Filename of the step output.

        See also :func:`build_step_output_tmp_filename()` regarding the actual processing.
        """
        pass

    @abstractmethod
    def build_step_output_tmp_filename(self, meta: Meta) -> Union[str, List[str]]:
        """
        Returns a filename to a temporary file to use in output of the current application.

        When an OTB (/External) application is harshly interrupted (crash or user interruption), it
        leaves behind an incomplete (and thus invalid) file.
        In order to ignore those files when a pipeline is restarted, an temporary filename is used
        by the application.
        Once the application exits with success, the file will be renamed into
        :func:`build_step_output_filename()`, and possibly moved into
        :func:`_FileProducingStepFactory.output_directory()` if this is a final product.
        """
        # TODO: Move to _ProducerStep ?
        pass

    def has_several_outputs(self) -> bool:
        """
        Tells whether this step produces several files.
        This method is meant to be overridden in :class:`_FileProducingStepFactory`.

        :return: False by default.
        """
        return False

    def update_filename_meta(self, meta: Meta) -> Meta:  # NOT to be overridden
        """
        Duplicates, completes, and returns, the `meta` dictionary with specific information for the
        current factory regarding tasks analysis.

        This method is used:

        - while analysing the dependencies to build the task graph -- in this use case the relevant
          information are the file names and paths.
        - and indirectly before instanciating a new :class:`Step`

        Other metadata not filled here:

        - :func:`get_task_name` which is deduced from `out_filename`  by default
        - :func:`out_extended_filename_complement`

        It's possible to inject some other metadata (that could be used from
        :func:`_get_canonical_input()` for instance) thanks to
        :func:`_update_filename_meta_pre_hook()`.

        This method is not meant to be overridden. Instead it implements the `template method`
        design pattern, and expects the customization to be done through the specialization of the
        hooks:

        - :func:`_update_filename_meta_pre_hook()`,
        - :func:`_update_filename_meta_post_hook()`.

        """
        meta = meta.copy()
        self._update_filename_meta_pre_hook(meta)

        meta.pop('task_name',                  None)
        meta.pop('task_basename',              None)
        meta.pop('output_parameter',           None)
        meta.pop('update_out_filename',        None)
        meta.pop('accept_as_compatible_input', None)
        meta.pop('does_product_exist',         None)
        # for k in list(meta.keys()):  # Remove all entries associated to reduce_* keys
        #     if k.startswith('reduce_'):
        #         del meta[k]

        meta['in_filename']  = out_filename(meta)
        meta['out_filename'] = self.build_step_output_filename(meta)
        meta['current_step'] = self.__class__.__name__
        meta['pipe']         = meta.get('pipe', []) + [self.__class__.__name__]

        if self.has_several_outputs():
            meta['does_product_exist'] = lambda: check_several_products(out_filename(meta), meta.get('current_step', '??'))
        # else:  # this is already what is done by default
        #     meta['does_product_exist'] = lambda: check_one_product(out_filename(meta), meta.get('current_step', '??'))

        self._update_filename_meta_post_hook(meta)
        return meta

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:  # to be overridden
        """
        Hook meant to be overridden to complete product metadata before they are used to produce
        filenames or tasknames.

        Called from :func:`update_filename_meta()`

        :meta public:
        """
        return meta

    def _update_filename_meta_post_hook(self, meta: Meta) -> None:  # to be overridden
        """
        Hook meant to be overridden to fix product metadata by overriding their default definition.

        Called from :func:`update_filename_meta()`

        :meta public:
        """
        pass

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:  # to be overridden
        """
        Duplicates, completes, and returns, the `meta` dictionary with specific information for the
        current factory regarding :class:`Step` instanciation.
        """
        meta.pop('out_extended_filename_complement', None)
        # logger.debug("OLD inputs (%s): %s", self.__class__.__name__, set().union(*(input.keys() for input in meta.get('inputs', []))))
        meta['inputs'] = all_inputs + meta.get('inputs', [])
        # logger.debug("NEW inputs (%s): %s", self.__class__.__name__, set().union(*(input.keys() for input in meta['inputs'])))
        meta = self.update_filename_meta(meta)  # copy on-the-fly
        meta['out_tmp_filename']   = self.build_step_output_tmp_filename(meta)
        return meta

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:  # pylint: disable=unused-argument
        """
        Root implementation of :func:`update_image_metadata` that shall be specialized in every file
        producing Step Factory.
        """
        if 'image_metadata' not in meta:
            meta['image_metadata'] = {}
        imd = meta['image_metadata']
        imd['TIFFTAG_DATETIME'] = str(datetime.datetime.now().strftime('%Y:%m:%d %H:%M:%S'))
        imd['TIFFTAG_SOFTWARE'] = f'S1 Tiling v{__version__}'
        if self.image_description:
            if isinstance(self.image_description, list):
                imd['TIFFTAG_IMAGEDESCRIPTION'] = [
                    id.format(
                        **meta,
                        flying_unit_code_short=meta.get('flying_unit_code', 'S1?')[1:].upper())
                    for id in self.image_description
                ]
            else:
                imd['TIFFTAG_IMAGEDESCRIPTION'] = self.image_description.format(
                    **meta,
                    flying_unit_code_short=meta.get('flying_unit_code', 'S1?')[1:].upper())
        for key, value in self.__extra_metadata.items():
            imd[key] = value

    def _get_inputs(self, previous_steps: List[InputList]) -> InputList:
        """
        Extract the last inputs to use at the current level from all previous products seen in the
        pipeline.

        This method will need to be overridden in classes like :class:`_ComputeLIA` in order to
        fetch N-1 "xyz" input.

        Postcondition:
            :``_check_input_step_type(result)`` is True
        """
        # By default, simply return the last step information
        assert len(previous_steps) > 0
        inputs = previous_steps[-1]
        _check_input_step_type(inputs)
        return inputs

    def _get_canonical_input(self, inputs: InputList) -> AbstractStep:
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
            # If this error is raised, this means the current step has several inputs, we need to
            # tell explicitely how the "main" input is found.
            keys = set().union(*(input.keys() for input in inputs))
            raise TypeError(f"No way to handle a multiple-inputs ({keys}) step from StepFactory: {self.__class__.__name__}")

    def create_step(
        self,
        execution_parameters: Dict,
        previous_steps: List[InputList]
    ) -> AbstractStep:
        """
        Instanciates the step related to the current :class:`StepFactory`, that consumes results
        from the previous `input` steps.

        1. This methods starts by updating metadata information through: :func:`complete_meta()` on
           the ``input`` metadatas.

        2. Then it updates the GDAL image metadata information that will need to be written in the
           pipeline output image through :func:`update_image_metadata()`.

        3. Eventually the actual step creation method is executed according to the exact kind of
           step factory (:class:`ExecutableStepFactory`, :class:`AnyProducerStepFactory`,
           :class:`OTBStepFactory`) through the variation point :func:`_do_create_actual_step()`.

        While this method is not meant to be overridden, for simplity it will be in :class:`Store`
        factory.

        Note: it's possible to override this method to return no step (``None``). In that case, no
        OTB Application would be registered in the actual :class:`Pipeline`.
        """
        inputs     = self._get_inputs(previous_steps)
        input_step = self._get_canonical_input(inputs)
        meta       = self.complete_meta(input_step.meta, inputs)
        self.update_image_metadata(meta, inputs)  # Needs to be done after complete_meta!
        return self._do_create_actual_step(execution_parameters, input_step, meta)

    def _do_create_actual_step(  # pylint: disable=unused-argument
        self, execution_parameters: Dict, input_step: AbstractStep, meta: Meta
    ) -> AbstractStep:
        """
        Generic variation point for the exact step creation.
        The default implementation returns a new :class:`AbstractStep`.

        :meta public:
        """
        return AbstractStep(**meta)


class StoreStep(_ProducerStep):
    """
    Artificial Step that takes care of executing the last OTB application in the pipeline.
    """

    def __init__(self, previous: _OTBStep) -> None:
        assert not previous.is_first_step
        super().__init__(*[], **previous.meta)
        self._app = previous._app
        self._out = previous.param_out

    @property
    def shall_store(self) -> bool:
        return True

    def _set_out_parameters(self) -> None:
        """
        Takes care of setting all output parameters.
        """
        p_out = as_list(self._out)
        files = as_list(self.output_parameter)
        assert len(p_out) == len(files), f"Mismatching number of files parameters and ouput files: {p_out} VS {files}"
        assert self._app
        nb = len(files)
        ef_meta = out_extended_filename_complement(self.meta)
        extended_filenames = ef_meta if isinstance(ef_meta, list) else nb * [ef_meta]
        assert len(extended_filenames) == nb, f"Mismatching number of files parameters and ouput files+EF: {p_out} VS {files} VS {ef_meta}"
        for po, tmp, ef in zip(p_out, files, extended_filenames):
            assert isinstance(po,  str), f"String expected for param_out={po}"
            assert isinstance(tmp, str), f"String expected for output tmp filename={tmp}"
            logger.debug(" - set ouput param: '%s' = '%s' + '%s'", po, tmp, ef)
            self._app.SetParameterString(po, tmp + ef)

    def _do_execute(self, parameters, dryrun: bool) -> None:
        """
        Takes care of positionning the `out` parameter of the OTB applications pipeline, and trigger
        the execution of the (in-memory, or not) pipeline.

        :meta public:
        """
        assert self._app
        if dryrun:
            return
        with Utils.RedirectStdToLogger(logging.getLogger('s1tiling.OTB')):
            # For OTB application execution, redirect stdout/stderr messages to s1tiling.OTB
            self._set_out_parameters()
            self._app.ExecuteAndWriteOutput()

    def release_app(self) -> None:
        self._app = None

    def _do_call_hook(self, hook: Callable) -> None:
        """
        Specializes hook execution in case of OTB applications: we also pass the otb application.

        :meta public:
        """
        assert self._app
        hook(self.meta, self._app)


# ======================================================================
# Some specific steps
class FirstStep(AbstractStep):
    """
    First step instances are the pipeline staring points.
    They store meta data that'll be used by the :class:`StepFactories
    <s1tiling.libs.steps.StepFactory>` to:

    1. first tell what could be be produced
    2. then to instanciate the :class:`steps <s1tiling.libs.steps.Step>`

    - no application executed
    """

    def __init__(self, *argv, **kwargs) -> None:
        super().__init__(*argv, **kwargs)
        if 'out_filename' not in self._meta:
            # If not set through the parameters, set it from the basename + out dir
            self._meta['out_filename'] = self._meta['basename']
        _, basename = os.path.split(self._meta['basename'])
        self._meta['basename'] = basename
        self._meta['pipe'] = [self._meta['out_filename']]

    def __str__(self) -> str:
        return f'FirstStep{self._meta}'

    def __repr__(self) -> str:
        return f'FirstStep{self._meta}'

    @property
    def input_metas(self) -> List[Meta]:
        """
        Specific to :class:`MergeStep` and :class:`FirstStep`: returns the metas from the inputs as
        a list.
        """
        return [self._meta]


class MergeStep(AbstractStep):
    """
    Kind of FirstStep that merges the result of one or several other steps of same kind.

    Used in input of :class:`Concatenate`

    - no application executed
    """

    def __init__(self, input_steps_metas: Dict, *argv, **kwargs) -> None:
        meta = {**(input_steps_metas[0]), **kwargs}  # kwargs override step0.meta
        super().__init__(*argv, **meta)
        self.__input_steps_metas = input_steps_metas
        self._meta['out_filename'] = [out_filename(s) for s in input_steps_metas]

    def __str__(self) -> str:
        return f'MergeStep{self.__input_steps_metas}'

    def __repr__(self) -> str:
        return f'MergeStep{self.__input_steps_metas}'

    @property
    def input_metas(self) -> Dict:
        """
        Specific to :class:`MergeStep` and :class:`FirstStep`: returns the metas from the inputs as
        a list.
        """
        return self.__input_steps_metas


class _FileProducingStepFactory(StepFactory):
    """
    Abstract class that factorizes filename transformations and parameter handling for Steps that
    produce files, either with OTB or through external calls.

    :func:`create_step` is kind of *abstract* at this point.
    """

    def __init__(
        self,
        cfg                : FileProducingConfiguration,
        gen_tmp_dir        : str,
        gen_output_dir     : Optional[str],
        gen_output_filename: OutputFilenameGenerator,
        *argv,
        **kwargs,
    ) -> None:
        """
        Constructor

        See :func:`output_directory`, :func:`tmp_directory`, :func:`build_step_output_filename` and
        :func:`build_step_output_tmp_filename` for the usage of ``gen_tmp_dir``, ``gen_output_dir``
        and ``gen_output_filename``.
        """
        super().__init__(*argv, extra_metadata=cfg.extra_metadata, **kwargs)
        is_a_final_step = gen_output_dir and gen_output_dir != gen_tmp_dir
        # logger.debug("%s -> final: %s <== gen_tmp=%s    gen_out=%s", self.name, is_a_final_step, gen_tmp_dir, gen_output_dir)

        self.__gen_tmp_dir         = gen_tmp_dir
        self.__gen_output_dir      = gen_output_dir if gen_output_dir else gen_tmp_dir
        self.__gen_output_filename = gen_output_filename
        self.__ram_per_process     = cfg.ram_per_process
        # TODO: for a domain independent StepFactory, extract the following directory names handling
        #       to an external domain specific strategy returned by the configuration object, and
        #       interrogated by the leaf StepFactories.
        self.__directories            = cfg.extra_directories
        self.__directories['tmp_dir'] = cfg.tmpdir
        self.__directories['out_dir'] = cfg.output_preprocess if is_a_final_step else cfg.tmpdir

        self.__has_several_outputs    = self.__gen_output_filename.has_several_outputs()
        logger.debug("new _FileProducingStepFactory(%s) -> directories = %s", self.name, self.__directories)

    def has_several_outputs(self) -> bool:
        """
        Tells whether this step produces several files
        """
        return self.__has_several_outputs

    def output_directory(self, meta: Meta) -> str:
        """
        Accessor to where output files will be stored in case their production is required (i.e. not
        in-memory processing)

        This property is built from ``gen_output_dir`` construction parameter.
        Typical values for the parameter are:

        - ``os.path.join(cfg.output_preprocess, '{tile_name}'),`` where ``tile_name`` is looked into
          ``meta`` parameter
        - ``None``, in that case the result will be the same as :func:`tmp_directory`.
          This case will make sense for steps that don't produce required products
        """
        return str(self.__gen_output_dir).format(
            **meta,
            **self.__directories,
        )

    def _get_nominal_output_basename(self, meta: Meta) -> Union[str, List[str]]:
        """
        Returns the pathless basename of the produced file (internal).
        """
        return self.__gen_output_filename.generate(meta['basename'], meta)

    def build_step_output_filename(self, meta: Meta) -> Union[str, List[str]]:
        """
        Returns the names of typical result files in case their production is required (i.e. not
        in-memory processing).

        This specialization uses ``gen_output_filename`` naming policy parameter to build the output
        filename. See the :ref:`Available naming policies`.
        """
        filename = self._get_nominal_output_basename(meta)

        def in_dir(fn: str) -> str:
            # in_dir = lambda fn : os.path.join(self.output_directory(meta), fn)
            return os.path.join(self.output_directory(meta), fn)

        if isinstance(filename, str):
            return in_dir(filename)
        else:
            return [in_dir(fn) for fn in filename]

    def tmp_directory(self, meta) -> str:
        """
        Directory used to store temporary files before they are renamed into their final version.

        This property is built from ``gen_tmp_dir`` construction parameter.
        Typical values for the parameter are:

        - ``os.path.join(cfg.tmpdir, 'S1')``
        - ``os.path.join(cfg.tmpdir, 'S2', '{tile_name}')`` where ``tile_name`` is looked into
          ``meta`` parameter
        """
        return self.__gen_tmp_dir.format(**meta)

    def build_step_output_tmp_filename(self, meta: Meta) -> Union[str, List[str]]:
        """
        This specialization of :func:`StepFactory.build_step_output_tmp_filename` will automatically
        insert ``.tmp`` before the filename extension.
        """
        filename = self._get_nominal_output_basename(meta)

        def add_tmp(fn: str) -> str:
            return os.path.join(self.tmp_directory(meta), re.sub(re_any_ext, r'.tmp\g<0>', fn))

        if isinstance(filename, str):
            return add_tmp(filename)
        else:
            return [add_tmp(fn) for fn in filename]

    def parameters(self, meta: Meta) -> Union[ExeParameters, OTBParameters]:
        """
        Most steps that produce files will expect parameters.

        Warning: parameters that designate output filenames are expected to use :func:`tmp_filename`
        and not :func:`out_filename`. Indeed products are meant to be first produced with temporary
        names before being renamed with their final names, once the operation producing them has
        succeeded.

        Note: This method is kind-of abstract -- :class:`SelectBestCoverage
        <s1tiling.libs.otbwrappers.SelectBestCoverage>` is a :class:`_FileProducingStepFactory` but,
        it doesn't actualy consume parameters.
        """
        raise TypeError(f"An {self.__class__.__name__} step don't produce anything!")

    @property
    def ram_per_process(self):
        """Property ram_per_process"""
        return self.__ram_per_process


class OTBStepFactory(_FileProducingStepFactory):
    """
    Abstract StepFactory for all OTB Applications.

    All step factories that wrap OTB applications are meant to inherit from :class:`OTBStepFactory`.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        cfg                : FileProducingConfiguration,
        *,
        appname            : str,
        gen_tmp_dir        : str,
        gen_output_dir     : Optional[str],
        gen_output_filename: OutputFilenameGenerator,
        extended_filename  : Optional[Union[str, List[str]]] = None,
        pixel_type         : Optional[Union[int, List[int]]] = None,
        **kwargs,
    ) -> None:
        """
        Constructor.

        See:
            :func:`_FileProducingStepFactory.__init__`

        Parameters:
            :cfg:                 Request configuration for current S1Tiling session
            :appname:             Name of the OTB application
            :gen_tmp_dir:         Dirname format for the temporary product
            :gen_output_dir:      Optional Dirname format for the final product -- ``None`` if not required.
            :gen_output_filename: Ouput filename generator.
            :extended_filename:   Optional extra :external+OTB:std:doc:`OTB extended filename extension <ExtendedFilenames>`.
            :param_in:            Flag used by the default OTB application for the input file (default: "in")
            :param_out:           Flag used by the default OTB application for the ouput file (default: "out")
        """
        super().__init__(cfg, gen_tmp_dir, gen_output_dir, gen_output_filename, **kwargs)
        # is_a_final_step = gen_output_dir and gen_output_dir != gen_tmp_dir
        # logger.debug("%s -> final: %s <== gen_tmp=%s    gen_out=%s", self.name, is_a_final_step, gen_tmp_dir, gen_output_dir)

        self._in                   = kwargs.get('param_in',  'in')
        self._out                  = kwargs.get('param_out', 'out')
        # param_in is only used in connected mode. As such a string is expected.
        assert self.param_in  is None or isinstance(self.param_in, str), f"String expected for {appname} param_in={self.param_in}"
        # param_out is always used.
        assert isinstance(self.param_out, (str, list)), f"String or list expected for {appname} param_out={self.param_out}"
        self._appname              = appname
        self._extended_filename    = extended_filename
        self._pixel_type           = pixel_type
        logger.debug("new OTBStepFactory(%s) -> app=%s // pt=%s", self.name, appname, pixel_type)

    @property
    def appname(self) -> str:
        """OTB Application property."""
        return self._appname

    @abstractmethod
    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Override of :func:`parameters()` to precise covariant return type to `OTBParameters`
        """
        raise TypeError(f"An {self.__class__.__name__} step don't produce anything!")

    @property
    def param_in(self) -> str:
        """
        Name of the "in" parameter used by the OTB Application.
        Default is likely to be "in", while some applications use "io.in", often "il" for list of
        files...
        """
        return self._in

    @property
    def param_out(self) -> str:
        """
        Name of the "out" parameter used by the OTB Application.
        Default is likely to be "out", whie some applications use "io.out".
        """
        return self._out

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        Propagates the optional :external+OTB:std:doc:`extended filename <ExtendedFilenames>` set in
        the construtor to the step meta data.

        .. note::

            :func:`StepFactory.complete_meta()` already takes care of clearing any residual
            ``out_extended_filename_complement`` metadata from previous steps
        """
        meta = super().complete_meta(meta, all_inputs)
        if self._extended_filename:
            meta['out_extended_filename_complement'] = self._extended_filename
        return meta

    def set_output_pixel_type(self, app, meta: Meta) -> None:  # pylint: disable=unused-argument
        """
        Permits to have steps force the output pixel data.
        """

        def do_set(name: str, ptype: Optional[int]) -> None:
            if ptype is not None:
                assert app
                logger.debug("%s.SetParameterOutputImagePixelType(%s, %s)", self.appname, name, ptype)
                app.SetParameterOutputImagePixelType(name, ptype)

        if isinstance(self.param_out, list):
            assert isinstance(self._pixel_type, list)
            assert len(self.param_out) == len(self._pixel_type)
            for name, ptype in zip(self.param_out, self._pixel_type):
                do_set(name, ptype)
        elif isinstance(self._pixel_type, int):
            do_set(self.param_out, self._pixel_type)

    def _do_create_actual_step(
        self,
        execution_parameters: Dict,
        input_step: AbstractStep,
        meta: Meta
    ) -> AbstractStep:
        """
        Instanciates the step related to the current :class:`StepFactory`, that consumes results
        from the previous `input` step.

        0. We expect the step metadata and the GDAL image metadata to have been updated.

        1. Steps that wrap an OTB application will instanciate this application object, and:

           - either pipe the new application to the one from the `input` step if it wasn't a first
             step
           - or fill in the "in" parameter of the application with the :func:`out_filename` of the
             `input` step.

        1-bis. in case the new step isn't related to an OTB application, nothing specific is done,
        we'll just return an :class:`AbstractStep`

        :meta public:
        """
        assert self.appname

        parameters = self.parameters(meta)
        # Otherwise: step with an OTB application...
        if is_running_dry(execution_parameters):
            logger.warning('DRY RUN mode: ignore step and OTB Application creation')
            lg_from = input_step.out_filename if input_step.is_first_step else 'app'
            parameters = self.parameters(meta)
            logger.info('Register app: %s (from %s) %s', self.appname, lg_from, ' '.join(f'-{k} {v!r}' for k, v in parameters.items()))
            meta['param_out'] = self.param_out
            return Step('FAKEAPP', **meta)
        with Utils.RedirectStdToLogger(logging.getLogger('s1tiling.OTB')):
            # For OTB application execution, redirect stdout/stderr messages to s1tiling.OTB
            app = otb.Registry.CreateApplication(self.appname)
            if not app:
                raise RuntimeError("Cannot create OTB application '" + self.appname + "'")
            left_over_parameters : Set[str] = set()
            if input_step.is_first_step:
                if not files_exist(input_step.out_filename):  # pragma: no cover
                    logger.critical(
                        "Cannot create OTB pipeline starting with %s as some input files don't exist (%s)", self.appname, input_step.out_filename
                    )
                    raise RuntimeError(
                        f"Cannot create OTB pipeline starting with {self.appname}: some input files don't exist ({input_step.out_filename})"
                    )
                # parameters[self.param_in] = input_step.out_filename
                lg_from = input_step.out_filename
            else:
                assert isinstance(input_step, _OTBStep)
                assert isinstance(self.param_in, str), f"String expected for {self.param_in}"
                assert isinstance(input_step.param_out, str), f"String expected for {self.param_out}"
                app.ConnectImage(self.param_in, input_step.app, input_step.param_out)
                this_step_is_in_memory = execution_parameters.get('in_memory', True) and not input_step.shall_store
                # logger.debug("Chaining %s in memory: %s", self.appname, this_step_is_in_memory)
                app.PropagateConnectMode(this_step_is_in_memory)
                if this_step_is_in_memory:
                    # When this is not a store step, we need to clear the input parameters
                    # from its list, otherwise some OTB applications may complain
                    in_parameters = parameters[self.param_in]
                    if isinstance(in_parameters, list):
                        # However, if the input is a list, and previous app provide only a subset of
                        # the piped inputs => We still need a AddImageToInputImageList
                        crt_in_parameter_set   = set(in_parameters)
                        prv_out_parameter_set  = {input_step.out_filename}  # TODO what if it was a list?
                        params_piped_in_memory = crt_in_parameter_set & prv_out_parameter_set
                        left_over_parameters   = crt_in_parameter_set - params_piped_in_memory

                    del parameters[self.param_in]
                lg_from = 'app'

            logger.debug(
                'Register app: %s (from %s) %s -%s %s',
                self.appname,
                lg_from,
                ' '.join(f'-{k} {v!r}' for k, v in parameters.items()),
                self.param_out,
                as_app_shell_param(meta.get('out_filename', '???')),
            )
            try:
                app.SetParameters(parameters)

                for input_param in left_over_parameters:
                    logger.debug(" - register leftover list parameter '%s': %s", self.param_in, input_param)
                    app.AddParameterStringList(self.param_in, input_param)
                self.set_output_pixel_type(app, meta)
            except Exception:  # pragma: no cover
                logger.exception(
                    "Cannot set parameters to %s (from %s) %s", self.appname, lg_from, ' '.join(f'-{k} {v!r}' for k, v in parameters.items())
                )
                raise

        meta['param_out'] = self.param_out
        return Step(app, **meta)

    def check_requirements(self) -> Optional[Tuple[str, str]]:
        """
        This specialization of :func:`check_requirements` checks whether the related OTB application
        can correctly be executed from S1Tiling.

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

    def requirement_context(self) -> str:
        """
        Returns the requirement context that permits to fix missing requirements.
        By default, OTB applications requires... OTB!
        """
        return "Please install OTB."


class ExecutableStepFactory(_FileProducingStepFactory):
    """
    Abstract StepFactory for executing any external program.

    All step factories that wrap GDAL applications, or any other executable are meant to inherit
    from :class:`ExecutableStepFactory`.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        cfg:                 FileProducingConfiguration,
        *,
        exename:             str,
        gen_tmp_dir:         str,
        gen_output_dir:      Optional[str],
        gen_output_filename: OutputFilenameGenerator,
        **kwargs,
    ) -> None:
        """
        Constructor

        See:
            :func:`_FileProducingStepFactory.__init__`
        """
        super().__init__(cfg, gen_tmp_dir, gen_output_dir, gen_output_filename, **kwargs)
        self._exename = exename
        logger.debug("new ExecutableStepFactory(%s) -> exe=%s", self.name, exename)

    def _do_create_actual_step(
        self,
        execution_parameters: Dict,
        input_step: AbstractStep,
        meta: Meta
    ) -> ExecutableStep:
        """
        This Step creation method does more than just creating the step.
        It also executes immediately the external process.

        :meta public:
        """
        logger.debug("Directly execute %s step", self.name)
        res        = ExecutableStep(self._exename, **meta)
        parameters = self.parameters(meta)
        res.execute_and_write_output(parameters, execution_parameters)
        return res


class AnyProducerStepFactory(_FileProducingStepFactory):
    """
    Abstract StepFactory for executing any Python made step.

    All step factories that wrap calls to Python code are meant to inherit from
    :class:`AnyProducerStepFactory`.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        cfg:                 FileProducingConfiguration,
        *,
        action:              Callable,
        gen_tmp_dir:         str,
        gen_output_dir:      Optional[str],
        gen_output_filename: OutputFilenameGenerator,
        **kwargs,
    ) -> None:
        """
        Constructor

        See:
            :func:`_FileProducingStepFactory.__init__`
        """
        super().__init__(cfg, gen_tmp_dir, gen_output_dir, gen_output_filename, **kwargs)
        self._action = action
        logger.debug("new AnyProducerStepFactory(%s)", self.name)

    def _do_create_actual_step(
        self,
        execution_parameters: Dict,
        input_step: AbstractStep,
        meta: Meta
    ) -> AnyProducerStep:
        """
        This Step creation method does more than just creating the step.
        It also executes immediately the external process.

        :meta public:
        """
        logger.debug("Directly execute %s step", self.name)
        res        = AnyProducerStep(self._action, **meta)
        parameters = self.parameters(meta)
        res.execute_and_write_output(parameters, execution_parameters)
        return res


class Store(StepFactory):
    """
    Factory for Artificial Step that forces the result of the previous app sequence to be stored on
    disk by breaking in-memory connection.

    While it could be used manually, it's meant to be automatically appended at the end of a
    pipeline if any step is actually related to OTB.
    """

    def __init__(self, appname: str, *argv, **kwargs) -> None:  # pylint: disable=unused-argument
        super().__init__('(StoreOnFile)', "(StoreOnFile)", *argv, **kwargs)
        # logger.debug('Creating Store Factory: %s', appname)

    def create_step(
        self,
        execution_parameters: Dict,
        previous_steps: List[InputList]
    ) -> Union[AbstractStep, StoreStep]:
        """
        Specializes :func:`StepFactory.create_step` to trigger
        :func:`StoreStep.execute_and_write_output` on the last step that relates to an OTB
        Application.

        In case the input step is a `first step`, we simply return a :class:`AbstractStep`. Indeed
        :class:`StoreStep` doesn't transform anything: it just makes sure the registered
        transformations have been applied.

        Eventually, it makes sure all the OTB applications have been released with
        :func:`Step.release_app()`.
        """
        inputs     = self._get_inputs(previous_steps)
        input_step = self._get_canonical_input(inputs)
        if input_step.is_first_step:
            # TODO: The boolean tested is incorrectly named! Fix that.
            # | This case may happen when StepFactories skips their actions by returning an
            # | AbstractStep instead of the usual Step; meaning no OTB application will be called.
            # | This may happen in the case of the concatenation when there is only one input image
            # | that will be renamed.
            # logger.debug(f"Unexpected case where StoreStep is build from: {input_step}")
            meta = input_step.meta.copy()
            return AbstractStep(**meta)

        logger.debug('Creating StoreStep from %s', input_step)
        assert isinstance(input_step, _OTBStep)
        res = StoreStep(input_step)
        try:
            res.execute_and_write_output(None, execution_parameters)  # Parameters have already been set for OTB applications
        finally:
            # logger.debug("Collecting memory!")
            # Collect memory now!
            res.release_app()  # <- StoreStep._app = None
            for inps in reversed(previous_steps):  # delete all /*OTB*/Step._app
                for inp in inps:
                    for _, step in inp.items():
                        step.release_app()
        return res

    # abstract methods...

    def build_step_output_filename(self, meta: Meta) -> NoReturn:
        """
        Deleted method: No way to ask for the output filename of a Store Factory
        """
        raise TypeError("No way to ask for the output filename of a Store Factory")

    def build_step_output_tmp_filename(self, meta: Meta) -> NoReturn:
        """
        Deleted method: No way to ask for the output temporary filename of a Store Factory
        """
        raise TypeError("No way to ask for the output temporary filename of a Store Factory")
