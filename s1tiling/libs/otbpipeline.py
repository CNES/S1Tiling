#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2024 (c) CNES.
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
import re
import copy
from itertools import filterfalse
import logging
import logging.handlers
import multiprocessing
from typing import Dict, List, Optional, Set, Tuple, Type, Union

# memory leaks
from distributed import get_worker
import objgraph
from pympler import tracker  # , muppy
# from memory_profiler import profile

from .                  import Utils
from .                  import exceptions
from .S1DateAcquisition import S1DateAcquisition
from .configuration     import Configuration
from .file_naming       import CannotGenerateFilename
from .meta              import (
        Meta, accept_as_compatible_input, is_running_dry, get_task_name, product_exists, out_filename,
)
from .node_queue        import node_queue
from .outcome           import PipelineOutcome
from .steps             import (
        AbstractStep, FirstStep, InputList, OTBStepFactory, StepFactory, MergeStep, Store,
        files_exist,
)
# from ..__meta__         import __version__

logger = logging.getLogger('s1tiling.pipeline')

re_tiff    = re.compile(r'\.tiff?$')
re_any_ext = re.compile(r'\.[^.]+$')  # Match any kind of file extension


class Pipeline:
    """
    Pipeline of OTB applications.

    It's instanciated as a list of :class:`AbstractStep` s.
    :func:`Step.execute_and_write_output()` will be executed on the last step
    of the pipeline.

    Internal class only meant to be used by :class:`PipelineDescriptionSequence`.
    """
    # Should we inherit from contextlib.ExitStack?
    def __init__(
        self,
        execution_parameters: Dict,
        do_watch_ram:         bool,
        name:                 Optional[str] = None,
        output:               Optional[str] = None
    ) -> None:
        self.__pipeline             : List[StepFactory] = []
        self.__execution_parameters = execution_parameters
        self.__do_watch_ram         = do_watch_ram
        self.__name                 = name
        self.__output               = output
        self.__inputs               : InputList = []

    def __repr__(self) -> str:
        return self.name

    def set_inputs(self, inputs: Dict) -> None:
        """
        Set the input(s) of the instanciated pipeline.
        The `inputs` is parameter expected to be a list of {'key': [metas...]} that'll
        get tranformed into a dictionary of {'key': :class:`AbstractStep`}.

        Some :class:`AbstractStep` will actually be :class:`MergeStep` instances.
        """
        logger.debug("   Pipeline(%s).set_inputs(%s)", self.__name, inputs)
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
    def _input_filenames(self) -> List[str]:
        """
        Property _input_filenames
        """
        return [input[k].out_filename for input in self.__inputs for k in input]

    @property
    def appname(self) -> str:
        """
        Name of the pipeline application(s).
        """
        # assert: if self.__name is not set, all crt are instances of Step
        appname = self.__name or '|'.join(crt.appname for crt in self.__pipeline)
        return appname

    @property
    def name(self) -> str:
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
    def shall_watch_ram(self) -> bool:
        """
        Tells whether objects in RAM shall be watched for memory leaks.
        """
        return self.__do_watch_ram

    def push(self, otbstep: StepFactory) -> None:
        """
        Registers a StepFactory into the pipeline.
        """
        assert isinstance(otbstep, StepFactory)
        self.__pipeline.append(otbstep)

    def check_requirements(self) -> Optional[Tuple[str, Set]]:
        """
        Check all the :class:`StepFactory`'s registered in the pipeline can be
        exexuted.

        :return: ``None`` if requirements are fulfilled.
        :return: A message indicating what is missing otherwise, and some
                 context how to fix it.
        """
        sing_plur = {True: 'are', False: 'is'}
        reqs : List[Tuple[str, str]] = list(filter(None, (sf.check_requirements() for sf in self.__pipeline)))
        missing_reqs = [rq for rq, _ in reqs]
        contexts = set(ctx for _, ctx in reqs)
        if reqs:
            return f"{' and '.join(missing_reqs)} {sing_plur[len(missing_reqs) > 1]} required", contexts
        else:
            return None

    def do_execute(self) -> PipelineOutcome:
        """
        Execute the pipeline.

        1. Makes sure the inputs exist -- unless in dry-run mode
        2. Incrementaly create the steps of the pipeline.
        3. Return the resulting output filename, or the caught errors.
        """
        assert self.__inputs
        logger.debug("INPUTS: %s", self.__inputs)
        tested_files = list(Utils.flatten_stringlist(
            [v.out_filename for inp in self.__inputs for _, v in inp.items()]))
        logger.debug("Testing whether input files exist: %s", tested_files)
        missing_inputs = list(filterfalse(files_exist, tested_files))
        if len(missing_inputs) > 0 and not is_running_dry(self.__execution_parameters):
            msg = f"Cannot execute {self} as the following input(s) {missing_inputs} do(es)n't exist"
            logger.warning(msg)
            return PipelineOutcome(RuntimeError(msg))
        # logger.debug("LOG OTB: %s", os.environ.get('OTB_LOGGER_LEVEL'))
        assert self.__pipeline  # shall not be empty!
        steps = [self.__inputs]
        for crt in self.__pipeline:  # crt is a StepFactory
            step = crt.create_step(self.__execution_parameters, steps)
            if step:  # a StepFactory may return no step so it can be skipped
                steps.append([{'__last': step}])

        assert len(steps[-1]) == 1
        res = steps[-1][0]['__last'].out_filename
        assert res == self.output, (
            f"Step output {self.output!r} doesn't match expected output {res!r}."
            "\nThis is likely happenning because pipeline name generation isn't incremental."
        )
        steps = None  # type: ignore  # force reset local variable, in doubt...
        # logger.debug('Pipeline "%s" terminated -> %s', self, res)
        return PipelineOutcome(res)


# TODO: try to make it static...
def execute4dask(pipeline: Optional[Pipeline], *args, **unused_kwargs) -> PipelineOutcome:
    """
    Internal worker function used by Dask to execute a pipeline.

    Returns the product filename(s) or the caught error in case of failure.
    """
    assert pipeline is not None
    logger.debug('Parameters for %s:\n|--> %s', pipeline, args)
    watch_ram = pipeline.shall_watch_ram
    if watch_ram:
        logger.info("=== objgraph growth (before pipeline exection) ===")
        objgraph.show_growth(limit=5)
    try:
        assert len(args) == 1
        for arg in args[0]:
            # logger.info('ARG: %s (%s)', arg, type(arg))
            if isinstance(arg, PipelineOutcome) and not arg:
                logger.warning('Cancel execution of %s because an error has occured upstream on a dependent input file: %s', pipeline, arg)
                return copy.deepcopy(arg).add_related_filename(pipeline.output)
        # Any exceptions leaking to Dask Scheduler would end the execution of the scheduler.
        # That's why errors need to be caught and transformed here.
        logger.info('Execute %s', pipeline)
        res = pipeline.do_execute().add_related_filename(pipeline.output)
    except Exception as ex:  # pylint: disable=broad-except  # Use in nominal code
    # except RuntimeError as ex:  # pylint: disable=broad-except  # Use when debugging...
        logger.exception('Execution of %s failed', pipeline)
        logger.debug('(ERROR) %s has been executed with the following parameters: %s', pipeline, args)
        return PipelineOutcome(ex).add_related_filename(pipeline.output).set_pipeline_name(pipeline.appname)

    del pipeline  # Release the pipeline
    if watch_ram:
        logger.info("=== objgraph growth (after pipeline exection) ===")
        objgraph.show_growth()
        objgraph.show_most_common_types()

        # all_objects = muppy.get_objects()
        # sum1 = summary.summarize(all_objects)
        # summary.print_(sum1)
        w = get_worker()
        if not hasattr(w, 'tracker'):
            setattr(w, "tr",  tracker.SummaryTracker())
        getattr(w, "tr").print_diff()
    return res


class PipelineDescription:
    """
    Pipeline description:
    - stores the various factory steps that describe a pipeline,
    - can tell the expected product name given an input.
    - tells whether its product is required
    """
    def __init__(  # pylint: disable=too-many-arguments
        self,
        factory_steps:       List[StepFactory],
        execution_parameters: Dict,
        name:                Optional[str]  = None,
        product_required:    bool           = False,
        is_name_incremental: bool           = False,
        inputs:              Optional[Dict] = None
    ) -> None:
        """
        constructor
        """
        assert factory_steps  # shall not be None or empty
        self.__factory_steps        = factory_steps
        self.__is_name_incremental  = is_name_incremental
        self.__is_product_required  = product_required
        self.__execution_parameters = execution_parameters
        if name:
            self.__name = name
        else:
            self.__name = '|'.join([step.name for step in self.__factory_steps])
        assert inputs
        self.__inputs              = inputs
        # logger.debug("New pipeline: %s; required: %s, incremental: %s",
        #     '|'.join([step.name for step in self.__factory_steps]), self.__is_product_required, self.__is_name_incremental)

    def expected(self, input_meta: Meta) -> Optional[Dict]:
        """
        Returns the expected name of the product(s) of this pipeline
        """
        assert self.__factory_steps  # shall not be None or empty
        try:
            # logger.debug("INCREMENTAL: %s in %s", self.__is_name_incremental, self)
            if self.__is_name_incremental:
                res = input_meta
                for step in self.__factory_steps:
                    # logger.debug("   in %s, updating %s", step.name, res)
                    res = step.update_filename_meta(res)
            else:
                res = self.__factory_steps[-1].update_filename_meta(input_meta)
            logger.debug("    expected: %s(%s) -> %s", self.__name, input_meta['out_filename'], out_filename(res))
            # logger.debug("    -> full meta: %s", res)
            return res
        except exceptions.NotCompatibleInput as e:
            logger.warning('%s => rejecting expected(%s)', e, input_meta)
            return None
        except CannotGenerateFilename as e:
            # This warning may happen, when incremental name building hasn't been activated:
            # indeed, later calls to update_filename_meta, may require meta data set on
            # earlier steps.
            logger.warning('%s => rejecting expected(%s)', e, input_meta)
            return None

    @property
    def inputs(self) -> Dict:
        """
        Property inputs
        """
        return self.__inputs

    @property
    def sources(self) -> List[str]:
        """
        Property sources
        """
        # logger.debug("SOURCES(%s) = %s", self.name, self.__inputs)
        res = [(val if isinstance(val, str) else val.name) for (_, val) in self.__inputs.items()]
        return res

    @property
    def name(self) -> str:
        """
        Descriptive name of the pipeline specification.
        """
        assert isinstance(self.__name, str)
        return self.__name

    @property
    def product_is_required(self) -> bool:
        """
        Tells whether the product if this pipeline is required.
        """
        return self.__is_product_required

    def instanciate(self, file: str, do_measure: bool, in_memory: bool, do_watch_ram: bool) -> Pipeline:
        """
        Instanciates the pipeline specified.

        Note: It systematically registers a :class:`Store` step at the end
        if any :class:`StepFactory` is actually an :class:`OTBStepFactory`

        Returns:
            A :class:`Pipeline` instance
        """
        execution_parameters = {
                **self.__execution_parameters,
                'in_memory' : in_memory,
                'do_measure': do_measure
        }
        pipeline = Pipeline(execution_parameters, do_watch_ram, self.name, file)
        need_OTB_store = False
        for factory_step in self.__factory_steps + []:
            pipeline.push(factory_step)
            need_OTB_store = need_OTB_store or isinstance(factory_step, OTBStepFactory)  # TODO: use a dedicated function
            # logger.debug(f"{self.name}.push({factory_step.name}) -> need store: {need_OTB_store}")
        if need_OTB_store:
            pipeline.push(Store('noappname'))
            # logger.debug("Store pushed!")
        return pipeline

    def __repr__(self) -> str:
        res = f'PipelineDescription: {self.name} ## Sources: {self.sources}'
        return res


def to_dask_key(pathname: str) -> str:
    """
    Generate a simplified graph key name from a full pathname.
    - Strip directory name
    - Replace '-' with '_' as Dask has a special interpretation for '-' in key names.
    """
    return pathname.replace('-', '_')


def register_task(tasks: Dict, key: str, value) -> None:
    """
    Register a task named `key` in the right format.
    """
    tasks[key] = value


class TaskInputInfo:
    """
    Abstraction of the input(s) information associated to a particular task.

    Used to merge, or to stack, information about inputs.
    """
    def __init__(self, pipeline: PipelineDescription) -> None:
        """
        constructor
        """
        self.__pipeline     = pipeline
        self._inputs        : Dict[str, List[Dict]] = {}  # map<source, meta / meta list>

    def add_input(self, origin: str, input_meta: Meta, destination_meta: Meta) -> bool:
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
        logger.debug('    add_input[%s # %s]: not empty <<-- %s', origin, out_filename(destination_meta), out_filename(input_meta))
        logger.debug('    -> check %s in %s', f'reduce_inputs_{origin}', destination_meta.keys())
        if f'reduce_inputs_{origin}' in destination_meta.keys():
            # logger.debug('add_input[%s]: self.__inputs[%s]= %s <--- %s',
            #     origin, origin, self._inputs[origin], destination_meta[f'reduce_inputs_{origin}'](self._inputs[origin] + [input_meta]))
            self._inputs[origin] = destination_meta[f'reduce_inputs_{origin}'](
                    self._inputs[origin] + [input_meta])
            return False
        else:
            self._inputs[origin].append(input_meta)
            return True

    def clear(self) -> None:
        """
        Clear the TaskInputInfo and make it ``~False``
        """
        self._inputs = {}

    def __bool__(self) -> bool:
        """
        Tells whether the object has a definition.
        """
        return len(self.inputs) > 0

    @property
    def pipeline(self) -> PipelineDescription:
        """
        Property pipeline
        """
        return self.__pipeline

    @property
    def inputs(self) -> Dict[str, List[Dict]]:
        """
        Inputs associated to the task.

        It's organized as a dictionary that associates a source type to a meta or a list of meta
        information.
        """
        return self._inputs

    @property
    def input_task_names(self) -> List[str]:
        """
        List of input tasks the current task depends on.
        """
        tns = [get_task_name(meta) for meta in self.input_metas]
        logger.debug('   input_task_names(%s) --> %s', self.pipeline.name, tns)
        return tns

    @property
    def input_metas(self) -> List[Dict]:
        """
        List of input meta information the current task depends on.
        """
        metas = [meta for inputs in self.inputs.values() for meta in inputs]
        return metas

    def __repr__(self) -> str:
        res = 'TaskInputInfo:\n- inputs:\n'
        for k, inps in self.inputs.items():
            res += f'  - "{k}":\n'
            for val in inps:
                res += f'    - {val}\n'
        res += f'- pipeline: {self.pipeline}\n'
        return res


def fetch_input_data(key: str, inputs: InputList) -> AbstractStep:
    """
    Helper function that extract the meta data associated to a key from a
    multiple-inputs list of inputs.
    """
    keys = set().union(*(input.keys() for input in inputs))
    assert key in keys, f"Cannot find input '{key}' among {keys}"
    return [input[key] for input in inputs if key in input.keys()][0]


def fetch_input_data_all_inputs(keys: Set[str], all_inputs: List[InputList]) -> Dict[str, AbstractStep]:
    """
    Helper function that extract the meta data associated to a key from a
    multiple-inputs list of list of inputs.

    Unlike :func:`fetch_input_data`, this flavor is able to dig in inputs from
    all levels to find the requested one.
    """
    data : Dict[str, List] = {k: [] for k in keys}  # NB: can't use dict.fromkeys(keys, []) as [] is mutable and will be shared
    # for inputs in all_inputs:
    for _, inputs in enumerate(all_inputs):
        for inp in inputs:
            for key in keys & inp.keys() :
                # logger.debug('#%s -> key: %s, input: %s\n   +++---> %s', lvl, key, inp[key], data[key])
                data[key].append(inp[key])
    res = {}
    for k, i in data.items():
        assert len(i) == 1, f"Only {len(i)} input(s) found instead of 1. Found: {i!r}"
        res[k] = i[0]
    return res


def _update_out_filename(updated_meta, with_meta) -> None:
    """
    Helper function to update the `out_filename` from metadata.
    Meant to be used metadata associated to products made of several inputs
    like Concatenate.
    """
    if 'update_out_filename' in updated_meta:
        updated_meta['update_out_filename'](updated_meta, with_meta)


def _register_new_input_and_update_out_filename(
    tasks:         Dict[str, TaskInputInfo],
    origin:        str,
    input_meta:    Dict,
    new_task_meta: Meta,
    outputs:       List[Dict],  # List<Meta>
) -> None:
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
        logger.debug('    The %s task depends on one more input, updating its metadata to reflect the situation.\nUpdating from %s ...', task_name, new_task_meta)
        _update_out_filename(new_task_meta, task_inputs)  # Required for concatenation dates handling
        logger.debug('    ...to %s', new_task_meta)
        logger.debug("  Next inputs: %s", [get_task_name(ni) for ni in outputs])

        def simplified_task_name(meta: Meta) -> str:
            tn = get_task_name(meta)
            return tn[0] if isinstance(tn, list) else tn
        already_registered_next_input = [ni for ni in outputs if simplified_task_name(ni) == task_name]
        assert len(already_registered_next_input) == 1, \
                f'Task {task_name!r}: 1!={len(already_registered_next_input)} => {already_registered_next_input} inputs have already been registered'
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
    def __init__(self, cfg: Configuration, dryrun: bool, debug_caches: bool) -> None:
        """
        constructor
        """
        assert cfg
        self.__cfg                  = cfg
        self.__pipelines            : List[PipelineDescription] = []
        self.__execution_parameters = {
                'dryrun'      : dryrun,
                'debug_caches': debug_caches,
        }

    def register_pipeline(self, factory_steps: List[Type], *args, **kwargs) -> PipelineDescription:
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
        pipeline = PipelineDescription(steps, self.__execution_parameters, *args, **kwargs)
        logger.debug('--> Register pipeline %s as %s', pipeline.name, [fs.__name__ for fs in factory_steps])
        self.__pipelines.append(pipeline)
        return pipeline

    def _build_dependencies(  # pylint: disable=too-many-locals
            self, tile_name: str, raster_list: List[Dict]
    ) -> Tuple[Set[str], Dict, Dict]:
        """
        Runs the inputs through all pipeline descriptions to build the full list
        of intermediary and final products and what they require to be built.
        """
        first_inputs = _generate_first_steps_from_manifests(tile_name=tile_name, raster_list=raster_list)
        assert first_inputs, "A non empty list of raster inputs is expected"
        # the tile_origin meta from all input is actually the same and it's actually the S2 tile footprint
        tile_origin = first_inputs[0]["tile_origin"]

        pipelines_outputs = {
                'basename': first_inputs,  # TODO: find the right name _0/__/_firststeps/...?
                'tilename': [
                    FirstStep(
                        tile_name=tile_name,
                        tile_origin=tile_origin,  # S2 tile footprint
                        basename=f"S2info_{tile_name}",
                        out_filename=self.__cfg.output_grid,  # Trick existing file detection
                        does_product_exist=lambda: True,
                    ).meta],
        }
        logger.debug('FIRST: %s', pipelines_outputs['basename'])

        required = {}  # (first batch) Final products identified as _needed to be produced_
        previous : Dict[str, TaskInputInfo] = {}  # Graph of deps: for a product tells how it's produced (pipeline + inputs)
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
                inputs = pipelines_outputs[source_name][:]

                logger.debug('  FROM all %s inputs as "%s": %s', len(inputs), origin, [out_filename(i) for i in inputs])
                dropped = []
                for inp in inputs:  # inputs are meta
                    logger.debug('  ----------------------------------------------------------------------')
                    logger.debug('  - GIVEN "%s" "%s": %s', origin, out_filename(inp), inp)
                    expected = pipeline.expected(inp)
                    if not expected:
                        logger.debug("    No '%s' product can be generated from '%s' input '%s' ==> Ignore for now",
                                pipeline.name, origin, out_filename(inp))
                        dropped.append(inp)  # remember that source/input will be used differently
                        continue
                    expected_taskname = get_task_name(expected)
                    logger.debug('    task %s <-- from input: %s', expected_taskname, out_filename(inp))
                    logger.debug('    --> file "%s": %s', out_filename(expected), expected)
                    # TODO: Correctly handle the case where a task produce
                    # several filenames. In that case we shall have only one
                    # task, but possibly, several following tasks may depend on
                    # the current task.
                    # For the moment, just keep the first, and use product
                    # selection pattern as in filter_LIA().
                    if isinstance(expected_taskname, list):
                        expected_taskname = expected_taskname[0]  # TODO: see comment above

                    # We cannot analyse early whether a task product is already
                    # there as some product have names that depend on all
                    # inputs (see Concatenate).
                    # This is why the full dependencies tree is produced at
                    # this time. Unrequired parts will be trimmed in the next
                    # task producing step.
                    if expected_taskname not in previous:
                        outputs.append(expected)
                        previous[expected_taskname] = TaskInputInfo(pipeline=pipeline)
                        previous[expected_taskname].add_input(origin, inp, expected)
                        logger.debug('    Is a new product? YES! %s, with a source from "%s"', expected_taskname, origin)
                    elif (input_task_name := get_task_name(inp)) not in previous[expected_taskname].input_task_names:
                        logger.debug("    Is a new product? NO!  %s, but input task %s NOT registered in input_task_names(%s)", expected_taskname, input_task_name, previous[expected_taskname].pipeline.name)
                        _register_new_input_and_update_out_filename(
                                tasks=previous,
                                origin=origin,
                                input_meta=inp,
                                new_task_meta=expected,
                                outputs=outputs)
                    logger.debug("    Keys in previous: %s", previous.keys())
                    if pipeline.product_is_required:
                        # logger.debug("    %s' products are required => register %s", pipeline.name, expected_taskname)
                        # assert (expected_taskname not in required) or (required[expected_taskname] == expected)
                        required[expected_taskname] = expected
                    task_names_to_output_files_table[expected_taskname] = out_filename(expected)
                # endfor inp in inputs:  # inputs are meta
                if dropped:
                    dropped_inputs[origin] = dropped
            # endfor origin, sources in pipeline.inputs.items():

            # For all new outputs, check which dropped inputs would be compatible
            logger.debug('* Checking dropped inputs: %s', list(dropped_inputs.keys()))
            for output in outputs:
                for origin, inputs in dropped_inputs.items():
                    for inp in inputs:
                        logger.debug("  - Is '%s' a '%s' input for '%s' ?", out_filename(inp), origin, out_filename(output))
                        # Does the output accepts the inpu as compatible?
                        if accept_as_compatible_input(output, inp):
                            logger.debug('    => YES')
                            _register_new_input_and_update_out_filename(
                                    tasks=previous,
                                    origin=origin,
                                    input_meta=inp,
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
                previous[name].clear()  # for the next log
            else:
                required_task_names.add(name)

        logger.debug("Dependencies found:")
        for task_name, prev in previous.items():
            if prev:
                logger.debug('- %s requires %s on %s', task_name, prev.pipeline.name, [out_filename(i) for i in prev.input_metas])
            else:
                logger.debug('- %s already exists, no need to produce it', task_name)
        return required_task_names, previous, task_names_to_output_files_table

    def _build_tasks_from_dependencies(  # pylint: disable=too-many-locals
        self,
        required :                        Set[str],
        previous :                        Dict,
        task_names_to_output_files_table: Dict,
        do_watch_ram:                     bool
    ) -> Dict[str, Union[Tuple, "FirstStep"]]:  # Dict of FirstStep or Tuple parameter for execute4dask
        """
        Generates the actual list of tasks for :func:`dask.client.get()`.

        `previous` is made of:
        - "pipeline": reference to the :class:`PipelineDescription`
        - "inputs": list of the inputs (metadata)
        """
        tasks : Dict[str, Union[Tuple, FirstStep]] = {}
        logger.debug('#############################################################################')
        logger.debug('#############################################################################')
        logger.debug('Building all tasks')
        required_tasks = node_queue(required)
        for task_name in required_tasks:
            logger.debug("* Checking if task '%s' needs to be executed", os.path.basename(task_name))
            assert (task_name in previous) and previous[task_name], \
                    f"No previous task registered for {task_name}.\nOnly the following have previous tasks: {previous.keys()} "
            base_task_name = to_dask_key(task_name)
            task_inputs    = previous[task_name].inputs
            pipeline_descr = previous[task_name].pipeline

            def first(files: Union[str, List[str]]) -> str:
                return str(files[0]) if isinstance(files, list) else str(files)
            input_task_keys = [to_dask_key(first(tn))
                    for tn in previous[task_name].input_task_names]
            assert list(input_task_keys)
            logger.debug(' - It depends on %s --> %s', [os.path.basename(tn) for tn in input_task_keys], task_inputs)
            output_filename = task_names_to_output_files_table[task_name]
            pipeline_instance = pipeline_descr.instanciate(output_filename, True, True, do_watch_ram)
            pipeline_instance.set_inputs(task_inputs)
            logger.debug(' ~~> TASKS[%s] += %s(keys=%s)', os.path.basename(base_task_name), pipeline_descr.name, [os.path.basename(tn) for tn in input_task_keys])
            register_task(tasks, base_task_name, (execute4dask, pipeline_instance, input_task_keys))

            logger.debug(" - Analysing whether its inputs needs to be registered for production...")
            logger.debug("   Already registered: %s", [os.path.basename(tn) for tn in required_tasks])
            # logger.debug("   Already registered: %s", already_registered)
            for t in previous[task_name].input_metas:  # TODO: check whether the inputs need to be produced as well
                tn = first(get_task_name(t))
                logger.debug("   - About task '%s': %s?", os.path.basename(tn), t)
                # logger.debug("   - About task '%s': %s?", tn, t)
                if tn in required_tasks:
                    logger.info("      ~> Ignoring '%s' which is already registered for production", os.path.basename(tn))
                elif not product_exists(t):
                    logger.info("      => Need to register production of task '%s' (for %s)", os.path.basename(tn), pipeline_descr.name)
                    required_tasks.add_if_new(tn)
                else:
                    logger.info("      => Starting %s from existing '%s' task", pipeline_descr.name, os.path.basename(tn))
                    register_task(tasks, to_dask_key(tn), FirstStep(**t))
        return tasks

    def _check_static_task_requirements(self, tasks: Dict[str, Union[Tuple, "FirstStep"]]) -> None:
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
        missing_apps : Dict[str, List[str]] = {}
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
            raise exceptions.MissingApplication(missing_apps, contexts)
        else:
            logger.debug('All required applications are correctly available')

    def generate_tasks(
        self, tile_name:    str, raster_list:  List[Dict], do_watch_ram=False
    ) -> Tuple[Dict[str, Union[Tuple, "FirstStep"]], List[str]]:
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


def _generate_first_steps_from_manifests(
    raster_list:  List[Dict],
    tile_name:    str,
) -> List[Dict]:  # List[meta(FirstStep)]
    """
    Flatten all rasters from the manifest as a list of :class:`FirstStep`
    """
    inputs = []
    # Log commented and kept for filling in unit tests
    # logger.debug('Generate first steps from: %s', raster_list)
    for raster_info in raster_list:
        raster: S1DateAcquisition = raster_info['raster']

        manifest = raster.get_manifest()
        for image in raster.get_images_list():
            start = FirstStep(tile_name=tile_name,
                              tile_origin=raster_info['tile_origin'],
                              tile_coverage=raster_info['tile_coverage'],
                              manifest=manifest,
                              basename=image)
            inputs.append(start.meta)
    return inputs


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
            debug_otb) -> None:
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
    def __init__(self, cfg, debug_otb) -> None:
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
