#!/usr/bin/env python
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
#          Fabien CONTIVAL (CS Group)
#
# =========================================================================

from __future__ import annotations

from collections.abc import Iterable
import fnmatch
import logging
import os
import re
from typing import Callable, Dict, List, Literal, Union
from s1tiling.libs.Utils import get_shape_from_polygon
from unittest import TestCase

# WARNING: Update these lists everytime an OTB application with an original
# naming scheme for its parameters is used.
k_input_keys  = [
    'io.in',
    'in',
    'il',
    'in.normals',
    'in.xyz',
    'insar',
    'indem',
    'indemproj',
    'xyz',
    'inr',
    'inm',

    'ingammaarea',
    'insigmanaught',
]
k_output_keys = ['io.out', 'out', 'out.deg', 'out.cos', 'out.sin', 'out.tan', 'out.gamma_area']


def isfile(filename, existing_files) -> bool:
    """
    Mock-replacement for :func:`os.path.isfile`
    """
    res = filename in existing_files
    logging.debug("mock.isfile(%s) = %s ∈ %s", filename, res, existing_files)
    return res


def isdir(dirname, existing_dirs) -> bool:
    """
    Mock-replacement for :func:`os.path.isdir`
    """
    res = dirname in existing_dirs
    logging.debug("mock.isdir(%s) = %s ∈ %s", dirname, res, existing_dirs)
    return res


def makedirs(dirname, existing_dirs) -> Literal[True]:
    """
    Mock-replacement for :func:`os.makedirs`
    """
    logging.debug("mock.makedirs(%s) Added into %s", dirname, existing_dirs)
    return True


class MockDirEntry:
    """
    Mock-replacement for :class:`os.DirEntry` type returned by :func:`scandir`
    and :func:`listdir` functions.
    """
    def __init__(self, pathname, inputdir) -> None:
        """
        constructor
        """
        self.path = pathname
        # `name`: relative to scandir...
        self.name = os.path.relpath(pathname, inputdir)
        self.inputdir = inputdir

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'MockDirEntry("{self.path}", "{self.inputdir}") --> {self.name}'


def list_dirs(dir, pattern, known_dirs, inputdir) -> List[MockDirEntry]:
    """
    Mock-replacement for :func:`Utils.list_dirs`
    """
    logging.debug('mock.list_dirs(%r, %r) ---> %s', dir, pattern, known_dirs)
    if pattern:
        filt = lambda path: '/' not in path.name and fnmatch.fnmatch(path.name, pattern)
    else:
        filt = lambda path: '/' not in path.name
    dir_entries = [MockDirEntry(kd, inputdir) for kd in known_dirs]
    res = [de for de in dir_entries if filt(de)]
    logging.debug('res --> %s', res)
    return res


def list_files(dir, pattern, known_files, inputdir) -> List[MockDirEntry]:
    """
    Mock-replacement for :func:`Utils.list_files`
    """
    logging.debug('mock.list_files(%r, %r) ---> %s', dir, pattern, known_files)
    if not pattern:
        filt = lambda path: '/' not in path.name
    elif isinstance(pattern, re.Pattern):
        filt = lambda path: '/' not in path.name and re.match(pattern, path.name)
    else:
        filt = lambda path: '/' not in path.name and fnmatch.fnmatch(path.name, pattern)
    dir_entries = [MockDirEntry(kd, inputdir) for kd in known_files]
    res = [de for de in dir_entries if filt(de)]
    logging.debug('res --> %s', res)
    return res


def glob(pat, known_files) -> List[str]:
    """
    Mock-replacement for :func:`glob.glob`
    """
    res = [fn for fn in known_files if fnmatch.fnmatch(fn, pat)]
    logging.debug('mock.glob(%s) ---> %s', pat, res)
    return res


def compute_coverage(image_footprint_polygon, reference_tile_footprint_polygon) -> float:
    image_footprint          = get_shape_from_polygon(image_footprint_polygon[:4])
    reference_tile_footprint = get_shape_from_polygon(reference_tile_footprint_polygon[:4])
    intersection = image_footprint.Intersection(reference_tile_footprint)
    coverage = intersection.GetArea() / reference_tile_footprint.GetArea()
    assert coverage > 0   # We wouldn't have selected this pair S2 tile + S1 image otherwise
    assert coverage <= 1  # the ratio intersection / S2 tile should be <= 1!!
    return coverage


def dirname(path, depth) -> str:
    """
    Helper function to return dirname at ``depth`` up-level.
    """
    for _ in range(depth):
        path = os.path.dirname(path)
    return path


def _as_cmdline_call(d) -> str:
    if isinstance(d, dict):
        return ' '.join(["-%s '%s'" %(k, v) for k,v in d.items()])
    else:
        return ' '.join(f"{i!r}" for i in d)


class MockOTBApplication:
    """
    Mock-replacement for :class:`otbApplication.Application`
    """
    def __init__(self, appname: str, mock_ctx: OTBApplicationsMockContext) -> None:
        """
        Constructor that answers to
        # app = otb.Registry.CreateApplication(self.appname)
        """
        self.__appname      : str = appname
        self.__params       : Dict = {}
        self.__pixel_types  : Dict = {}
        # self.__metadata     : Dict = {}
        # self.__expectations : Dict = {}
        self.__mock_ctx     = mock_ctx

    def __del__(self) -> None:
        """
        destructor
        """
        self.unregister()

    def __str__(self) -> str:
        return f"MockOTBApplication({self.appname}) => params: {self.__params}"

    def __repr__(self) -> str:
        return f"MockOTBApplication({self.appname}, {self.__mock_ctx})"

    def add_unknown_parameter(self, key, value) -> None:
        logging.debug('  %s.add_unknown_parameter(%s, %s) in %s', self.appname, key, value, self.__params)
        if key not in self.__params:
            self.__params[key] = value

    def unregister(self) -> None:
        self.__mock_ctx = None

    def ConnectImage(self, param_in, input_app, input_app_param_out) -> None:
        logging.debug("%s.ConnectImage(): %s -> %s", self.appname, param_in, input_app.appname)
        input_app.add_unknown_parameter(input_app_param_out, self)
        self.add_unknown_parameter(param_in, input_app)

    def PropagateConnectMode(self, in_memory) -> None:
        pass

    def SetParameterOutputImagePixelType(self, param_out, pixel_type) -> None:
        logging.debug("Setting pixel_types[%s] <- %s // (%s) %s", param_out, pixel_type, self.appname, id(self))
        self.__pixel_types[param_out] = pixel_type

    def SetParameters(self, parameters) -> None:
        logging.debug("%s.SetParameters(): %s", self.appname, parameters)
        self.__params.update(parameters)

    def AddParameterStringList(self, key, lvalues) -> None:
        logging.debug("%s.AddParameterStringList(): %s = %s", self.appname, key, lvalues)
        if key not in self.__params:
            self.__params[key] = []
        elif not isinstance(self.__params, list):
            self.__params[key] = [self.__params[key]]
        assert isinstance(self.__params[key], list), f"params[{key}] is a {type(self.__params[key])} : {self.__params[key]}"
        self.__params[key].append(lvalues)

    def SetParameterString(self, key, svalue) -> None:
        logging.debug("%s.SetParameterString(): %s = %s", self.appname, key, svalue)
        assert isinstance(svalue, str)
        self.__params[key] = svalue

    @property
    def parameters(self):
        return self.__params

    @property
    def appname(self):
        return self.__appname

    @property
    def out_filenames(self):
        # We may actually have several ouputs => always return a list
        filenames = [self.__params[kv] for kv in k_output_keys if kv in self.__params]
        assert filenames, ('%s has no output filename (--> %s)' % (self.appname, _as_cmdline_call(self.__params)))
        return filenames

    @property
    def unextended_out_filenames(self):
        """
        Remove Extended Filename from ``out_filenames``
        """
        return [re.sub(r'\?.*$', '', filename) for filename in self.out_filenames]

    def execute_and_write_output(self, is_top_level) -> None:
        assert self.__mock_ctx
        # Simulate app at the start of the pipeline first
        for k in k_input_keys:
            if k in self.parameters:
                parameters = self.parameters[k] if isinstance(self.parameters[k], list) else [self.parameters[k]]
                for param in parameters:
                    if isinstance(param, MockOTBApplication):
                        logging.info('mock.ExecuteAndWriteOutput: %s: recursing to... %s', self.appname, param.appname)
                        param.execute_and_write_output(False)

            # elif  k in self.parameters:
            #     logging.debug("mock.ExecuteAndWriteOutput: %s PARAM: -'%s' -> '%s'", self.appname, k, type(self.parameters[k]))
        logging.info('mock.ExecuteAndWriteOutput: %s %s', self.appname, _as_cmdline_call(self.parameters))
        logging.debug("pixel type(%s): %s // %s", self.appname, self.__pixel_types, id(self))
        self.__mock_ctx.assert_app_is_expected(self.appname, self.parameters, self.__pixel_types)

    def ExecuteAndWriteOutput(self) -> None:
        assert self.__mock_ctx
        self.execute_and_write_output(True)
        # register output as a known file from now on
        for filename in self.out_filenames:
            file_produced = self.__mock_ctx.tmp_to_out(filename)
            logging.debug('Register new known file %s -> %s', filename, file_produced)
            self.__mock_ctx.known_files.append(file_produced)


class CommandLine:
    """
    Helper class that contains either:
    - a dictionary of "-paramname value"
    - or a sequenced list of parameters
    """
    def __init__(self, exename: Union[Callable, str], parameters: Union[List, Dict]) -> None:
        """
        constructor
        """
        self.__exename    = exename
        if isinstance(parameters, list):
            self.__parameters = [exename] + parameters
        else:
            self.__parameters = parameters

    def __contains__(self, key) -> bool:
        """
        Implements "in" operator
        """
        assert self.is_dict(), 'Current command-line is not a dictionary of "--key value"'
        return key in self.__parameters

    def __setitem__(self, key, value) -> None:
        """
        Implements Write-only [] operator
        """
        assert self.is_dict(), 'Current command-line is not a dictionary of "--key value"'
        self.__parameters[key] = value

    def is_dict(self) -> bool:
        """
        Tells whether the current command-line is a dictionary of "--key value"'
        """
        return isinstance(self.__parameters, dict)

    def assert_have_same_keys(self, actual_parameters: Dict) -> None:
        assert isinstance(actual_parameters, dict) and self.is_dict()
        actual_keys   = actual_parameters.keys()
        expected_keys = self.__parameters.keys()
        assert actual_keys == expected_keys, f'actual={actual_keys} != expected={expected_keys}'

    def __eq__(self, rhs) -> bool:
        """
        Implements == operator
        """
        # logging.debug('CMP expected: %s\nactual: %s\n--> %s', self.__parameters, rhs, self.__parameters == rhs)
        return self.__parameters == rhs

    def __str__(self) -> str:
        return _as_cmdline_call(self.__parameters)

    def __repr__(self) -> str:
        return self.__str__()


class OTBApplicationsMockContext:
    """
    Mocking context where OTB/S1Tiling expected application calls are cached.
    """

    def __init__(self, cfg, mocker, tmp_to_out_map, dem_files) -> None:
        """
        constructor
        """
        self.__applications           : List[MockOTBApplication] = []
        self.__expectations           : List[Dict]               = []
        self.__configuration                                     = cfg
        self.__known_files                                       = dem_files[:] + [os.path.join(cfg.tmpdir, 'geoid', os.path.basename(cfg.GeoidFile))]
        self.__tmp_to_out_map                                    = tmp_to_out_map

        # Cached data to help detect mismatching between actual and expected image_metadata,
        # files_removed, etc.
        self.__current_step                                      = ""
        self.__last_expected_metadata                            = {}
        self.__mismatching_metadata                              = []
        self.__last_expected_files_to_remove                     = set()
        self.__mismatching_removed_files                         = []

        # Register a few known_files & dirs
        self.__known_files.append(cfg.dem_db_filepath)
        self.__known_files.append(cfg.output_grid)

        # Mock various functions, that do stuff, from otbpipeline/steps...
        mocker.patch('s1tiling.libs.steps.otb.Registry.CreateApplication', lambda a : self.create_application(a))
        mocker.patch('s1tiling.libs.steps.ExecutableStep._do_execute',     lambda slf, params, dryrun : self.execute_process(slf, params, dryrun))
        mocker.patch('s1tiling.libs.steps.AnyProducerStep._do_execute',    lambda slf, params, dryrun : self.execute_function(slf._action, params, dryrun))
        mocker.patch('s1tiling.libs.Utils.remove_files',                   lambda files: self.mock_remove_files(files))

    @property
    def known_files(self):
        return self.__known_files

    def tmp_to_out(self, tmp_filename: str) -> str:
        # Remove queued applications
        parts = tmp_filename.split('|>')
        res = '|>'.join(self.__tmp_to_out_map.get(p, p) for p in parts)
        return res

    def execute_process(self, step, params: List, dryrun) -> None:
        cmdlinelist = [step._exename] + params
        msg = ' '.join([str(p) for p in cmdlinelist])
        logging.info('Mocking execution of: %s', msg)
        self.assert_execution_is_expected(cmdlinelist)
        # It's quite complex to deduce the name of the "out" product for all situation.
        # As so far there is only one executable: gdalbuildvrt and as the output is the
        # first parameter, let's rely on this!
        file_produced = self.tmp_to_out(step.out_filename)
        logging.debug('Register new known file %s -> %s', cmdlinelist[1], file_produced)
        self.known_files.append(file_produced)

    def execute_function(self, action, params, dryrun) -> None:
        msg = ' '.join([str(p) for p in params])
        logging.info('Mocking execution of: %s(%s)', action.__name__, msg)
        self.assert_execution_is_expected([action]+params)
        # It's quite complex to deduce the name of the "out" product for all situation.
        # As so far there is only one executable: gdalbuildvrt and as the output is the
        # first parameter, let's rely on this!
        file_produced = self.tmp_to_out(params[0])
        logging.debug('Register new known file %s -> %s', params[0], file_produced)
        self.known_files.append(file_produced)

    def create_application(self, appname: str) -> MockOTBApplication:
        logging.info('Creating mocked application: %s', appname)
        app = MockOTBApplication(appname, self)
        self.__applications.append(app)
        return app

    def clear(self) -> None:
        self.__applications = []

    def set_expectations(
        self,
        appname: Union[Callable, str],
        cmdline: Union[List,Dict],
        pixel_types,
        metadata,
        files_to_remove : Iterable[str]=(),
    ) -> None:
        expectation = {'appname': appname, 'cmdline': CommandLine(appname, cmdline)}
        logging.debug("Register expectation: %s", expectation)
        if pixel_types:
            expectation['pixel_types'] = pixel_types
        if metadata:
            expectation['metadata'] = metadata
        if files_to_remove:
            expectation['files_to_remove'] = set(files_to_remove)
        self.__expectations.append(expectation)

    def _remaining_expectations_as_str(self, appname = None) -> str:
        if appname:
            msgs = [ f"\n * {exp['appname']} {exp['cmdline']}" for  exp in self.__expectations if appname == exp['appname']]
        else:
            msgs = [ f"\n * {exp['appname']} {exp['cmdline']}" for  exp in self.__expectations]
        msg = ('(%s)' % len(msgs,)) + ''.join(msgs)
        return msg

    def _update_output_to_final_filename(self, params):
        # logging.debug("  O2R from: %s", params)
        for kv in k_output_keys:
            if kv in params:
                if isinstance(params[kv], MockOTBApplication):
                    params[kv] =  params[kv].appname + '|>' + self._update_output_to_final_filename(params[kv].parameters)
                # logging.debug("  O2R[%s] to: %s", kv, params)
                return params[kv]

    def _update_input_to_root_filename(self, params: Union[Dict, List]) -> Union[List[str], str]:
        assert isinstance(params, dict) # of parameters
        # logging.debug("  I2R from: %s", params)
        in_param_keys = [kv for kv in k_input_keys if kv in params]
        # assert len(in_param_keys) > 0, f"No input keys found in {params.keys()}"
        for kv in in_param_keys:
            if isinstance(params[kv], MockOTBApplication):
                # logging.debug("    case 1: RECURSE (%s, %s)", kv, params[kv].parameters)
                updated = self._update_input_to_root_filename(params[kv].parameters)
                if isinstance(updated, list):
                    updated = [u + '|>'+params[kv].appname for u in updated]
                else:
                    updated = updated + '|>'+params[kv].appname
                # logging.debug("    case 1: BACK %s <-- %s", kv, updated)
                params[kv] = updated
            elif isinstance(params[kv], list):
                # logging.debug("    case 2")
                ps = []
                for p in params[kv]:
                    logging.debug("    %s is MockOTBApplication: %s // %s", kv, isinstance(p, MockOTBApplication), p)
                    if isinstance(p, MockOTBApplication):
                        p = self._update_input_to_root_filename(p.parameters) + '|>'+p.appname
                    ps.append(p)
                    assert isinstance(p, str)
                params[kv] = ps
                assert isinstance(params[kv], list) # of str...
            # logging.debug("  I2R[%s] to: %s", kv, params)
            return params[kv]
        # logging.debug("    case 3: no %s IN %s: -> %s", params.keys(), k_input_keys, in_param_keys)
        # logging.debug("  I2R[??] to: %s", params)
        # Some upstream OTB applications upstream in a pipeline have no input images
        return 'Ø'

    def register_any_unexpected_image_metadata(self, new_metadata: Dict, name: str, filename: str) -> None:
        # Clean some useless/instable metadata
        # Assertions failures are cached for later as the AssertionError exception would be cached
        # in otbpipeline while running the pipeline
        new_metadata.pop('TIFFTAG_SOFTWARE', None)
        new_metadata.pop('TIFFTAG_DATETIME', None)
        if new_metadata != self.__last_expected_metadata:
            self.__mismatching_metadata.append({
                "expected": self.__last_expected_metadata,
                "actual": new_metadata,
                "context": f"\nMismatching metadata for {name}.",
            })
        self.__last_expected_metadata        = {}  # Make sure to clear before existing

    def assert_all_metadata_match(self) -> None:
        tc = TestCase()
        tc.maxDiff = None
        for metadata_mismatch in self.__mismatching_metadata:
            tc.assertDictEqual(
                metadata_mismatch["actual"],
                metadata_mismatch["expected"],
                metadata_mismatch.get("context", "")+" (ACTUAL <-> EXPECTED)",
            )
        for removed_files_mismatch in self.__mismatching_removed_files:
            tc.assertSetEqual(
                removed_files_mismatch["actual"],
                removed_files_mismatch["expected"],
                removed_files_mismatch.get("context", "")+" (ACTUAL <-> EXPECTED)",
            )
        self.__last_expected_files_to_remove = {}  # Make sure to clear before existing

    def mock_remove_files(self, files: list) -> None:
        """
        Mock-replacement for :func:`Utils.remove_files`
        It does assert on-the-fly that expectations match actual files removed.

        Assertions failures are cached for later as the AssertionError exception would be cached in
        otbpipeline while running the pipeline
        """
        logging.debug('mock.remove_files(%r) ---> %s', files, self.known_files)
        for f in files:
            self.__known_files.remove(f)
        expected_files_to_remove = self.__last_expected_files_to_remove
        actual_files_removed     = set(files)
        logging.debug("Checking files_to_remove.\nEXPECTED %s\nACTUAL   %s", expected_files_to_remove, actual_files_removed)
        if expected_files_to_remove != actual_files_removed:
            self.__mismatching_removed_files.append({
                "expected": expected_files_to_remove,
                "actual":   actual_files_removed,
                "context":  f"\nMismatching files to remove for {self.__current_step}.",
            })

    def assert_app_is_expected(self, appname, params, pixel_types) -> None:
        # Find out what the root input filename is (as we may not have any
        # input filename when dealing with in-memory processing
        logging.debug("In %s case:", appname)
        self._update_input_to_root_filename(params)
        self._update_output_to_final_filename(params)
        # logging.info('SEARCHING %s %s among %s', appname, _as_cmdline_call(params), self._remaining_expectations_as_str())
        for exp in self.__expectations:
            # logging.debug('TEST %s against %s', appname, exp)
            if appname != exp['appname']:
                continue
            assert exp['cmdline'].is_dict()
            if 'elev.dem' in exp['cmdline']:
                # Override the value w/ S1FileManager's one that wasn't known at the beginning
                    exp['cmdline']['elev.dem'] = self.__configuration.tmp_dem_dir
            exp['cmdline'].assert_have_same_keys(params)
            # logging.debug('TEST: %s <- %s == %s', params == exp['cmdline'], params, exp['cmdline'])
            if params == exp['cmdline']:
                exp_pixel_type = exp.get('pixel_types', {})
                assert pixel_types == exp_pixel_type, f'Pixel type set to "{pixel_types}" for {appname}. "{exp_pixel_type}" was expected.'
                logging.debug('Expectation found for %s', params)
                logging.info('FOUND and removing %s among %s', exp, self._remaining_expectations_as_str())
                if img_metadata := exp.get('metadata', None):
                    self.__last_expected_metadata.update(img_metadata)
                if files_to_remove := exp.get('files_to_remove', ()):
                    self.__last_expected_files_to_remove = files_to_remove
                self.__current_step = f"{appname} {params}"
                self.__expectations.remove(exp)
                logging.info('REMAINING: %s', self._remaining_expectations_as_str())
                return  # Found! => return "true"
        logging.error('NO expectation FOUND for %s %s', appname, _as_cmdline_call(params))
        assert False, f"Cannot find any matching expectation for\n-> {appname}: {_as_cmdline_call(params)}\namong {self._remaining_expectations_as_str(appname)}"

    def assert_execution_is_expected(self, cmdlinelist: List) -> None:
        # self._update_input_to_root_filename(cmdlinelist)
        # self._update_output_to_final_filename(cmdlinelist)
        assert len(cmdlinelist) > 0
        appname = cmdlinelist[0]
        logging.debug("TESTING: %s -> %s", appname, cmdlinelist)
        for exp in self.__expectations:
            logging.debug("AGAINST: %s", exp)
            if appname != exp['appname']:
                continue
            if cmdlinelist == exp['cmdline']:
                logging.debug('Expectation found for %s', _as_cmdline_call(cmdlinelist))
                logging.info('FOUND and removing %s among %s', exp, self._remaining_expectations_as_str())
                if exp.get('metadata', None):
                    self.__last_expected_metadata.update(exp['metadata'])
                if files_to_remove := exp.get('files_to_remove', ()):
                    self.__last_expected_files_to_remove = files_to_remove
                self.__current_step = appname
                self.__expectations.remove(exp)
                logging.info('REMAINING: %s', self._remaining_expectations_as_str())
                return  # Found! => return "true"
        logging.error('Expectation NOT FOUND')
        assert False, f"Cannot find any matching expectation for\n-> {appname}: {_as_cmdline_call(cmdlinelist[1:])}\namong {self._remaining_expectations_as_str(appname)}"

    def assert_all_have_been_executed(self) -> None:
        assert len(self.__expectations) == 0, f"The following applications haven't executed: {self._remaining_expectations_as_str()}"
