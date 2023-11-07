#!/usr/bin/env python
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
#
# =========================================================================

import fnmatch
import logging
import os
import re
from s1tiling.libs.Utils import get_shape_from_polygon
from unittest import TestCase

# WARNING: Update these lists everytime an OTB application with an original
# naming scheme for its parameters is used.
k_input_keys  = ['io.in', 'in', 'il', 'in.normals', 'in.xyz', 'insar', 'indem', 'indemproj', 'xyz']
k_output_keys = ['io.out', 'out', 'out.lia', 'out.sin']


def isfile(filename, existing_files):
    """
    Mock-replacement for :func:`os.path.isfile`
    """
    res = filename in existing_files
    logging.debug("mock.isfile(%s) = %s ∈ %s", filename, res, existing_files)
    return res


def isdir(dirname, existing_dirs):
    """
    Mock-replacement for :func:`os.path.isdir`
    """
    res = dirname in existing_dirs
    logging.debug("mock.isdir(%s) = %s ∈ %s", dirname, res, existing_dirs)
    return res


def makedirs(dirname, existing_dirs):
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
    def __init__(self, pathname, inputdir):
        """
        constructor
        """
        self.path = pathname
        # `name`: relative to scandir...
        self.name = os.path.relpath(pathname, inputdir)
        self.inputdir = inputdir

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'MockDirEntry("{self.path}", "{self.inputdir}") --> {self.name}'


def list_dirs(dir, pattern, known_dirs, inputdir):
    """
    Mock-replacement for :func:`Utils.list_dirs`
    """
    logging.debug('mock.list_dirs(%s, %s) ---> %s', dir, pattern, known_dirs)
    if pattern:
        filt = lambda path: '/' not in path.name and fnmatch.fnmatch(path.name, pattern)
    else:
        filt = lambda path: '/' not in path.name
    dir_entries = [MockDirEntry(kd, inputdir) for kd in known_dirs]
    res = [de for de in dir_entries if filt(de)]
    logging.debug('res --> %s', res)
    return res


def glob(pat, known_files):
    """
    Mock-replacement for :func:`glob.glob`
    """
    res = [fn for fn in known_files if fnmatch.fnmatch(fn, pat)]
    logging.debug('mock.glob(%s) ---> %s', pat, res)
    return res


def compute_coverage(image_footprint_polygon, reference_tile_footprint_polygon):
    image_footprint          = get_shape_from_polygon(image_footprint_polygon[:4])
    reference_tile_footprint = get_shape_from_polygon(reference_tile_footprint_polygon[:4])
    intersection = image_footprint.Intersection(reference_tile_footprint)
    coverage = intersection.GetArea() / reference_tile_footprint.GetArea()
    assert coverage > 0   # We wouldn't have selected this pair S2 tile + S1 image otherwise
    assert coverage <= 1  # the ratio intersection / S2 tile should be <= 1!!
    return coverage


def dirname(path, depth):
    """
    Helper function to return dirname at ``depth`` up-level.
    """
    for i in range(depth):
        path = os.path.dirname(path)
    return path


def _as_cmdline_call(d):
    if isinstance(d, dict):
        return ' '.join(["-%s '%s'" %(k, v) for k,v in d.items()])
    else:
        return ' '.join(d)


class MockOTBApplication:
    """
    Mock-replacement for :class:`otbApplication.Application`
    """
    def __init__(self, appname, mock_ctx):
        """
        Constructor that answers to
        # app = otb.Registry.CreateApplication(self.appname)
        """
        self.__appname      = appname
        self.__params       = {}
        self.__pixel_types  = {}
        self.__metadata     = {}
        self.__expectations = {}
        self.__mock_ctx     = mock_ctx

    def __del__(self):
        """
        destructor
        """
        self.unregister()

    def add_unknown_parameter(self, key, value):
        if key not in self.__params:
            self.__params[key] = value

    def unregister(self):
        self.__mock_ctx = None

    def ConnectImage(self, param_in, input_app, input_app_param_out):
        input_app.add_unknown_parameter(input_app_param_out, self)
        self.add_unknown_parameter(param_in, input_app)

    def PropagateConnectMode(self, in_memory):
        pass

    def SetParameterOutputImagePixelType(self, param_out, pixel_type):
        self.__pixel_types[param_out] = pixel_type

    def SetParameters(self, parameters):
        self.__params.update(parameters)

    def SetParameterString(self, key, svalue):
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
        assert filenames, ('%s has no output filename (--> %s)' % (self.__appname, _as_cmdline_call(self.__params)))
        return filenames

    @property
    def unextended_out_filenames(self):
        """
        Remove Extended Filename from ``out_filenames``
        """
        return [re.sub(r'\?.*$', '', filename) for filename in self.out_filenames]

    def execute_and_write_output(self, is_top_level):
        # Simulate app at the start of the pipeline first
        for k in k_input_keys:
            if k in self.parameters and isinstance(self.parameters[k], MockOTBApplication):
                logging.info('mock.ExecuteAndWriteOutput: %s: recursing...', self.__appname)
                self.parameters[k].execute_and_write_output(False)
            # elif  k in self.parameters:
            #     logging.debug("mock.ExecuteAndWriteOutput: %s PARAM: -'%s' -> '%s'", self.__appname, k, type(self.parameters[k]))
        logging.info('mock.ExecuteAndWriteOutput: %s %s', self.__appname, _as_cmdline_call(self.parameters))
        self.__mock_ctx.assert_app_is_expected(self.__appname, self.parameters, self.__pixel_types)

    def ExecuteAndWriteOutput(self):
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
    def __init__(self, exename, parameters):
        """
        constructor
        """
        self.__exename    = exename
        if isinstance(parameters, list):
            self.__parameters = [exename] + parameters
        else:
            self.__parameters = parameters

    def __contains__(self, key):
        """
        Implements "in" operator
        """
        assert self.is_dict(), 'Current command-line is not a dictionary of "--key value"'
        return key in self.__parameters

    def __setitem__(self, key, value):
        """
        Implements Write-only [] operator
        """
        assert self.is_dict(), 'Current command-line is not a dictionary of "--key value"'
        self.__parameters[key] = value

    def is_dict(self):
        """
        Tells whether the current command-line is a dictionary of "--key value"'
        """
        return isinstance(self.__parameters, dict)

    def assert_have_same_keys(self, actual_parameters):
        assert isinstance(actual_parameters, dict) and self.is_dict()
        actual_keys   = actual_parameters.keys()
        expected_keys = self.__parameters.keys()
        assert actual_keys == expected_keys, f'actual={actual_keys} != expected={expected_keys}'

    def __eq__(self, rhs):
        """
        Implements == operator
        """
        # logging.debug('CMP expected: %s\nactual: %s\n--> %s', self.__parameters, rhs, self.__parameters == rhs)
        return self.__parameters == rhs

    def __str__(self):
        return _as_cmdline_call(self.__parameters)


class OTBApplicationsMockContext:
    """
    Mocking context where OTB/S1Tiling expected application calls are cached.
    """

    def __init__(self, cfg, mocker, tmp_to_out_map, dem_files):
        """
        constructor
        """
        self.__applications           = []
        self.__expectations           = []
        self.__configuration          = cfg
        self.__known_files            = dem_files[:]
        self.__tmp_to_out_map         = tmp_to_out_map
        self.__last_expected_metadata = {}

        self.__known_files.append(cfg.srtm_db_filepath)
        mocker.patch('s1tiling.libs.otbpipeline.otb.Registry.CreateApplication', lambda a : self.create_application(a))
        mocker.patch('s1tiling.libs.otbpipeline.execute', lambda cmdlinelist, dryrun : self.execute_process(cmdlinelist, dryrun))

    @property
    def known_files(self):
        return self.__known_files

    def tmp_to_out(self, tmp_filename):
        # Remove queued applications
        parts = tmp_filename.split('|>')
        res = '|>'.join(self.__tmp_to_out_map.get(p, p) for p in parts)
        return res

    def execute_process(self, cmdlinelist, dryrun):
        msg = ' '.join([str(p) for p in cmdlinelist])
        logging.info('Mocking execution of: %s', msg)
        self.assert_execution_is_expected(cmdlinelist)
        # It's quite complex to deduce the name of the "out" product for all situation.
        # As so far there is only one executable: gdalbuildvrt and as the output is the
        # first parameter, let's rely on this!
        assert cmdlinelist[0] == 'gdalbuildvrt'
        file_produced = self.tmp_to_out(cmdlinelist[1])
        logging.debug('Register new known file %s -> %s', cmdlinelist[1], file_produced)
        self.known_files.append(file_produced)

    def create_application(self, appname):
        logging.info('Creating mocked application: %s', appname)
        app = MockOTBApplication(appname, self)
        self.__applications.append(app)
        return app

    def clear(self):
        self.__applications = []

    def set_expectations(self, appname, cmdline, pixel_types, metadata):
        expectation = {'cmdline': CommandLine(appname, cmdline), 'appname': appname}
        if pixel_types:
            expectation['pixel_types'] = pixel_types
        if metadata:
            expectation['metadata'] = metadata
        self.__expectations.append(expectation)

    def _remaining_expectations_as_str(self, appname = None):
        if appname:
            msgs = [ f"\n * {exp['appname']} {exp['cmdline']}" for  exp in self.__expectations if appname == exp['appname']]
            # msgs = ['\n * ' + exp['appname'] + ' ' + _as_cmdline_call(exp['cmdline']) for exp in self.__expectations if appname == exp['appname']]
        else:
            msgs = [ f"\n * {exp['appname']} {exp['cmdline']}" for  exp in self.__expectations]
            # msgs = ['\n * ' + exp['appname'] + ' ' + _as_cmdline_call(exp['cmdline']) for exp in self.__expectations]
        msg = ('(%s)' % len(msgs,)) + ''.join(msgs)
        return msg

    def _update_output_to_final_filename(self, params):
        for kv in k_output_keys:
            if kv in params:
                if isinstance(params[kv], MockOTBApplication):
                    params[kv] =  params[kv].appname + '|>' + self._update_output_to_final_filename(params[kv].parameters)
                return params[kv]

    def _update_input_to_root_filename(self, params):
        assert isinstance(params, dict) # of parameters
        in_param_keys = [kv for kv in k_input_keys if kv in params]
        for kv in in_param_keys:
            if isinstance(params[kv], MockOTBApplication):
                updated = self._update_input_to_root_filename(params[kv].parameters)
                if isinstance(updated, list):
                    updated = [u + '|>'+params[kv].appname for u in updated]
                else:
                    updated = updated + '|>'+params[kv].appname
                params[kv] = updated
            elif isinstance(params[kv], list):
                ps = []
                for p in params[kv]:
                    if isinstance(p, MockOTBApplication):
                        p = self._update_input_to_root_filename(p.parameters) + '|>'+params[kv].appname
                    ps.append(p)
                    assert isinstance(p, str)
                params[kv] = ps
                assert isinstance(params[kv], list) # of str...
            return params[kv]

    def assert_these_metadata_are_expected(self, filename, new_metadata):
        # Clean some useless/instable metadata
        new_metadata.pop('TIFFTAG_SOFTWARE', None)
        new_metadata.pop('TIFFTAG_DATETIME', None)
        # assert new_metadata == self.__last_expected_metadata
        tc = TestCase()
        tc.maxDiff = None
        last_expected_metadata = self.__last_expected_metadata
        self.__last_expected_metadata = {} # Make sure to clear before the assert
        tc.assertDictEqual(new_metadata , last_expected_metadata)
        pass

    def assert_app_is_expected(self, appname, params, pixel_types):
        # Find out what the root input filename is (as we may not have any
        # input filename when dealing with in-memory processing
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
                    exp['cmdline']['elev.dem'] = self.__configuration.tmp_srtm_dir
            exp['cmdline'].assert_have_same_keys(params)
            # logging.debug('TEST: %s <- %s == %s', params == exp['cmdline'], params, exp['cmdline'])
            if params == exp['cmdline']:
                exp_pixel_type = exp.get('pixel_types', {})
                assert pixel_types == exp_pixel_type, f'Pixel type set to "{pixel_types}" for {appname}. "{exp_pixel_type}" was expected.'
                logging.debug('Expectation found for %s', params)
                logging.info('FOUND and removing %s among %s', exp, self._remaining_expectations_as_str())
                if exp.get('metadata', None):
                    self.__last_expected_metadata.update(exp['metadata'])
                self.__expectations.remove(exp)
                logging.info('REMAINING: %s', self._remaining_expectations_as_str())
                return  # Found! => return "true"
        logging.error('NO expectation FOUND for %s %s', appname, _as_cmdline_call(params))
        assert False, f"Cannot find any matching expectation for\n-> {appname} {_as_cmdline_call(params)}\namong {self._remaining_expectations_as_str(appname)}"

    def assert_execution_is_expected(self, cmdlinelist):
        # self._update_input_to_root_filename(cmdlinelist)
        # self._update_output_to_final_filename(cmdlinelist)
        assert len(cmdlinelist) > 0
        appname = cmdlinelist[0]
        for exp in self.__expectations:
            if appname != exp['appname']:
                continue
            if cmdlinelist == exp['cmdline']:
                logging.debug('Expectation found for %s', _as_cmdline_call(cmdlinelist))
                logging.info('FOUND and removing %s among %s', exp, self._remaining_expectations_as_str())
                if exp.get('metadata', None):
                    self.__last_expected_metadata.update(exp['metadata'])
                self.__expectations.remove(exp)
                logging.info('REMAINING: %s', self._remaining_expectations_as_str())
                return  # Found! => return "true"
        logging.error('Expectation NOT FOUND')
        assert False, f"Cannot find any matching expectation for\n-> {appname} {_as_cmdline_call(cmdlinelist)}\namong {self._remaining_expectations_as_str(appname)}"

    def assert_all_have_been_executed(self):
        assert len(self.__expectations) == 0, f"The following applications haven't executed: {self._remaining_expectations_as_str()}"
