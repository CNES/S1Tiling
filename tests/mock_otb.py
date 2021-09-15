#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fnmatch
import logging
import os
import re

k_input_keys  = ['io.in', 'in', 'il']
k_output_keys = ['io.out', 'out']


def isfile(filename, existing_files):
    res = filename in existing_files
    logging.debug("mock.isfile(%s) = %s ∈ %s", filename, res, existing_files)
    return res


def isdir(dirname, existing_dirs):
    res = dirname in existing_dirs
    logging.debug("mock.isdir(%s) = %s ∈ %s", dirname, res, existing_dirs)
    return res


class MockDirEntry:
    def __init__(self, pathname, inputdir):
        """
        constructor
        """
        self.path = pathname
        # `name`: relative to scandir...
        self.name = os.path.relpath(pathname, inputdir)


def list_dirs(dir, pat, known_dirs, inputdir):
    logging.debug('mock.list_dirs(%s, %s) ---> %s', dir, pat, known_dirs)
    return [MockDirEntry(kd, inputdir) for kd in known_dirs]


def glob(pat, known_files):
    res = [fn for fn in known_files if fnmatch.fnmatch(fn, pat)]
    logging.debug('mock.glob(%s) ---> %s', pat, res)
    return res


def dirname(path, depth):
    for i in range(depth):
        path = os.path.dirname(path)
    return path



def _as_cmdline_call(d):
    line = ' '.join(["-%s '%s'" %(k, v) for k,v in d.items()])
    return line


class MockApplication:
    def __init__(self, appname, mock_ctx):
        """
        Constructor that answers to
        # app = otb.Registry.CreateApplication(self.appname)
        """
        self.__appname      = appname
        self.__params       = {}
        self.__pixel_types  = {}
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
    def out_filename(self):
        for kv in k_output_keys:
            if kv in self.__params:
                return self.__params[kv]
        assert ('%s has no output filename (--> %s)' % (self.__appname, _as_cmdline_call(self.__params)))

    @property
    def unextended_out_filename(self):
        """
        Remove Extended Filename from ``out_filename``
        """
        return re.sub(r'\?.*$', '', self.out_filename)

    # def set_expectations(self, cmdline, pixel_types):
    #     assert cmdline
    #     self.__expectations['cmdline'] = cmdline
    #     if pixel_types:
    #         self.__expectations['pixel_types'] = pixel_types

    def execute_and_write_output(self, is_top_level):
        # Simulate app at the start of the pipeline first
        for k in k_input_keys:
            if k in self.parameters and isinstance(self.parameters[k], MockApplication):
                logging.info('mock.ExecuteAndWriteOutput: %s: recursing...', self.__appname)
                self.parameters[k].execute_and_write_output(False)
        logging.info('mock.ExecuteAndWriteOutput: %s %s', self.__appname, _as_cmdline_call(self.__params))
        self.__mock_ctx.assert_app_is_expected(self.__appname, self.__params, self.__pixel_types)

    def ExecuteAndWriteOutput(self):
        self.execute_and_write_output(True)
        # register output as a known file from now on
        # self.__mock_ctx.known_files.append(self.unextended_out_filename)
        file_produced = self.__mock_ctx.tmp_to_out(self.out_filename)
        logging.debug('Register new know file %s -> %s', self.out_filename, file_produced)
        self.__mock_ctx.known_files.append(file_produced)
        # assert self.__params == self.__expectations['cmdline']
        # assert self.__pixel_types == get(self.__expectations, 'pixel_types', {})


class OTBApplicationsMockContext:
    """
    «class documentation»
    """

    def __init__(self, cfg, mocker, tmp_to_out_map):
        """
        constructor
        """
        self.__applications   = []
        self.__expectations   = []
        self.__configuration  = cfg
        self.__known_files    = []
        self.__tmp_to_out_map = tmp_to_out_map
        mocker.patch('s1tiling.libs.otbpipeline.otb.Registry.CreateApplication', lambda a : self.create_application(a))

    @property
    def known_files(self):
        return self.__known_files

    def tmp_to_out(self, tmp_filename):
        # TODO: Handle App|>App|>file and file|>App|App
        parts = tmp_filename.split('|>')
        res = '|>'.join(self.__tmp_to_out_map.get(p, p) for p in parts)
        return res

    def create_application(self, appname):
        logging.info('Creating mocked application: %s', appname)
        app = MockApplication(appname, self)
        self.__applications.append(app)
        return app

    def clear(self):
        self.__applications = []

    def set_expectations(self, appname, cmdline, pixel_types):
        expectation = {'cmdline': cmdline, 'appname': appname}
        if pixel_types:
            expectation['pixel_types'] = pixel_types
        self.__expectations.append(expectation)
        # app = self.__find_right_app(appname)
        # app.set_expectations(cmdline, pixel_types)

    def _remaining_expectations_as_str(self, appname = None):
        if appname:
            msgs = ['\n * ' + exp['appname'] + ' ' + _as_cmdline_call(exp['cmdline']) for exp in self.__expectations if appname == exp['appname']]
        else:
            msgs = ['\n * ' + exp['appname'] + ' ' + _as_cmdline_call(exp['cmdline']) for exp in self.__expectations]
        msg = ('(%s)' % len(msgs,)) + ''.join(msgs)
        return msg

    def _update_output_to_final_filename(self, params):
        for kv in k_output_keys:
            if kv in params:
                if isinstance(params[kv], MockApplication):
                    params[kv] =  params[kv].appname + '|>' + self._update_output_to_final_filename(params[kv].parameters)
                return params[kv]

    def _update_input_to_root_filename(self, params):
        for kv in k_input_keys:
            if kv in params:
                if isinstance(params[kv], MockApplication):
                    params[kv] = self._update_input_to_root_filename(params[kv].parameters) + '|>'+params[kv].appname
                return params[kv]

    def assert_app_is_expected(self, appname, params, pixel_types):
        # Find out what the root input filename is (as we may not have any
        # input filename when dealing with in-memory processing
        self._update_input_to_root_filename(params)
        self._update_output_to_final_filename(params)
        # logging.info('SEARCHING %s %s among %s', appname, _as_cmdline_call(params), self._remaining_expectations_as_str())
        for exp in self.__expectations:
            if appname != exp['appname']:
                continue
            if 'elev.dem' in exp['cmdline']:
                # Override the value w/ S1FileManager's one that wasn't known at the beginning
                    exp['cmdline']['elev.dem'] = self.__configuration.tmp_srtm_dir
            ## Fix output filename:
            ## - Ignore .tmp from filename
            ## - TODO: calls to OTB applications produce files in temporary directory...
            #for kv in k_output_keys:
            #    if kv in params and isinstance(params[kv], str):
            #        # params[kv] = params[kv].replace('.tmp', '')
            #        params[kv] = self.tmp_to_out(params[kv])
            # And the check...
            assert params.keys() == exp['cmdline'].keys(), f'actual={params.keys()} != expected={exp["cmdline"].keys()}'
            # logging.debug('TEST: %s <- %s == %s', params == exp['cmdline'], params, exp['cmdline'])
            if params == exp['cmdline']:
                assert pixel_types == exp.get('pixel_types', {})
                logging.debug('Expectation found for %s', params)
                # logging.info('FOUND and removing %s among %s', exp, self._remaining_expectations_as_str())
                self.__expectations.remove(exp)
                # logging.info('REMAINING: %s', self._remaining_expectations_as_str())
                return  # Found! => return "true"
        logging.error('NOT FOUND')
        assert False, f"Cannot find any matching expectation for {appname} {_as_cmdline_call(params)} among {self._remaining_expectations_as_str(appname)}"

    def assert_all_have_been_executed(self):
        assert len(self.__expectations) == 0, f"The following applications haven't executed: {self._remaining_expectations_as_str()}"
