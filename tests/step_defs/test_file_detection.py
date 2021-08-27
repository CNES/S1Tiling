#!/usr/bin/env python
# -*- coding: utf-8 -*-
import fnmatch
import logging
import os
from pathlib import Path
import pytest
from pytest_bdd import scenarios, given, when, then, parsers

import s1tiling.libs.Utils
from s1tiling.libs.S1FileManager import S1FileManager
from s1tiling.libs.S1DateAcquisition import S1DateAcquisition

# ======================================================================
# Scenarios
scenarios('../features/test_file_detection.feature')

# ======================================================================
# Test Data

FILES = [
        {
            's1dir': 'S1A_IW_GRDH_1SDV_20200108T044150_20200108T044215_030704_038506_C7F5',
            's1file': 's1a-iw-grd-vv-20200108t044150-20200108t044215-030704-038506-001.tiff',
            },
        {
            's1dir': 'S1A_IW_GRDH_1SDV_20200108T044215_20200108T044240_030704_038506_D953',
            's1file': 's1a-iw-grd-vv-20200108t044215-20200108t044240-030704-038506-001.tiff',
            },
        {
            's1dir': 'S1A_IW_GRDH_1SDV_20200108T044150_20200108T044215_030704_038506_C7F5',
            's1file': 's1a-iw-grd-vh-20200108t044150-20200108t044215-030704-038506-001.tiff',
            },
        {
            's1dir': 'S1A_IW_GRDH_1SDV_20200108T044215_20200108T044240_030704_038506_D953',
            's1file': 's1a-iw-grd-vh-20200108t044215-20200108t044240-030704-038506-001.tiff',
            }
        ]

TMPDIR = 'TMP'
INPUT  = 'data_raw'
OUTPUT = 'OUTPUT'

def safe_dir(idx):
    s1dir  = FILES[idx]['s1dir']
    return f'{INPUT}/{s1dir}/{s1dir}.SAFE'

def input_file(idx):
    s1dir  = FILES[idx]['s1dir']
    s1file = FILES[idx]['s1file']
    return f'{INPUT}/{s1dir}/{s1dir}.SAFE/measurement/{s1file}'

def input_file_vv(idx):
    assert idx < 2
    return input_file(idx)

def input_file_vh(idx):
    assert idx < 2
    return input_file(idx+2)

# ======================================================================
# Mocks

class Configuration():
    def __init__(self, inputdir, tmpdir, outputdir, *argv):
        """
        constructor
        """
        self.first_date        = '2020-01-01'
        self.last_date         = '2020-01-10'
        self.download          = False
        self.raw_directory     = inputdir
        self.tmpdir            = tmpdir
        self.output_preprocess = outputdir

def isfile(filename, existing_files):
    res = filename in existing_files
    logging.debug("mock.isfile(%s) = %s ∈ %s", filename, res, existing_files)
    return res

def isdir(dirname, existing_dirs):
    res = dirname in existing_dirs
    logging.debug("mock.isdir(%s) = %s ∈ %s", dirname, res, existing_dirs)
    return res

class MockDirEntry:
    def __init__(self, pathname):
        """
        constructor
        """
        self.path = pathname
        # `name`: relative to scandir...
        self.name = os.path.relpath(pathname, INPUT)


def list_dirs(dir, pat, known_dirs):
    logging.debug('mock.list_dirs(%s, %s) ---> %s', dir, pat, known_dirs)
    return [MockDirEntry(kd) for kd in known_dirs]

def glob(pat, known_files):
    res = [fn for fn in known_files if fnmatch.fnmatch(fn, pat)]
    logging.debug('mock.glob(%s) ---> %s', pat, res)
    return res

@pytest.fixture
def known_files():
    kf = []
    return kf

@pytest.fixture
def known_dirs():
    kd = set()
    return kd

@pytest.fixture
def image_list():
    rl = []
    return rl

@pytest.fixture
def configuration(mocker):
    known_dirs = [INPUT, TMPDIR, OUTPUT, safe_dir(0), safe_dir(1)]
    mocker.patch('os.path.isdir', lambda f: isdir(f, known_dirs))
    cfg = Configuration(INPUT, TMPDIR, OUTPUT)
    return cfg

def dirname(path, depth):
    for i in range(depth):
        path = os.path.dirname(path)
    return path

# ======================================================================
# Given steps

def _declare_know_files(mocker, known_files, known_dirs, patterns):
    # logging.debug('_declare_know_files(%s)', patterns)
    all_files = [input_file(idx) for idx in range(len(FILES))]
    # logging.debug('- all_files: %s', all_files)
    files = []
    for pattern in patterns:
        files += [fn for fn in all_files if fnmatch.fnmatch(fn, '*'+pattern+'*')]
    known_files.extend(files)
    known_dirs.update([dirname(fn, 3) for fn in known_files])
    logging.debug('Mocking w/ %s --> %s', patterns, files)
    # Utils.list_dirs has been imported in S1FileManager. This is the one that needs patching!
    # mocker.patch('s1tiling.libs.S1FileManager.list_dirs', return_value=files)
    mocker.patch('s1tiling.libs.S1FileManager.list_dirs', lambda dir, pat : list_dirs(dir, pat, known_dirs))
    mocker.patch('glob.glob', lambda pat : glob(pat, known_files))


@given('All files are known')
def given_all_files_are_know(mocker, known_files, known_dirs):
    _declare_know_files(mocker, known_files, known_dirs, ['vv', 'vh'])

@given('All VV files are known')
def given_all_VV_files_are_know(mocker, known_files, known_dirs):
    _declare_know_files(mocker, known_files, known_dirs, ['vv'])

@given('All VH files are known')
def given_all_VH_files_are_know(mocker, known_files, known_dirs):
    _declare_know_files(mocker, known_files, known_dirs, ['vh'])


# ======================================================================
# When steps

def _search(configuration, image_list, polarisation):
    configuration.polarisation = polarisation
    manager = S1FileManager(configuration)
    manager._update_s1_img_list()
    logging.debug('_search(%s) --> += %s', polarisation, manager.get_raster_list())
    for p in manager.get_raster_list():
        for im in p.get_images_list():
            image_list.append(im)

@when('VV-VH files are searched')
def when_searching_VV_VH(configuration, image_list, mocker):
    _search(configuration, image_list, 'VV VH')

@when('VV files are searched')
def when_searching_VV(configuration, image_list, mocker):
    _search(configuration, image_list, 'VV')

@when('VH files are searched')
def when_searching_VH(configuration, image_list, mocker):
    _search(configuration, image_list, 'VH')


# ======================================================================
# Then steps

@then('No (other) files are found')
def then_no_other_files_are_found(image_list):
    assert len(image_list) == 0

@then('VV files are found')
def then_VV_files_are_found(image_list):
    assert len(image_list) >= 2
    for i in [0, 1]:
        assert input_file_vv(i) in image_list
        image_list.remove(input_file_vv(i))

@then('VH files are found')
def then_VH_files_are_found(image_list):
    assert len(image_list) >= 2
    for i in [0, 1]:
        assert input_file_vh(i) in image_list
        image_list.remove(input_file_vh(i))
