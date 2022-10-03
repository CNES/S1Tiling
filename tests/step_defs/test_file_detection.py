#!/usr/bin/env python
# -*- coding: utf-8 -*-
import fnmatch
import logging
import os
from pathlib import Path
import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from tests.mock_otb import isfile, isdir, glob, dirname
from tests.mock_data import FileDB
import s1tiling.libs.Utils
from s1tiling.libs.S1FileManager import S1FileManager
from s1tiling.libs.S1DateAcquisition import S1DateAcquisition

# ======================================================================
# Scenarios
scenarios('../features/test_file_detection.feature')

# ======================================================================
# Test Data

TMPDIR = 'TMP'
INPUT  = 'data_raw'
OUTPUT = 'OUTPUT'
LIADIR = 'LIADIR'
TILE   = '33NWB'

file_db = FileDB(INPUT, TMPDIR, OUTPUT, LIADIR, TILE, 'unused', 'unused')

def safe_dir(idx):
    return file_db.safe_dir(idx)

def input_file(idx, polarity):
    return file_db.input_file(idx, polarity)

def input_file_vv(idx):
    return file_db.input_file(idx, 'vv')

def input_file_vh(idx):
    return file_db.input_file(idx, 'vh')

# ======================================================================
# Mocks

class Configuration():
    def __init__(self, inputdir, tmpdir, outputdir, *argv):
        """
        constructor
        """
        self.first_date          = '2020-01-01'
        self.last_date           = '2020-01-10'
        self.download            = False
        self.raw_directory       = inputdir
        self.tmpdir              = tmpdir
        self.output_preprocess   = outputdir
        self.cache_srtm_by       = 'symlink'
        self.fname_fmt           = {}
        self.orbit_direction     = None
        self.relative_orbit_list = []

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

# ======================================================================
# Given steps

def _declare_known_S1_files(mocker, known_files, known_dirs, patterns):
    # logging.debug('_declare_know_files(%s)', patterns)
    # all_files = [input_file(idx) for idx in range(len(FILES))]
    all_files = file_db.all_vvvh_files()
    # logging.debug('All files:')
    # for a in all_files:
    #     logging.debug(' - %s', Path(*Path(a).parts[-3:]))
    assert file_db.input_file(0, 'vv') != file_db.input_file(0, 'vh')
    assert all_files[0] != all_files[1]
    files = []
    for pattern in patterns:
        files += [fn for fn in all_files if fnmatch.fnmatch(fn, '*'+pattern+'*')]
    known_files.extend(files)
    # for k in known_files:
        # logging.debug(' - %s', k)
    known_dirs.update([dirname(fn, 3) for fn in known_files])
    logging.debug('Mocking w/ %s --> %s', patterns, files)
    mocker.patch('glob.glob', lambda pat : glob(pat, known_files))
    # Utils.list_dirs has been imported in S1FileManager. This is the one that needs patching!
    mocker.patch('s1tiling.libs.S1FileManager.list_dirs', lambda dir, pat : list_dirs(dir, pat, known_dirs))
    # Utils.get_orbit_direction has been imported in S1FileManager. This is the one that needs patching!
    mocker.patch('s1tiling.libs.S1FileManager.get_orbit_direction', lambda manifest : 'DES')
    mocker.patch('s1tiling.libs.S1FileManager.get_relative_orbit', lambda manifest : 7)
    mocker.patch('s1tiling.libs.S1FileManager.S1FileManager._filter_products_with_enough_coverage', lambda slf, pi: slf._products_info)


@given('All S1 files are known')
def given_all_S1_files_are_know(mocker, known_files, known_dirs):
    _declare_known_S1_files(mocker, known_files, known_dirs, ['vv', 'vh'])

@given('All S1 VV files are known')
def given_all_S1_VV_files_are_know(mocker, known_files, known_dirs):
    _declare_known_S1_files(mocker, known_files, known_dirs, ['vv'])

@given('All S1 VH files are known')
def given_all_S1_VH_files_are_know(mocker, known_files, known_dirs):
    _declare_known_S1_files(mocker, known_files, known_dirs, ['vh'])


def _declare_known_S2_files(mocker, known_files, known_dirs):
    pass


@given('All S2 files are known')
def given_all_S2_files_are_know(mocker, known_files, known_dirs):
    _declare_known_S2_files(mocker, known_files, known_dirs, ['vv', 'vh'])

# ======================================================================
# When steps

def _search(configuration, image_list, polarisation):
    configuration.polarisation = polarisation
    manager = S1FileManager(configuration)
    manager._refresh_s1_product_list()
    manager._update_s1_img_list_for('33NWB')
    logging.debug('_search(%s) --> += %s', polarisation, manager.get_raster_list())
    for p in manager.get_raster_list():
        # logging.debug(" * %s", p.get_manifest())
        for im in p.get_images_list():
            # logging.debug("   -> %s", im)
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
