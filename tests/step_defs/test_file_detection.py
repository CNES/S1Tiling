#!/usr/bin/env python
# -*- coding: utf-8 -*-
import fnmatch
import logging
from pathlib import Path
import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from s1tiling.libs.S1FileManager import S1FileManager

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

INPUT  = 'data_raw'

def input_file(idx):
    s1dir  = FILES[idx]['s1dir']
    s1file = FILES[idx]['s1file']
    return f'{INPUT}/{s1dir}/{s1dir}.SAFE/measurement/{s1file}'

def raster_vv(idx):
    s1dir  = FILES[idx]['s1dir']
    return (S1DateAcquisition(
        f'{INPUT}/{s1dir}/{s1dir}.SAFE/manifest.safe',
        [input_file(idx)]),
        [(14.9998201759, 1.8098185887), (15.9870050338, 1.8095484335), (15.9866155411, 0.8163071941), (14.9998202469, 0.8164290331000001)])

def raster_vh(idx):
    s1dir  = FILES[idx-2]['s1dir']
    return (S1DateAcquisition(
        f'{INPUT}/{s1dir}/{s1dir}.SAFE/manifest.safe',
        [input_file(idx)]),
        [(14.9998201759, 1.8098185887), (15.9870050338, 1.8095484335), (15.9866155411, 0.8163071941), (14.9998202469, 0.8164290331000001)])

# ======================================================================
# Mocks

# resource_dir = Path(__file__).parent.parent.parent.absolute() / 's1tiling/resources'

class Configuration():
    def __init__(self, *argv):
        """
        constructor
        """
        self.first_date      = '2020-01-01'
        self.last_date       = '2020-01-10'
        self.download        = False
        self.raw_directory   = INPUT
        # self.polarisation    = polarisation

def isfile(filename, existing_files):
    # assert False
    res = filename in existing_files
    logging.debug("isfile(%s) = %s ∈ %s", filename, res, existing_files)
    return res

@pytest.fixture
def raster_list():
    rl = []
    return rl

@pytest.fixture
def configuration():
    cfg = Configuration()
    return cfg

# ======================================================================
# Given steps

def _declare_know_files(mocker, patterns):
    all_files = [input_file(idx) for idx in range(len(FILES))]
    files = []
    for pattern in patterns:
        files += [fn for fn in all_files if fnmatch.fnmatch(fn, pattern)]
    logging.debug('Mocking w/ %s', files)
    mocker.patch('s1tiling.libs.Utils.list_dirs', return_value=files)


@given('All files are known')
def given_all_files_are_know(mocker):
    _declare_know_files(mocker, ['vv', 'vh'])

@given('All VV files are known')
def given_all_VV_files_are_know(mocker):
    _declare_know_files(mocker, ['vv'])

@given('All VH files are known')
def given_all_VH_files_are_know(mocker):
    _declare_know_files(mocker, ['vh'])

# ======================================================================
# When steps

@when('VV-VH files are searched')
def when_searching_VV_VH(configuration, raster_list):
    configuration.polarisation = 'VV VH'
    manager = S1FileManager(configuration)
    raster_list.extend(manager._update_s1_img_list())

@when('VV files are searched')
def when_searching_VV(configuration, raster_list):
    configuration.polarisation = 'VV'
    raster_list.extend(manager._update_s1_img_list())

@when('VH files are searched')
def when_searching_VH(configuration, raster_list):
    configuration.polarisation = 'VH'
    raster_list.extend(manager._update_s1_img_list())


# ======================================================================
# Then steps

@then('No (other) files are found')
def then_no_other_files_are_found(raster_list):
    assert len(raster_list) == 0

@then('VV files are found')
def then_VV_files_are_found(raster_list):
    assert len(products) >= 2
    for i in [0, 1]:
        assert raster_vv(i) in raster_list
        raster_list.remove(raster_vv(i))

@then('VH files are found')
def then_VH_files_are_found(raster_list):
    assert len(products) >= 2
    for i in [0, 1]:
        assert raster_vh(i) in raster_list
        raster_list.remove(raster_vh(i))
