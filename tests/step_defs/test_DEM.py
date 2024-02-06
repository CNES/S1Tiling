#!/usr/bin/env python
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
#
# =========================================================================

import fnmatch
import logging
from pathlib import Path
from typing import List, Set


# import shapely

import pytest
from pytest_bdd import scenarios, given, when, then

# from tests.mock_otb  import isdir, glob, dirname
from tests.mock_data import FileDB
import s1tiling.libs.Utils as Utils
from s1tiling.libs.configuration import resource_dir

# ======================================================================
# Scenarios
scenarios(
        '../features/test_DEM_tile_search.feature',
        )

# ======================================================================
# Test Data

TMPDIR = 'TMP'
INPUT  = 'INPUT'
OUTPUT = 'OUTPUT'
LIADIR = 'LIADIR'
TILE   = '33NWB'

file_db = FileDB(INPUT, TMPDIR, OUTPUT, LIADIR, TILE, 'unused', 'unused')

# ======================================================================
# Fixtures

def _declare_known_S1_files(known_files, patterns) -> None:
    # logging.debug('_declare_known_files(%s)', patterns)
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
    logging.debug('Mocking w/ S1: %s', patterns)
    for file in files:
        logging.debug('--> %s', file)


@pytest.fixture
def known_files() -> List[str]:
    kf = []
    return kf


@pytest.fixture
def mocked_get_origin(mocker):
    mocker.patch('s1tiling.libs.Utils.get_origin', lambda manifest : file_db.get_origin(manifest))
    return mocker


# ======================================================================
# Given steps

@given('All S1 VV files are known')
def given_all_S1_VV_files_are_known(known_files) -> None:
    # logging.debug('Given: All S1 VV files are known')
    _declare_known_S1_files(known_files, ['vv'])


@given("The shipped WGS84 DEM database", target_fixture="dem_db")
def given_the_shipped_WGS84_DEM_database(mocked_get_origin) -> Path:
    DEMShapefile       = Path(resource_dir / 'shapefile' / 'srtm_tiles.gpkg')
    assert DEMShapefile.exists(), f"{DEMShapefile} doesn't exist"
    return DEMShapefile


@given("The DEM database converted to Lambert93", target_fixture="dem_db")
def given_the_DEM_database_converted_to_Lambert93(mocked_get_origin, baseline_dir) -> Path:
    DEMShapefile       = Path(baseline_dir / 'MNT' / 'srtm-lambert.gpkg')
    assert DEMShapefile.exists(), f"{DEMShapefile} doesn't exist"
    return DEMShapefile


# ======================================================================
# When steps

@when("S1 files footprints are searched in DEM database", target_fixture="matching_DEMS")
def when_S1_files_footprints_are_searched_in_DEM_database(dem_db: Path, known_files) -> List[Set[str]]:
    the_dem_db = Utils.Layer(str(dem_db))
    res = []
    for s1_file in known_files:
        shape = Utils.get_shape(s1_file)
        logging.debug("shape of %s = %s", s1_file, shape)
        dems = Utils.find_dem_intersecting_poly(shape, the_dem_db, ["id"], "id")
        # dems = Utils.find_dem_intersecting_raster(s1_file, str(dem_db[0]), ["id"], "id")
        logging.debug("%s intersects %s", s1_file, dems)
        res.append(set(dems.keys()))
    return res


# ======================================================================
# Then steps

@then("The expected DEM files are found")
def then_The_expected_DEM_files_are_found(matching_DEMS, known_files) -> None:
    for s1_file, dems in zip(known_files, matching_DEMS):
        idx = file_db._find_image(s1_file)
        assert set(file_db.dem_coverage(idx)) == dems
