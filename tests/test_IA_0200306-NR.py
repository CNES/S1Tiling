#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2025 (c) CNES.
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
# Authors:
# - Thierry KOLECK (CNES)
# - Luc HERMITTE (CSGROUP)
#
# =========================================================================

"""Functional / non regression tests for (Ellipsoid) Incidence Angle Computations"""


import logging
import os
import pathlib
import shutil
import subprocess

import otbApplication as otb

import pytest

from s1tiling.libs.outcome import DownloadOutcome

# import pytest_check
from .mock_otb     import OTBApplicationsMockContext
from .mock_data    import FileDB
from .mock_helpers import declare_know_files
import s1tiling.libs.configuration
from s1tiling.libs.api         import s1_process_ia
from s1tiling.libs.steps       import ram as param_ram

# ======================================================================
# Full processing versions
# ======================================================================

nodata_SAR=0
nodata_DEM=-32768
nodata_XYZ='nan'
nodata_IA='nan'


def remove_dirs(dir_list) -> None:
    for dir in dir_list:
        if os.path.isdir(dir):
            logging.info("rm -r '%s'", dir)
            shutil.rmtree(dir)


def process(tmpdir, outputdir, iadir, baseline_reference_outputs, test_file, watch_ram, dirs_to_clean=None):
    '''
    Executes the S1Processor
    '''
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    src_dir       = crt_dir.parent.absolute()
    dirs_to_clean = dirs_to_clean or [outputdir, tmpdir/'S1', tmpdir/'S2', iadir]

    logging.info('$S1TILING_TEST_DATA_INPUT  -> %s', os.environ['S1TILING_TEST_DATA_INPUT'])
    logging.info('$S1TILING_TEST_DATA_OUTPUT -> %s', os.environ['S1TILING_TEST_DATA_OUTPUT'])
    logging.info('$S1TILING_TEST_DATA_IA     -> %s', os.environ['S1TILING_TEST_DATA_IA'])
    logging.info('$S1TILING_TEST_SRTM        -> %s', os.environ['S1TILING_TEST_SRTM'])
    logging.info('$S1TILING_TEST_TMPDIR      -> %s', os.environ['S1TILING_TEST_TMPDIR'])
    logging.info('$S1TILING_TEST_DOWNLOAD    -> %s', os.environ['S1TILING_TEST_DOWNLOAD'])
    logging.info('$S1TILING_TEST_RAM         -> %s', os.environ['S1TILING_TEST_RAM'])

    remove_dirs(dirs_to_clean)

    args = ['python3', src_dir / 's1tiling/S1IAMap.py', test_file]
    if watch_ram:
        args.append('--watch-ram')
    # args.append('--cache-before-ortho')
    logging.info('Running: %s', args)
    return subprocess.call(args, cwd=crt_dir)


# ======================================================================
# Mocked versions
# ======================================================================

def set_environ_mocked(inputdir, outputdir, iadir, tmpdir, ram):
    os.environ['S1TILING_TEST_DOWNLOAD']       = 'False'
    os.environ['S1TILING_TEST_DATA_INPUT']         = str(inputdir)
    os.environ['S1TILING_TEST_DATA_OUTPUT']        = str(outputdir.absolute())
    os.environ['S1TILING_TEST_DATA_IA']            = str(iadir.absolute())
    os.environ['S1TILING_TEST_SRTM']               = 'UNUSED'
    os.environ['S1TILING_TEST_TMPDIR']             = str(tmpdir.absolute())
    os.environ['S1TILING_TEST_RAM']                = str(ram)


class MockedSentinelOrbitFile:
    def __init__(self, filename: str, mission: str):
        self.filename = filename
        self.mission  = mission


# ======================================================================
def mock_IA(application_mocker: OTBApplicationsMockContext, file_db: FileDB):
    spacing=10.0

    # ComputeGroundAndSatPositionsOnEllipsoid
    application_mocker.set_expectations('SARComputeGroundAndSatPositionsOnEllipsoid', {
        'ram'        : param_ram(2048),
        'outputs.spacingx': 10.0,
        'outputs.spacingy': -10.0,
        'outputs.sizex'   : 10980,
        'outputs.sizey'   : 10980,
        'map'             : 'utm',
        'map.utm.zone'    : 33,
        'map.utm.northhem': True,
        'outputs.ulx'     : 499979.99999484676,
        'outputs.uly'     : 200040.0000009411,
        'ineof'      : file_db.eof_for_s2(),
        'inrelorb'   : file_db.relorb_for_s2(),
        'withxyz'    : True,
        'withsatpos' : True,
        'nodata'     : nodata_XYZ,
        'out'        : file_db.xyz_ellipsoid_on_s2(True),
    }, None, {
        # 'ACQUISITION_DATETIME'       : file_db.start_time(0),
        # 'DEM_LIST'                   : ', '.join(exp_dem_names),
        'S2_TILE_CORRESPONDING_CODE' : '33NWB',
        'SPATIAL_RESOLUTION'         : f"{spacing}",
        'EOF_FILE'                   : os.path.basename(file_db.eof_for_s2()),
        'IMAGE_TYPE'                 : 'XYZ',
        'TIFFTAG_IMAGEDESCRIPTION'   : 'XYZ surface and satellite positions on S2 tile on ellipsoid',
        'ORTHORECTIFIED'             : 'true',
        # meta already set during orthorectification
        # 'ORBIT_DIRECTION'          : 'DES',
        # 'ORBIT_NUMBER'             : '030704',
        # 'RELATIVE_ORBIT_NUMBER'    : '{:0>3d}'.format(file_db.relorb_for_s2()),
    })

    # ExtractNormalVector
    application_mocker.set_expectations('ExtractNormalVectorToEllipsoid', {
        'ram'             : param_ram(2048),
        'outputs.spacingx': 10.0,
        'outputs.spacingy': -10.0,
        'outputs.sizex'   : 10980,
        'outputs.sizey'   : 10980,
        'map'             : 'utm',
        'map.utm.zone'    : 33,
        'map.utm.northhem': True,
        'outputs.ulx'     : 499979.99999484676,
        'outputs.uly'     : 200040.0000009411,
        'out'             : 'SARComputeIncidenceAngle|>'+file_db.degia_on_s2(True),
    }, None, {
        'IMAGE_TYPE'                 : 'Normals',
        'ORTHORECTIFIED'             : 'true',
        'S2_TILE_CORRESPONDING_CODE' : '33NWB',
        'SPATIAL_RESOLUTION'         : f"{spacing}",
        'TIFFTAG_IMAGEDESCRIPTION' : 'Image normals on Sentinel-{flying_unit_code_short} IW GRD',
    })

    # ComputeIA
    application_mocker.set_expectations('SARComputeIncidenceAngle', {
        'in.normals'      : 'Ø|>ExtractNormalVectorToEllipsoid', #'ComputeNormals|>'+file_db.normalsfile(idx),
        'ram'             : param_ram(2048),
        'in.xyz'          : file_db.xyz_ellipsoid_on_s2(False),
        'nodata'          : nodata_IA,
        'out.sin'         : file_db.sinia_on_s2(True),
        'out.deg'         : file_db.degia_on_s2(True),
    }, {'out.deg': otb.ImagePixelType_uint16}, {
        'DATA_TYPE'                : ['sin(IA)', '100 * degrees(IA)'],
        'IMAGE_TYPE'               : 'IA',
        'TIFFTAG_IMAGEDESCRIPTION' : ['sin(IA) on S2 grid', '100 * degrees(IA) on S2 grid'],
    })


# ======================================================================
@pytest.mark.parametrize("maps",
                         [
                             # ["deg"],
                             ["deg", "sin"],
                             # ["cos", "tan"],
                             # ["cos", "deg", "sin", "tan"],
                         ])
def test_33NWB_202001_ia_mocked(
        baselinedir, outputdir, iadir, eofdir, tmpdir, ram,
        mocker,
        maps,
):
    """
    Mocked test of production of IA maps
    """
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)

    inputdir = str((baselinedir/'inputs').absolute())
    set_environ_mocked(inputdir, outputdir, iadir, tmpdir, ram)

    tile = '33NWB'

    # baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    configuration = s1tiling.libs.configuration.Configuration(test_file, do_show_configuration=False)
    configuration.extra_directories['ia_dir'] = iadir.absolute()
    configuration.ia_maps_to_produce          = maps
    configuration.relative_orbit_list         = [7]
    configuration.show_configuration()
    logging.info("Sigma0 NORMLIM mocked test")

    # file_db = FileDB(inputdir, eofdir, tmpdir.absolute(), outputdir.absolute(), iadir.absolute(), tile, demdir="UNUSED", geoid_file=None)
    file_db = FileDB(
        inputdir=inputdir,
        eofdir=eofdir,
        tmpdir=tmpdir.absolute(),
        outputdir=outputdir.absolute(),
        xiadir=iadir.absolute(),
        gamma_areadir="UNUSED",
        tile=tile,
        demdir="UNUSED",
        geoid_file=None,
    )
    mocker.patch('s1tiling.libs.otbtools.otb_version', lambda : '7.4.0')
    eof_file = os.path.join(eofdir, 'S1A_OPER_AUX_POEORB_OPOD_20210316T205443_V20200108T225942_20200110T005942.EOF')
    mocked_eof = MockedSentinelOrbitFile(eof_file, 'S1A')
    mocker.patch('s1tiling.libs.orbit._manager.EOFFileManager.search_for',
                 lambda slf, obts: [DownloadOutcome({obt: mocked_eof}) for obt in obts])

    application_mocker = OTBApplicationsMockContext(configuration, mocker, file_db.tmp_to_out_map, dem_files=[])
    known_files = application_mocker.known_files
    known_files.append(eof_file)
    known_dirs = set()
    declare_know_files(mocker, known_files, known_dirs, tile, ['vv'], file_db, application_mocker)
    assert os.path.isfile(file_db.input_file_vv(0))  # Check mocking
    assert os.path.isfile(file_db.input_file_vv(1))

    mock_IA(application_mocker, file_db)

    s1_process_ia(config_opt=configuration,
              dryrun=False, debug_otb=True, watch_ram=False,
              debug_tasks=False)
    application_mocker.assert_all_have_been_executed()
    application_mocker.assert_all_metadata_match()
