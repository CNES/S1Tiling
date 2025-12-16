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
#          Fabien CONTIVAL (CS Group)
#
# =========================================================================

from datetime import datetime, timedelta
import logging
import os
import pathlib
import subprocess
from typing import cast

import otbApplication as otb

import pytest
from s1tiling.libs import Utils

from s1tiling.libs.otbtools import otb_version
from s1tiling.libs.outcome import DownloadOutcome
from s1tiling.libs.utils.formatters import ExtendedFormatter
# from unittest.mock import patch

# import pytest_check
from .helpers      import otb_compare, comparable_metadata
from .mock_otb     import OTBApplicationsMockContext
from .mock_data    import FileDB
from .mock_helpers import declare_know_files, remove_dirs
# import s1tiling.S1Processor
import s1tiling.libs.configuration
from s1tiling.libs.api         import (
    s1_process,
    s1_process_lia_v0, register_LIA_pipelines_v0, s1_process_lia_v1_1, s1_process_lia_v1_2,
    s1_process_gamma_area, register_GAMMA_AREA_pipelines,
)
from s1tiling.libs.steps       import ram as param_ram
from s1tiling.libs.otbwrappers import AgglomerateDEMOnS1, AgglomerateDEMOnS2, AnalyseBorders, Concatenate


def to_datetime(s: str) -> datetime:
    return datetime.strptime(s, '%Y:%m:%d %H:%M:%S')


@pytest.fixture
def naming_policy() -> str:
    return 'with_calibration'


# ======================================================================
# Full processing versions
# ======================================================================

nodata_SAR=0
nodata_DEM=-32768
nodata_XYZ='nan'
nodata_LIA='nan'
nodata_RTC='nan'


def process(tmpdir, outputdir, liadir, gamma_areadir, baseline_reference_outputs, test_file, watch_ram, dirs_to_clean=None):
    '''
    Executes the S1Processor
    '''
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    src_dir       = crt_dir.parent.absolute()
    dirs_to_clean = dirs_to_clean or [outputdir, tmpdir/'S1', tmpdir/'S2', liadir, gamma_areadir]

    logging.info('$S1TILING_TEST_DATA_INPUT      -> %s', os.environ['S1TILING_TEST_DATA_INPUT'])
    logging.info('$S1TILING_TEST_DATA_OUTPUT     -> %s', os.environ['S1TILING_TEST_DATA_OUTPUT'])
    logging.info('$S1TILING_TEST_DATA_LIA        -> %s', os.environ['S1TILING_TEST_DATA_LIA'])
    logging.info('$S1TILING_TEST_DATA_GAMMA_AREA -> %s', os.environ['S1TILING_TEST_DATA_GAMMA_AREA'])
    logging.info('$S1TILING_TEST_SRTM            -> %s', os.environ['S1TILING_TEST_SRTM'])
    logging.info('$S1TILING_TEST_TMPDIR          -> %s', os.environ['S1TILING_TEST_TMPDIR'])
    logging.info('$S1TILING_TEST_DOWNLOAD        -> %s', os.environ['S1TILING_TEST_DOWNLOAD'])
    logging.info('$S1TILING_TEST_RAM             -> %s', os.environ['S1TILING_TEST_RAM'])

    remove_dirs(dirs_to_clean)

    args = ['python3', src_dir / 's1tiling/S1Processor.py', test_file]
    if watch_ram:
        args.append('--watch-ram')
    # args.append('--cache-before-ortho')
    logging.info('Running: %s', args)
    return subprocess.call(args, cwd=crt_dir)


@pytest.mark.slow
def test_33NWB_202001_NR_execute_OTB(baselinedir, outputdir, liadir, gamma_areadir, tmpdir, demdir, ram, download, watch_ram):
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)
    # In all cases, the baseline is required for the reference outputs
    # => We need it
    assert os.path.isdir(baselinedir), \
        ("No baseline found in '%s', please run minio-client to fetch it with:\n"+\
        "?> mc cp --recursive minio-otb/s1-tiling/baseline '%s'") % (baselinedir, baselinedir.absolute(),)

    if download:
        inputdir                                   = (tmpdir/'data_raw').absolute()
        os.environ['S1TILING_TEST_DOWNLOAD']       = 'True'
        os.environ['S1TILING_TEST_OVERRIDE_CUT_Y'] = 'None' # Use default
    else:
        inputdir                                   = str((baselinedir/'inputs').absolute())
        os.environ['S1TILING_TEST_DOWNLOAD']       = 'False'
        os.environ['S1TILING_TEST_OVERRIDE_CUT_Y'] = 'False' # keep everything

    os.environ['S1TILING_TEST_DATA_INPUT']         = str(inputdir)
    os.environ['S1TILING_TEST_DATA_OUTPUT']        = str(outputdir.absolute())
    os.environ['S1TILING_TEST_DATA_LIA']           = str(liadir.absolute())
    os.environ['S1TILING_TEST_DATA_GAMMA_AREA']    = str(gamma_areadir.absolute())
    os.environ['S1TILING_TEST_SRTM']               = str(demdir.absolute())
    os.environ['S1TILING_TEST_TMPDIR']             = str(tmpdir.absolute())
    os.environ['S1TILING_TEST_RAM']                = str(ram)

    resources_dir = crt_dir.parent.absolute() / 's1tiling/resources'
    os.environ['S1TILING_RESOURCES']               = str(resources_dir)

    # images = [
    #         '33NWB/s1a_33NWB_vh_DES_007_20200108txxxxxx.tif',
    #         '33NWB/s1a_33NWB_vv_DES_007_20200108txxxxxx.tif',
    #         '33NWB/s1a_33NWB_vh_DES_007_20200108txxxxxx_BorderMask.tif',
    #         '33NWB/s1a_33NWB_vv_DES_007_20200108txxxxxx_BorderMask.tif',
    #         ]
    baseline_path = baselinedir / 'expected'
    if otb_version() >= '8.0.0':
        baseline_path = baseline_path / 'otb8'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    logging.info("Full test")
    EX = process(tmpdir, outputdir, liadir, gamma_areadir, baseline_path, test_file, watch_ram)
    assert EX == 0
    descr_ortho = 'sigma calibrated orthorectified Sentinel-1A IW GRD'
    descr_mask  = 'Orthorectified Sentinel-1A IW GRD smoothed border mask S2 tile'
    for kind, descr, image_type in zip(['', '_BorderMask'], [descr_ortho, descr_mask], ['BACKSCATTERING', 'MASK']):
        for polar in ['vh', 'vv']:
            im = f'33NWB/s1a_33NWB_{polar}_DES_007_20200108txxxxxx{kind}.tif'
            expected = baseline_path / im
            produced = outputdir / im
            assert os.path.isfile(produced)
            assert otb_compare(expected, produced) == 0, \
                    ("Comparison of %s against %s failed" % (produced, expected))
            # expected_md = comparable_metadata(expected)
            expected_md = {
                'ACQUISITION_DATETIME'           : '2020:01:08T04:41:50Z',
                'ACQUISITION_DATETIME_1'         : '2020:01:08T04:41:50Z',
                'ACQUISITION_DATETIME_2'         : '2020:01:08T04:42:15Z',
                'AREA_OR_POINT'                  : 'Area',
                'CALIBRATION'                    : 'sigma',
                'DEM_INFO'                       : 'SRTM_30_hgt',
                'FLYING_UNIT_CODE'               : 's1a',
                'IMAGE_TYPE'                     : image_type,
                'INPUT_S1_IMAGES'                : 'S1A_IW_GRDH_1SDV_20200108T044150_20200108T044215_030704_038506_C7F5, S1A_IW_GRDH_1SDV_20200108T044215_20200108T044240_030704_038506_D953',
                'NOISE_REMOVED'                  : 'False',
                'ORBIT_DIRECTION'                : 'DES',
                'ORBIT_NUMBER'                   : '030704',
                'ORTHORECTIFICATION_INTERPOLATOR': 'nn',
                'ORTHORECTIFIED'                 : 'true',
                'POLARIZATION'                   : polar,
                'RELATIVE_ORBIT_NUMBER'          : '007',
                'S2_TILE_CORRESPONDING_CODE'     : '33NWB',
                'SPATIAL_RESOLUTION'             : '10.0',
                'TIFFTAG_IMAGEDESCRIPTION'       : descr,
                'TIFFTAG_SOFTWARE'               : 'S1 Tiling',
            }
            assert expected_md == comparable_metadata(produced)
        # The following line permits to test otb_compare correctly detect differences when
        # called from pytest.
        # assert otb_compare(baseline_path+images[0], result_path+images[1]) == 0


@pytest.mark.slow
def test_33NWB_202001_NR_masks_only_execute_OTB(baselinedir, outputdir, liadir, gamma_areadir, tmpdir, demdir, ram, download, watch_ram):
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)
    # In all cases, the baseline is required for the reference outputs
    # => We need it
    assert os.path.isdir(baselinedir), \
        ("No baseline found in '%s', please run minio-client to fetch it with:\n"+\
        "?> mc cp --recursive minio-otb/s1-tiling/baseline '%s'") % (baselinedir, baselinedir.absolute(),)

    if download:
        inputdir                                   = (tmpdir/'data_raw').absolute()
        os.environ['S1TILING_TEST_DOWNLOAD']       = 'True'
        os.environ['S1TILING_TEST_OVERRIDE_CUT_Y'] = 'None' # Use default
    else:
        inputdir                                   = str((baselinedir/'inputs').absolute())
        os.environ['S1TILING_TEST_DOWNLOAD']       = 'False'
        os.environ['S1TILING_TEST_OVERRIDE_CUT_Y'] = 'False' # keep everything

    os.environ['S1TILING_TEST_DATA_INPUT']         = str(inputdir)
    os.environ['S1TILING_TEST_DATA_OUTPUT']        = str(outputdir.absolute())
    os.environ['S1TILING_TEST_DATA_LIA']           = str(liadir.absolute())
    os.environ['S1TILING_TEST_DATA_GAMMA_AREA']    = str(gamma_areadir.absolute())
    os.environ['S1TILING_TEST_SRTM']               = str(demdir.absolute())
    os.environ['S1TILING_TEST_TMPDIR']             = str(tmpdir.absolute())
    os.environ['S1TILING_TEST_RAM']                = str(ram)

    resources_dir = crt_dir.parent.absolute() / 's1tiling/resources'
    os.environ['S1TILING_RESOURCES']               = str(resources_dir)

    images = [
            '33NWB/s1a_33NWB_vh_DES_007_20200108txxxxxx_BorderMask.tif',
            '33NWB/s1a_33NWB_vv_DES_007_20200108txxxxxx_BorderMask.tif',
    ]
    baseline_path = baselinedir / 'expected'
    if otb_version() >= '8.0.0':
        baseline_path = baseline_path / 'otb8'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'

    logging.info("Mask only test")
    # Fake remaining things to do
    remove_dirs([outputdir])
    os.makedirs(outputdir / '33NWB')
    start_points = [
            '33NWB/s1a_33NWB_vh_DES_007_20200108txxxxxx.tif',
            '33NWB/s1a_33NWB_vv_DES_007_20200108txxxxxx.tif'
    ]
    for pt in start_points:
        os.symlink(baseline_path/pt, outputdir/pt)


    dirs_to_clean = [tmpdir/'S1', tmpdir/'S2'] # do not clear outputdir in that case
    EX = process(tmpdir, outputdir, liadir, gamma_areadir, baseline_path, test_file, watch_ram, dirs_to_clean)
    assert EX == 0
    for im, polar in zip(images, ['vh', 'vv']):
        expected = baseline_path / im
        produced = outputdir / im
        assert os.path.isfile(produced)
        assert otb_compare(expected, produced) == 0, \
                ("Comparison of %s against %s failed" % (produced, expected))
        # expected_md = comparable_metadata(expected)
        expected_md = {
            'ACQUISITION_DATETIME'           : '2020:01:08T04:41:50Z',  # Start point has it
            # For now, the start points don't have this...
            'ACQUISITION_DATETIME_1'         : '2020:01:08T04:41:50Z',
            'ACQUISITION_DATETIME_2'         : '2020:01:08T04:42:15Z',
            'AREA_OR_POINT'                  : 'Area',
            'CALIBRATION'                    : 'sigma',
            'DEM_INFO'                       : 'SRTM_30_hgt',
            'FLYING_UNIT_CODE'               : 's1a',
            'IMAGE_TYPE'                     : 'MASK',
            'INPUT_S1_IMAGES'                : 'S1A_IW_GRDH_1SDV_20200108T044150_20200108T044215_030704_038506_C7F5, S1A_IW_GRDH_1SDV_20200108T044215_20200108T044240_030704_038506_D953',
            # For now, the start points don't have this...
            'NOISE_REMOVED'                  : 'False',
            'ORBIT_DIRECTION'                : 'DES',
            'ORBIT_NUMBER'                   : '030704',
            'ORTHORECTIFICATION_INTERPOLATOR': 'nn',
            'ORTHORECTIFIED'                 : 'true',
            'POLARIZATION'                   : polar,
            'RELATIVE_ORBIT_NUMBER'          : '007',
            'S2_TILE_CORRESPONDING_CODE'     : '33NWB',
            'SPATIAL_RESOLUTION'             : '10.0',
            'TIFFTAG_IMAGEDESCRIPTION'       : 'Orthorectified Sentinel-1A IW GRD smoothed border mask S2 tile',
            'TIFFTAG_SOFTWARE'               : 'S1 Tiling',
        }
        assert expected_md == comparable_metadata(produced)


# ======================================================================
# Mocked versions
# ======================================================================

def set_environ_mocked(inputdir, outputdir, liadir, gamma_areadir, demdir, tmpdir, ram):
    os.environ['S1TILING_TEST_DOWNLOAD']       = 'False'
    os.environ['S1TILING_TEST_OVERRIDE_CUT_Y'] = 'False' # keep everything

    os.environ['S1TILING_TEST_DATA_INPUT']         = str(inputdir)
    os.environ['S1TILING_TEST_DATA_OUTPUT']        = str(outputdir.absolute())
    os.environ['S1TILING_TEST_DATA_LIA']           = str(liadir.absolute())
    os.environ['S1TILING_TEST_DATA_GAMMA_AREA']    = str(gamma_areadir.absolute())
    os.environ['S1TILING_TEST_SRTM']               = str(demdir.absolute())
    os.environ['S1TILING_TEST_TMPDIR']             = str(tmpdir.absolute())
    os.environ['S1TILING_TEST_RAM']                = str(ram)

    crt_dir       = pathlib.Path(__file__).parent.absolute()
    resources_dir = crt_dir.parent.absolute() / 's1tiling/resources'
    os.environ['S1TILING_RESOURCES']               = str(resources_dir)


k_calib_convert = {
    'normlim': 'beta',
    'gamma_naught_rtc': 'sigma',
}


def _expected_calibration(
        file_db           : FileDB,
    naming_policy: str,
    calibration  : str,
) -> str:
    return ExtendedFormatter().format(file_db.CALIBRATION_CONVERTER[naming_policy], calibration=calibration)


def mock_upto_concat_S2(
        application_mocker: OTBApplicationsMockContext,
        file_db           : FileDB,
        naming_policy     : str,
        calibration       : str,
        N                 : int,
        old_IPF           : bool=False
):
    assert calibration[0] != '_'
    raw_calibration = k_calib_convert.get(calibration, calibration)
    for i in range(N):
        orbit_info = file_db.get_orbit_information(i)
        input_file = file_db.input_file_vv(i)
        # expected_ortho_file = file_db.orthofile(i, False)

        orthofile = file_db.orthofile(i, True, naming_policy='with_calibration', calibration=f'_{raw_calibration}')
        expected_calibration = _expected_calibration(file_db, 'with_calibration', raw_calibration)
        assert f'_{expected_calibration}' in orthofile
        assert '__' not in orthofile

        # Workaround defect on skipping cut margins
        out_calib = ('ResetMargin|>OrthoRectification|>' if old_IPF else 'OrthoRectification|>' )+orthofile
        in_ortho  = input_file+('|>SARCalibration|>ResetMargin' if old_IPF else '|>SARCalibration')
        # out_calib = ('ResetMargin|>OrthoRectification|>')+orthofile
        # in_ortho  = input_file+('|>SARCalibration|>ResetMargin')

        application_mocker.set_expectations(
            'SARCalibration',
            {
                'ram'        : param_ram(2048),
                'in'         : input_file,
                'lut'        : raw_calibration,
                'removenoise': False,
                # 'out'        : 'ResetMargin|>OrthoRectification|>'+orthofile,
                'out'        : out_calib,
            }, None,
            {
                'ACQUISITION_DATETIME'        : file_db.start_time(i),
                'CALIBRATION'                 : raw_calibration,
                'FLYING_UNIT_CODE'            : 's1a',
                'IMAGE_TYPE'                  : 'GRD',
                'INPUT_S1_IMAGES'             : file_db.product_name(i),
                'NOISE_REMOVED'               : 'False',
                'RELATIVE_ORBIT_NUMBER'       : '{:0>3d}'.format(orbit_info['relative_orbit']),
                'ORBIT_NUMBER'                : '{:0>6d}'.format(orbit_info['absolute_orbit']),
                'ORBIT_DIRECTION'             : 'DES',
                'POLARIZATION'                : 'vv',
                'AbsoluteCalibrationConstant' : '',
                'AcquisitionDate'             : '',
                'AcquisitionStartTime'        : '',
                'AcquisitionStopTime'         : '',
                'AverageSceneHeight'          : '',
                'BeamMode'                    : '',
                'BeamSwath'                   : '',
                'BlueDisplayChannel'          : '',
                'GreenDisplayChannel'         : '',
                'Instrument'                  : '',
                'LineSpacing'                 : '',
                'Mission'                     : '',
                'Mode'                        : '',
                'NumberOfColumns'             : '',
                'NumberOfLines'               : '',
                'OrbitDirection'              : '',
                'OrbitNumber'                 : '',
                'PRF'                         : '',
                'PixelSpacing'                : '',
                'RadarFrequency'              : '',
                'RedDisplayChannel'           : '',
                'SAR'                         : '',
                'SARCalib*'                   : '',
                'SensorID'                    : '',
                'Swath'                       : '',
            })

        if old_IPF:     #  workaround defect on skipping cutmargin
            application_mocker.set_expectations(
                'ResetMargin',
                {
                    'in'               : input_file+'|>SARCalibration',
                    'ram'              : param_ram(2048),
                    'threshold.x'      : 1000 if old_IPF else 0,
                    'threshold.y.start': 0,
                    'threshold.y.end'  : 0,
                    'mode'             : 'threshold',
                    'out'              : 'OrthoRectification|>'+orthofile,
                }, None, None)

        application_mocker.set_expectations(
            'OrthoRectification',
            {
                # 'io.in'           : input_file+'|>SARCalibration|>ResetMargin',
                'io.in'           : in_ortho,
                'opt.ram'         : param_ram(2048),
                'interpolator'    : 'nn',
                'outputs.spacingx': 10.0,
                'outputs.spacingy': -10.0,
                'outputs.sizex'   : 10980,
                'outputs.sizey'   : 10980,
                'opt.gridspacing' : 40.0,
                'map'             : 'utm',
                'map.utm.zone'    : 33,
                'map.utm.northhem': True,
                'outputs.ulx'     : 499979.99999484676,
                'outputs.uly'     : 200040.0000009411,
                'elev.dem'        : file_db.dem_file(),
                'elev.geoid'      : file_db.GeoidFile,
                'io.out'          : orthofile,
            }, None,
            {
                'DEM_INFO'                       : 'SRTM_30_hgt',
                'ORTHORECTIFICATION_INTERPOLATOR': 'nn',
                'ORTHORECTIFIED'                 : 'true',
                'S2_TILE_CORRESPONDING_CODE'     : '33NWB',
                'SPATIAL_RESOLUTION'             : '10.0',
                'TIFFTAG_IMAGEDESCRIPTION'       : f'{raw_calibration} calibrated orthorectified Sentinel-1A IW GRD',
                'AbsoluteCalibrationConstant'    : '',
                'AcquisitionDate'                : '',
                'AcquisitionStartTime'           : '',
                'AcquisitionStopTime'            : '',
                'AverageSceneHeight'             : '',
                'BeamMode'                       : '',
                'BeamSwath'                      : '',
                'BlueDisplayChannel'             : '',
                'GreenDisplayChannel'            : '',
                'Instrument'                     : '',
                'LineSpacing'                    : '',
                'Mission'                        : '',
                'Mode'                           : '',
                'NumberOfColumns'                : '',
                'NumberOfLines'                  : '',
                'OrbitDirection'                 : '',
                'OrbitNumber'                    : '',
                'PRF'                            : '',
                'PixelSpacing'                   : '',
                'RadarFrequency'                 : '',
                'RedDisplayChannel'              : '',
                'SAR'                            : '',
                'SARCalib*'                      : '',
                'SensorID'                       : '',
                'Swath'                          : '',
            })

    if N == 1:
        # In this case, there is not a Synthetize but a call to rename.
        orthofile = file_db.orthofile(0,   False, naming_policy='with_calibration', calibration=f'_{raw_calibration}')
        out       = file_db.concatfile_from_one(0, True, naming_policy=naming_policy, calibration=f'_{calibration}')
        application_mocker.set_expectations(
            Concatenate.rename,
            [orthofile, out],
            None,
            {
                'ACQUISITION_DATETIME'     : file_db.start_time(0),
                'ACQUISITION_DATETIME_1'   : file_db.start_time(0),
                'IMAGE_TYPE'               : 'BACKSCATTERING',
                'INPUT_S1_IMAGES'          : file_db.product_name(0),
                'TIFFTAG_IMAGEDESCRIPTION' : f'{raw_calibration} calibrated orthorectified Sentinel-1A IW GRD',
            }
        )
    else:
        for i in range((N+1)//2):
            orthofile1 = file_db.orthofile(2*i,   False, naming_policy='with_calibration', calibration=f'_{raw_calibration}')
            orthofile2 = file_db.orthofile(2*i+1, False, naming_policy='with_calibration', calibration=f'_{raw_calibration}')
            application_mocker.set_expectations(
                'Synthetize',
                {
                    'ram'      : param_ram(2048),
                    'il'       : [orthofile1, orthofile2],
                    'out'      : file_db.concatfile_from_two(i, True, naming_policy=naming_policy, calibration=f'_{calibration}'),
                }, None,
                {
                    'ACQUISITION_DATETIME'     : file_db.start_time_for_two(i),
                    'ACQUISITION_DATETIME_1'   : file_db.start_time(2*i),
                    'ACQUISITION_DATETIME_2'   : file_db.start_time(2*i+1),
                    'IMAGE_TYPE'               : 'BACKSCATTERING',
                    'INPUT_S1_IMAGES'          : '%s, %s' % (file_db.product_name(2*i), file_db.product_name(2*i+1)),
                    'TIFFTAG_IMAGEDESCRIPTION' : f'{raw_calibration} calibrated orthorectified Sentinel-1A IW GRD',
                },
                {orthofile1, orthofile2},
            )


def mock_masking(application_mocker: OTBApplicationsMockContext, file_db, calibration, N, naming_policy):
    k_calibration_table = {
        'normlim'          : 'NormLim',
        'gamma_naught_rtc':  'GammaNaughtRTC',
    }
    raw_calibration = k_calibration_table.get(calibration, calibration)
    if N >= 2:
        outfile = lambda idx, tmp, calibration: file_db.maskfile_from_two(idx, tmp, naming_policy=naming_policy, calibration=calibration)
        if calibration == 'normlim':
            infile = lambda idx, tmp: file_db.sigma0_normlim_file_from_two(idx, tmp, naming_policy=naming_policy)
        elif calibration == 'gamma_naught_rtc':
            infile = lambda idx, tmp: file_db.gamma0_rtc_file_from_two(idx, tmp, naming_policy=naming_policy)
        else:
            infile = lambda idx, tmp: file_db.concatfile_from_two(idx, tmp, naming_policy=naming_policy)
    else:
        outfile = lambda idx, tmp, calibration: file_db.maskfile_from_one(idx//2, tmp, naming_policy=naming_policy, calibration=calibration)
        if calibration == 'normlim':
            infile = lambda idx, tmp: file_db.sigma0_normlim_file_from_one(idx//2, tmp, naming_policy=naming_policy)
        elif calibration == 'gamma_naught_rtc':
            infile = lambda idx, tmp: file_db.gamma0_rtc_file_from_one(idx // 2, tmp, naming_policy=naming_policy)
        else:
            infile = lambda idx, tmp: file_db.concatfile_from_one(idx//2, tmp, naming_policy=naming_policy)

    for i in range((N+1) // 2):  # Make sure to iterate even with odd number of inputs
        assert raw_calibration
        out_mask = outfile(i, True, calibration=f'_{raw_calibration}')
        expected_calibration = _expected_calibration(file_db, naming_policy, raw_calibration)
        assert f'_{expected_calibration}' in out_mask
        application_mocker.set_expectations(
            'BandMath',
            {
                'ram'      : param_ram(2048),
                'il'       : [infile(i, False)],
                'exp'      : 'im1b1==0?0:1',
                'out'      : 'BinaryMorphologicalOperation|>'+out_mask,
            }, {
                'out': otb.ImagePixelType_uint8
            }, {
                'IMAGE_TYPE'                : 'MASK',
                'TIFFTAG_IMAGEDESCRIPTION'  : 'Orthorectified Sentinel-1A IW GRD border mask S2 tile',
            })
        application_mocker.set_expectations(
            'BinaryMorphologicalOperation',
            {
                'in'       : [infile(i, False)+'|>BandMath'],
                'ram'      : param_ram(2048),
                'structype': 'ball',
                'xradius'  : 5,
                'yradius'  : 5,
                'filter'   : 'opening',
                'out'      : out_mask,
            }, {
                'out': otb.ImagePixelType_uint8
            }, {
                'TIFFTAG_IMAGEDESCRIPTION'  : 'Orthorectified Sentinel-1A IW GRD smoothed border mask S2 tile',
            })


def mock_LIA_v1_0(application_mocker: OTBApplicationsMockContext, file_db: FileDB):
    demdir = file_db.demdir
    for idx in range(2):
        orbit_info        = file_db.get_orbit_information(idx)
        cov               = file_db.dem_coverage(idx)
        exp_dem_names     = sorted(cov)
        exp_out_vrt       = file_db.vrtfile(idx, False)
        exp_out_dem       = file_db.sardemprojfile(idx, False)
        exp_in_dem_files  = [f"{demdir}/{dem}.hgt" for dem in exp_dem_names]

        application_mocker.set_expectations(
            AgglomerateDEMOnS1.agglomerate,
            [file_db.vrtfile(idx, True)] + exp_in_dem_files,
            None,
            None,
        )

        application_mocker.set_expectations(
            'SARDEMProjection2',
            {
                'ram'        : param_ram(2048),
                'insar'      : file_db.input_file_vv(idx),
                'indem'      : exp_out_vrt,
                'withxyz'    : True,
                'nodata'     : -32768,
                'out'        : file_db.sardemprojfile(idx, True),
            }, None,
            {
                'ACQUISITION_DATETIME'     : file_db.start_time(idx),
                'DEM_INFO'                 : 'SRTM_30_hgt',
                'DEM_LIST'                 : ', '.join(exp_dem_names),
                'FLYING_UNIT_CODE'         : 's1a',
                'IMAGE_TYPE'               : 'GRD',
                'INPUT_S1_IMAGES'          : file_db.product_name(idx),
                'ORBIT_DIRECTION'          : 'DES',
                'ORBIT_NUMBER'             : '{:0>6d}'.format(orbit_info['absolute_orbit']),
                'POLARIZATION'             : '',  # <=> removing the key
                'RELATIVE_ORBIT_NUMBER'    : '{:0>3d}'.format(orbit_info['relative_orbit']),
                'TIFFTAG_IMAGEDESCRIPTION' : 'SARDEM projection onto DEM list',
            },
            # {exp_out_vrt}
        )

        application_mocker.set_expectations(
            'SARCartesianMeanEstimation2',
            {
                'ram'             : param_ram(2048),
                'insar'           : file_db.input_file_vv(idx),
                'indem'           : exp_out_vrt,
                'indemproj'       : exp_out_dem,
                'indirectiondemc' : 24,
                'indirectiondeml' : 12,
                'mlran'           : 1,
                'mlazi'           : 1,
                'out'             : file_db.xyzfile(idx, True),
            }, None,
            {
                'PRJ.DIRECTIONTOSCANDEMC'  : '',  # <=> expect key removal
                'PRJ.DIRECTIONTOSCANDEML'  : '',  # <=> expect key removal
                'PRJ.GAIN'                 : '',  # <=> expect key removal
                'TIFFTAG_IMAGEDESCRIPTION' : 'Cartesian XYZ coordinates estimation',
            },
            {exp_out_vrt, exp_out_dem},
        )

        application_mocker.set_expectations(
            'ExtractNormalVector',
            {
                'ram'             : param_ram(2048),
                'xyz'             : file_db.xyzfile(idx, False),
                'nodata'          : 'nan',
                # 'nodata'          : '-32768',
                'out'             : 'SARComputeIncidenceAngle|>'+file_db.degLIAfile(idx, True),
            }, None,
            {
                'TIFFTAG_IMAGEDESCRIPTION' : 'Image normals on Sentinel-1A IW GRD',
            },
            {file_db.xyzfile(idx, False)},
        )

        application_mocker.set_expectations(
            'SARComputeIncidenceAngle',
            {
                'ram'             : param_ram(2048),
                'in.normals'      : file_db.xyzfile(idx, False)+'|>ExtractNormalVector', #'ComputeNormals|>'+file_db.normalsfile(idx),
                'in.xyz'          : file_db.xyzfile(idx, False),
                'out.deg'         : file_db.degLIAfile(idx, True),
                'out.sin'         : file_db.sinLIAfile(idx, True),
                'nodata'          : 'nan',
                # 'nodata'          : '-32768',
            }, {'out.deg': otb.ImagePixelType_uint16},
            {
                'DATA_TYPE'                : ['sin(LIA)', '100 * degrees(LIA)'],
                'IMAGE_TYPE'               : 'LIA',
                'TIFFTAG_IMAGEDESCRIPTION' : ['sin(LIA) on Sentinel-1A IW GRD', '100 * degrees(LIA) on Sentinel-1A IW GRD'],
            },
        )

        application_mocker.set_expectations(
            'OrthoRectification',
            {
                'opt.ram'         : param_ram(2048),
                'io.in'           : file_db.degLIAfile(idx, False),
                'interpolator'    : 'nn',
                'outputs.spacingx': 10.0,
                'outputs.spacingy': -10.0,
                'outputs.sizex'   : 10980,
                'outputs.sizey'   : 10980,
                'opt.gridspacing' : 40.0,
                'map'             : 'utm',
                'map.utm.zone'    : 33,
                'map.utm.northhem': True,
                'outputs.ulx'     : 499979.99999484676,
                'outputs.uly'     : 200040.0000009411,
                'elev.dem'        : file_db.dem_file(),
                'elev.geoid'      : file_db.GeoidFile,
                'io.out'          : file_db.orthodegLIAfile(idx, True),
            }, {'io.out': otb.ImagePixelType_int16},
            {
                'DATA_TYPE'                      : '100 * degrees(LIA)',
                'DEM_INFO'                       : 'SRTM_30_hgt',
                'ORTHORECTIFICATION_INTERPOLATOR': 'nn',
                'ORTHORECTIFIED'                 : 'true',
                'S2_TILE_CORRESPONDING_CODE'     : '33NWB',
                'SPATIAL_RESOLUTION'             : '10.0',
                'TIFFTAG_IMAGEDESCRIPTION'       : 'Orthorectified LIA Sentinel-1A IW GRD',
                'ORBIT_DIRECTION'                : 'DES',
                'ORBIT_NUMBER'                   : '030704',
                'RELATIVE_ORBIT_NUMBER'          : '007',
                'AbsoluteCalibrationConstant'    : '',
                'AcquisitionDate'                : '',
                'AcquisitionStartTime'           : '',
                'AcquisitionStopTime'            : '',
                'AverageSceneHeight'             : '',
                'BeamMode'                       : '',
                'BeamSwath'                      : '',
                'BlueDisplayChannel'             : '',
                'GreenDisplayChannel'            : '',
                'Instrument'                     : '',
                'LineSpacing'                    : '',
                'Mission'                        : '',
                'Mode'                           : '',
                'NumberOfColumns'                : '',
                'NumberOfLines'                  : '',
                'OrbitDirection'                 : '',
                'OrbitNumber'                    : '',
                'PRF'                            : '',
                'PixelSpacing'                   : '',
                'RadarFrequency'                 : '',
                'RedDisplayChannel'              : '',
                'SAR'                            : '',
                'SARCalib*'                      : '',
                'SensorID'                       : '',
                'Swath'                          : '',
            })

        application_mocker.set_expectations(
            'OrthoRectification',
            {
                'opt.ram'         : param_ram(2048),
                'io.in'           : file_db.sinLIAfile(idx, False),
                'interpolator'    : 'nn',
                'outputs.spacingx': 10.0,
                'outputs.spacingy': -10.0,
                'outputs.sizex'   : 10980,
                'outputs.sizey'   : 10980,
                'opt.gridspacing' : 40.0,
                'map'             : 'utm',
                'map.utm.zone'    : 33,
                'map.utm.northhem': True,
                'outputs.ulx'     : 499979.99999484676,
                'outputs.uly'     : 200040.0000009411,
                'elev.dem'        : file_db.dem_file(),
                'elev.geoid'      : file_db.GeoidFile,
                'io.out'          : file_db.orthosinLIAfile(idx, True),
            }, None,
            {
                'DATA_TYPE'                      : 'sin(LIA)',
                'DEM_INFO'                       : 'SRTM_30_hgt',
                'ORTHORECTIFICATION_INTERPOLATOR': 'nn',
                'ORTHORECTIFIED'                 : 'true',
                'S2_TILE_CORRESPONDING_CODE'     : '33NWB',
                'SPATIAL_RESOLUTION'             : '10.0',
                'TIFFTAG_IMAGEDESCRIPTION'       : 'Orthorectified sin_LIA Sentinel-1A IW GRD',
                'ORBIT_DIRECTION'                : 'DES',
                'ORBIT_NUMBER'                   : '030704',
                'RELATIVE_ORBIT_NUMBER'          : '007',
                'AbsoluteCalibrationConstant'    : '',
                'AcquisitionDate'                : '',
                'AcquisitionStartTime'           : '',
                'AcquisitionStopTime'            : '',
                'AverageSceneHeight'             : '',
                'BeamMode'                       : '',
                'BeamSwath'                      : '',
                'BlueDisplayChannel'             : '',
                'GreenDisplayChannel'            : '',
                'Instrument'                     : '',
                'LineSpacing'                    : '',
                'Mission'                        : '',
                'Mode'                           : '',
                'NumberOfColumns'                : '',
                'NumberOfLines'                  : '',
                'OrbitDirection'                 : '',
                'OrbitNumber'                    : '',
                'PRF'                            : '',
                'PixelSpacing'                   : '',
                'RadarFrequency'                 : '',
                'RedDisplayChannel'              : '',
                'SAR'                            : '',
                'SARCalib*'                      : '',
                'SensorID'                       : '',
                'Swath'                          : '',
            })

    # endfor on 2 consecutive images

    application_mocker.set_expectations(
        'Synthetize',
        {
            'ram'      : param_ram(2048),
            'il'       : [file_db.orthodegLIAfile(0, False), file_db.orthodegLIAfile(1, False)],
            'out'      : file_db.concatLIAfile_from_two(0, True),
        }, {'out': otb.ImagePixelType_int16},
        {
            'ACQUISITION_DATETIME'     : file_db.start_time_for_two(0),
            'ACQUISITION_DATETIME_1'   : file_db.start_time(0),
            'ACQUISITION_DATETIME_2'   : file_db.start_time(1),
            'DEM_INFO'                 : 'SRTM_30_hgt',
            'DEM_LIST'                 : '',  # <=> Removing the key
            'INPUT_S1_IMAGES'          : '%s, %s' % (file_db.product_name(0), file_db.product_name(1)),
            'TIFFTAG_IMAGEDESCRIPTION' : 'Orthorectified LIA Sentinel-1A IW GRD',
            # meta already set during orthorectification
            # 'ORBIT_DIRECTION'          : 'DES',
            # 'ORBIT_NUMBER'             : '030704',
            # 'RELATIVE_ORBIT_NUMBER'    : '007',
        },
        {file_db.orthodegLIAfile(0, False), file_db.orthodegLIAfile(1, False)},
    )

    application_mocker.set_expectations(
        'Synthetize',
        {
            'ram'      : param_ram(2048),
            'il'       : [file_db.orthosinLIAfile(0, False), file_db.orthosinLIAfile(1, False)],
            'out'      : file_db.concatsinLIAfile_from_two(0, True),
        }, None,
        {
            'ACQUISITION_DATETIME'     : file_db.start_time_for_two(0),
            'ACQUISITION_DATETIME_1'   : file_db.start_time(0),
            'ACQUISITION_DATETIME_2'   : file_db.start_time(1),
            'DEM_INFO'                 : 'SRTM_30_hgt',
            'DEM_LIST'                 : '',  # <=> Removing the key
            'INPUT_S1_IMAGES'          : '%s, %s' % (file_db.product_name(0), file_db.product_name(1)),
            'TIFFTAG_IMAGEDESCRIPTION' : 'Orthorectified sin_LIA Sentinel-1A IW GRD',
            # meta already set during orthorectification
            # 'ORBIT_DIRECTION'          : 'DES',
            # 'ORBIT_NUMBER'             : '030704',
            # 'RELATIVE_ORBIT_NUMBER'    : '007',
        },
        {file_db.orthosinLIAfile(0, False), file_db.orthosinLIAfile(1, False)}
    )


def mock_LIA_v1_1(application_mocker: OTBApplicationsMockContext, file_db: FileDB):
    tmpdir = file_db.tmpdir
    exp_dem_names       = file_db.dems_on_s2()
    exp_out_vrt         = file_db.vrtfile_on_s2(False)
    exp_out_dem_s2      = file_db.demfile_on_s2(False)
    # exp_out_geoid_s2  = file_db.geoidfile_on_s2(False)
    exp_out_height_s2   = file_db.height_on_s2(False)
    exp_out_xyz_s2      = file_db.xyz_on_s2(False)
    # exp_out_normals_s2  = file_db.normals_on_s2(False)
    # TODO: Don't hardcode the mocked tmp subdir for DEMs
    exp_in_dem_files  = [f"{tmpdir}/TMP_DEM/{dem}.hgt" for dem in exp_dem_names]

    application_mocker.set_expectations(
        AgglomerateDEMOnS2.agglomerate,
        [file_db.vrtfile_on_s2(True)] + exp_in_dem_files,
        None, None,
    )

    # ProjectDEMToS2Tile
    spacing=10.0
    extent = cast(dict[str, float], file_db.TILE_DATA['33NWB']['extent'])
    application_mocker.set_expectations(
        'gdalwarp', [
            "-wm", f'{2048*1024*1024}',
            "-multi", "-wo", "2",
            "-t_srs", f"epsg:{extent['epsg']}",
            "-tr", f"{spacing}", f"-{spacing}",
            "-ot", "Float32",
            # "-crop_to_cutline",
            "-te", f"{extent['xmin']}", f"{extent['ymin']}", f"{extent['xmax']}", f"{extent['ymax']}",
            "-r", "cubic",
            "-dstnodata", str(nodata_DEM),
            exp_out_vrt,
            file_db.demfile_on_s2(True),
        ], None, {
            'S2_TILE_CORRESPONDING_CODE' : '33NWB',
            'SPATIAL_RESOLUTION'         : f"{spacing}",
            'DEM_INFO'                   : 'SRTM_30_hgt',
            'DEM_RESAMPLING_METHOD'      : 'cubic',
            'TIFFTAG_IMAGEDESCRIPTION'   : 'Warped DEM to S2 tile',
            'ORTHORECTIFIED'             : 'true',
        },
        {exp_out_vrt},
    )

    # ProjectGeoidToS2Tile
    application_mocker.set_expectations(
        'Superimpose',
        {
            'ram'                     : param_ram(2048),
            'inr'                     : exp_out_dem_s2,
            'inm'                     : file_db.GeoidFile,
            'interpolator'            : 'nn',
            'interpolator.bco.radius' : 2,
            'fv'                      : nodata_DEM,
            'out'                     : 'BandMath|>' + file_db.height_on_s2(True),
        }, None, {
            # 'ACQUISITION_DATETIME'       : file_db.start_time(0),
            # 'DEM_LIST'                   : ', '.join(exp_dem_names),
            'S2_TILE_CORRESPONDING_CODE'           : '33NWB',
            'SPATIAL_RESOLUTION'                   : f"{spacing}",
            'TIFFTAG_IMAGEDESCRIPTION'             : 'Geoid superimposed on S2 tile',
            'GEOID_ORTHORECTIFICATION_INTERPOLATOR': 'nn',
            'ORTHORECTIFIED'                       : 'true',
        },
        {exp_out_dem_s2},
    )

    # Sum DEM + GEOID
    is_nodata_DEM_bandmath = Utils.test_nodata_for_bandmath(bandname="im2b1", nodata=nodata_DEM)
    application_mocker.set_expectations(
        'BandMath',
        {
            'il'         : [
                exp_out_dem_s2+"|>Superimpose",
                exp_out_dem_s2,
                # exp_out_geoid_s2
            ],
            'ram'        : param_ram(2048),
            'exp'        : f'{is_nodata_DEM_bandmath} ? {nodata_DEM} : im1b1+im2b1',
            'out'        : file_db.height_on_s2(True),
        }, None, {
            'TIFFTAG_GDAL_NODATA'      : '-32768',
            'TIFFTAG_IMAGEDESCRIPTION' : 'DEM + GEOID height info projected on S2 tile',
            'DEM_RESAMPLING_METHOD'    : 'cubic',
            'ORTHORECTIFIED'           : 'true',
        })
    # ComputeGroundAndSatPositionsOnDEM
    application_mocker.set_expectations(
        'SARDEMProjection2', {
            'ram'        : param_ram(2048),
            'insar'      : file_db.input_file_vv(0),
            'indem'      : exp_out_height_s2,
            'elev.geoid' : '@',
            'withcryz'   : False,
            'withxyz'    : True,
            'withsatpos' : True,
            'nodata'     : nodata_XYZ,
            'out'        : file_db.xyz_on_s2(True),
        }, None, {
            # 'ACQUISITION_DATETIME'     : file_db.start_time(0),
            'DEM_INFO'                 : 'SRTM_30_hgt',
            'DEM_LIST'                 : ', '.join(exp_dem_names),
            'IMAGE_TYPE'               : 'XYZ',
            'TIFFTAG_IMAGEDESCRIPTION' : 'XYZ ground and satellite positions on S2 tile',
            'ORTHORECTIFIED'           : 'true',
            'POLARIZATION'             : '',
            'band.DirectionToScanDEM*' : '',
            'band.Gain'                : '',
            # THE *ORBIT* meta data are set by the OTB application and can't be checked when mocking
            # 'ORBIT_DIRECTION'          : 'DES',
            # 'ORBIT_NUMBER'             : '030704',
            # 'RELATIVE_ORBIT_NUMBER'    : '007',
        },
        {exp_out_height_s2},
    )

    # ExtractNormalVector
    application_mocker.set_expectations(
        'ExtractNormalVector',
        {
            'ram'             : param_ram(2048),
            'xyz'             : exp_out_xyz_s2,
            'nodata'          : nodata_XYZ,
            'out'             : 'SARComputeIncidenceAngle|>'+file_db.deglia_on_s2(True),
        }, None, {
            'TIFFTAG_IMAGEDESCRIPTION' : 'Image normals on Sentinel-{flying_unit_code_short} IW GRD',
        },
    )

    # ComputeLIA
    application_mocker.set_expectations(
        'SARComputeIncidenceAngle',
        {
            'in.normals'      : file_db.xyz_on_s2(False)+'|>ExtractNormalVector', #'ComputeNormals|>'+file_db.normalsfile(idx),
            'ram'             : param_ram(2048),
            'in.xyz'          : file_db.xyz_on_s2(False),
            'out.deg'         : file_db.deglia_on_s2(True),
            'out.sin'         : file_db.sinlia_on_s2(True),
            'nodata'          : nodata_LIA,
        }, {'out.deg': otb.ImagePixelType_uint16}, {
            'DATA_TYPE'                : ['sin(LIA)', '100 * degrees(LIA)'],
            'IMAGE_TYPE'               : 'LIA',
            'TIFFTAG_IMAGEDESCRIPTION' : ['sin(LIA) on S2 grid', '100 * degrees(LIA) on S2 grid'],
        },
        {exp_out_xyz_s2},
    )


def mock_GAMMA_AREA_v1_2(application_mocker: OTBApplicationsMockContext, file_db: FileDB):
    demdir = file_db.demdir
    for idx in range(2):
        orbit_info            = file_db.get_orbit_information(idx)
        cov                   = file_db.dem_coverage(idx)
        exp_dem_names         = sorted(cov)
        exp_out_vrt           = file_db.vrtfile(idx, False)
        exp_out_resampled_dem = file_db.resampleddemfile(idx, False)
        exp_out_dem           = file_db.sardemprojfile(idx, False)
        exp_in_dem_files      = [f"{demdir}/{dem}.hgt" for dem in exp_dem_names]

        # Build VRT
        application_mocker.set_expectations(
            AgglomerateDEMOnS1.agglomerate,
            [file_db.vrtfile(idx, True)] + exp_in_dem_files,
            None,
            None
        )

        # Resample DEM
        # > NaNifyNoData
        is_nodata_DEM_bandmath = Utils.test_nodata_for_bandmath(bandname="im1b1", nodata=nodata_DEM)
        application_mocker.set_expectations(
            'BandMath', {
                'il'         : [ exp_out_vrt ],
                'ram'        : param_ram(2048),
                'exp'        : f'{is_nodata_DEM_bandmath} ? 0./0. : im1b1',
                'out'        : 'RigidTransformResample|>'+file_db.resampleddemfile(idx, True),
            }, None, {
                'TIFFTAG_IMAGEDESCRIPTION' : 'The same but with NaN',
            },
            files_to_remove={exp_out_vrt},
        )

        # > RigidTransformResample
        application_mocker.set_expectations(
            'RigidTransformResample', {
                'in'                      : [exp_out_vrt + '|>BandMath'],
                'ram'                     : param_ram(2048),
                'transform.type'          : "id",
                'transform.type.id.scalex': 2.0,
                'transform.type.id.scaley': 2.0,
                'out'                     : file_db.resampleddemfile(idx, True),
            }, None, {
                'DEM_RESAMPLING_METHOD'    : 'X*2.0, Y*2.0',
                'TIFFTAG_IMAGEDESCRIPTION' : 'DEM resampled X*2.0 Y*2.0',
            },
            # files_to_remove={exp_out_vrt},
        )

        # SumAllHeights
        # > ProjectGeoidToS2Tile
        spacing=10.0
        application_mocker.set_expectations(
            'Superimpose', {
                'ram'                     : param_ram(2048),
                'inr'                     : exp_out_resampled_dem,
                'inm'                     : file_db.GeoidFile,
                'interpolator'            : 'nn',
                'interpolator.bco.radius' : 2,
                'fv'                      : nodata_DEM,
                'out'                     : 'BandMath|>' + file_db.height_4rtc(idx, True),
            }, None, {
                # 'ACQUISITION_DATETIME'       : file_db.start_time(0),
                # 'DEM_LIST'                   : ', '.join(exp_dem_names),
                'SPATIAL_RESOLUTION'                   : f"{spacing}",
                'TIFFTAG_IMAGEDESCRIPTION'             : 'Geoid superimposed on DEM',
                'GEOID_ORTHORECTIFICATION_INTERPOLATOR': 'nn',
            },
        )

        # > DEM + GEOID
        is_nodata_DEM_bandmath = Utils.test_nodata_for_bandmath(bandname="im2b1", nodata=nodata_DEM)
        application_mocker.set_expectations(
            'BandMath', {
                'il'         : [
                    exp_out_resampled_dem+"|>Superimpose",
                    exp_out_resampled_dem,
                    # exp_out_geoid_s2
                ],
                'ram'        : param_ram(2048),
                'exp'        : f'{is_nodata_DEM_bandmath} ? {nodata_DEM} : im1b1+im2b1',
                'out'        : file_db.height_4rtc(idx, True),
            }, None, {
                'TIFFTAG_GDAL_NODATA'      : '-32768',
                'TIFFTAG_IMAGEDESCRIPTION' : 'DEM + GEOID',
                'DEM_RESAMPLING_METHOD'    : 'cubic',
            },
            {exp_out_resampled_dem},
        )

        # Project DEM
        application_mocker.set_expectations(
            'SARDEMProjection2', {
                'ram'        : param_ram(2048),
                'insar'      : file_db.input_file_vv(idx),
                # 'indem'      : exp_out_resampled_dem,
                'indem'      : file_db.height_4rtc(idx, False),
                'withxyz'    : True,
                'nodata'     : str(nodata_RTC),
                'elev.geoid' : '@',
                'out'        : file_db.sardemprojfile(idx, True),
            }, {
                # 'out': otb.ImagePixelType_float
            }, {
                'ACQUISITION_DATETIME'     : file_db.start_time(idx),
                'DEM_INFO'                 : 'SRTM_30_hgt',
                'DEM_LIST'                 : ', '.join(exp_dem_names),
                'FLYING_UNIT_CODE'         : 's1a',
                'IMAGE_TYPE'               : 'GRD',
                'INPUT_S1_IMAGES'          : file_db.product_name(idx),
                'ORBIT_DIRECTION'          : 'DES',
                'ORBIT_NUMBER'             : '{:0>6d}'.format(orbit_info['absolute_orbit']),
                'POLARIZATION'             : '',  # <=> removing the key
                'RELATIVE_ORBIT_NUMBER'    : '{:0>3d}'.format(orbit_info['relative_orbit']),
                'TIFFTAG_IMAGEDESCRIPTION' : 'SARDEM projection onto DEM list',
            },
        )

        application_mocker.set_expectations(
            'SARGammaAreaImageEstimation', {
                'ram'                   : param_ram(2048),
                'distributearea'        : False,
                # 'indem'                 : exp_out_resampled_dem,
                'indem'                 : file_db.height_4rtc(idx, False),
                'indemproj'             : exp_out_dem,
                'indirectiondemc'       : 24,
                'indirectiondeml'       : 12,
                'inputwindow'           : 'everything',
                'insar'                 : file_db.input_file_vv(idx),
                'mlazi'                 : 1,
                'mlran'                 : 1,
                'streaming'             : 'enable',
                'warn'                  : 'once',
                'innermarginratio'      : 0.01,
                'innermarginratiostatus': True,
                'outermarginratio'      : 0.04,
                'outermarginratiostatus': True,
                'out'                   : file_db.gamma_areafile(idx, True),
            }, None, {
                'Polarization'                    : '',  # <=> removing the key
                'PRJ.DIRECTIONTOSCANDEMC'         : '',  # <=> removing the key
                'PRJ.DIRECTIONTOSCANDEML'         : '',  # <=> removing the key
                'PRJ.GAIN'                        : '',  # <=> removing the key
                'TIFFTAG_IMAGEDESCRIPTION'        : 'Gamma area image estimation',
                'band.LLFracDistributedGammaArea' : '',   # <=> removing the key
                'band.LRFracDistributedGammaArea' : '',   # <=> removing the key
                'band.ULFracDistributedGammaArea' : '',   # <=> removing the key
                'band.URFracDistributedGammaArea' : '',   # <=> removing the key
            },
            {
                exp_out_dem,
                file_db.height_4rtc(idx, False)
            },
        )

        application_mocker.set_expectations(
            'OrthoRectification', {
                'opt.ram'         : param_ram(2048),
                'io.in'           : file_db.gamma_areafile(idx, False),
                'interpolator'    : 'nn',
                'outputs.spacingx': 10.0,
                'outputs.spacingy': -10.0,
                'outputs.sizex'   : 10980,
                'outputs.sizey'   : 10980,
                'opt.gridspacing' : 40.0,
                'map'             : 'utm',
                'map.utm.zone'    : 33,
                'map.utm.northhem': True,
                'outputs.ulx'     : 499979.99999484676,
                'outputs.uly'     : 200040.0000009411,
                'elev.dem'        : file_db.dem_file(),
                'elev.geoid'      : file_db.GeoidFile,
                'io.out'          : file_db.orthoGAMMA_AREAfile(idx, True),
            }, None, {
                'DATA_TYPE'                      : 'meters^2',
                'DEM_INFO'                       : 'SRTM_30_hgt',
                'ORTHORECTIFICATION_INTERPOLATOR': 'nn',
                'ORTHORECTIFIED'                 : 'true',
                'S2_TILE_CORRESPONDING_CODE'     : '33NWB',
                'SPATIAL_RESOLUTION'             : '10.0',
                'TIFFTAG_IMAGEDESCRIPTION'       : 'Orthorectified GAMMA_AREA Sentinel-1A IW GRD',
                'ORBIT_DIRECTION'                : 'DES',
                # 'ORBIT_NUMBER'                   : '030704',
                'RELATIVE_ORBIT_NUMBER'          : '007',
                'AbsoluteCalibrationConstant'    : '',
                'AcquisitionDate'                : '',
                'AcquisitionStartTime'           : '',
                'AcquisitionStopTime'            : '',
                'AverageSceneHeight'             : '',
                'BeamMode'                       : '',
                'BeamSwath'                      : '',
                'BlueDisplayChannel'             : '',
                'GreenDisplayChannel'            : '',
                'Instrument'                     : '',
                # LineSpacing & PixelSpacing are inherited from input SAR image
                # => they see no metadata update
                # 'LineSpacing'                  : '10.0',
                'Mission'                        : '',
                'Mode'                           : '',
                'NumberOfColumns'                : '',
                'NumberOfLines'                  : '',
                'OrbitDirection'                 : '',
                'OrbitNumber'                    : '',
                'PRF'                            : '',
                # 'PixelSpacing'                 : '10.0',
                'RadarFrequency'                 : '',
                'RedDisplayChannel'              : '',
                'SAR'                            : '',
                'SARCalib*'                      : '',
                'SensorID'                       : '',
                'Swath'                          : '',
            })

    # endfor on 2 consecutive images

    application_mocker.set_expectations(
        'Synthetize', {
            'ram'      : param_ram(2048),
            'il'       : [file_db.orthoGAMMA_AREAfile(0, False), file_db.orthoGAMMA_AREAfile(1, False)],
            'out'      : file_db.concatGAMMA_AREAfile_from_two(0, True),
        }, None, {
            'ACQUISITION_DATETIME'     : file_db.start_time_for_two(0),
            'ACQUISITION_DATETIME_1'   : file_db.start_time(0),
            'ACQUISITION_DATETIME_2'   : file_db.start_time(1),
            'DEM_INFO'                 : 'SRTM_30_hgt',
            'DEM_LIST'                 : '',  # <=> Removing the key
            'INPUT_S1_IMAGES'          : '%s, %s' % (file_db.product_name(0), file_db.product_name(1)),
            'TIFFTAG_IMAGEDESCRIPTION' : 'Orthorectified GAMMA_AREA Sentinel-1A IW GRD',
            # meta already set during orthorectification
            # 'ORBIT_DIRECTION'          : 'DES',
            # 'ORBIT_NUMBER'             : '030704',
            # 'RELATIVE_ORBIT_NUMBER'    : '007',
        },
        {
            file_db.orthoGAMMA_AREAfile(0, False),
            file_db.orthoGAMMA_AREAfile(1, False),
        },
    )


def mock_LIA_v1_2(application_mocker: OTBApplicationsMockContext, file_db: FileDB):
    tmpdir = file_db.tmpdir
    exp_dem_names       = file_db.dems_on_s2()
    exp_out_vrt         = file_db.vrtfile_on_s2(False)
    exp_out_dem_s2      = file_db.demfile_on_s2(False)
    # exp_out_geoid_s2  = file_db.geoidfile_on_s2(False)
    exp_out_height_s2   = file_db.height_on_s2(False)
    exp_out_xyz_s2      = file_db.xyz_on_s2(False)
    # exp_out_normals_s2  = file_db.normals_on_s2(False)
    # TODO: Don't hardcode the mocked tmp subdir for DEMs
    exp_in_dem_files  = [f"{tmpdir}/TMP_DEM/{dem}.hgt" for dem in exp_dem_names]

    application_mocker.set_expectations(
        AgglomerateDEMOnS2.agglomerate,
        [file_db.vrtfile_on_s2(True)] + exp_in_dem_files,
        None,
        None,
    )

    # ProjectDEMToS2Tile
    spacing=10.0
    extent = cast(dict[str, float], file_db.TILE_DATA['33NWB']['extent'])
    application_mocker.set_expectations(
        'gdalwarp', [
            "-wm", f'{2048*1024*1024}',
            "-multi", "-wo", "2",
            "-t_srs", f"epsg:{extent['epsg']}",
            "-tr", f"{spacing}", f"-{spacing}",
            "-ot", "Float32",
            # "-crop_to_cutline",
            "-te", f"{extent['xmin']}", f"{extent['ymin']}", f"{extent['xmax']}", f"{extent['ymax']}",
            "-r", "cubic",
            "-dstnodata", str(nodata_DEM),
            exp_out_vrt,
            file_db.demfile_on_s2(True),
        ], None, {
            'S2_TILE_CORRESPONDING_CODE' : '33NWB',
            'SPATIAL_RESOLUTION'         : f"{spacing}",
            'DEM_INFO'                   : 'SRTM_30_hgt',
            'DEM_RESAMPLING_METHOD'      : 'cubic',
            'TIFFTAG_IMAGEDESCRIPTION'   : 'Warped DEM to S2 tile',
            'ORTHORECTIFIED'             : 'true',
        },
        {exp_out_vrt},
    )

    # ProjectGeoidToS2Tile
    application_mocker.set_expectations(
        'Superimpose',
        {
            'ram'                     : param_ram(2048),
            'inr'                     : exp_out_dem_s2,
            'inm'                     : file_db.GeoidFile,
            'interpolator'            : 'nn',
            'interpolator.bco.radius' : 2,
            'fv'                      : nodata_DEM,
            'out'                     : 'BandMath|>' + file_db.height_on_s2(True),
        }, None, {
            # 'ACQUISITION_DATETIME'       : file_db.start_time(0),
            # 'DEM_LIST'                   : ', '.join(exp_dem_names),
            'S2_TILE_CORRESPONDING_CODE'           : '33NWB',
            'SPATIAL_RESOLUTION'                   : f"{spacing}",
            'TIFFTAG_IMAGEDESCRIPTION'             : 'Geoid superimposed on S2 tile',
            'GEOID_ORTHORECTIFICATION_INTERPOLATOR': 'nn',
            'ORTHORECTIFIED'                       : 'true',
        })

    # Sum DEM + GEOID
    is_nodata_DEM_bandmath = Utils.test_nodata_for_bandmath(bandname="im2b1", nodata=nodata_DEM)
    application_mocker.set_expectations(
        'BandMath',
        {
            'il'         : [
                exp_out_dem_s2+"|>Superimpose",
                exp_out_dem_s2,
                # exp_out_geoid_s2
            ],
            'ram'        : param_ram(2048),
            'exp'        : f'{is_nodata_DEM_bandmath} ? {nodata_DEM} : im1b1+im2b1',
            'out'        : file_db.height_on_s2(True),
        }, None, {
            'TIFFTAG_GDAL_NODATA'      : '-32768',
            'TIFFTAG_IMAGEDESCRIPTION' : 'DEM + GEOID height info projected on S2 tile',
            'DEM_RESAMPLING_METHOD'    : 'cubic',
            'ORTHORECTIFIED'           : 'true',
        },
        {exp_out_dem_s2},
    )
    # ComputeGroundAndSatPositionsOnDEM
    application_mocker.set_expectations(
        'SARComputeGroundAndSatPositionsOnDEM',
        {
            'ram'        : param_ram(2048),
            'ineof'      : file_db.eof_for_s2(),
            'indem'      : exp_out_height_s2,
            'inrelorb'   : file_db.relorb_for_s2(),
            'elev.geoid' : '@',
            'withxyz'    : True,
            'withsatpos' : True,
            'nodata'     : nodata_XYZ,
            'out'        : file_db.xyz_on_s2(True),
        }, None, {
            # 'ACQUISITION_DATETIME'     : file_db.start_time(0),
            'DEM_INFO'                 : 'SRTM_30_hgt',
            'DEM_LIST'                 : ', '.join(exp_dem_names),
            'EOF_FILE'                 : os.path.basename(file_db.eof_for_s2()),
            # 'FLYING_UNIT_CODE'         : 's1a',
            'IMAGE_TYPE'               : 'XYZ',
            'TIFFTAG_IMAGEDESCRIPTION' : 'XYZ ground and satellite positions on S2 tile',
            'ORTHORECTIFIED'           : 'true',
            'POLARIZATION'             : '',
            'band.DirectionToScanDEM*' : '',
            'band.Gain'                : '',
            # THE *ORBIT* meta data are set by the OTB application and can't be checked when mocking
            # 'ORBIT_DIRECTION'          : 'DES',
            # 'ORBIT_NUMBER'             : '030704',
            # 'RELATIVE_ORBIT_NUMBER'    : '{:0>3d}'.format(file_db.relorb_for_s2()),
        },
        {exp_out_height_s2},
    )

    # ExtractNormalVector
    application_mocker.set_expectations(
        'ExtractNormalVector',
        {
            'ram'             : param_ram(2048),
            'xyz'             : exp_out_xyz_s2,
            'nodata'          : nodata_XYZ,
            'out'             : 'SARComputeIncidenceAngle|>'+file_db.deglia_on_s2(True),
        }, None, {
            'TIFFTAG_IMAGEDESCRIPTION' : 'Image normals on Sentinel-{flying_unit_code_short} IW GRD',
        })

    # ComputeLIA
    application_mocker.set_expectations(
        'SARComputeIncidenceAngle',
        {
            'in.normals'      : exp_out_xyz_s2+'|>ExtractNormalVector', #'ComputeNormals|>'+file_db.normalsfile(idx),
            'ram'             : param_ram(2048),
            'in.xyz'          : exp_out_xyz_s2,
            'out.deg'         : file_db.deglia_on_s2(True),
            'out.sin'         : file_db.sinlia_on_s2(True),
            'nodata'          : nodata_LIA,
        }, {'out.deg': otb.ImagePixelType_uint16}, {
            'DATA_TYPE'                : ['sin(LIA)', '100 * degrees(LIA)'],
            'IMAGE_TYPE'               : 'LIA',
            'TIFFTAG_IMAGEDESCRIPTION' : ['sin(LIA) on S2 grid', '100 * degrees(LIA) on S2 grid'],
        },
        {exp_out_xyz_s2},
    )


@pytest.mark.parametrize("naming_policy",
                         [
                             'with_calibration',
                             'theia'
                         ])
def test_33NWB_202001_NR_core_mocked_with_concat(tmpdir, demdir, ram, mocker, naming_policy):
    """
    Mocked test of production of S2 sigma0 calibrated images.

    In this flavour, we emulate old IPF 002.50 where image borders needed to be cut.
    """
    baselinedir   = pathlib.Path('/BASELINE')
    eofdir        = pathlib.Path('/UNUSED')
    outputdir     = pathlib.Path('/OUTPUT')
    liadir        = pathlib.Path('/UNUSED')
    gamma_areadir = pathlib.Path('/UNUSED')
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)

    inputdir = str((baselinedir/'inputs').absolute())
    set_environ_mocked(inputdir, outputdir, liadir, gamma_areadir, demdir, tmpdir, ram)

    tile = '33NWB'

    # baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    configuration = s1tiling.libs.configuration.Configuration(test_file, do_show_configuration=False)
    # Force the use of "_{calibration}" in mocked tests
    configuration.fname_fmt['concatenation'] = FileDB.CONCATENATION_NAMING[naming_policy]
    configuration.dname_fmt['tiled']         = '{out_dir}/{tile_name}/tiled'
    configuration.show_configuration()
    logging.info("Full mocked test")

    file_db = FileDB(
        inputdir=inputdir,
        eofdir=eofdir,
        tmpdir=tmpdir.absolute(),
        outputdir=outputdir.absolute(),
        xiadir=liadir.absolute(),
        gamma_areadir=gamma_areadir.absolute(),
        tile=tile,
        demdir=demdir,
        geoid_file=configuration.GeoidFile,
        dname_fmt_tiled=configuration.dname_fmt['tiled'],
    )
    mocker.patch('s1tiling.libs.otbtools.otb_version', lambda : '7.4.0')

    application_mocker = OTBApplicationsMockContext(configuration, mocker, file_db.tmp_to_out_map, file_db.dem_files)
    known_files = application_mocker.known_files
    known_dirs = set()
    declare_know_files(mocker, known_files, known_dirs, tile, ['vv'], file_db, application_mocker)
    assert os.path.isfile(file_db.input_file_vv(0))  # Check mocking
    assert os.path.isfile(file_db.input_file_vv(1))

    def mock__AnalyseBorders_complete_meta(slf, meta, all_inputs):
        meta = super(AnalyseBorders, slf).complete_meta(meta, all_inputs)
        meta['cut'] = {
            'threshold.x'      : 1000,
            'threshold.y.start': 0,
            'threshold.y.end'  : 0,
            'skip'             : False,
        }
        return meta
    mocker.patch('s1tiling.libs.otbwrappers.AnalyseBorders.complete_meta', mock__AnalyseBorders_complete_meta)

    mock_upto_concat_S2(application_mocker, file_db, naming_policy=naming_policy, calibration='sigma', N=2, old_IPF=True)
    mock_masking(application_mocker, file_db, 'sigma', 2, naming_policy)
    s1_process(config_opt=configuration, searched_items_per_page=0,
            dryrun=False, debug_otb=True, watch_ram=False,
            debug_tasks=False, cache_before_ortho=False)
    application_mocker.assert_all_have_been_executed()
    application_mocker.assert_all_metadata_match()


@pytest.mark.parametrize("naming_policy",
                         [
                             'with_calibration',
                             'theia'
                         ])
def test_33NWB_202001_NR_core_mocked_no_concat(tmpdir, demdir, ram, mocker, naming_policy):
    """
    Mocked test of production of S2 sigma0 calibrated images.
    """
    baselinedir   = pathlib.Path('/BASELINE')
    eofdir        = pathlib.Path('/UNUSED')
    outputdir     = pathlib.Path('/OUTPUT')
    liadir        = pathlib.Path('/UNUSED')
    gamma_areadir = pathlib.Path('/UNUSED')
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)

    inputdir = str((baselinedir/'inputs').absolute())
    set_environ_mocked(inputdir, outputdir, liadir, gamma_areadir, demdir, tmpdir, ram)

    tile = '33NWB'

    # baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    configuration = s1tiling.libs.configuration.Configuration(test_file, do_show_configuration=False)
    # Force the use of "_{calibration}" in mocked tests
    configuration.fname_fmt['concatenation'] = FileDB.CONCATENATION_NAMING[naming_policy]
    configuration.show_configuration()
    logging.info("Full mocked test")

    file_db = FileDB(inputdir, eofdir, tmpdir.absolute(), outputdir.absolute(), liadir.absolute(), gamma_areadir.absolute(), tile, demdir, configuration.GeoidFile)
    mocker.patch('s1tiling.libs.otbtools.otb_version', lambda : '7.4.0')

    application_mocker = OTBApplicationsMockContext(configuration, mocker, file_db.tmp_to_out_map, file_db.dem_files)
    known_files = application_mocker.known_files
    known_dirs = set()
    declare_know_files(mocker, known_files, known_dirs, tile, ['vv-20200108t044150-20200108t044215'], file_db, application_mocker)
    assert os.path.isfile(file_db.input_file_vv(0))  # Check mocking
    assert not os.path.isfile(file_db.input_file_vv(1))

    def mock__AnalyseBorders_complete_meta(slf, meta, all_inputs):
        meta = super(AnalyseBorders, slf).complete_meta(meta, all_inputs)
        meta['cut'] = {
            'threshold.x'      : 0,
            'threshold.y.start': 0,
            'threshold.y.end'  : 0,
            'skip'             : True,
        }
        return meta
    mocker.patch('s1tiling.libs.otbwrappers.AnalyseBorders.complete_meta', mock__AnalyseBorders_complete_meta)

    mock_upto_concat_S2(application_mocker, file_db, naming_policy=naming_policy, calibration='sigma', N=1)
    mock_masking(application_mocker, file_db, 'sigma', 1, naming_policy)
    s1_process(config_opt=configuration, searched_items_per_page=0,
            dryrun=False, debug_otb=True, watch_ram=False,
            debug_tasks=False, cache_before_ortho=False)
    application_mocker.assert_all_have_been_executed()
    application_mocker.assert_all_metadata_match()


class MockedSentinelOrbitFile:
    def __init__(self, filename: str, mission: str):
        self.filename = filename
        self.mission  = mission


@pytest.mark.parametrize("register_expectations,processor",
                         [
                             (mock_LIA_v1_0, s1_process_lia_v0),
                             (mock_LIA_v1_1, s1_process_lia_v1_1),
                             (mock_LIA_v1_2, s1_process_lia_v1_2),
                         ])
def test_33NWB_202001_lia_mocked(tmpdir, demdir, ram, mocker, register_expectations, processor):
    """
    Mocked test of production of LIA and sin LIA files
    """
    baselinedir   = pathlib.Path('/BASELINE')
    eofdir        = pathlib.Path('/_EOF')
    outputdir     = pathlib.Path('/OUTPUT')
    liadir        = pathlib.Path('/_LIA')
    gamma_areadir = pathlib.Path('/UNUSED')
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)

    inputdir = str((baselinedir/'inputs').absolute())
    set_environ_mocked(inputdir, outputdir, liadir, gamma_areadir, demdir, tmpdir, ram)

    tile = '33NWB'

    # baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    configuration = s1tiling.libs.configuration.Configuration(test_file, do_show_configuration=False)
    configuration.calibration_type             = 'normlim'
    configuration.extra_directories['lia_dir'] = liadir.absolute()
    configuration.produce_lia_map              = True
    configuration.relative_orbit_list          = [7]
    configuration.show_configuration()
    logging.info("Sigma0 NORMLIM mocked test")

    file_db = FileDB(
        inputdir=inputdir,
        eofdir=eofdir,
        tmpdir=tmpdir.absolute(),
        outputdir=outputdir.absolute(),
        xiadir=liadir.absolute(),
        gamma_areadir="",
        tile=tile,
        demdir=demdir,
        geoid_file=configuration.GeoidFile,
    )
    mocker.patch('s1tiling.libs.otbtools.otb_version', lambda : '7.4.0')
    eof_file = os.path.join(eofdir, 'S1A_OPER_AUX_POEORB_OPOD_20210316T205443_V20200108T225942_20200110T005942.EOF')
    mocked_eof = MockedSentinelOrbitFile(eof_file, 'S1A')
    mocker.patch('s1tiling.libs.orbit._manager.EOFFileManager.search_for',
                 lambda slf, obts: [DownloadOutcome({obt: mocked_eof}) for obt in obts])

    application_mocker = OTBApplicationsMockContext(configuration, mocker, file_db.tmp_to_out_map, file_db.dem_files)
    known_files = application_mocker.known_files
    known_files.append(eof_file)
    known_dirs = set()
    declare_know_files(mocker, known_files, known_dirs, tile, ['vv'], file_db, application_mocker)
    assert os.path.isfile(file_db.input_file_vv(0))  # Check mocking
    assert os.path.isfile(file_db.input_file_vv(1))

    register_expectations(application_mocker, file_db)

    processor(config_opt=configuration,
              dryrun=False, debug_otb=True, watch_ram=False,
              debug_tasks=False)
    application_mocker.assert_all_have_been_executed()
    application_mocker.assert_all_metadata_match()


def test_33NWB_202001_normlim_v1_0_mocked_one_date(tmpdir, demdir, ram, mocker, naming_policy):
    """
    Mocked test of production of S2 normlim calibrated images.
    """
    baselinedir   = pathlib.Path('/BASELINE')
    eofdir        = pathlib.Path('/UNUSED')
    outputdir     = pathlib.Path('/OUTPUT')
    liadir        = pathlib.Path('/_LIA')
    gamma_areadir = pathlib.Path('/UNUSED')
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)

    inputdir = str((baselinedir/'inputs').absolute())

    set_environ_mocked(inputdir, outputdir, liadir, gamma_areadir, demdir, tmpdir, ram)

    tile = '33NWB'

    # baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    configuration = s1tiling.libs.configuration.Configuration(test_file, do_show_configuration=False)
    configuration.calibration_type             = 'normlim'
    configuration.extra_directories['lia_dir'] = liadir.absolute()
    configuration.produce_lia_map              = True
    configuration.fname_fmt['concatenation']   = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_tmpbeta.tif'
    configuration.relative_orbit_list          = [7]
    configuration.show_configuration()
    logging.info("Sigma0 NORMLIM mocked test")

    file_db = FileDB(
        inputdir=inputdir,
        eofdir=eofdir,
        tmpdir=tmpdir.absolute(),
        outputdir=outputdir.absolute(),
        xiadir=liadir.absolute(),
        gamma_areadir="",
        tile=tile,
        demdir=demdir,
        geoid_file=configuration.GeoidFile,
    )
    mocker.patch('s1tiling.libs.otbtools.otb_version', lambda : '7.4.0')

    application_mocker = OTBApplicationsMockContext(configuration, mocker, file_db.tmp_to_out_map, file_db.dem_files)
    known_files = application_mocker.known_files
    known_dirs = set()
    declare_know_files(mocker, known_files, known_dirs, tile, ['vv'], file_db, application_mocker)
    assert os.path.isfile(file_db.input_file_vv(0))  # Check mocking
    assert os.path.isfile(file_db.input_file_vv(1))

    def mock__AnalyseBorders_complete_meta(slf, meta, all_inputs):
        meta = super(AnalyseBorders, slf).complete_meta(meta, all_inputs)
        meta['cut'] = {
            'threshold.x'      : 0,
            'threshold.y.start': 0,
            'threshold.y.end'  : 0,
            'skip'             : True,
        }
        return meta
    mocker.patch('s1tiling.libs.otbwrappers.AnalyseBorders.complete_meta', mock__AnalyseBorders_complete_meta)

    mock_upto_concat_S2(application_mocker, file_db, naming_policy=naming_policy, calibration='normlim', N=2)
    mock_LIA_v1_0(application_mocker, file_db)
    mock_masking(application_mocker, file_db, 'normlim', 2, naming_policy)

    is_nodata_SAR_bandmath = Utils.test_nodata_for_bandmath(bandname='im1b1', nodata=nodata_SAR)
    is_nodata_LIA_bandmath = Utils.test_nodata_for_bandmath(bandname='im2b1', nodata=nodata_LIA)
    application_mocker.set_expectations(
        'BandMath',
        {
            'ram'      : param_ram(2048),
            'il'       : [
                file_db.concatfile_from_two(0, False, naming_policy=naming_policy, calibration='_normlim'),
                file_db.selectedsinLIAfile()
            ],
            'exp'      : f'({is_nodata_LIA_bandmath} || {is_nodata_SAR_bandmath}) ? {nodata_SAR} : max(1e-07, im1b1*im2b1)',
            'out'      : file_db.sigma0_normlim_file_from_two(0, True, naming_policy=naming_policy),
        }, None,
        {
            'CALIBRATION'              : 'Normlim',
            'IMAGE_TYPE'               : 'BACKSCATTERING',
            'LIA_FILE'                 : os.path.basename(file_db.selectedsinLIAfile()),
            'TIFFTAG_IMAGEDESCRIPTION' : 'Sigma0 Normlim Calibrated Sentinel-1A IW GRD',
        },
        {file_db.concatfile_from_two(0, False, naming_policy=naming_policy, calibration='_normlim')},
    )

    s1_process(
        config_opt=configuration, searched_items_per_page=0,
        dryrun=False, debug_otb=True, watch_ram=False, debug_tasks=False,
        lia_process=register_LIA_pipelines_v0,
    )
    application_mocker.assert_all_have_been_executed()
    application_mocker.assert_all_metadata_match()


def test_33NWB_202001_normlim_v1_0_mocked_all_dates(tmpdir, demdir, ram, mocker, naming_policy):
    """
    Mocked test of production of S2 normlim calibrated images.
    """
    number_dates = 3

    baselinedir   = pathlib.Path('/BASELINE')
    eofdir        = pathlib.Path('/UNUSED')
    outputdir     = pathlib.Path('/OUTPUT')
    liadir        = pathlib.Path('/_LIA')
    gamma_areadir = pathlib.Path('/UNUSED')
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)

    inputdir = str((baselinedir/'inputs').absolute())

    set_environ_mocked(inputdir, outputdir, liadir, gamma_areadir, demdir, tmpdir, ram)

    tile = '33NWB'

    # baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    configuration = s1tiling.libs.configuration.Configuration(test_file, do_show_configuration=False)
    configuration.calibration_type             = 'normlim'
    configuration.extra_directories['lia_dir'] = liadir.absolute()
    configuration.fname_fmt['concatenation']   = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_tmpbeta.tif'
    configuration.relative_orbit_list          = [7]
    logging.info("Sigma0 NORMLIM mocked test")

    file_db = FileDB(
        inputdir=inputdir,
        eofdir=eofdir,
        tmpdir=tmpdir.absolute(),
        outputdir=outputdir.absolute(),
        xiadir=liadir.absolute(),
        gamma_areadir="",
        tile=tile,
        demdir=demdir,
        geoid_file=configuration.GeoidFile,
    )
    configuration.first_date       = (to_datetime(file_db.CONCATS[0]['start_time']) - timedelta(1)).strftime('%Y-%m-%d')
    configuration.last_date        = (to_datetime(file_db.CONCATS[number_dates-1]['start_time']) + timedelta(1)).strftime('%Y-%m-%d')
    configuration.produce_lia_map  = True
    configuration.show_configuration()

    mocker.patch('s1tiling.libs.otbtools.otb_version', lambda : '7.4.0')

    application_mocker = OTBApplicationsMockContext(configuration, mocker, file_db.tmp_to_out_map, file_db.dem_files)
    known_files = application_mocker.known_files
    known_dirs = set()
    declare_know_files(mocker, known_files, known_dirs, tile, ['vv'], file_db, application_mocker)
    for i in range(number_dates):
        assert os.path.isfile(file_db.input_file_vv(i))  # Check mocking

    def mock__AnalyseBorders_complete_meta(slf, meta, all_inputs):
        meta = super(AnalyseBorders, slf).complete_meta(meta, all_inputs)
        meta['cut'] = {
            'threshold.x'      : 0,
            'threshold.y.start': 0,
            'threshold.y.end'  : 0,
            'skip'             : True,
        }
        return meta
    mocker.patch('s1tiling.libs.otbwrappers.AnalyseBorders.complete_meta', mock__AnalyseBorders_complete_meta)

    mock_upto_concat_S2(application_mocker, file_db, naming_policy=naming_policy, calibration='normlim', N=number_dates*2)  # 2x2 inputs images
    mock_LIA_v1_0(application_mocker, file_db)  # always N=2
    mock_masking(application_mocker, file_db, 'normlim', number_dates*2, naming_policy)  # 2x2 inputs images

    is_nodata_SAR_bandmath = Utils.test_nodata_for_bandmath(bandname='im1b1', nodata=nodata_SAR)
    is_nodata_LIA_bandmath = Utils.test_nodata_for_bandmath(bandname='im2b1', nodata=nodata_LIA)
    for idx in range(number_dates):
        s2_input = file_db.concatfile_from_two(idx, False, naming_policy=naming_policy, calibration='_normlim')
        application_mocker.set_expectations(
            'BandMath',
            {
                'ram' : param_ram(2048),
                'il'  : [
                    s2_input,
                    file_db.selectedsinLIAfile()
                ],
                'exp' : f'({is_nodata_LIA_bandmath} || {is_nodata_SAR_bandmath}) ? {nodata_SAR} : max(1e-07, im1b1*im2b1)',
                'out' : file_db.sigma0_normlim_file_from_two(idx, True, naming_policy=naming_policy),
            }, None,
            {
                'CALIBRATION'              : 'Normlim',
                'IMAGE_TYPE'               : 'BACKSCATTERING',
                'LIA_FILE'                 : os.path.basename(file_db.selectedsinLIAfile()),
                'TIFFTAG_IMAGEDESCRIPTION' : 'Sigma0 Normlim Calibrated Sentinel-1A IW GRD',
            },
            {s2_input if configuration.tmpdir in s2_input else ""},
        )

    s1_process(
            config_opt=configuration, searched_items_per_page=0,
            dryrun=False, debug_otb=True, watch_ram=False, debug_tasks=False,
            lia_process=register_LIA_pipelines_v0,
    )
    application_mocker.assert_all_have_been_executed()
    application_mocker.assert_all_metadata_match()


@pytest.mark.parametrize("register_expectations,processor",
                         [
                             (mock_GAMMA_AREA_v1_2, s1_process_gamma_area),
                         ])
def test_33NWB_202001_gamma_area_mocked(
        tmpdir, demdir, ram,
        mocker,
        register_expectations, processor
):
    """
    Mocked test of production of GAMMA_AREA file
    """
    baselinedir   = pathlib.Path('/BASELINE')
    # eofdir        = pathlib.Path('/UNUSED')
    outputdir     = pathlib.Path('/OUTPUT')
    liadir        = pathlib.Path('/UNUSED')
    gamma_areadir = pathlib.Path('/GAMMA_AREA')
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)

    inputdir = str((baselinedir/'inputs').absolute())
    set_environ_mocked(inputdir, outputdir, liadir, gamma_areadir, demdir, tmpdir, ram)

    tile = '33NWB'

    # baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    configuration = s1tiling.libs.configuration.Configuration(test_file, do_show_configuration=False)
    configuration.calibration_type                    = 'gamma_naught_rtc'
    configuration.extra_directories['gamma_area_dir'] = gamma_areadir.absolute()
    configuration.show_configuration()
    print(configuration)
    logging.info("Sigma0 GAMMA_AREA mocked test")

    file_db = FileDB(
        inputdir=inputdir,
        eofdir="",
        tmpdir=tmpdir.absolute(),
        outputdir=outputdir.absolute(),
        xiadir="",
        gamma_areadir=gamma_areadir.absolute(),
        tile=tile,
        demdir=demdir,
        geoid_file=configuration.GeoidFile,
    )
    mocker.patch('s1tiling.libs.otbtools.otb_version', lambda : '7.4.0')

    application_mocker = OTBApplicationsMockContext(configuration, mocker, file_db.tmp_to_out_map, file_db.dem_files)
    known_files = application_mocker.known_files
    known_dirs = set()
    declare_know_files(mocker, known_files, known_dirs, tile, ['vv'], file_db, application_mocker)
    assert os.path.isfile(file_db.input_file_vv(0))  # Check mocking
    assert os.path.isfile(file_db.input_file_vv(1))

    register_expectations(application_mocker, file_db)

    processor(config_opt=configuration, searched_items_per_page=0,
            dryrun=False, debug_otb=True, watch_ram=False,
            debug_tasks=False)
    application_mocker.assert_all_have_been_executed()
    application_mocker.assert_all_metadata_match()


def test_33NWB_202001_gamma_naught_rtc_v1_0_mocked_one_date(tmpdir, demdir, ram, mocker, naming_policy):
    """
    Mocked test of production of S2 normlim calibrated images.
    """
    baselinedir   = pathlib.Path('/BASELINE')
    # eofdir        = pathlib.Path('/UNUSED')
    outputdir     = pathlib.Path('/OUTPUT')
    liadir        = pathlib.Path('/UNUSED')
    gamma_areadir = pathlib.Path('/GAMMA_AREA')
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)

    inputdir = str((baselinedir/'inputs').absolute())

    set_environ_mocked(inputdir, outputdir, liadir, gamma_areadir, demdir, tmpdir, ram)

    tile = '33NWB'

    # baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    configuration = s1tiling.libs.configuration.Configuration(test_file, do_show_configuration=False)
    configuration.calibration_type                    = 'gamma_naught_rtc'
    configuration.extra_directories['gamma_area_dir'] = gamma_areadir.absolute()
    configuration.show_configuration()
    configuration.fname_fmt['concatenation'] = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_tmpsigma.tif'
    logging.info("Gamma0 RTC mocked test")

    file_db = FileDB(
        inputdir=inputdir,
        eofdir="",
        tmpdir=tmpdir.absolute(),
        outputdir=outputdir.absolute(),
        xiadir="",
        gamma_areadir=gamma_areadir.absolute(),
        tile=tile,
        demdir=demdir,
        geoid_file=configuration.GeoidFile,
    )
    mocker.patch('s1tiling.libs.otbtools.otb_version', lambda : '7.4.0')

    application_mocker = OTBApplicationsMockContext(configuration, mocker, file_db.tmp_to_out_map, file_db.dem_files)
    known_files = application_mocker.known_files
    known_dirs = set()
    declare_know_files(mocker, known_files, known_dirs, tile, ['vv'], file_db, application_mocker)
    assert os.path.isfile(file_db.input_file_vv(0))  # Check mocking
    assert os.path.isfile(file_db.input_file_vv(1))

    def mock__AnalyseBorders_complete_meta(slf, meta, all_inputs):
        meta = super(AnalyseBorders, slf).complete_meta(meta, all_inputs)
        meta['cut'] = {
            'threshold.x'      : 0,
            'threshold.y.start': 0,
            'threshold.y.end'  : 0,
            'skip'             : True,
        }
        return meta
    mocker.patch('s1tiling.libs.otbwrappers.AnalyseBorders.complete_meta', mock__AnalyseBorders_complete_meta)

    mock_upto_concat_S2(application_mocker, file_db, naming_policy=naming_policy, calibration='gamma_naught_rtc', N=2)
    mock_GAMMA_AREA_v1_2(application_mocker, file_db)
    mock_masking(application_mocker, file_db, 'gamma_naught_rtc', 2, naming_policy)

    insigmanaught = file_db.concatfile_from_two(0, False, naming_policy=naming_policy, calibration='_gamma_naught_rtc')
    application_mocker.set_expectations(
        'SARGammaAreaToGammaNaughtRTCImageEstimation',
        {
            'ram'             : param_ram(2048),
            'ingammaarea'     : file_db.selectedGAMMA_AREAfile(),
            'insigmanaught'   : insigmanaught,
            'mingammaarea'    : 1.0,
            'calibfactor'     : 1.0,
            'streaming'       : 'enable',
            'producedatamask' : False,
            'nodata'          : '0',
            'out'             : file_db.gamma0_rtc_file_from_two(0, True, naming_policy=naming_policy),
        }, None, {
            'CALIBRATION'              : 'GammaNaughtRTC',
            'GAMMA_AREA_FILE'          : os.path.basename(file_db.selectedGAMMA_AREAfile()),
            'TIFFTAG_IMAGEDESCRIPTION' : 'Gamma0 RTC Calibrated Sentinel-1A IW GRD',
        })

    s1_process(
        config_opt=configuration, searched_items_per_page=0,
        dryrun=False, debug_otb=True, watch_ram=False, debug_tasks=False,
        gamma_area_process=register_GAMMA_AREA_pipelines,
    )
    application_mocker.assert_all_have_been_executed()
    application_mocker.assert_all_metadata_match()


def test_33NWB_202001_gamma_naught_rtc_v1_0_mocked_all_dates(tmpdir, demdir, ram, mocker, naming_policy):
    """
    Mocked test of production of S2 normlim calibrated images.
    """
    number_dates = 3

    baselinedir   = pathlib.Path('/BASELINE')
    # eofdir        = pathlib.Path('/UNUSED')
    outputdir     = pathlib.Path('/OUTPUT')
    liadir        = pathlib.Path('/UNUSED')
    gamma_areadir = pathlib.Path('/GAMMA_AREA')
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)

    inputdir = str((baselinedir/'inputs').absolute())

    set_environ_mocked(inputdir, outputdir, liadir, gamma_areadir, demdir, tmpdir, ram)

    tile = '33NWB'

    # baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    configuration = s1tiling.libs.configuration.Configuration(test_file, do_show_configuration=False)
    configuration.calibration_type                    = 'gamma_naught_rtc'
    configuration.extra_directories['gamma_area_dir'] = gamma_areadir.absolute()
    configuration.fname_fmt['concatenation'] = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_tmpsigma.tif'
    logging.info("Gamma0 RTC mocked test")

    file_db = FileDB(
        inputdir=inputdir,
        eofdir="",
        tmpdir=tmpdir.absolute(),
        outputdir=outputdir.absolute(),
        xiadir="",
        gamma_areadir=gamma_areadir.absolute(),
        tile=tile,
        demdir=demdir,
        geoid_file=configuration.GeoidFile,
    )
    configuration.first_date       = (to_datetime(file_db.CONCATS[0]['start_time']) - timedelta(1)).strftime('%Y-%m-%d')
    configuration.last_date        = (to_datetime(file_db.CONCATS[number_dates-1]['start_time']) + timedelta(1)).strftime('%Y-%m-%d')
    configuration.show_configuration()

    mocker.patch('s1tiling.libs.otbtools.otb_version', lambda : '7.4.0')

    application_mocker = OTBApplicationsMockContext(configuration, mocker, file_db.tmp_to_out_map, file_db.dem_files)
    known_files = application_mocker.known_files
    known_dirs = set()
    declare_know_files(mocker, known_files, known_dirs, tile, ['vv'], file_db, application_mocker)
    for i in range(number_dates):
        assert os.path.isfile(file_db.input_file_vv(i))  # Check mocking

    def mock__AnalyseBorders_complete_meta(slf, meta, all_inputs):
        meta = super(AnalyseBorders, slf).complete_meta(meta, all_inputs)
        meta['cut'] = {
            'threshold.x'      : 0,
            'threshold.y.start': 0,
            'threshold.y.end'  : 0,
            'skip'             : True,
        }
        return meta
    mocker.patch('s1tiling.libs.otbwrappers.AnalyseBorders.complete_meta', mock__AnalyseBorders_complete_meta)

    mock_upto_concat_S2(application_mocker, file_db, naming_policy=naming_policy, calibration='gamma_naught_rtc', N=number_dates*2)  # 2x2 inputs images
    mock_GAMMA_AREA_v1_2(application_mocker, file_db)  # always N=2
    mock_masking(application_mocker, file_db, 'gamma_naught_rtc', number_dates*2, naming_policy)  # 2x2 inputs images

    for idx in range(number_dates):
        insigmanaught = file_db.concatfile_from_two(idx, False, naming_policy=naming_policy, calibration='_gamma_naught_rtc')
        application_mocker.set_expectations(
            'SARGammaAreaToGammaNaughtRTCImageEstimation',
            {
                'ram'             : param_ram(2048),
                'ingammaarea'     : file_db.selectedGAMMA_AREAfile(),
                'insigmanaught'   : insigmanaught,
                'mingammaarea'    : 1.0,
                'streaming'       : 'enable',
                'calibfactor'     : 1.0,
                'producedatamask' : False,
                'nodata'          : '0',
                'out'             : file_db.gamma0_rtc_file_from_two(idx, True, naming_policy=naming_policy),
            }, None, {
                'CALIBRATION'             : 'GammaNaughtRTC',
                'GAMMA_AREA_FILE'         : os.path.basename(file_db.selectedGAMMA_AREAfile()),
                'TIFFTAG_IMAGEDESCRIPTION': 'Gamma0 RTC Calibrated Sentinel-1A IW GRD',
            })

    s1_process(
            config_opt=configuration, searched_items_per_page=0,
            dryrun=False, debug_otb=True, watch_ram=False, debug_tasks=False,
            lia_process=register_LIA_pipelines_v0,
    )
    application_mocker.assert_all_have_been_executed()
    application_mocker.assert_all_metadata_match()
