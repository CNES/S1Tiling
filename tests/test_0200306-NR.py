#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import fnmatch
import logging
import os
import pathlib
import shutil
import subprocess

import otbApplication as otb

from unittest.mock import patch

# import pytest_check
from .helpers import otb_compare, comparable_metadata
from .mock_otb import OTBApplicationsMockContext, isfile, isdir, list_dirs, glob, dirname, makedirs
from .mock_data import FileDB
import s1tiling.S1Processor
import s1tiling.libs.configuration

from s1tiling.libs.otbpipeline import _fetch_input_data


# ======================================================================
# Full processing versions
# ======================================================================

def remove_dirs(dir_list):
    for dir in dir_list:
        if os.path.isdir(dir):
            logging.info("rm -r '%s'", dir)
            shutil.rmtree(dir)


def process(tmpdir, outputdir, liadir, baseline_reference_outputs, test_file, watch_ram, dirs_to_clean=None):
    '''
    Executes the S1Processor
    '''
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    src_dir       = crt_dir.parent.absolute()
    dirs_to_clean = dirs_to_clean or [outputdir, tmpdir/'S1', tmpdir/'S2', liadir]

    logging.info('$S1TILING_TEST_DATA_INPUT  -> %s', os.environ['S1TILING_TEST_DATA_INPUT'])
    logging.info('$S1TILING_TEST_DATA_OUTPUT -> %s', os.environ['S1TILING_TEST_DATA_OUTPUT'])
    logging.info('$S1TILING_TEST_DATA_LIA    -> %s', os.environ['S1TILING_TEST_DATA_LIA'])
    logging.info('$S1TILING_TEST_SRTM        -> %s', os.environ['S1TILING_TEST_SRTM'])
    logging.info('$S1TILING_TEST_TMPDIR      -> %s', os.environ['S1TILING_TEST_TMPDIR'])
    logging.info('$S1TILING_TEST_DOWNLOAD    -> %s', os.environ['S1TILING_TEST_DOWNLOAD'])
    logging.info('$S1TILING_TEST_RAM         -> %s', os.environ['S1TILING_TEST_RAM'])

    remove_dirs(dirs_to_clean)

    args = ['python3', src_dir / 's1tiling/S1Processor.py', test_file]
    if watch_ram:
        args.append('--watch-ram')
    # args.append('--cache-before-ortho')
    logging.info('Running: %s', args)
    return subprocess.call(args, cwd=crt_dir)


def test_33NWB_202001_NR_execute_OTB(baselinedir, outputdir, liadir, tmpdir, srtmdir, ram, download, watch_ram):
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
    os.environ['S1TILING_TEST_SRTM']               = str(srtmdir.absolute())
    os.environ['S1TILING_TEST_TMPDIR']             = str(tmpdir.absolute())
    os.environ['S1TILING_TEST_RAM']                = str(ram)

    images = [
            '33NWB/s1a_33NWB_vh_DES_007_20200108txxxxxx.tif',
            '33NWB/s1a_33NWB_vv_DES_007_20200108txxxxxx.tif',
            '33NWB/s1a_33NWB_vh_DES_007_20200108txxxxxx_BorderMask.tif',
            '33NWB/s1a_33NWB_vv_DES_007_20200108txxxxxx_BorderMask.tif',
            ]
    baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    logging.info("Full test")
    EX = process(tmpdir, outputdir, liadir, baseline_path, test_file, watch_ram)
    assert EX == 0
    for im in images:
        expected = baseline_path / im
        produced = outputdir / im
        assert os.path.isfile(produced)
        assert otb_compare(expected, produced) == 0, \
                ("Comparison of %s against %s failed" % (produced, expected))
        assert comparable_metadata(expected) == comparable_metadata(produced)
    # The following line permits to test otb_compare correctly detect differences when
    # called from pytest.
    # assert otb_compare(baseline_path+images[0], result_path+images[1]) == 0


def test_33NWB_202001_NR_masks_only_execute_OTB(baselinedir, outputdir, liadir, tmpdir, srtmdir, ram, download, watch_ram):
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
    os.environ['S1TILING_TEST_SRTM']               = str(srtmdir.absolute())
    os.environ['S1TILING_TEST_TMPDIR']             = str(tmpdir.absolute())
    os.environ['S1TILING_TEST_RAM']                = str(ram)

    images = [
            '33NWB/s1a_33NWB_vh_DES_007_20200108txxxxxx_BorderMask.tif',
            '33NWB/s1a_33NWB_vv_DES_007_20200108txxxxxx_BorderMask.tif',
            ]
    baseline_path = baselinedir / 'expected'
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
    EX = process(tmpdir, outputdir, liadir, baseline_path, test_file, watch_ram, dirs_to_clean)
    assert EX == 0
    for im in images:
        expected = baseline_path / im
        produced = outputdir / im
        assert os.path.isfile(produced)
        assert otb_compare(expected, produced) == 0, \
                ("Comparison of %s against %s failed" % (produced, expected))
        assert comparable_metadata(expected) == comparable_metadata(produced)


# ======================================================================
# Mocked versions
# ======================================================================

def _declare_know_files(mocker, known_files, known_dirs, patterns, file_db):
    # logging.debug('_declare_know_files(%s)', patterns)
    all_files = file_db.all_files()
    # logging.debug('- all_files: %s', all_files)
    files = []
    for pattern in patterns:
        files += [fn for fn in all_files if fnmatch.fnmatch(fn, '*'+pattern+'*')]
    known_files.extend(files)
    known_dirs.update([dirname(fn, 3) for fn in known_files])
    known_dirs.update([dirname(fn, 2) for fn in known_files])
    # known_dirs.update([dirname(fn, 1) for fn in known_files])
    logging.debug('Mocking w/ %s --> %s', patterns, files)
    # Utils.list_dirs has been imported in S1FileManager. This is the one that needs patching!
    mocker.patch('s1tiling.libs.S1FileManager.list_dirs', lambda dir, pat : list_dirs(dir, pat, known_dirs, file_db.inputdir))
    mocker.patch('glob.glob',        lambda pat  : glob(pat, known_files))
    mocker.patch('os.path.isfile',   lambda file : isfile(file, known_files))
    mocker.patch('os.path.isdir',    lambda dir  : isdir(dir, known_dirs))
    mocker.patch('os.makedirs',      lambda dir, **kw  : makedirs(dir, known_dirs))
    mocker.patch('os.path.getctime', lambda file : 0)
    # TODO: Test written meta data as well
    mocker.patch('s1tiling.libs.otbwrappers.OrthoRectify.add_ortho_metadata',    lambda slf, mt, app : True)
    mocker.patch('s1tiling.libs.otbwrappers.OrthoRectifyLIA.add_ortho_metadata', lambda slf, mt, app : True)
    mocker.patch('s1tiling.libs.otbpipeline.commit_execution',                   lambda tmp, out : True)
    mocker.patch('s1tiling.libs.Utils.get_origin',          lambda manifest : file_db.get_origin(manifest))
    mocker.patch('s1tiling.libs.Utils.get_orbit_direction', lambda manifest : file_db.get_orbit_direction(manifest))
    mocker.patch('s1tiling.libs.Utils.get_relative_orbit',  lambda manifest : file_db.get_relative_orbit(manifest))

    def mock_commit_execution_for_SelectLIA(inp, out):
        logging.debug('mock.mv %s %s', inp, out)
        assert os.path.isfile(inp)
        known_files.append(out)
        known_files.remove(inp)
    mocker.patch('s1tiling.libs.otbwrappers.commit_execution', mock_commit_execution_for_SelectLIA)

    def mock_add_image_metadata(slf, mt, *args, **kwargs):
        # TODO: Problem: how can we pass around meta from different pipelines???
        fullpath = mt.get('out_filename')
        logging.debug('Mock Set metadata in %s', fullpath)
        assert 'inputs' in mt, f'Looking for "inputs" in {mt.keys()}'
        inputs = mt['inputs']
        # indem = _fetch_input_data('indem', inputs)
        # assert 'srtms' in indem.meta, f"Metadata don't contain 'srtms', only: {indem.meta.keys()}"
        assert 'srtms' in mt, f"Metadata don't contain 'srtms', only: {mt.keys()}"
        return mt
    mocker.patch('s1tiling.libs.otbwrappers.SARDEMProjection.add_image_metadata', mock_add_image_metadata)

    def mock_direction_to_scan(slf, meta):
        logging.debug('Mocking direction to scan')
        meta['directiontoscandeml'] = 12
        meta['directiontoscandemc'] = 24
        meta['gain']                = 42
        return meta
    mocker.patch('s1tiling.libs.otbwrappers.SARCartesianMeanEstimation.fetch_direction', lambda slf, ip, mt : mock_direction_to_scan(slf, mt))


def set_environ_mocked(inputdir, outputdir, liadir, srtmdir, tmpdir, ram):
    os.environ['S1TILING_TEST_DOWNLOAD']       = 'False'
    os.environ['S1TILING_TEST_OVERRIDE_CUT_Y'] = 'False' # keep everything

    os.environ['S1TILING_TEST_DATA_INPUT']         = str(inputdir)
    os.environ['S1TILING_TEST_DATA_OUTPUT']        = str(outputdir.absolute())
    os.environ['S1TILING_TEST_DATA_LIA']           = str(liadir.absolute())
    os.environ['S1TILING_TEST_SRTM']               = str(srtmdir.absolute())
    os.environ['S1TILING_TEST_TMPDIR']             = str(tmpdir.absolute())
    os.environ['S1TILING_TEST_RAM']                = str(ram)


def mock_upto_concat_S2(application_mocker, file_db, calibration, N=2):
    raw_calibration = 'beta' if calibration == 'normlim' else calibration
    for i in range(N):
        input_file = file_db.input_file_vv(i)
        expected_ortho_file = file_db.orthofile(i, False)

        application_mocker.set_expectations('SARCalibration', {
            'ram'        : '2048',
            'in'         : input_file,
            'lut'        : raw_calibration,
            'removenoise': False,
            'out'        : 'ResetMargin|>OrthoRectification|>'+file_db.orthofile(i, True),
            }, None)

        application_mocker.set_expectations('ResetMargin', {
            'in'               : input_file+'|>SARCalibration',
            'ram'              : '2048',
            'threshold.x'      : 1000,
            'threshold.y.start': 0,
            'threshold.y.end'  : 0,
            'mode'             : 'threshold',
            'out'              : 'OrthoRectification|>'+file_db.orthofile(i, True),
            }, None)

        application_mocker.set_expectations('OrthoRectification', {
            'io.in'           : input_file+'|>SARCalibration|>ResetMargin',
            'opt.ram'         : '2048',
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
            'io.out'          : file_db.orthofile(i, True),
            }, None)

    for i in range(N//2):
        application_mocker.set_expectations('Synthetize', {
            'ram'      : '2048',
            'il'       : [file_db.orthofile(2*i, False), file_db.orthofile(2*i+1, False)],
            'out'      : file_db.concatfile_from_two(i, True),
            }, None)


def mock_masking(application_mocker, file_db, calibration, N=2):
    if calibration == 'normlim':
        infile = file_db.sigma0_normlim_file_from_two
    else:
        infile = file_db.concatfile_from_two

    for i in range(N // 2):
        application_mocker.set_expectations('BandMath', {
            'ram'      : '2048',
            'il'       : [infile(i, False)],
            'exp'      : 'im1b1==0?0:1',
            'out'      : 'BinaryMorphologicalOperation|>'+file_db.maskfile_from_two(i, True),
            }, {'out': otb.ImagePixelType_uint8})
        application_mocker.set_expectations('BinaryMorphologicalOperation', {
            'in'       : [infile(i, False)+'|>BandMath'],
            'ram'      : '2048',
            'structype': 'ball',
            'xradius'  : 5,
            'yradius'  : 5,
            'filter'   : 'opening',
            'out'      : file_db.maskfile_from_two(i, True),
            }, {'out': otb.ImagePixelType_uint8})


def mock_LIA(application_mocker, file_db):
    srtmdir = file_db.srtmdir
    for idx in range(2):
        cov               = file_db.dem_coverage(idx)
        exp_srtm_names    = sorted(cov)
        exp_out_vrt       = file_db.vrtfile(idx, False)
        exp_out_dem       = file_db.sardemprojfile(idx, False)
        exp_in_srtm_files = [f"{srtmdir}/{srtm}.hgt" for srtm in exp_srtm_names]

        application_mocker.set_expectations('gdalbuildvrt', [file_db.vrtfile(idx, True)] + exp_in_srtm_files, None)

        application_mocker.set_expectations('SARDEMProjection', {
            'ram'        : '2048',
            'insar'      : file_db.input_file_vv(idx),
            'indem'      : exp_out_vrt,
            'withxyz'    : True,
            'nodata'     : -32768,
            'out'        : file_db.sardemprojfile(idx, True),
            }, None)

        application_mocker.set_expectations('SARCartesianMeanEstimation', {
            'ram'             : '2048',
            'insar'           : file_db.input_file_vv(idx),
            'indem'           : exp_out_vrt,
            'indemproj'       : exp_out_dem,
            'indirectiondemc' : 24,
            'indirectiondeml' : 12,
            'mlran'           : 1,
            'mlazi'           : 1,
            'out'             : file_db.xyzfile(idx, True),
            }, None)

        application_mocker.set_expectations('ExtractNormalVector', {
            'ram'             : '2048',
            'xyz'             : file_db.xyzfile(idx, False),
            'out'             : 'SARComputeLocalIncidenceAngle|>'+file_db.LIAfile(idx, True),
            }, None)

        application_mocker.set_expectations('SARComputeLocalIncidenceAngle', {
            'ram'             : '2048',
            'in.normals'      : file_db.xyzfile(idx, False)+'|>ExtractNormalVector', #'ComputeNormals|>'+file_db.normalsfile(idx),
            'in.xyz'          : file_db.xyzfile(idx, False),
            'out.lia'         : file_db.LIAfile(idx, True),
            'out.sin'         : file_db.sinLIAfile(idx, True),
            }, None)

        application_mocker.set_expectations('OrthoRectification', {
            'opt.ram'         : '2048',
            'io.in'           : file_db.LIAfile(idx, False),
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
            'io.out'          : file_db.orthoLIAfile(idx, True),
            }, None)

        application_mocker.set_expectations('OrthoRectification', {
            'opt.ram'         : '2048',
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
            }, None)

    # endfor on 2 consecutive images

    application_mocker.set_expectations('Synthetize', {
        'ram'      : '2048',
        'il'       : [file_db.orthoLIAfile(0, False), file_db.orthoLIAfile(1, False)],
        'out'      : file_db.concatLIAfile_from_two(0, True),
        }, None)

    application_mocker.set_expectations('Synthetize', {
        'ram'      : '2048',
        'il'       : [file_db.orthosinLIAfile(0, False), file_db.orthosinLIAfile(1, False)],
        'out'      : file_db.concatsinLIAfile_from_two(0, True),
        }, None)


def test_33NWB_202001_NR_core_mocked(baselinedir, outputdir, liadir, tmpdir, srtmdir, ram, download, watch_ram, mocker):
    """
    Mocked test of production of S2 sigma0 calibrated images.
    """
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)

    inputdir = str((baselinedir/'inputs').absolute())
    set_environ_mocked(inputdir, outputdir, liadir, srtmdir, tmpdir, ram)

    baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    configuration = s1tiling.libs.configuration.Configuration(test_file)
    logging.info("Full mocked test")

    file_db = FileDB(inputdir, tmpdir.absolute(), outputdir.absolute(), liadir.absolute(), '33NWB', srtmdir, configuration.GeoidFile)
    mocker.patch('s1tiling.libs.otbwrappers.otb_version', lambda : '7.4.0')

    application_mocker = OTBApplicationsMockContext(configuration, mocker, file_db.tmp_to_out_map)
    known_files = application_mocker.known_files
    known_dirs = set()
    _declare_know_files(mocker, known_files, known_dirs, ['vv'], file_db)
    assert os.path.isfile(file_db.input_file_vv(0))  # Check mocking
    assert os.path.isfile(file_db.input_file_vv(1))
    mock_upto_concat_S2(application_mocker, file_db, 'sigma')
    mock_masking(application_mocker, file_db, 'sigma')
    s1tiling.S1Processor.s1_process(config_opt=configuration, searched_items_per_page=0,
            dryrun=False, debug_otb=True, watch_ram=False,
            debug_tasks=False, cache_before_ortho=False)
    application_mocker.assert_all_have_been_executed()


def test_33NWB_202001_lia_mocked(baselinedir, outputdir, liadir, tmpdir, srtmdir, ram, download, watch_ram, mocker):
    """
    Mocked test of production of LIA and sin LIA files
    """
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)

    inputdir = str((baselinedir/'inputs').absolute())

    set_environ_mocked(inputdir, outputdir, liadir, srtmdir, tmpdir, ram)

    tile_name = '33NWB'
    baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    configuration = s1tiling.libs.configuration.Configuration(test_file)
    configuration.calibration_type = 'normlim'
    configuration.lia_directory = liadir.absolute()
    logging.info("Sigma0 NORMLIM mocked test")

    file_db = FileDB(inputdir, tmpdir.absolute(), outputdir.absolute(), liadir.absolute(), tile_name, srtmdir, configuration.GeoidFile)
    mocker.patch('s1tiling.libs.otbwrappers.otb_version', lambda : '7.4.0')

    application_mocker = OTBApplicationsMockContext(configuration, mocker, file_db.tmp_to_out_map)
    known_files = application_mocker.known_files
    known_dirs = set()
    _declare_know_files(mocker, known_files, known_dirs, ['vv'], file_db)
    assert os.path.isfile(file_db.input_file_vv(0))  # Check mocking
    assert os.path.isfile(file_db.input_file_vv(1))

    mock_LIA(application_mocker, file_db)

    s1tiling.S1Processor.s1_process_lia(config_opt=configuration, searched_items_per_page=0,
            dryrun=False, debug_otb=True, watch_ram=False,
            debug_tasks=False)
    application_mocker.assert_all_have_been_executed()


def test_33NWB_202001_normlim_mocked_one_date(baselinedir, outputdir, liadir, tmpdir, srtmdir, ram, download, watch_ram, mocker):
    """
    Mocked test of production of S2 normlim calibrated images.
    """
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)

    inputdir = str((baselinedir/'inputs').absolute())

    set_environ_mocked(inputdir, outputdir, liadir, srtmdir, tmpdir, ram)

    tile_name = '33NWB'
    baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    configuration = s1tiling.libs.configuration.Configuration(test_file, do_show_configuration=False)
    configuration.calibration_type = 'normlim'
    configuration.lia_directory = liadir.absolute()
    configuration.show_configuration()
    logging.info("Sigma0 NORMLIM mocked test")

    file_db = FileDB(inputdir, tmpdir.absolute(), outputdir.absolute(), liadir.absolute(), tile_name, srtmdir, configuration.GeoidFile)
    mocker.patch('s1tiling.libs.otbwrappers.otb_version', lambda : '7.4.0')

    application_mocker = OTBApplicationsMockContext(configuration, mocker, file_db.tmp_to_out_map)
    known_files = application_mocker.known_files
    known_dirs = set()
    _declare_know_files(mocker, known_files, known_dirs, ['vv'], file_db)
    assert os.path.isfile(file_db.input_file_vv(0))  # Check mocking
    assert os.path.isfile(file_db.input_file_vv(1))

    mock_upto_concat_S2(application_mocker, file_db, 'normlim')
    mock_LIA(application_mocker, file_db)
    mock_masking(application_mocker, file_db, 'normlim')

    application_mocker.set_expectations('BandMath', {
        'ram'      : '2048',
        'il'       : [file_db.concatfile_from_two(0, False), file_db.selectedsinLIAfile()],
        'exp'      : 'im1b1*im2b1',
        'out'      : file_db.sigma0_normlim_file_from_two(0, True),
        }, None)

    s1tiling.S1Processor.s1_process(config_opt=configuration, searched_items_per_page=0,
            dryrun=False, debug_otb=True, watch_ram=False,
            debug_tasks=False)
    application_mocker.assert_all_have_been_executed()


def test_33NWB_202001_normlim_mocked_all_dates(baselinedir, outputdir, liadir, tmpdir, srtmdir, ram, download, watch_ram, mocker):
    """
    Mocked test of production of S2 normlim calibrated images.
    """
    number_dates = 3

    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)

    inputdir = str((baselinedir/'inputs').absolute())

    set_environ_mocked(inputdir, outputdir, liadir, srtmdir, tmpdir, ram)

    tile_name = '33NWB'
    baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    configuration = s1tiling.libs.configuration.Configuration(test_file, do_show_configuration=False)
    configuration.calibration_type = 'normlim'
    configuration.lia_directory = liadir.absolute()
    logging.info("Sigma0 NORMLIM mocked test")

    file_db = FileDB(inputdir, tmpdir.absolute(), outputdir.absolute(), liadir.absolute(), tile_name, srtmdir, configuration.GeoidFile)
    configuration.first_date = file_db.CONCATS[0]['first_date']
    configuration.last_date  = file_db.CONCATS[number_dates-1]['last_date']
    configuration.show_configuration()

    mocker.patch('s1tiling.libs.otbwrappers.otb_version', lambda : '7.4.0')

    application_mocker = OTBApplicationsMockContext(configuration, mocker, file_db.tmp_to_out_map)
    known_files = application_mocker.known_files
    known_dirs = set()
    _declare_know_files(mocker, known_files, known_dirs, ['vv'], file_db)
    for i in range(number_dates):
        assert os.path.isfile(file_db.input_file_vv(i))  # Check mocking

    mock_upto_concat_S2(application_mocker, file_db, 'normlim', number_dates*2)  # 2x2 inputs images
    mock_LIA(application_mocker, file_db)  # always N=2
    mock_masking(application_mocker, file_db, 'normlim', number_dates*2)  # 2x2 inputs images

    for idx in range(number_dates):
        application_mocker.set_expectations('BandMath', {
            'ram'      : '2048',
            'il'       : [file_db.concatfile_from_two(idx, False), file_db.selectedsinLIAfile()],
            'exp'      : 'im1b1*im2b1',
            'out'      : file_db.sigma0_normlim_file_from_two(idx, True),
            }, None)

    s1tiling.S1Processor.s1_process(config_opt=configuration, searched_items_per_page=0,
            dryrun=False, debug_otb=True, watch_ram=False,
            debug_tasks=False)
    application_mocker.assert_all_have_been_executed()

