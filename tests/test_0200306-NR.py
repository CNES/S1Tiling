#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import fnmatch
import logging
import os
import pathlib
import shutil
import subprocess

# from unittest.mock import Mock
from unittest.mock import patch
# import unittest.mock

# import pytest_check
from .helpers import otb_compare, comparable_metadata
from .mock_otb import OTBApplicationsMockContext
import s1tiling.S1Processor
import s1tiling.libs.configuration


# ======================================================================
# Full prcessing versions
# ======================================================================

def remove_dirs(dir_list):
    for dir in dir_list:
        if os.path.isdir(dir):
            logging.info("rm -r '%s'", dir)
            shutil.rmtree(dir)


def process(tmpdir, outputdir, baseline_reference_outputs, test_file, watch_ram, dirs_to_clean=None):
    '''
    Executes the S1Processor
    '''
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    src_dir       = crt_dir.parent.absolute()
    dirs_to_clean = dirs_to_clean or [outputdir, tmpdir/'S1', tmpdir/'S2']

    logging.info('$S1TILING_TEST_DATA_INPUT  -> %s', os.environ['S1TILING_TEST_DATA_INPUT'])
    logging.info('$S1TILING_TEST_DATA_OUTPUT -> %s', os.environ['S1TILING_TEST_DATA_OUTPUT'])
    logging.info('$S1TILING_TEST_SRTM        -> %s', os.environ['S1TILING_TEST_SRTM'])
    logging.info('$S1TILING_TEST_TMPDIR      -> %s', os.environ['S1TILING_TEST_TMPDIR'])
    logging.info('$S1TILING_TEST_DOWNLOAD    -> %s', os.environ['S1TILING_TEST_DOWNLOAD'])
    logging.info('$S1TILING_TEST_RAM         -> %s', os.environ['S1TILING_TEST_RAM'])

    remove_dirs(dirs_to_clean)

    args = ['python3', src_dir / 's1tiling/S1Processor.py', test_file]
    if watch_ram:
        args.append('--watch-ram')
    logging.info('Running: %s', args)
    return subprocess.call(args, cwd=crt_dir)


def test_33NWB_202001_NR(baselinedir, outputdir, tmpdir, srtmdir, ram, download, watch_ram):
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
    EX = process(tmpdir, outputdir, baseline_path, test_file, watch_ram)
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


def test_33NWB_202001_NR_masks_only(baselinedir, outputdir, tmpdir, srtmdir, ram, download, watch_ram):
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
    EX = process(tmpdir, outputdir, baseline_path, test_file, watch_ram, dirs_to_clean)
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

class FileDB:
    FILES = [
            {
                's1dir'      : 'S1A_IW_GRDH_1SDV_20200108T044150_20200108T044215_030704_038506_C7F5',
                's1file'     : 's1a-iw-grd-vv-20200108t044150-20200108t044215-030704-038506-001.tiff',
                'cal_ok'     : 's1a-iw-grd-vv-20200108t044150-20200108t044215-030704-038506-001_CalOk.tiff',
                'ortho_ready': 's1a-iw-grd-vv-20200108t044150-20200108t044215-030704-038506-001_OrthoReady.tiff',
                'orthofile'  : 's1a_33NWB_vv_DES_007_20200108t044150'
                },
            {
                's1dir'      : 'S1A_IW_GRDH_1SDV_20200108T044215_20200108T044240_030704_038506_D953',
                's1file'     : 's1a-iw-grd-vv-20200108t044215-20200108t044240-030704-038506-001.tiff',
                'cal_ok'     : 's1a-iw-grd-vv-20200108t044215-20200108t044240-030704-038506-001_CalOk.tiff',
                'ortho_ready': 's1a-iw-grd-vv-20200108t044215-20200108t044240-030704-038506-001_OrthoReady.tiff',
                'orthofile'  : 's1a_33NWB_vv_DES_007_20200108t044215'
                }
            ]

    def __init__(self, inputdir, tmpdir, outputdir, tile):
        self.__input_dir  = inputdir
        self.__tmp_dir    = tmpdir
        self.__output_dir = outputdir
        self.__tile       = tile

    @property
    def inputdir(self):
        """
        Property inputdir
        """
        return self.__input_dir

    @property
    def outputdir(self):
        """
        Property outputdir
        """
        return self.__output_dir

    def all_files(self):
        return [self.input_file(idx) for idx in range(len(self.FILES))]

    def safe_dir(self, idx):
        s1dir  = self.FILES[idx]['s1dir']
        return f'{self.__input_dir}/{s1dir}/{s1dir}.SAFE'

    def input_file(self, idx):
        s1dir  = self.FILES[idx]['s1dir']
        s1file = self.FILES[idx]['s1file']
        return f'{self.__input_dir}/{s1dir}/{s1dir}.SAFE/measurement/{s1file}'

    def input_file_vv(self, idx):
        assert idx < 2
        return self.input_file(idx)

    def raster_vv(self, idx):
        s1dir  = self.FILES[idx]['s1dir']
        return (S1DateAcquisition(
            f'{self.__input_dir}/{s1dir}/{s1dir}.SAFE/manifest.safe',
            [self.input_file(idx)]),
            [(14.9998201759, 1.8098185887), (15.9870050338, 1.8095484335), (15.9866155411, 0.8163071941), (14.9998202469, 0.8164290331000001)])

    def cal_ok(self, idx):
        return f'{self.__tmp_dir}/S1/{self.FILES[idx]["cal_ok"]}'

    def ortho_ready(self, idx):
        return f'{self.__tmp_dir}/S1/{self.FILES[idx]["ortho_ready"]}'

    def orthofile(self, idx):
        return f'{self.__tmp_dir}/S2/{self.__tile}/{self.FILES[idx]["orthofile"]}.tif'

    def concatfile(self, idx):
        if idx is None:
            return f'{self.__output_dir}/{self.__tile}/s1a_33NWB_vv_DES_007_20200108txxxxxx.tif'
        else:
            return f'{self.__output_dir}/{self.__tile}/{self.FILES[idx]["orthofile"]}.tif'

    def masktmp(self, idx):
        if idx is None:
            return f'{self.__output_dir}/{self.__tile}/s1a_33NWB_vv_DES_007_20200108txxxxxx_BorderMaskTmp.tif'
        else:
            return f'{self.__output_dir}/{self.__tile}/{self.FILES[idx]["orthofile"]}_BorderMaskTmp.tif'

    def maskfile(self, idx):
        if idx is None:
            return f'{self.__output_dir}/{self.__tile}/s1a_33NWB_vv_DES_007_20200108txxxxxx_BorderMask.tif'
        else:
            return f'{self.__output_dir}/{self.__tile}/{self.FILES[idx]["orthofile"]}_BorderMask.tif'

    def dem_file(self):
        return f'{self.__tmp_dir}/TMP'

    # def geoid_file(self):
    #     return f'resources/Geoid/egm96.grd'


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


def _declare_know_files(mocker, known_files, known_dirs, patterns, file_db):
    # logging.debug('_declare_know_files(%s)', patterns)
    all_files = file_db.all_files()
    # logging.debug('- all_files: %s', all_files)
    files = []
    for pattern in patterns:
        files += [fn for fn in all_files if fnmatch.fnmatch(fn, '*'+pattern+'*')]
    known_files.extend(files)
    known_dirs.update([dirname(fn, 3) for fn in known_files])
    logging.debug('Mocking w/ %s --> %s', patterns, files)
    # Utils.list_dirs has been imported in S1FileManager. This is the one that needs patching!
    # mocker.patch('s1tiling.libs.S1FileManager.list_dirs', return_value=files)
    mocker.patch('s1tiling.libs.S1FileManager.list_dirs', lambda dir, pat : list_dirs(dir, pat, known_dirs, file_db.inputdir))
    mocker.patch('glob.glob', lambda pat : glob(pat, known_files))


def test_33NWB_202001_NR_mocked(baselinedir, outputdir, tmpdir, srtmdir, ram, download, watch_ram, mocker):
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)

    inputdir = str((baselinedir/'inputs').absolute())

    os.environ['S1TILING_TEST_DOWNLOAD']       = 'False'
    os.environ['S1TILING_TEST_OVERRIDE_CUT_Y'] = 'False' # keep everything

    os.environ['S1TILING_TEST_DATA_INPUT']         = str(inputdir)
    os.environ['S1TILING_TEST_DATA_OUTPUT']        = str(outputdir.absolute())
    os.environ['S1TILING_TEST_SRTM']               = str(srtmdir.absolute())
    os.environ['S1TILING_TEST_TMPDIR']             = str(tmpdir.absolute())
    os.environ['S1TILING_TEST_RAM']                = str(ram)

    images = [
            # '33NWB/s1a_33NWB_vh_DES_007_20200108txxxxxx.tif',
            '33NWB/s1a_33NWB_vv_DES_007_20200108txxxxxx.tif',
            # '33NWB/s1a_33NWB_vh_DES_007_20200108txxxxxx_BorderMask.tif',
            '33NWB/s1a_33NWB_vv_DES_007_20200108txxxxxx_BorderMask.tif',
            ]
    baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    configuration = s1tiling.libs.configuration.Configuration(test_file)
    logging.info("Full mocked test")

    file_db = FileDB(inputdir, tmpdir.absolute(), outputdir.absolute(), '33NWB')
    # known_files = [ file_db.input_file_vv(0) ]
    # logging.error('os.path.isfile: %s', os.path.isfile)
    # mocker.patch('os.path.isfile', lambda f: isfile(f, known_files))
    # logging.error('mock.isfile: %s', os.path.isfile)
    # assert not os.path.isfile('foo')
    # assert os.path.isfile(file_db.input_file_vv(0))
    # assert not os.path.isfile(file_db.input_file_vv(1))

    known_files = []
    known_dirs = set()
    _declare_know_files(mocker, known_files, known_dirs, ['vv'], file_db)
    assert os.path.isfile(file_db.input_file_vv(0))  # Check mocking
    assert os.path.isfile(file_db.input_file_vv(1))
    application_mocker = OTBApplicationsMockContext(configuration, mocker)
    for i in range(2):
        #SARCalibration -ram '4096' -in '/home/luc/dev/S1tiling/tests/20200306-NR/data_raw-pol/S1A_IW_GRDH_1SDV_20200108T044215_20200108T044240_030704_038506_D953/S1A_IW_GRDH_1SDV_20200108T044215_20200108T044240_030704_038506_D953.SAFE/measurement/s1a-iw-grd-vh-20200108t044215-20200108t044240-030704-038506-002.tiff' -lut 'sigma' -noise False
        application_mocker.set_expectations('SARCalibration', {
            'ram': 2048,
            'in': file_db.input_file_vv(i),
            'out': file_db.cal_ok(i),
            'lut': 'sigma',
            'noise': False,
            }, None)

        # ResetMargin (from app) -ram '4096' -threshold.x 1000 -threshold.y.start 0 -threshold.y.end 0 -mode 'threshold'
        application_mocker.set_expectations('ResetMargin', {
            'ram': 2048,
            'in': file_db.cal_ok(i),
            'out': file_db.ortho_ready(i),
            'threshold.x': 1000,
            'threshold.y.start': 0,
            'threshold.y.end': 0,
            'mode': 'threshold',
            }, None)

        #  OrthoRectification (from app) -opt.ram '4096' -interpolator 'nn' -outputs.spacingx '10.0' -outputs.spacingy '-10.0' -outputs.sizex 10980 -outputs.sizey 10980 -opt.gridspacing '40.0' -map 'utm' -map.utm.zone 33 -map.utm.northhem True -outputs.ulx '499979.99999484676' -outputs.uly '200040.0000009411' -elev.dem '/home/luc/dev/S1tiling/tests/20200306-NR/tmp/tmpuv76wv2i' -elev.geoid '/home/luc/dev/S1tiling/s1tiling/s1tiling/resources/Geoid/egm96.grd'
        application_mocker.set_expectations('OrthoRectification', {
            'io.in': file_db.ortho_ready(i),
            'opt.ram': 2048,
            'interpolator': 'nn',
            'outputs.spacingx': 10.0,
            'outputs.spacingy': -10.0,
            'outputs.sizex': 10980,
            'outputs.sizey': 10980,
            'opt.gridspacing': 40.0,
            'map': 'utm',
            'map.utm.zone': 33,
            'map.utm.northhem': True,
            'outputs.ulx': 499979.99999484676,
            'outputs.uly': 200040.0000009411,
            'elev.dem': file_db.dem_file(),
            'elev.geoid': configuration.GeoidFile,
            'io.out': file_db.orthofile(i)+'?&writegeom=false&gdal:co:COMPRESS=DEFLATE',
            }, None)
#

    for i in range(1):
        # Synthetize -ram '4096' -il '/home/luc/dev/S1tiling/tests/20200306-NR/tmp/S2/33NWB/s1a_33NWB_vh_DES_007_20200108t044150.tif' '/home/luc/dev/S1tiling/tests/20200306-NR/tmp/S2/33NWB/s1a_33NWB_vh_DES_007_20200108t044215.tif'
        application_mocker.set_expectations('Synthetize', {
            '-ram': 2048,
            '-il': [file_db.orthofile(2*i), file_db.orthofile(2*i+1)],
            '-out': file_db.concatfile(i),
            }, None)
        application_mocker.set_expectations('BandMath', {
            '-ram': 2048,
            '-il': file_db.concatfile(i),
            '-out': file_db.masktmp(i),
            '-exp': 'im1b1==0?0:1'
            }, None)
        application_mocker.set_expectations('BinaryMorphologicalOperation', {
            '-ram': 2048,
            '-in': file_db.masktmp(i),
            '-out': file_db.maskfile(i),
            '-structype': 'ball',
            '-xradius': 5,
            '-yradius': 5,
            '-filter': 'opening'
            }, None)

    s1tiling.S1Processor.s1_process(0, dryrun=False, debug_otb=True, watch_ram=False, debug_tasks=False,
            cache_before_ortho=False, config_opt=configuration)
    application_mocker.assert_all_have_been_executed()

