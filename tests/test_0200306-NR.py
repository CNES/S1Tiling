#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import pathlib
import os
import sys
import shutil
import logging
from helpers import otb_compare

def remove_dirs(dir_list):
    for dir in dir_list:
        if os.path.exists(dir):
            logging.info("rm -r '%s'", dir)
            shutil.rmtree(dir)


def process(tmpdir, outputdir, baseline_reference_outputs, test_file):
    '''
    Executes the S1Processor
    '''
    crt_dir = pathlib.Path(__file__).parent.absolute()
    src_dir = crt_dir.parent.absolute()
    args = ['python3', src_dir / 's1tiling/S1Processor.py', test_file]
    remove_dirs([outputdir, tmpdir/'S1', tmpdir/'S2'])
    logging.info('Running: %s', args)
    return subprocess.call(args, cwd=crt_dir)


def test_33NWB_202001_NR(baselinedir, outputdir, tmpdir, srtmdir, download):
    logging.info("Baseline expected in '%s'", baselinedir)
    # In all cases, the baseline is required for the reference outputs
    # => We need it
    assert os.path.exists(baselinedir), \
        ("No baseline found in '%s', please run minio-client to fetch it with:\n"+\
        "?> mc cp --recursive minio-otb/s1-tiling/baseline '%s'") % (baselinedir, baselinedir.absolute(),)

    if download:
        os.environ['S1TILING_TEST_DATA_INPUT']     = str((tmpdir/'data_raw').absolute())
        os.environ['S1TILING_TEST_DOWNLOAD']       = 'True'
        os.environ['S1TILING_TEST_OVERRIDE_CUT_Y'] = 'None'
    else:
        os.environ['S1TILING_TEST_DATA_INPUT']     = str((baselinedir/'inputs').absolute())
        os.environ['S1TILING_TEST_DOWNLOAD']       = 'False'
        os.environ['S1TILING_TEST_OVERRIDE_CUT_Y'] = 'False' # keep everything

    os.environ['S1TILING_TEST_DATA_OUTPUT']        = str(outputdir.absolute())
    os.environ['S1TILING_TEST_SRTM']               = str(srtmdir.absolute())
    os.environ['S1TILING_TEST_TMPDIR']             = str(tmpdir.absolute())

    logging.info('$S1TILING_TEST_DATA_INPUT  -> %s', os.environ['S1TILING_TEST_DATA_INPUT'])
    logging.info('$S1TILING_TEST_DATA_OUTPUT -> %s', os.environ['S1TILING_TEST_DATA_OUTPUT'])
    logging.info('$S1TILING_TEST_SRTM        -> %s', os.environ['S1TILING_TEST_SRTM'])
    logging.info('$S1TILING_TEST_TMPDIR      -> %s', os.environ['S1TILING_TEST_TMPDIR'])
    logging.info('$S1TILING_TEST_DOWNLOAD    -> %s', os.environ['S1TILING_TEST_DOWNLOAD'])

    images = [
            '33NWB/s1a_33NWB_vh_DES_007_20200108txxxxxx.tif',
            '33NWB/s1a_33NWB_vv_DES_007_20200108txxxxxx.tif']
    baseline_path =  baselinedir/'expected'
    test_file = baselinedir / 'test_33NWB_202001.cfg'
    EX = process(tmpdir, outputdir, baseline_path, test_file)
    assert EX == 0
    for im in images:
        assert otb_compare(baseline_path / im, outputdir / im) == 0
    # The following line permits to test otb_compare correctly detect differences when
    # called from pytest.
    # assert otb_compare(baseline_path+images[0], result_path+images[1]) == 0
