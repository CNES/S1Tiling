#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import pathlib
from helpers import otb_compare

def process(cwd):
    '''
    Executes the S1Processor
    '''
    crt_dir = pathlib.Path(__file__).parent.absolute()
    src_dir = crt_dir.parent.absolute()
    args=[src_dir/'S1Processor.py', 'test.cfg']
    print(args)
    return subprocess.call(args, cwd=cwd)


def test_0200306_NR():
    images = ['33NWB/s1a_33NWB_vh_DES_007_20200108txxxxxx.tif', '33NWB/s1a_33NWB_vv_DES_007_20200108txxxxxx.tif']
    cwd = '/work/scratch/hermittel/dev/S1Tiling/tests/20200306-NR/'
    process(cwd)
    baseline_path = cwd+'data_baseline_out/'
    result_path   = cwd+'data_out2/'
    for im in images:
        assert otb_compare(baseline_path+im, result_path+im) == 0
    # The following line permits to test otb_compare correctly detect differences when called from pytest.
    # assert otb_compare(baseline_path+images[0], result_path+images[1]) == 0
