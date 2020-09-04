#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import pathlib
import os
import shutil
from helpers import otb_compare


def process(cwd):
    '''
    Executes the S1Processor
    '''
    crt_dir = pathlib.Path(__file__).parent.absolute()
    src_dir = crt_dir.parent.absolute()
    args = ['python3', src_dir / 's1tiling/S1Processor.py', 'test.cfg']
    # TODO: extract the result path from the config file
    result_path = cwd + 'data_out/'
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    print(args)
    return subprocess.call(args, cwd=cwd), result_path


def test_0200306_NR():
    images = [
            '33NWB/s1a_33NWB_vh_DES_007_20200108txxxxxx.tif',
            '33NWB/s1a_33NWB_vv_DES_007_20200108txxxxxx.tif']
    # cwd = '/work/scratch/hermittel/dev/S1Tiling/tests/20200306-NR/'
    cwd = '/home/luc/dev/S1tiling/tests/20200306-NR/'
    EX, result_path = process(cwd)
    assert EX == 0
    baseline_path = cwd + 'data_baseline_out/'
    for im in images:
        assert otb_compare(baseline_path + im, result_path + im) == 0
    # The following line permits to test otb_compare correctly detect differences when
    # called from pytest.
    # assert otb_compare(baseline_path+images[0], result_path+images[1]) == 0
