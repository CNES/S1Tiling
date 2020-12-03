#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import subprocess


def otb_compare(baseline, result):
    """
    Compare the images produced by the test
    """
    args = ['otbTestDriver',
            '--compare-image', '1e-12', baseline, result,
            'Execute', 'echo', '"running OTB Compare"',
            '-testenv']
    print(args)
    return subprocess.call(args)

def metadata_compare(baseline, result):
    """
    Compare the metadata of the images produced by the test
    """
    # This is really dirty and non portable... for now
    arg = 'bash -c "diff -I PROCESSED_DATETIME -I TIFFTAG_DATETIME -I "Files:" <(gdalinfo %s) <(gdalinfo %s)"' % (baseline, result)
    print(arg)
    return os.system(arg)
