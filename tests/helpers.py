#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import subprocess
from osgeo import gdal


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

def comparable_metadata(image):
    """
    Return the metadata from the specified image minus
    - PROCESSED_DATETIME
    - TIFFTAG_DATETIME
    - version in TIFFTAG_SOFTWARE
    """
    ds = gdal.Open(str(image))
    md = ds.GetMetadata()
    del ds
    if 'PROCESSED_DATETIME' in md:
        del md['PROCESSED_DATETIME']
    del md['TIFFTAG_DATETIME']
    return md

def metadata_compare(baseline, result):
    """
    Compare the metadata of the images produced by the test
    """
    # This is really dirty and non portable... for now
    arg = 'bash -c "diff -I PROCESSED_DATETIME -I TIFFTAG_DATETIME -I "Files:" <(gdalinfo %s) <(gdalinfo %s)"' % (baseline, result)
    print(arg)
    return os.system(arg)

