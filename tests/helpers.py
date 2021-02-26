#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import subprocess
import logging
from osgeo import gdal

re_TIFFTAG_SOFTWARE = r'(.*) \bv\d+\.\d+.*'

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

    md.pop('TIFFTAG_DATETIME',  None)
    md.pop('PROCESSED_DATETIME', None)

    if 'TIFFTAG_SOFTWARE' in md:
        # logging.error('PERFECT')
        ts = re.sub(re_TIFFTAG_SOFTWARE, r'\1', md['TIFFTAG_SOFTWARE'])
        logging.info('TIFFTAG_SOFTWARE changed from "%s" to "%s" [in "%s"]' % (md['TIFFTAG_SOFTWARE'], ts, image))
        md['TIFFTAG_SOFTWARE'] = ts
    else:
        logging.error('WHY???')

    return md


def metadata_compare(baseline, result):
    """
    Compare the metadata of the images produced by the test
    """
    # This is really dirty and non portable... for now
    arg = 'bash -c "diff -I PROCESSED_DATETIME -I TIFFTAG_DATETIME -I "Files:" <(gdalinfo %s) <(gdalinfo %s)"' % (baseline, result)
    print(arg)
    return os.system(arg)

