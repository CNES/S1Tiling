#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2024 (c) CNES.
#   Copyright 2022-2024 (c) CS GROUP France.
#
#   This file is part of S1Tiling project
#       https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
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
#
# =========================================================================

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

    md.pop('DataType', None)
    md.pop('FACILITY_IDENTIFIER', None)
    md.pop('METADATATYPE', None)
    md.pop('OTB_VERSION', None)
    md.pop('PROCESSED_DATETIME', None)
    md.pop('ProductType', None)
    md.pop('ProductionDate', None)
    md.pop('TIFFTAG_DATETIME',  None)
    md.pop('TileHintX', None)
    md.pop('TileHintY', None)

    if 'TIFFTAG_SOFTWARE' in md:
        # logging.error('PERFECT')
        ts = re.sub(re_TIFFTAG_SOFTWARE, r'\1', md['TIFFTAG_SOFTWARE'])
        logging.info('TIFFTAG_SOFTWARE changed from "%s" to "%s" [in "%s"]' % (md['TIFFTAG_SOFTWARE'], ts, image))
        md['TIFFTAG_SOFTWARE'] = ts
    else:
        logging.error('There is no TIFFTAG_SOFTWARE information in %s image metadata!', image)

    return md


def metadata_compare(baseline, result):
    """
    Compare the metadata of the images produced by the test
    """
    # This is really dirty and non portable... for now
    arg = 'bash -c "diff -I PROCESSED_DATETIME -I TIFFTAG_DATETIME -I "Files:" <(gdalinfo %s) <(gdalinfo %s)"' % (baseline, result)
    print(arg)
    return os.system(arg)
