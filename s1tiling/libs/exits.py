#!/usr/bin/env python3
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

"""
This module lists EXIT codes
"""

import logging
from typing import Dict, Type
from s1tiling.libs import exceptions

OK                  = 0
TASK_FAILED         = 66
DOWNLOAD_ERROR      = 67
OFFLINE_DATA        = 68
OUTPUT_DISK_FULL    = 69
TMP_DISK_FULL       = 70
CORRUPTED_DATA_SAFE = 71
CONFIG_ERROR        = 72
NO_S2_TILE          = 73
NO_S1_IMAGE         = 74
MISSING_DEM         = 75
MISSING_GEOID       = 76
MISSING_APP         = 77
UNKNOWN_REASON      = 78

logger = logging.getLogger('s1tiling.exists')


k_exit_table : Dict[Type[BaseException], int] = {
        exceptions.ConfigurationError    : CONFIG_ERROR,
        exceptions.CorruptedDataSAFEError: CORRUPTED_DATA_SAFE,
        exceptions.DownloadEOFFileError  : DOWNLOAD_ERROR,
        exceptions.DownloadS1FileError   : DOWNLOAD_ERROR,
        exceptions.NoS2TileError         : NO_S2_TILE,
        exceptions.NoS1ImageError        : NO_S1_IMAGE,
        exceptions.MissingDEMError       : MISSING_DEM,
        exceptions.MissingGeoidError     : MISSING_GEOID,
        exceptions.InvalidOTBVersionError: CONFIG_ERROR,
        exceptions.MissingApplication    : MISSING_APP,
}


def translate_exception_into_exit_code(exception: BaseException) -> int:
    """
    This function re-couple S1Tiling internal exception into excepted exit code.
    """
    return k_exit_table.get(exception.__class__, UNKNOWN_REASON)


class Situation:
    """
    Class to help determine the exit value from processing function.

    The computed ``code`` to return will be:

    - ``exits.TASK_FAILED`` if computation errors have been observed;
    - ``exits.DOWNLOAD_ERROR`` if some input S1 products could not be downloaded;
    - ``exits.OFFLINE_DATA`` if some input S1 products could not be downloaded
      in time because they were off-line.
    - ``exits.OK`` if no issue has been observed
    """
    def __init__(
            self,
            nb_computation_errors: int,
            nb_search_failures   : int,
            nb_download_failures : int,
            nb_download_timeouts : int) -> None:
        """
        constructor
        """
        logger.info('Situation: %s computations errors. %s search failures. %s download failures. %s download timeouts',
                     nb_computation_errors, nb_search_failures, nb_download_failures, nb_download_timeouts)
        if nb_computation_errors > 0:
            self.code = TASK_FAILED
        elif (nb_download_failures > nb_download_timeouts) or (nb_search_failures > 0):
            # So far, timeouts are counted as failures as well
            self.code = DOWNLOAD_ERROR
        elif nb_download_timeouts > 0:
            self.code = OFFLINE_DATA
        else:
            self.code = OK
