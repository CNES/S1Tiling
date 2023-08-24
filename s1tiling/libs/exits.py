#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2023 (c) CNES. All rights reserved.
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
MISSING_SRTM        = 75
MISSING_GEOID       = 76
MISSING_APP         = 77

logger = logging.getLogger('s1tiling.exists')

class Situation:
    """
    Class to help determine the exit value from processing function
    """
    def __init__(self, nb_computation_errors, nb_download_failures, nb_download_timeouts):
        """
        constructor
        """
        logger.debug('Situation: %s computations errors. %s download failures. %s download timeouts',
                nb_computation_errors, nb_download_failures, nb_download_timeouts)
        if nb_computation_errors > 0:
            self.code = TASK_FAILED
        elif nb_download_failures > nb_download_timeouts:
            # So far, timeouts are counted as failures as well
            self.code = DOWNLOAD_ERROR
        elif nb_download_timeouts > 0:
            self.code = OFFLINE_DATA
        else:
            self.code = OK
