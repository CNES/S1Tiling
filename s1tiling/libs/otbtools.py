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
# =========================================================================

"""
Submodule defining OTB related helper functions and classes
"""

import logging
import subprocess
import re

logger = logging.getLogger('s1tiling.otb.tools')


def otb_version() -> str:
    """
    Returns the current version on OTB (through a call to ResetMargin -version)
    The result is cached
    """
    if not hasattr(otb_version, "_version"):
        try:
            r = subprocess.run(['otbcli_ResetMargin', '-version'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            version = r.stdout.decode('utf-8').strip('\n')
            match_v = re.search(r'\d+(\.\d+)+$', version)
            if not match_v:
                raise RuntimeError(f"Cannot extract OTB version from {version}")
            version = match_v[0]
            logger.info("OTB version detected on the system is %s", version)
            setattr(otb_version, "_version", version)
        except Exception as ex:
            logger.exception(ex)
            raise RuntimeError("Cannot determine current OTB version") from ex
    return getattr(otb_version, "_version")
