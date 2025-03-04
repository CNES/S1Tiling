#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2025 (c) CNES.
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
# Authors:
# - Thierry KOLECK (CNES)
# - Luc HERMITTE (CSGROUP)
#
# =========================================================================

"""Centralizes EODAG heper functions"""

import logging
import os
from typing import Optional, Protocol

from eodag.api.core import EODataAccessGateway


logger = logging.getLogger('s1tiling.utils.eodag')


class EODAGConfiguration(Protocol):
    """
    Specialized protocol for configuration information related to function:`EODataAccessGateway
    factory<create>`.

    Can be seen an a ISP compliant concept for Configuration object regarding eodag object
    construction.
    """
    download     : bool
    raw_directory: str
    eodag_config : str


def create(cfg: EODAGConfiguration) -> Optional[EODataAccessGateway]:
    """
    :class:`EODataAccessGateway` factory from S1Tiling configuration obejct.
    """
    if not cfg.download:
        return None

    logger.debug('Using %s EODAG configuration file', cfg.eodag_config or 'user default')
    dag = EODataAccessGateway(cfg.eodag_config)
    # TODO: update once eodag directly offers "DL directory setting" feature v1.7? +?
    dest_dir = os.path.abspath(cfg.raw_directory)
    logger.debug('Override EODAG output directory to %s', dest_dir)
    for provider in dag.providers_config.keys():
        if hasattr(dag.providers_config[provider], 'download'):
            dag.providers_config[provider].download.update({'output_dir': dest_dir})
            logger.debug(' - for %s', provider)
        else:
            logger.debug(' - NOT for %s', provider)
    return dag
