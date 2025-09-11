#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2025 (c) CNES.
#   Copyright 2022-2024 (c) CS GROUP France.
#
#   This file is part of S1Tiling project
#       https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
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
This module defines helper functions used by S1Tiling Step Factories.
"""

import logging
import re

from ..meta import Meta

logger = logging.getLogger('s1tiling.wrappers.lia')


def remove_polarization_marks(name: str) -> str:
    """
    Clean filename of any specific polarization mark like ``vv``, ``vh``, or
    the ending in ``-001`` and ``002``.
    """
    # (?=  marks a 0-length match to ignore the dot
    return re.sub(r'[hv][hv]-|[HV][HV]_|-00[12](?=\.)', '', name)


def depolarize_4_filename_pre_hook(meta: Meta) -> None:
    """
    Provide names clear from polar related information.
    """
    # Ignore polarization in filenames
    if 'polarless_basename' in meta:
        assert meta['polarless_basename'] == remove_polarization_marks(meta['basename'])
    else:
        meta['polarless_basename'] = remove_polarization_marks(meta['basename'])


def does_sin_lia_match_s2_tile_for_orbit(output_meta: Meta, input_meta: Meta) -> bool:
    """
    Tells whether a given ComputeGroundAndSatPositionsOnDEM input is compatible
    with the the current S2 tile.

    ``tile_name``, ``flying_unit_code`` and ``orbit`` have to be identical.
    """
    fields = ['flying_unit_code', 'tile_name', 'orbit']
    # logger.debug("checking %s among %s VS %s", fields, input_meta, output_meta)
    return all(str(input_meta[k]) == str(output_meta[k]) for k in fields)


def does_gamma_area_match_s2_tile_for_orbit(output_meta: Meta, input_meta: Meta) -> bool:
    """
    Tells whether a given ComputeGroundAndSatPositionsOnDEM input is compatible
    with the the current S2 tile.

    ``tile_name`` has to be identical.
    """
    fields = ['flying_unit_code', 'tile_name', 'orbit_direction', 'orbit']
    return all(input_meta[k] == output_meta[k] for k in fields)


def does_s2_data_match_s2_tile(output_meta: Meta, input_meta: Meta) -> bool:
    """
    Tells whether a given sin_LIA input is compatible with the the current S2 tile.

    ``tile_name`` has to be identical.
    """
    fields = ['tile_name']
    return all(input_meta[k] == output_meta[k] for k in fields)
