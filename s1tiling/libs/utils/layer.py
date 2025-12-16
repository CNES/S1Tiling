#!/usr/bin/env python3
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
# Authors:
# - Thierry KOLECK (CNES)
# - Luc HERMITTE (CSGROUP)
#
# =========================================================================

"""Layer and OGR related toolbox"""

import logging
from typing import Dict, List

from osgeo.ogr import Geometry

from .path import AnyPath
from ..Utils import Layer, Polygon

logger = logging.getLogger('s1tiling.utils.layer')


def tile_exists(mgrs_grid_name: AnyPath, tile_name_field: str) -> bool:
    """
    This function checks if a given MGRS tile exists in the database

    Args:
      mgrs_grid_name:  MGRS grid database
      tile_name_field: MGRS tile identifier

    Returns:
      True if the tile exists, False otherwise
    """
    layer = Layer(mgrs_grid_name)

    for current_tile in layer:
        # logger.debug("%s", current_tile.GetField('NAME'))
        if current_tile.GetField('NAME') == tile_name_field:
            return True
    return False


def filter_existing_tiles(mgrs_grid_name: AnyPath, tile_names: List[str]) -> List[str]:
    """
    Sanitize tile name list.

    :param mgrs_grid_name: MGRS grid database
    :param tile_names:     List of tile names to sanitize

    :return: a sorted list of all tile names that exist in MGRS grid database.
    """
    valid_tiles = set()

    layer = Layer(mgrs_grid_name)

    for current_tile in layer:
        # logger.debug("%s", current_tile.GetField('NAME'))
        if (tile_name := current_tile.GetField('NAME')) in tile_names:
            valid_tiles.add(tile_name)

    unknown_tiles = sorted(set(tile_names) - valid_tiles)
    for tile_name in unknown_tiles:
        logger.warning("Tile '%s' does not exist, skipping ...", tile_name)

    return sorted(valid_tiles)


def polygon2extent(polygon: Polygon) -> Dict[str, float]:
    """
    Transforms an OGR polygon into an extent dictionary.

    :return: dictionary made of the keys: "lonmin", "lonmax", "latmin", "latmax"
    """
    extent = {
        'lonmin': min(a[0] for a in polygon),
        'lonmax': max(a[0] for a in polygon),
        'latmin': min(a[1] for a in polygon),
        'latmax': max(a[1] for a in polygon),
    }
    return extent


def footprint2extent(footprint: Geometry) -> Dict[str, float]:
    """
    Transforms an OGR :class:`osgeo.ogr.Geometry` into an extent dictionary.

    :return: dictionary made of the keys: "lonmin", "lonmax", "latmin", "latmax"
    """
    return polygon2extent(footprint.GetPoints())
