#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2024 (c) CNES.
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

"""Layer related toolbox"""

import logging
from typing import Dict, List

from ..Utils import Layer, find_dem_intersecting_poly, get_mgrs_tile_geometry_by_name

logger = logging.getLogger('s1tiling.utils.layer')


def tile_exists(mgrs_grid_name: str, tile_name_field: str) -> bool:
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


def filter_existing_tiles(mgrs_grid_name: str, tile_names: List[str]) -> List[str]:
    """
    Sanitize tile name list.

    :param mgrs_grid_name: MGRS grid database
    :param tile_names:     List of tile names to sanitize

    :return: list of all tile names that exist in MGRS grid database.
    """
    valid_tiles = set()

    layer = Layer(mgrs_grid_name)

    for current_tile in layer:
        # logger.debug("%s", current_tile.GetField('NAME'))
        if (tile_name := current_tile.GetField('NAME')) in tile_names:
            valid_tiles.add(tile_name)

    unknown_tiles = set(tile_names) - valid_tiles
    for tile_name in unknown_tiles:
        logger.warning("Tile '%s' does not exist, skipping ...", tile_name)

    return list(valid_tiles)


def check_dem_coverage(
        mgrs_grid_name   : str,
        dem_db_filepath  : str,
        tiles_to_process : List[str],
        dem_field_ids    : List[str],
        dem_main_field_id: str,
) -> Dict[str, Dict]:
    """
    Given a set of MGRS tiles to process, this method
    returns the needed DEM tiles and the corresponding coverage.

    Args:
      tile_to_process: The list of MGRS tiles identifiers to process

    Return:
      A list of tuples (DEM tile id, coverage of MGRS tiles).
      Coverage range is [0,1]
    """
    dem_layer  = Layer(dem_db_filepath)
    mgrs_layer = Layer(mgrs_grid_name)

    needed_dem_tiles = {}

    for tile in tiles_to_process:
        logger.debug("Check DEM tiles for %s", tile)
        mgrs_footprint = get_mgrs_tile_geometry_by_name(tile, mgrs_layer)
        logger.debug("%s original %s footprint is %s", tile, mgrs_footprint.GetSpatialReference().GetName(), mgrs_footprint)
        dem_tiles = find_dem_intersecting_poly(
                mgrs_footprint, dem_layer, dem_field_ids, dem_main_field_id)
        needed_dem_tiles[tile] = dem_tiles
        logger.info("S2 tile %s is covered by %s DEM tiles", tile, len(dem_tiles))
    logger.info("DEM ok")
    return needed_dem_tiles
