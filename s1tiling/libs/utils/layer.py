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

from .timer import timethis
from ..Utils import Layer, Polygon, find_dem_intersecting_mulitiple_polygons, find_dem_intersecting_poly, get_mgrs_tile_geometry_by_name, get_tile_geometries

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


@timethis("Extracting DEM coverage of requested tiles")
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

    mgrs_footprints = get_tile_geometries(tiles_to_process, mgrs_layer)

    logger.debug("Check DEM files for all requested tiles")
    needed_dem_tiles = find_dem_intersecting_mulitiple_polygons(
        mgrs_footprints, dem_layer, dem_field_ids, dem_main_field_id)

    logger.debug("Summary of S2 tiles intersection with DEM tiles")
    for tile in tiles_to_process:
        # mgrs_footprint = mgrs_footprints[tile]
        # logger.debug("%s original %s footprint is %s", tile, mgrs_footprint.GetSpatialReference().GetName(), mgrs_footprint)
        logger.debug(" - S2 tile %s is covered by %s DEM tiles", tile, len(needed_dem_tiles[tile]))
    logger.info("DEM ok")
    return needed_dem_tiles


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
