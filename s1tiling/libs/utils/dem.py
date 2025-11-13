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

"""Submodule dedicated to DEM files"""


import logging
from typing import Dict, List, Optional, Tuple, Union

from osgeo import ogr
from rtree import index

from .layer import Layer
from .path  import AnyPath
from .timer import timethis


logger = logging.getLogger('s1tiling.utils.dem')


def ogr_bbox_to_rtree_coordinates(
    minX: float, maxX: float,
    minY: float, maxY: float,
) -> Tuple[float, float, float, float]:
    """
    Converts a GDAL bounding box in the format expected by :class:`rtree.Index` API.
    """
    return minX, minY, maxX, maxY


class DEMInformation:
    """
    Aggregate information associated to a DEM tile:

    - name (/id)
    - footprint
    - information used for generating actual filenames
    """
    def __init__(self, dem_tile_name: str, dem_footprint: ogr.Geometry, dem_tile_info: Dict[str, str]):
        """ constructor """
        self.__tile_name = dem_tile_name
        self.__footprint = dem_footprint
        self.__tile_info = dem_tile_info

    @property
    def tile_name(self) -> str:
        """ returns DEM tile name """
        return self.__tile_name

    @property
    def footprint(self) -> ogr.Geometry:
        """ returns DEM tile footprint """
        return self.__footprint

    @property
    def tile_info(self) -> Dict[str, str]:
        """
        returns DEM tile information -- the while dictionary so it can be used for Formatting
        strings.
        """
        return self.__tile_info


@timethis("Loading DEM tile footprints")
def load_dem_tiles_information(
    dem_layer:     Layer,
    dem_field_ids: List[str],
    main_id:       str
) -> Dict[str, DEMInformation]:
    """
    Extracts dem information and footprint for each DEM tile.
    """
    dem_information = {}

    dem_layer.reset_reading()
    for dem_tile in dem_layer:
        dem_footprint = dem_tile.GetGeometryRef().Clone()

        tile_info = {}
        for field_id in dem_field_ids:
            tile_info[field_id] = dem_tile.GetField(field_id)
        dem_name = tile_info[main_id]
        tile_info['footprint'] = dem_footprint

        dem_information[dem_name] = DEMInformation(dem_name, dem_footprint, tile_info)

    return dem_information


class Index:
    """
    Tree indexed DEM database.
    """

    def __init__(self, dem_db: Union[AnyPath, Layer], *, dem_field_ids: List[str], main_id: str):
        """
        constructor
        """
        layer           = dem_db if isinstance(dem_db, Layer) else Layer(dem_db)
        dem_information = load_dem_tiles_information(layer, dem_field_ids, main_id)

        self.__indexed_dem_information : Dict[int, DEMInformation] = {}
        self.__index  = index.Index()

        for i, dem_tile_name in enumerate(dem_information):
            info      = dem_information[dem_tile_name]
            footprint = info.footprint
            bbox      = footprint.GetEnvelope()  # (minX, maxX, minY, maxY)
            self.__index.insert(i, ogr_bbox_to_rtree_coordinates(*bbox))  # (minX, minY, maxX, maxY)

            self.__indexed_dem_information[i] = info

    def intersection(
        self, tgt_footprint: ogr.Geometry, tgt_tilename: Optional[str] = None
    ) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Return a dictionnary of all the DEM tiles that intersect the requested target footprint.

        Returned information contains for each DEM tile:

        - its relative coverage of the target footprint,
        - the DEM information that can be used to generate filenames through formatting functions.
        """
        tgt_area = tgt_footprint.GetArea()
        bbox     = tgt_footprint.GetEnvelope()
        candidates = self.__index.intersection(ogr_bbox_to_rtree_coordinates(*bbox))

        dem_tiles : Dict[str, Dict[str, Union[str, float]]] = {}
        for i in candidates:  # DEM candidates
            intersection = self.__indexed_dem_information[i].footprint.Intersection(tgt_footprint)
            intersection_area = intersection.GetArea()

            if intersection_area > 0.0:
                dem_tile_name = self.__indexed_dem_information[i].tile_name
                dem_tiles[dem_tile_name] = {
                    '_coverage': intersection_area / tgt_area,
                    **self.__indexed_dem_information[i].tile_info
                }
                logger.debug("%s ∩ %s => %6.02f%%", tgt_tilename or " - ", dem_tile_name, 100 * dem_tiles[dem_tile_name]['_coverage'])
        return dem_tiles
