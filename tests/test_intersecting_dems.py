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

"""
Unit test that compares different implementations for searching the DEM tiles that intersect a set
of S2 MGRS tiles.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from osgeo import gdal, ogr
from osgeo.ogr import osr
import pytest

from rtree import index

from s1tiling.libs.Utils import get_tile_geometries
from s1tiling.libs.utils import dem
from s1tiling.libs.utils.dem import DEMInformation, check_dem_coverage, load_dem_tiles_information
from s1tiling.libs.utils.layer import Layer

# Disable the log warning about exception and GDAL.
gdal.UseExceptions()


SCRIPT_DIR         = Path(__file__).parent.absolute()

DATA_DIR           = SCRIPT_DIR / 'data'
S2_TO_DEM_MAP_FILE = DATA_DIR / 'S2-DEM-map-v120.json'

RESOURCE_DIR       = SCRIPT_DIR.parent / 's1tiling/resources'
DEM_FILE           = RESOURCE_DIR / 'shapefile' / 'srtm_tiles.gpkg'
MGRS_FILE          = RESOURCE_DIR / 'shapefile' / 'Features.shp'

DEM_FIELD_IDS      = ['id']
DEM_MAIN_ID        = 'id'

@pytest.fixture
def reference_s2_to_dem_map()-> Dict[str, Set[str]]:
    with open(S2_TO_DEM_MAP_FILE, "r") as file:
        data = json.load(file)
        # return {k:set(v) for k,v in list(data.items())[0:100]}
        return {k:set(v) for k,v in data.items()}


def _keep_ids(d: Dict[str, Any]) -> Set[str]:
    return set(d.keys())


# ======================================================================
def _load_mgrs_footprints(
    dem_layer: Layer,
    reference_s2_to_dem_map: Dict[str, List[str]]
) -> Dict[str, ogr.Geometry]:
    mgrs_layer = Layer(MGRS_FILE)
    out_sr = dem_layer.get_spatial_reference()

    tiles_to_process = reference_s2_to_dem_map.keys()
    mgrs_footprints = get_tile_geometries(tiles_to_process, mgrs_layer)

    for tile_name, poly in mgrs_footprints.items():
        orig_spatial_reference = poly.GetSpatialReference()
        # out_sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if orig_spatial_reference.GetName() != out_sr.GetName():
            poly = poly.Clone()
            res = poly.TransformTo(out_sr)  # inplace update
            if res != 0:
                raise RuntimeError(
                    f"Cannot convert footprint from {orig_spatial_reference.GetName()!r} to {out_sr.GetName()!r}")
            mgrs_footprints[tile_name] = poly

    return mgrs_footprints


def _load_footprints(
    reference_s2_to_dem_map: Dict[str, List[str]]
) -> Tuple[Dict[str, DEMInformation], Dict[str, ogr.Geometry]]:
    dem_layer = Layer(DEM_FILE)
    dem_information = load_dem_tiles_information(dem_layer, DEM_FIELD_IDS, DEM_MAIN_ID)
    return dem_information, _load_mgrs_footprints(dem_layer, reference_s2_to_dem_map)


# ======================================================================
def find_dem_intersecting_mulitiple_polygons_v2(
    footprints:    Dict[str, ogr.Geometry],
    dem_layer:     Layer,
    dem_field_ids: List[str],
    main_id:       str
) -> Dict[str, Any]:
    """
    Searches the DEM tiles that intersect the specifid polygon

    precondition: Expect poly.GetSpatialReference() and dem_layer.get_spatial_reference() to be identical!

    Second implementation.
    It takes 300s on my machine for 2550 S2 tiles.
    """
    # main_ids = list(filter(lambda f: 'id' in f or 'ID' in f, dem_field_ids))
    # main_id = (main_ids or dem_field_ids)[0]
    # logging.debug('Using %s as DEM tile main id for name', main_id)

    referenced_footprints = {}
    # Makes sure footprint polygons are expressed in the DEM Layer SpatialReference
    out_sr = dem_layer.get_spatial_reference()
    logging.debug("Searching for DEM tiles intersecting %s tiles:", len(footprints))
    for tile, poly in footprints.items():
        orig_spatial_reference = poly.GetSpatialReference()
        # out_sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if orig_spatial_reference.GetName() != out_sr.GetName():
            poly = poly.Clone()
            res = poly.TransformTo(out_sr)
            if res != 0:
                raise RuntimeError(
                    f"Cannot convert footprint from {orig_spatial_reference.GetName()!r} to {out_sr.GetName()!r}")
        area = poly.GetArea()
        referenced_footprints[tile] = {
            'poly'     : poly,
            'area'     : area,
            'dem_tiles': {},
        }
        logging.debug(" - %s: %s/%s", tile, poly, poly.GetSpatialReference().GetName())

    logging.debug("DEM tiles intersections:")
    dem_layer.reset_reading()
    nb_dems = 0
    for dem_tile in dem_layer:
        nb_dems += 1
        intersected_tiles = []
        dem_footprint = dem_tile.GetGeometryRef()

        dem_info = {}
        for field_id in dem_field_ids:
            dem_info[field_id] = dem_tile.GetField(field_id)
        dem_name = dem_info[main_id]

        for tile, tile_data in referenced_footprints.items():
            poly = tile_data['poly']
            intersection = poly.Intersection(dem_footprint)
            if intersection.GetArea() > 0.0:
                intersected_tiles.append(tile)
                intersection_info = dem_info.copy()
                intersection_info['_coverage'] = intersection.GetArea() / tile_data['area']
                tile_data['dem_tiles'][dem_name] = intersection_info
        if len(intersected_tiles) > 0:
            logging.debug(' - DEM tile %s covers %s S2 tiles: %s', dem_name, len(intersected_tiles), intersected_tiles)

    dem_tiles = {}
    for tile, tile_data in referenced_footprints.items():
        dem_tiles[tile] = tile_data['dem_tiles']

    # logging.debug("Found %s DEM tiles among %s", found, nb_dems)
    logging.debug('%s DEM footprints analysed', nb_dems)
    return dem_tiles


@pytest.mark.slow
@pytest.mark.bench_aternatives
def test_check_dem_coverage_v2(reference_s2_to_dem_map: Dict[str, List[str]]):
    s2_tiles = reference_s2_to_dem_map.keys()

    dem_layer  = Layer(DEM_FILE)
    mgrs_layer = Layer(MGRS_FILE)
    mgrs_footprints = get_tile_geometries(s2_tiles, mgrs_layer)

    coverage_map = find_dem_intersecting_mulitiple_polygons_v2(mgrs_footprints, dem_layer, ['id'], 'id')
    computed_s2_to_dem_map = {s2: _keep_ids(dems) for s2, dems in coverage_map.items()}
    assert reference_s2_to_dem_map.keys() == computed_s2_to_dem_map.keys()
    assert reference_s2_to_dem_map['10TDP'] == computed_s2_to_dem_map['10TDP']
    assert reference_s2_to_dem_map == computed_s2_to_dem_map


# ======================================================================
@pytest.mark.bench_aternatives
def test_search_dems_in_mgrs_quadtree_gdal(reference_s2_to_dem_map: Dict[str, List[str]]):
    mgrs_footprints : Dict[str, ogr.Geometry]
    dem_information, mgrs_footprints = _load_footprints(reference_s2_to_dem_map)

    driver = ogr.GetDriverByName('Memory')
    ds = driver.CreateDataSource('memData')
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer('index', srs, ogr.wkbPolygon)

    referenced_mgrs_footprints = {}
    for i, tile_name in enumerate(mgrs_footprints):
        feature = ogr.Feature(layer.GetLayerDefn())
        footprint = mgrs_footprints[tile_name]
        feature.SetGeometry(footprint)
        feature.SetFID(i)
        layer.CreateFeature(feature)

        referenced_mgrs_footprints[i] = {
            'tilename' : tile_name,
            'poly': footprint,
            'area': footprint.GetArea(),
            'dem_tiles': {},
        }

    nb_dems = 0
    for dem_tile_name, dem_info in dem_information.items():
        # intersected_tiles = []
        nb_dems += 1

        dem_footprint : ogr.Geometry = dem_info.footprint
        assert isinstance(dem_footprint, ogr.Geometry)
        layer.SetSpatialFilter(dem_footprint)
        for feature in layer:
            # Intersection potentielle (basée sur bbox)
            if feature.GetGeometryRef().Intersects(dem_footprint):
                intersection = feature.GetGeometryRef().Intersection(dem_footprint)
                intersection_area = intersection.GetArea()

                if intersection_area > 0.0:
                    mgrs_tile_fid = feature.GetFID()
                    mgrs_tile_data = referenced_mgrs_footprints[mgrs_tile_fid]

                    # intersected_tiles.append(mgrs_tile_data['tilename'])
                    intersection_info = dem_info.tile_info.copy()
                    intersection_info['_coverage'] = intersection_area / mgrs_tile_data['area']
                    mgrs_tile_data['dem_tiles'][dem_tile_name] = intersection_info
                    logging.debug("%s ∩ %s => %6.02f%%", dem_tile_name, mgrs_tile_data['tilename'], 100 * intersection_info['_coverage'])
        layer.SetSpatialFilter(None)

    dem_tiles = {}
    for tile_fid, tile_data in referenced_mgrs_footprints.items():
        tilename = tile_data['tilename']
        dem_tiles[tilename] = tile_data['dem_tiles']
        # logging.debug('dem_tiles[%s] = %s', tilename, tile_data)

    # logging.debug("Found %s DEM tiles among %s", found, nb_dems)
    logging.debug('%s DEM footprints analysed', nb_dems)
    coverage_map = dem_tiles

    computed_s2_to_dem_map = {s2: _keep_ids(dems) for s2, dems in coverage_map.items()}
    assert reference_s2_to_dem_map.keys() == computed_s2_to_dem_map.keys()
    assert reference_s2_to_dem_map['10TDP'] == computed_s2_to_dem_map['10TDP']
    assert reference_s2_to_dem_map == computed_s2_to_dem_map


# ======================================================================
@pytest.mark.bench_aternatives
def test_search_dems_in_mgrs_quadtree_rbtree(reference_s2_to_dem_map: Dict[str, List[str]]):
    mgrs_footprints : Dict[str, ogr.Geometry]
    dem_information, mgrs_footprints = _load_footprints(reference_s2_to_dem_map)

    referenced_mgrs_footprints = {}
    mgrs_index = index.Index()
    for i, tile_name in enumerate(mgrs_footprints):
        footprint = mgrs_footprints[tile_name]
        bbox = footprint.GetEnvelope()  # (minX, maxX, minY, maxY)
        mgrs_index.insert(i, (bbox[0], bbox[2], bbox[1], bbox[3]))  # (minX, minY, maxX, maxY)

        referenced_mgrs_footprints[i] = {
            'tilename' : tile_name,
            'poly': footprint,
            'area': footprint.GetArea(),
            'dem_tiles': {},
        }

    nb_dems = 0
    for dem_tile_name, dem_info in dem_information.items():
        # intersected_tiles = []
        nb_dems += 1

        dem_footprint : ogr.Geometry = dem_info.footprint
        assert isinstance(dem_footprint, ogr.Geometry)
        bbox = dem_footprint.GetEnvelope()
        candidates = list(mgrs_index.intersection((bbox[0], bbox[2], bbox[1], bbox[3])))
        for i in candidates:
            intersection = referenced_mgrs_footprints[i]['poly'].Intersection(dem_footprint)
            intersection_area = intersection.GetArea()

            if intersection_area > 0.0:
                mgrs_tile_data = referenced_mgrs_footprints[i]

                # intersected_tiles.append(mgrs_tile_data['tilename'])
                intersection_info = dem_info.tile_info.copy()
                intersection_info['_coverage'] = intersection_area / mgrs_tile_data['area']
                mgrs_tile_data['dem_tiles'][dem_tile_name] = intersection_info
                logging.debug("%s ∩ %s => %6.02f%%", dem_tile_name, mgrs_tile_data['tilename'], 100 * intersection_info['_coverage'])

    dem_tiles = {}
    for tile_fid, tile_data in referenced_mgrs_footprints.items():
        tilename = tile_data['tilename']
        dem_tiles[tilename] = tile_data['dem_tiles']
        # logging.debug('dem_tiles[%s] = %s', tilename, tile_data)

    # logging.debug("Found %s DEM tiles among %s", found, nb_dems)
    logging.debug('%s DEM footprints analysed', nb_dems)
    coverage_map = dem_tiles

    computed_s2_to_dem_map = {s2: _keep_ids(dems) for s2, dems in coverage_map.items()}
    assert reference_s2_to_dem_map.keys() == computed_s2_to_dem_map.keys()
    assert reference_s2_to_dem_map['10TDP'] == computed_s2_to_dem_map['10TDP']
    assert reference_s2_to_dem_map == computed_s2_to_dem_map


# ======================================================================
@pytest.mark.bench_aternatives
def test_search_mgrs_in_dems_quadtree_rbtree(reference_s2_to_dem_map: Dict[str, List[str]]):
    mgrs_footprints : Dict[str, ogr.Geometry]
    dem_information, mgrs_footprints = _load_footprints(reference_s2_to_dem_map)

    referenced_dem_footprints = {}
    dem_index = index.Index()
    for i, dem_tile_name in enumerate(dem_information):
        footprint = dem_information[dem_tile_name].footprint
        bbox = footprint.GetEnvelope()  # (minX, maxX, minY, maxY)
        dem_index.insert(i, (bbox[0], bbox[2], bbox[1], bbox[3]))  # (minX, minY, maxX, maxY)

        referenced_dem_footprints[i] = dem_information[dem_tile_name]

    dem_tiles : dict[str, dict[str, Any]] = {}
    for tile_name, mgrs_footprint in mgrs_footprints.items():
        dem_tiles[tile_name] = {}
        mgrs_area = mgrs_footprint.GetArea()

        bbox = mgrs_footprint.GetEnvelope()
        candidates = dem_index.intersection((bbox[0], bbox[2], bbox[1], bbox[3]))
        for i in candidates:  # DEM candidates
            intersection = referenced_dem_footprints[i].footprint.Intersection(mgrs_footprint)
            intersection_area = intersection.GetArea()

            if intersection_area > 0.0:
                dem_tile_name = referenced_dem_footprints[i].tile_name
                dem_tiles[tile_name][dem_tile_name] = {
                    '_coverage': intersection_area / mgrs_area,
                    **referenced_dem_footprints[i].tile_info
                }

                logging.debug("%s ∩ %s => %6.02f%%", dem_tile_name, tile_name, 100 * dem_tiles[tile_name][dem_tile_name]['_coverage'])

    # logging.debug("Found %s DEM tiles among %s", found, nb_dems)
    coverage_map = dem_tiles

    computed_s2_to_dem_map = {s2: _keep_ids(dems) for s2, dems in coverage_map.items()}
    assert reference_s2_to_dem_map.keys() == computed_s2_to_dem_map.keys()
    assert reference_s2_to_dem_map['10TDP'] == computed_s2_to_dem_map['10TDP']
    assert reference_s2_to_dem_map == computed_s2_to_dem_map


# ======================================================================
def test_search_mgrs_in_dems_implemented_v1(reference_s2_to_dem_map: Dict[str, List[str]]):
    dem_layer = Layer(DEM_FILE)
    mgrs_footprints = _load_mgrs_footprints(dem_layer, reference_s2_to_dem_map)

    dem_index = dem.Index(dem_layer, field_ids=DEM_FIELD_IDS, main_id=DEM_MAIN_ID)

    dem_tiles = {}
    for tile_name, mgrs_footprint in mgrs_footprints.items():
        dem_tiles[tile_name] = dem_index.intersection(mgrs_footprint, tile_name)


    # logging.debug("Found %s DEM tiles among %s", found, nb_dems)
    coverage_map = dem_tiles

    computed_s2_to_dem_map = {s2: _keep_ids(dems) for s2, dems in coverage_map.items()}
    assert reference_s2_to_dem_map.keys() == computed_s2_to_dem_map.keys()
    assert reference_s2_to_dem_map['10TDP'] == computed_s2_to_dem_map['10TDP']
    assert reference_s2_to_dem_map == computed_s2_to_dem_map


def test_search_mgrs_in_dems_implemented(reference_s2_to_dem_map: Dict[str, List[str]]):
    coverage_map = check_dem_coverage(
        MGRS_FILE, DEM_FILE,
        reference_s2_to_dem_map.keys(),
        DEM_FIELD_IDS, DEM_MAIN_ID,
    )

    computed_s2_to_dem_map = {s2: _keep_ids(dems) for s2, dems in coverage_map.items()}
    assert reference_s2_to_dem_map.keys() == computed_s2_to_dem_map.keys()
    assert reference_s2_to_dem_map['10TDP'] == computed_s2_to_dem_map['10TDP']
    assert reference_s2_to_dem_map == computed_s2_to_dem_map
