#!/usr/bin/env python
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
# Authors:
# - Thierry KOLECK (CNES)
# - Luc HERMITTE (CS Group)
#
# =========================================================================

""" This module contains various utility functions"""

import fnmatch
import logging
import os
from pathlib import Path
import re
import sys
from timeit import default_timer as timer
from typing import Any, Callable, Dict, Generator, Iterator, List, Literal, KeysView, Optional, Set, Tuple, Union
import xml.etree.ElementTree as ET
from osgeo import ogr
import osgeo  # To test __version__
from osgeo import osr

from .S1DateAcquisition import S1DateAcquisition


Polygon = Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float], Tuple[float,float]]


EXTENSION_TO_DRIVER_MAP = {
    '.gpkg': 'GPKG',
    '.shp': 'ESRI Shapefile',
}


logger = logging.getLogger("s1tiling.utils")


class Layer:
    """
    Thin wrapper that requests GDL Layers and keep a living reference to intermediary objects.
    """
    def __init__(self, grid, driver_name: Optional[str]=None) -> None:
        if not driver_name:
            _, ext = os.path.splitext(grid)
            driver_name = EXTENSION_TO_DRIVER_MAP.get(ext, "ESRI Shapefile")
            logging.debug("'%s' database extension: '%s'; using '%s' driver", grid, ext, driver_name)

        self.__grid        = grid
        self.__driver      = ogr.GetDriverByName(driver_name)
        self.__data_source = self.__driver.Open(self.__grid, 0)
        if not self.__data_source:
            raise RuntimeError(f"Cannot open {grid} with {driver_name} driver")
        self.__layer       = self.__data_source.GetLayer()

    def __iter__(self) -> Iterator[ogr.Feature]:
        return self.__layer.__iter__()

    def reset_reading(self) -> None:
        """
        Reset feature reading to start on the first feature.

        This affects iteration.
        """
        self.__layer.ResetReading()

    def find_tile_named(self, tile_name_field: str) -> Optional[ogr.Feature]:
        """
        Search for a tile that maches the name.
        """
        for tile in self.__layer:  # tile is a Feature
            if tile.GetField('NAME') in tile_name_field:
                return tile
        return None

    def get_spatial_reference(self) -> osr.SpatialReference:
        """Returns the SpatialReference the Layer is in"""
        return self.__layer.GetSpatialRef()


# ======================================================================
## Technical helpers

def get_relative_orbit(manifest: Union[str, Path]) -> int:
    """
    Returns the relative orbit number of the product.
    """
    root = ET.parse(manifest)
    url = "{http://www.esa.int/safe/sentinel-1.0}"
    key = f"metadataSection/metadataObject/metadataWrap/xmlData/{url}orbitReference/{url}relativeOrbitNumber"
    ron = root.find(key)
    if ron is None or ron.text is None:
        raise RuntimeError(f"No relativeOrbitNumber key found in {manifest}")
    return int(ron.text)


def get_origin(
        manifest: Union[str, Path]
) -> Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float], Tuple[float,float], str]:
    """Parse the coordinate of the origin in the manifest file to return its footprint.

    Args:
      manifest: The manifest from which to parse the coordinates of the origin

    Returns:
      the parsed coordinates (or throw an exception if they could not be parsed)
    """
    prefix_map = {"safe": "http://www.esa.int/safe/sentinel-1.0"}
    root = ET.parse(manifest)
    node_footprint = root.find("metadataSection/metadataObject/metadataWrap/xmlData/safe:frameSet/safe:frame/safe:footPrint",
              prefix_map)
    if node_footprint is None:
        raise RuntimeError(f"Cannot find coordinates in manifest {manifest!r}")
    srsName = node_footprint.attrib['srsName']
    srsName = re.sub(r"http://www.opengis.net/gml/srs/(epsg).xml#(\d+)", r"\1:\2", srsName)

    coord_text = node_footprint.findtext('{http://www.opengis.net/gml}coordinates')
    if coord_text is None:
        raise RuntimeError(f"Cannot find coordinates in manifest {manifest!r}")

    assert coord_text
    coord = [(float(val.replace("\n", "").split(",")[0]),
              float(val.replace("\n", "").split(",")[1]))
              for val in coord_text.split(" ")]
    return coord[0], coord[1], coord[2], coord[3], srsName

    # with open(manifest, "r", encoding="utf-8") as save_file:
    #     for line in save_file:
    #         if "<gml:coordinates>" in line:
    #             coor = line.replace("                <gml:coordinates>", "")\
    #                        .replace("</gml:coordinates>", "").split(" ")
    #             coord = [(float(val.replace("\n", "").split(",")[0]),
    #                       float(val.replace("\n", "").split(",")[1]))
    #                       for val in coor]
    #             return coord[0], coord[1], coord[2], coord[3]
    #     raise RuntimeError(f"Coordinates not found in {manifest!r}")


def get_shape_from_polygon(
    polygon: Union[Polygon, List[Tuple[float,float]]]
) -> ogr.Geometry:
    """
    Returns the shape of the footprint of the S1 product.

    .. note:
        For the moment, return the Geometry in OAMS_TRADITIONAL_GIS_ORDER (lat, lon; as in GDAL 2.x for WGS84).
        This also means we expect the polygon returned in WGS84
    """
    nw_coord, ne_coord, se_coord, sw_coord = polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(nw_coord[1], nw_coord[0], 0)
    ring.AddPoint(ne_coord[1], ne_coord[0], 0)
    ring.AddPoint(se_coord[1], se_coord[0], 0)
    ring.AddPoint(sw_coord[1], sw_coord[0], 0)
    ring.AddPoint(nw_coord[1], nw_coord[0], 0)
    poly.AddGeometry(ring)
    return poly


def get_shape(manifest: Union[str, Path]) -> ogr.Geometry:
    """
    Returns the shape of the footprint of the S1 product.
    """
    nw_coord, ne_coord, se_coord, sw_coord, srsName = get_origin(manifest)
    shape = get_shape_from_polygon((nw_coord, ne_coord, se_coord, sw_coord))
    sr = osr.SpatialReference()
    sr.SetFromUserInput(srsName)
    assert sr.GetName() == "WGS 84", f"Expected SpatialReference name to be 'WGS 84', found {sr.GetName()}"
    sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    shape.AssignSpatialReference(sr)
    return shape

def get_s1image_poly(s1image: Union[str, S1DateAcquisition]) -> ogr.Geometry:
    """
    Return shape of the ``s1image`` as a polygon
    """
    if isinstance(s1image, str):
        manifest = Path(s1image).parents[1] / 'manifest.safe'
    else:
        manifest = s1image.get_manifest()

    logger.debug("Manifest: %s", manifest)
    assert manifest.exists()
    poly = get_shape(manifest)
    return poly


def get_tile_origin_intersect_by_s1(grid_path: str, image: S1DateAcquisition) -> List:
    """
    Retrieve the list of MGRS tiles interesected by S1 product.

    Args:
      grid_path: Path to the shapefile containing the MGRS tiles
      image: S1 image as instance of S1DateAcquisition class

    Returns:
      a list of string of MGRS tiles names
    """
    manifest = image.get_manifest()
    poly = get_shape(manifest)

    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.Open(grid_path, 0)
    layer = data_source.GetLayer()

    intersect_tile = []

    for current_tile in layer:
        tile_footprint = current_tile.GetGeometryRef()
        intersection = poly.Intersection(tile_footprint)
        if intersection.GetArea() != 0:
            intersect_tile.append(current_tile.GetField('NAME'))
    return intersect_tile


def find_dem_intersecting_poly(
    poly:          ogr.Geometry,
    dem_layer:     Layer,
    dem_field_ids: List[str],
    main_id:       str
) -> Dict[str,Any]:
    """
    Searches the DEM tiles that intersect the specifid polygon

    precondition: Expect poly.GetSpatialReference() and dem_layer.get_spatial_reference() to be identical!
    """
    # main_ids = list(filter(lambda f: 'id' in f or 'ID' in f, dem_field_ids))
    # main_id = (main_ids or dem_field_ids)[0]
    # logger.debug('Using %s as DEM tile main id for name', main_id)

    dem_tiles = {}

    # Makes sure poly is expressed in the Layer SpatialReference
    orig_spatial_reference = poly.GetSpatialReference()
    out_sr = dem_layer.get_spatial_reference()
    # out_sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    if orig_spatial_reference.GetName() != out_sr.GetName():
        poly = poly.Clone()
        res = poly.TransformTo(out_sr)
        if res != 0:
            raise RuntimeError(
                    f"Cannot convert footprint from {orig_spatial_reference.GetName()!r} to {out_sr.GetName()!r}")
    area = poly.GetArea()
    logger.debug("Searching for DEM intersecting %s/%s", poly, poly.GetSpatialReference().GetName())

    dem_layer.reset_reading()
    tested = 0
    found = 0
    for dem_tile in dem_layer:
        tested += 1
        dem_footprint = dem_tile.GetGeometryRef()
        intersection = poly.Intersection(dem_footprint)
        if intersection.GetArea() > 0.0:
            found += 1
            # logger.debug("Tile: %s", dem_tile)
            coverage = intersection.GetArea() / area
            dem_info = {}
            for field_id in dem_field_ids:
                dem_info[field_id] = dem_tile.GetField(field_id)
            dem_info['_coverage'] = coverage
            dem_tiles[dem_info[main_id]] = dem_info
    logger.debug("Found %s DEM tiles among %s", found, tested)
    return dem_tiles


def find_dem_intersecting_raster(
    s1image:         str,
    dem_db_filepath: str,
    dem_field_ids:   List[str],
    main_id:         str
) -> Dict[str, Any]:
    """
    Searches the DEM tiles that intersect the S1 Image.
    """
    poly = get_s1image_poly(s1image)
    assert dem_db_filepath
    assert os.path.isfile(dem_db_filepath)
    dem_layer = Layer(dem_db_filepath)

    logger.info("Shape of %s: %s/%s", os.path.basename(s1image), poly, poly.GetSpatialReference().GetName())
    return find_dem_intersecting_poly(poly, dem_layer, dem_field_ids, main_id)


def get_orbit_direction(manifest: Union[str, Path]) -> Literal['DES', 'ASC']:
    """This function returns the orbit direction from a S1 manifest file.

    Args:
      manifest: path to the manifest file

    Returns:
      "ASC" for ascending orbits, "DES" for descending
      orbits. Throws an exception if manifest can not be parsed.

    """
    with open(manifest, "r", encoding="utf-8") as save_file:
        for line in save_file:
            if "<s1:pass>" in line:
                if "DESCENDING" in line:
                    return "DES"
                if "ASCENDING" in line:
                    return "ASC"
        raise RuntimeError(f"Orbit Directiction not found in {manifest!r}")


def convert_coord(
    tuple_list: List[Tuple[float, float]],
    in_epsg:    int,
    out_epsg:   int,
) -> List[Tuple[float, ...]]:
    """
    Convert a list of coordinates from one epsg code to another

    Args:
      tuple_list: a list of tuples representing the coordinates
      in_epsg: the input epsg code
      out_epsg: the output epsg code

    Returns:
      a list of tuples representing the converted coordinates
    """
    if not tuple_list:
        return []

    tuple_out = []

    in_spatial_ref = osr.SpatialReference()
    in_spatial_ref.ImportFromEPSG(in_epsg)
    out_spatial_ref = osr.SpatialReference()
    out_spatial_ref.ImportFromEPSG(out_epsg)
    if int(osgeo.__version__[0]) >= 3:
        # GDAL 2.0 and GDAL 3.0 don't take the CoordinateTransformation() parameters
        # in the same order: https://github.com/OSGeo/gdal/issues/1546
        #
        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        in_spatial_ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        # out_spatial_ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    for in_coord in tuple_list:
        lon = in_coord[0]
        lat = in_coord[1]

        coord_trans = osr.CoordinateTransformation(in_spatial_ref, out_spatial_ref)
        coord = coord_trans.TransformPoint(lon, lat)
        # logger.debug("convert_coord(lon=%s, lat=%s): %s, %s ==> %s", in_epsg, out_epsg, lon, lat, coord)
        tuple_out.append(coord)
    return tuple_out


_k_prod_re = re.compile(r'S1._IW_...._...._(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2}).*')

def extract_product_start_time(product_name : str) -> Optional[Dict[str, str]]:
    """
    Extracts product start time from its name.

    Returns: dictionary of keys {'YYYY', 'MM', 'DD', 'hh', 'mm', 'ss'}
             or None if the product name cannot be decoded.
    """
    assert '/' not in product_name, f"Expecting a basename for {product_name}"
    match = _k_prod_re.match(product_name)
    if not match:
        return None
    YYYY, MM, DD, hh, mm, ss = match.groups()
    return {'YYYY': YYYY, 'MM': MM, 'DD': DD, 'hh': hh, 'mm': mm, 'ss': ss}


def get_date_from_s1_raster(path_to_raster: str) -> str:
    """
    Small utilty function that parses a s1 raster file name to extract date.

    Args:
      path_to_raster: path to the s1 raster file

    Returns:
      a string representing the date
    """
    return path_to_raster.split("/")[-1].split("-")[4]


def get_polar_from_s1_raster(path_to_raster: str) -> str:
    """
    Small utilty function that parses a s1 raster file name to
    extract polarization.

    Args:
      path_to_raster: path to the s1 raster file

    Returns:
      a string representing the polarization
    """
    return path_to_raster.split("/")[-1].split("-")[3]


def get_platform_from_s1_raster(path_to_raster: str) -> str:
    """
    Small utilty function that parses a s1 raster file name to extract platform

    Args:
      path_to_raster: path to the s1 raster file

    Returns:
      a string representing the platform
    """
    return path_to_raster.split("/")[-1].split("-")[0]


# ======================================================================
## Technical helpers

class _PartialFormatHelper(dict):
    """
    Helper class that return missing ``{key}`` as themselves
    """
    def __missing__(self, key:str) ->str:
        return "{" + key + "}"


def partial_format(format_str: str, **kwargs) -> str:
    """
    Permits to apply partial formatting to format string.

    Example:
    --------
    >>> s = "{ab}_bla_{cd}"
    >>> partial_format(s, ab="TOTO")
    'tot_bla_{cd}'
    """
    return format_str.format_map(_PartialFormatHelper(**kwargs))


def flatten_stringlist(itr) -> Generator[str, None, None]:
    """
    Flatten a list of lists.
    But don't decompose string.
    """
    if type(itr) in (str,bytes):
        yield itr
    else:
        for x in itr:
            try:
                yield from flatten_stringlist(x)
            except TypeError:
                yield x


class ExecutionTimer:
    """Context manager to help measure execution times

    Example:
    with ExecutionTimer("the code", True) as t:
        Code_to_measure()
    """
    def __init__(self, text, do_measure) -> None:
        self._text       = text
        self._do_measure = do_measure

    def __enter__(self) -> "ExecutionTimer":
        self._start = timer()  # pylint: disable=attribute-defined-outside-init
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> Literal[False]:
        if self._do_measure:
            end = timer()
            logger.info("%s took %ssec", self._text, end - self._start)
        return False


def list_files(directory: str, pattern=None) -> List[os.DirEntry]:
    """
    Efficient listing of files in requested directory.

    This version shall be faster than glob to isolate files only as it keeps in "memory"
    the kind of the entry without needing to stat() the entry again.

    Requires Python 3.5
    """
    if pattern:
        filt = lambda path: path.is_file() and fnmatch.fnmatch(path.name, pattern)
    else:
        filt = lambda path: path.is_file()

    with os.scandir(directory) as nodes:
        res = list(filter(filt, nodes))
        # res = [entry for entry in nodes if filt(entry)]
    return res


def list_dirs(directory: str, pattern=None) -> List[os.DirEntry]:
    """
    Efficient listing of sub-directories in requested directory.

    This version shall be faster than glob to isolate directories only as it keeps in
    "memory" the kind of the entry without needing to stat() the entry again.

    Requires Python 3.5
    """
    if pattern:
        filt = lambda path: path.is_dir() and fnmatch.fnmatch(path.name, pattern)
    else:
        filt = lambda path: path.is_dir()

    with os.scandir(directory) as nodes:
        res = list(filter(filt, nodes))
        # res = [entry for entry in nodes if filt(entry)]
    return res


class RedirectStdToLogger:
    """
    Yet another helper class to redirect messages sent to stdout and stderr to a proper
    logger.

    This is a very simplified version tuned to answer S1Tiling needs.
    It also acts as a context manager.
    """
    def __init__(self, logger_) -> None:
        self.__old_stdout = sys.stdout
        self.__old_stderr = sys.stderr
        self.__logger     = logger_
        sys.stdout = RedirectStdToLogger.__StdOutErrAdapter(self.__logger)
        sys.stderr = RedirectStdToLogger.__StdOutErrAdapter(self.__logger, logging.ERROR)

    def __enter__(self) -> 'RedirectStdToLogger':
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> Literal[False]:
        sys.stdout = self.__old_stdout
        sys.stderr = self.__old_stderr
        return False

    class __StdOutErrAdapter:
        """
        Internal adapter that redirects messages, initially sent to a file, to a logger.
        """
        def __init__(self, logger_, mode=None) -> None:
            self.__logger     = logger_
            self.__mode       = mode  # None => adapt DEBUG/INFO/ERROR/...
            self.__last_level = mode  # None => adapt DEBUG/INFO/ERROR/...
            self.__lvl_re     = re.compile(r'(\((DEBUG|INFO|WARNING|ERROR)\))')
            self.__lvl_map    = {
                    'DEBUG':   logging.DEBUG,
                    'INFO':    logging.INFO,
                    'WARNING': logging.WARNING,
                    'ERROR':   logging.ERROR}

        def write(self, message) -> None:
            """
            Overrides sys.stdout.write() method
            """
            messages = message.rstrip().splitlines()
            for msg in messages:
                if self.__mode:
                    lvl = self.__mode
                else:
                    match = self.__lvl_re.search(msg)
                    if match:
                        lvl = self.__lvl_map[match.group(2)]
                    else:
                        lvl = self.__last_level or logging.INFO
                # OTB may have multi line messages.
                # In that case, reset happens with a new message
                self.__last_level = lvl
                self.__logger.log(lvl, msg)

        def flush(self) -> None:
            """
            Overrides sys.stdout.flush() method
            """
            pass

        def isatty(self) -> bool:
            """
            Overrides sys.stdout.isatty() method.
            This is required by OTB Python bindings.
            """
            return False


def remove_files(files: list) -> None:
    """
    Removes the files from the disk
    """
    assert isinstance(files, list)
    logger.debug("Remove %s", files)
    for file_it in files:
        if os.path.exists(file_it):
            os.remove(file_it)


class TopologicalSorter:
    """
    Depth-first topological_sort implementation
    """
    def __init__(self, dag: Dict, fetch_successor_function: Optional[Callable]=None) -> None:
        """
        constructor
        """
        self.__table = dag
        if fetch_successor_function:
            self.__successor_fetcher = fetch_successor_function
            self.__successors        = self.__successors_lazy
        else:
            self.__successors        = self.__successors_direct

    def depth(self, start_nodes: Union[List, Set, KeysView]) -> List:
        """
        Depth-first topological sorting method
        """
        results       : List = []
        visited_nodes : Dict = {}
        self.__recursive_depth_first(start_nodes, results, visited_nodes)
        return reversed(results)

    def __successors_lazy(self, node: Any) -> List:
        node_info = self.__table.get(node, None)
        # logger.debug('node:%s ; infos=%s', node, node_info)
        return self.__successor_fetcher(node_info) if node_info else []

    def __successors_direct(self, node: Any) -> List:
        return self.__table.get(node, [])

    def __recursive_depth_first(self, start_nodes: Union[List, Set, KeysView], results: List, visited_nodes: Dict[Any, int]) -> None:
        # logger.debug('start_nodes: %s', start_nodes)
        for node in start_nodes:
            visited = visited_nodes.get(node, 0)
            if   visited == 1:
                continue # done
            elif visited == 2:
                raise ValueError(f"Tsort: cyclic graph detected {node}")
            visited_nodes[node] = 2 # visiting
            succs = self.__successors(node)
            try:
                self.__recursive_depth_first(succs, results, visited_nodes)
            except ValueError as e:
                # raise e.'>'.node
                raise e
            visited_nodes[node] = 1 # visited
            results.append(node)


def tsort(dag: Dict, start_nodes: Union[List, Set, KeysView], fetch_successor_function: Optional[Callable]=None):
    """
    Topological sorting function (depth-first)

    Parameters:
        :dag:                      Direct Acyclic Graph to sort topologically
        :start_nodes:              nodes from which the sorting star
        :fetch_successor_function: Used to override how node transition is done
    """
    ts = TopologicalSorter(dag, fetch_successor_function)
    return ts.depth(start_nodes)
