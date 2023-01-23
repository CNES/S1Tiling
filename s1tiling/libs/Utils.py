#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2022 (c) CNES. All rights reserved.
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
import xml.etree.ElementTree as ET
from osgeo import ogr
import osgeo  # To test __version__
from osgeo import osr


def flatten_stringlist(itr):
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


class Layer:
    """
    Thin wrapper that requests GDL Layers and keep a living reference to intermediary objects.
    """
    def __init__(self, grid, driver_name="ESRI Shapefile"):
        self.__grid        = grid
        self.__driver      = ogr.GetDriverByName(driver_name)
        self.__data_source = self.__driver.Open(self.__grid, 0)
        self.__layer       = self.__data_source.GetLayer()

    def __iter__(self):
        return self.__layer.__iter__()

    def reset_reading(self):
        """
        Reset feature reading to start on the first feature.

        This affects iteration.
        """
        return self.__layer.ResetReading()

    def find_tile_named(self, tile_name_field):
        """
        Search for a tile that maches the name.
        """
        for tile in self.__layer:
            if tile.GetField('NAME') in tile_name_field:
                return tile
        return None


def get_relative_orbit(manifest):
    """
    Returns the relative orbit number of the product.
    """
    root = ET.parse(manifest)
    return int(root.find("metadataSection/metadataObject/metadataWrap/xmlData/{http://www.esa.int/safe/sentinel-1.0}orbitReference/{http://www.esa.int/safe/sentinel-1.0}relativeOrbitNumber").text)


def get_origin(manifest):
    """Parse the coordinate of the origin in the manifest file to return its footprint.

    Args:
      manifest: The manifest from which to parse the coordinates of the origin

    Returns:
      the parsed coordinates (or throw an exception if they could not be parsed)
    """
    with open(manifest, "r") as save_file:
        for line in save_file:
            if "<gml:coordinates>" in line:
                coor = line.replace("                <gml:coordinates>", "")\
                           .replace("</gml:coordinates>", "").split(" ")
                coord = [(float(val.replace("\n", "").split(",")[0]),
                          float(val.replace("\n", "").split(",")[1]))
                          for val in coor]
                return coord[0], coord[1], coord[2], coord[3]
        raise Exception("Coordinates not found in " + str(manifest))


def get_shape_from_polygon(polygon):
    """
    Returns the shape of the footprint of the S1 product.
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


def get_shape(manifest):
    """
    Returns the shape of the footprint of the S1 product.
    """
    nw_coord, ne_coord, se_coord, sw_coord = get_origin(manifest)
    return get_shape_from_polygon((nw_coord, ne_coord, se_coord, sw_coord))

def get_s1image_poly(s1image):
    """
    Return shape of the ``s1image`` as a polygon
    """
    if isinstance(s1image, str):
        manifest = Path(s1image).parents[1] / 'manifest.safe'
    else:
        manifest = s1image.get_manifest()

    logging.debug("Manifest: %s", manifest)
    assert manifest.exists()
    poly = get_shape(manifest)
    return poly


def get_tile_origin_intersect_by_s1(grid_path, image):
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


def find_srtm_intersecting_raster(s1image, srtm_db_filepath):
    """
    Searches the SRTM tiles that intersect the S1 Image.
    """
    # srtm_layer = Layer(srtm_shapefile)
    srtm_tiles = {}
    poly = get_s1image_poly(s1image)
    logging.info("Shape of %s: %s", os.path.basename(s1image), poly)

    srtm_shapefile = srtm_db_filepath
    assert srtm_shapefile
    assert os.path.isfile(srtm_shapefile)
    srtm_layer = Layer(srtm_shapefile, driver_name='GPKG')
    for tile in srtm_layer:
        tile_footprint = tile.GetGeometryRef()
        intersection = poly.Intersection(tile_footprint)
        if intersection.GetArea() > 0.0:
            # logging.debug("Tile: %s", tile)
            file = tile.GetField('id')
            srtm_tiles[file] = 1
    return sorted(srtm_tiles.keys())


def get_orbit_direction(manifest):
    """This function returns the orbit direction from a S1 manifest file.

    Args:
      manifest: path to the manifest file

    Returns:
      "ASC" for ascending orbits, "DES" for descending
      orbits. Throws an exception if manifest can not be parsed.

    """
    with open(manifest, "r") as save_file:
        for line in save_file:
            if "<s1:pass>" in line:
                if "DESCENDING" in line:
                    return "DES"
                if "ASCENDING" in line:
                    return "ASC"
        raise Exception("Orbit Directiction not found in " + str(manifest))


def convert_coord(tuple_list, in_epsg, out_epsg):
    """
    Convert a list of coordinates from one epsg code to another

    Args:
      tuple_list: a list of tuples representing the coordinates
      in_epsg: the input epsg code
      out_epsg: the output epsg code

    Returns:
      a list of tuples representing the converted coordinates
    """
    tuple_out = []

    if tuple_list:
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
        # logging.debug("convert_coord(lon=%s, lat=%s): %s, %s ==> %s", in_epsg, out_epsg, lon, lat, coord)
        tuple_out.append(coord)
    return tuple_out


_k_prod_re = re.compile(r'S1._IW_...._...._(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2}).*')

def extract_product_start_time(product_name : str):
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


def get_date_from_s1_raster(path_to_raster):
    """
    Small utilty function that parses a s1 raster file name to extract date.

    Args:
      path_to_raster: path to the s1 raster file

    Returns:
      a string representing the date
    """
    return path_to_raster.split("/")[-1].split("-")[4]


def get_polar_from_s1_raster(path_to_raster):
    """
    Small utilty function that parses a s1 raster file name to
    extract polarization.

    Args:
      path_to_raster: path to the s1 raster file

    Returns:
      a string representing the polarization
    """
    return path_to_raster.split("/")[-1].split("-")[3]


def get_platform_from_s1_raster(path_to_raster):
    """
    Small utilty function that parses a s1 raster file name to extract platform

    Args:
      path_to_raster: path to the s1 raster file

    Returns:
      a string representing the platform
    """
    return path_to_raster.split("/")[-1].split("-")[0]


class ExecutionTimer:
    """Context manager to help measure execution times

    Example:
    with ExecutionTimer("the code", True) as t:
        Code_to_measure()
    """
    def __init__(self, text, do_measure):
        self._text       = text
        self._do_measure = do_measure

    def __enter__(self):
        self._start = timer()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if self._do_measure:
            end = timer()
            logging.info("%s took %ssec", self._text, end - self._start)
        return False


def list_files(directory, pattern=None):
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
        res = [entry for entry in nodes if filt(entry)]
    return res


def list_dirs(directory, pattern=None):
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
        res = [entry for entry in nodes if filt(entry)]
    return res


class RedirectStdToLogger:
    """
    Yet another helper class to redirect messages sent to stdout and stderr to a proper
    logger.

    This is a very simplified version tuned to answer S1Tiling needs.
    It also acts as a context manager.
    """
    def __init__(self, logger):
        self.__old_stdout = sys.stdout
        self.__old_stderr = sys.stderr
        self.__logger     = logger
        sys.stdout = RedirectStdToLogger.__StdOutErrAdapter(logger)
        sys.stderr = RedirectStdToLogger.__StdOutErrAdapter(logger, logging.ERROR)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        sys.stdout = self.__old_stdout
        sys.stderr = self.__old_stderr
        return False

    class __StdOutErrAdapter:
        """
        Internal adapter that redirects messages, initially sent to a file, to a logger.
        """
        def __init__(self, logger, mode=None):
            self.__logger     = logger
            self.__mode       = mode  # None => adapt DEBUG/INFO/ERROR/...
            self.__last_level = mode  # None => adapt DEBUG/INFO/ERROR/...
            self.__lvl_re     = re.compile(r'(\((DEBUG|INFO|WARNING|ERROR)\))')
            self.__lvl_map    = {
                    'DEBUG':   logging.DEBUG,
                    'INFO':    logging.INFO,
                    'WARNING': logging.WARNING,
                    'ERROR':   logging.ERROR}

        def write(self, message):
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

        def flush(self):
            """
            Overrides sys.stdout.flush() method
            """
            pass

        def isatty(self):  # pylint: disable=no-self-use
            """
            Overrides sys.stdout.isatty() method.
            This is required by OTB Python bindings.
            """
            return False


def remove_files(files: list):
    """
    Removes the files from the disk
    """
    assert isinstance(files, list)
    logging.debug("Remove %s", files)
    for file_it in files:
        if os.path.exists(file_it):
            os.remove(file_it)


class TopologicalSorter:
    """
    Depth-first topological_sort implementation
    """
    def __init__(self, dag, fetch_successor_function=None):
        """
        constructor
        """
        self.__table = dag
        if fetch_successor_function:
            self.__successor_fetcher = fetch_successor_function
            self.__successors        = self.__successors_lazy
        else:
            self.__successors        = self.__successors_direct

    def depth(self, start_nodes):
        """
        Depth-first topological sorting method
        """
        results = []
        visited_nodes = {}
        self.__recursive_depth_first(start_nodes, results, visited_nodes)
        return reversed(results)

    def __successors_lazy(self, node):
        node_info = self.__table.get(node, None)
        # logging.debug('node:%s ; infos=%s', node, node_info)
        return self.__successor_fetcher(node_info) if node_info else []

    def __successors_direct(self, node):
        return self.__table.get(node, [])

    def __recursive_depth_first(self, start_nodes, results, visited_nodes):
        # logging.debug('start_nodes: %s', start_nodes)
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

def tsort(dag, start_nodes, fetch_successor_function=None):
    """
    Topological sorting function (depth-first)

    Parameters:
        :dag:                      Direct Acyclic Graph to sort topologically
        :start_nodes:              nodes from which the sorting star
        :fetch_successor_function: Used to override how node transition is done
    """
    ts = TopologicalSorter(dag, fetch_successor_function)
    return ts.depth(start_nodes)
