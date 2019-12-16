#!/usr/bin/python
#-*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright (c) CESBIO. All rights reserved.
#
#   See LICENSE for details.
#
#   This software is distributed WITHOUT ANY WARRANTY; without even
#   the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#   PURPOSE.  See the above copyright notices for more information.
#
# =========================================================================
#
# Authors: Thierry KOLECK (CNES)
#
# =========================================================================

""" This module contains various utility functions"""

import ogr
from osgeo import osr
import xml.etree.ElementTree as ET


def get_relative_orbit(manifest):
    root=ET.parse(manifest)
    return int(root.find("metadataSection/metadataObject/metadataWrap/xmlData/{http://www.esa.int/safe/sentinel-1.0}orbitReference/{http://www.esa.int/safe/sentinel-1.0}relativeOrbitNumber").text)

def get_origin(manifest):
    """Parse the coordinate of the origin in the manifest file

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
                coord = [(float(val.replace("\n", "").split(",")[0]),\
                          float(val.replace("\n", "")\
                                .split(",")[1]))for val in coor]
                return coord[0], coord[1], coord[2], coord[3]
        raise Exception("Coordinates not found in "+str(manifest))

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
    s1_footprint = get_origin(manifest)
    poly = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(s1_footprint[0][1], s1_footprint[0][0])
    ring.AddPoint(s1_footprint[1][1], s1_footprint[1][0])
    ring.AddPoint(s1_footprint[2][1], s1_footprint[2][0])
    ring.AddPoint(s1_footprint[3][1], s1_footprint[3][0])
    ring.AddPoint(s1_footprint[0][1], s1_footprint[0][0])
    poly.AddGeometry(ring)

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
        raise Exception("Orbit Directiction not found in "+str(manifest))


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
    for in_coord in tuple_list:
        lon = in_coord[0]
        lat = in_coord[1]

        in_spatial_ref = osr.SpatialReference()
        in_spatial_ref.ImportFromEPSG(in_epsg)
        out_spatial_ref = osr.SpatialReference()
        out_spatial_ref.ImportFromEPSG(out_epsg)

        coord_trans = osr.CoordinateTransformation(in_spatial_ref,\
                                                   out_spatial_ref)
        coord = coord_trans.TransformPoint(lon, lat)
        tuple_out.append(coord)
    return tuple_out

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



   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
