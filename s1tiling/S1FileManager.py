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

""" This module contains the S1FileManager class"""

import os
import ogr
from s1tiling.Utils import get_origin
from s1tiling.S1DateAcquisition import S1DateAcquisition
import tempfile
import glob
import sys

class S1FileManager(object):
    """ Class to manage processed files (downloads, checks) """
    def __init__(self,cfg):

        self.cfg=cfg
        self.raw_raster_list = []
        self.nb_images = 0

        self.tmpsrtmdir=tempfile.mkdtemp(dir=cfg.tmpdir)

        self.vh_pattern = "measurement/*vh*-???.tiff"
        self.vv_pattern = "measurement/*vv*-???.tiff"
        self.hh_pattern = "measurement/*hh*-???.tiff"
        self.hv_pattern = "measurement/*hv*-???.tiff"
        self.manifest_pattern = "manifest.safe"

        self.processed_filenames = self.get_processed_filenames(cfg)
        self.get_s1_img()

        if self.cfg.pepsdownload == True:
            self.fd=cfg.first_date
            self.ld=cfg.last_date
            self.pepscommand = "python ./peps/peps_download/peps_download.py -c S1 -p"+ self.cfg.type_image+\
            " -a ./peps/peps_download/peps.txt -m IW -d "\
            +self.cfg.first_date+" -f "+self.cfg.last_date+ " --pol "+self.cfg.polarisation
            self.roi_by_coordinates = None
            self.roi_by_tiles = None

            try:
                self.roi_by_tiles = self.cfg.ROI_by_tiles
            except cfg.NoOptionError:
                try:
                    self.roi_by_coordinates\
                        = cfg.ROI_by_coordinates.split()
                except cfg.NoOptionError:
                    print("No ROI defined in the config file")
                    exit(-1)

        try:
            os.makedirs(self.cfg.raw_directory)
        except os.error:
            pass

    def download_images(self,tiles=None):
        """ This method downloads the required images if pepsdownload is True"""
        import numpy as np
        from subprocess import Popen
        import time
        if self.cfg.pepsdownload == True:

            if self.roi_by_tiles is not None:
                if tiles is None:
                    if "ALL" in self.roi_by_tiles:
                        tiles_list = self.cfg.tiles_list
                    else:
                        tiles_list = self.roi_by_tiles
                else:
                   tiles_list=tiles
                latmin = []
                latmax = []
                lonmin = []
                lonmax = []
                print(tiles_list)

                driver = ogr.GetDriverByName("ESRI Shapefile")
                data_source = driver.Open(self.cfg.output_grid, 0)
                layer = data_source.GetLayer()
                for current_tile in layer:

                    if current_tile.GetField('NAME') in tiles_list:
                        tile_footprint = current_tile.GetGeometryRef()\
                                                     .GetGeometryRef(0)
                        latmin = np.min([p[1] for p in tile_footprint\
                                         .GetPoints()])
                        latmax = np.max([p[1] for p in tile_footprint\
                                         .GetPoints()])
                        lonmin = np.min([p[0] for p in tile_footprint\
                                         .GetPoints()])
                        lonmax = np.max([p[0] for p in tile_footprint\
                                         .GetPoints()])
                        """
                        command = "python "+self.pepscommand+" --lonmin "\
                                  +str(lonmin)+" --lonmax "+str(lonmax)\
                                  +" --latmin "+str(latmin)+" --latmax "\
                                  +str(latmax)+" -w "+self.raw_directory
                        """
                        command = self.pepscommand+" --lonmin "\
                                  +str(lonmin)+" --lonmax "+str(lonmax)\
                                  +" --latmin "+str(latmin)+" --latmax "\
                                  +str(latmax)+" -w "+self.cfg.raw_directory\
                                  +" --tiledata "+os.path.join(self.cfg.output_preprocess,tiles_list)
                        print(command)
                        status = -1
                        while status != 0:
                            if self.cfg.cluster:
                                pid = Popen(command, stdout=open(current_tile.GetField('NAME')+".txt","a"), stderr=None,shell=True)
                            else:
                                pid = Popen(command, stdout=open("/dev/stdout", 'w'), stderr=open("/dev/stderr", 'w'),shell=True)
                            while pid.poll() is None:
                                self.unzip_images()
                                time.sleep(20)
                            status = pid.poll()
            else:
                command = "python "+self.pepscommand+" --lonmin "\
                          +str(self.roi_by_coordinates[0])+" --lonmax "\
                          +str(self.roi_by_coordinates[2])+" --latmin "\
                          +str(self.roi_by_coordinates[1])+" --latmax "\
                          +str(self.roi_by_coordinates[3])+" -w "\
                          +self.raw_directory \
                          +" --tiledata "+os.path.join(self.cfg.output_preprocess,current_tile)
                print(command)
                status = -1
                while status != 0:
                    if self.cfg.cluster:
                        pid = Popen(command, stdout=open(current_tile.GetField('NAME')+".txt","a"), stderr=None,shell=True)
                    else:
                        pid = Popen(command, stdout=open("/dev/stdout", 'w'), stderr=open("/dev/stderr", 'w'),shell=True)

                    while pid.poll() is None:
                        self.unzip_images()
                        time.sleep(20)
                    status = pid.poll()
            self.unzip_images()
            self.get_s1_img()

    def unzip_images(self):
        """This method handles unzipping of product archives"""
        import zipfile
        for file_it in os.walk(self.cfg.raw_directory).next()[2]:
            if ".zip" in file_it:
                print("unzipping "+file_it)
                try:
                    zip_ref = zipfile.ZipFile(self.cfg.raw_directory+"/"+\
                                              file_it, 'r')
                    zip_ref.extractall(self.cfg.raw_directory)
                    zip_ref.close()
                except  zipfile.BadZipfile:
                    print("WARNING: "+self.cfg.raw_directory+"/"+\
                        file_it+" is corrupted. This file will be removed")
                try:
                    os.remove(self.cfg.raw_directory+"/"+file_it)
                except:
                    pass


    def get_s1_img(self):
        """
        This method returns the list of S1 images available
        (from analysis of raw_directory)

        Returns:
           the list of S1 images available as instances
           of S1DateAcquisition class
        """
        import glob

        self.raw_raster_list=[]
        if os.path.exists(self.cfg.raw_directory) == False:
            os.makedirs(self.cfg.raw_directory)
            return
        content = os.listdir(self.cfg.raw_directory)

        for current_content in content:
            safe_dir = os.path.join(self.cfg.raw_directory, current_content)
            if os.path.isdir(safe_dir) == True:


                manifest = os.path.join(safe_dir, self.manifest_pattern)
                acquisition = S1DateAcquisition(manifest, [])
                vv_images = [f for f in\
                             glob.glob(os.path.join(safe_dir, self.vv_pattern))]
                if vv_images == []:
                    vv_images = [f.replace("_OrthoReady.tiff",".tiff") for f in\
                             glob.glob(os.path.join(safe_dir, self.vv_pattern.replace(".tiff","_OrthoReady.tiff")))]
                for vv_image in vv_images:
                    if vv_image not in self.processed_filenames:
                        acquisition.add_image(vv_image)
                        self.nb_images += 1
                vh_images = [f for f in\
                             glob.glob(os.path.join(safe_dir, self.vh_pattern))]
                if vh_images == []:
                    vh_images = [f.replace("_OrthoReady.tiff",".tiff") for f in\
                             glob.glob(os.path.join(safe_dir, self.vh_pattern.replace(".tiff","_OrthoReady.tiff")))]
                for vh_image in vh_images:
                    if vh_image not in self.processed_filenames:
                        acquisition.add_image(vh_image)
                        self.nb_images += 1
                hh_images = [f for f in\
                             glob.glob(os.path.join(safe_dir, self.hh_pattern))]
                if hh_images == []:
                    hh_images = [f.replace("_OrthoReady.tiff",".tiff") for f in\
                             glob.glob(os.path.join(safe_dir, self.hh_pattern.replace(".tiff","_OrthoReady.tiff")))]
                for hh_image in hh_images:
                    if hh_image not in self.processed_filenames:
                        acquisition.add_image(hh_image)
                        self.nb_images += 1
                hv_images = [f for f in\
                             glob.glob(os.path.join(safe_dir, self.hv_pattern))]
                if hv_images == []:
                    hv_images = [f.replace("_OrthoReady.tiff",".tiff") for f in\
                             glob.glob(os.path.join(safe_dir, self.hv_pattern.replace(".tiff","_OrthoReady.tiff")))]
                for hv_image in hv_images:
                    if hv_image not in self.processed_filenames:
                        acquisition.add_image(hv_image)
                        self.nb_images += 1

                self.raw_raster_list.append(acquisition)

    def tile_exists(self, tile_name_field):
        """
        This method check if a given MGRS tiles exists in the database

        Args:
          tile_name_field: MGRS tile identifier

        Returns:
          True if the tile exists, False otherwise
        """
        driver = ogr.GetDriverByName("ESRI Shapefile")
        data_source = driver.Open(self.cfg.output_grid, 0)
        layer = data_source.GetLayer()

        for current_tile in layer:
            #print(current_tile.GetField('NAME'))
            if current_tile.GetField('NAME') in tile_name_field:
                return True
        return False

    def get_tiles_covered_by_products(self):
        """
        This method returns the list of MGRS tiles covered
        by available S1 products.

        Returns:
           The list of MGRS tiles identifiers covered by product as string
        """
        tiles = []

        driver = ogr.GetDriverByName("ESRI Shapefile")
        data_source = driver.Open(self.cfg.output_grid, 0)
        layer = data_source.GetLayer()

        #Loop on images
        for image in self.raw_raster_list:
            manifest = image.get_manifest()
            nw_coord, ne_coord, se_coord, sw_coord = get_origin(manifest)

            poly = ogr.Geometry(ogr.wkbPolygon)
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(nw_coord[1], nw_coord[0], 0)
            ring.AddPoint(ne_coord[1], ne_coord[0], 0)
            ring.AddPoint(se_coord[1], se_coord[0], 0)
            ring.AddPoint(sw_coord[1], sw_coord[0], 0)
            ring.AddPoint(nw_coord[1], nw_coord[0], 0)
            poly.AddGeometry(ring)

            for current_tile in layer:
                tile_footprint = current_tile.GetGeometryRef()
                intersection = poly.Intersection(tile_footprint)
                if intersection.GetArea()/tile_footprint.GetArea()\
                   > self.cfg.tile_to_product_overlap_ratio:
                    tile_name = current_tile.GetField('NAME')
                    if tile_name not in tiles:
                        tiles.append(tile_name)
        return tiles

    def get_s1_intersect_by_tile(self, tile_name_field):
        """
        This method return the list of S1 product intersecting a given MGRS tile

        Args:
          tile_name_field: The MGRS tile identifier

        Returns:
          A list of tuple (image as instance of
          S1DateAcquisition class, [corners]) for S1 products
          intersecting the given tile
        """
        date_exist=[os.path.basename(f)[21:21+8] for f in glob.glob(os.path.join(self.cfg.output_preprocess,tile_name_field,"s1?_*.tif"))]
        intersect_raster = []
        driver = ogr.GetDriverByName("ESRI Shapefile")
        data_source = driver.Open(self.cfg.output_grid, 0)
        layer = data_source.GetLayer()
        current_tile = None
        for current_tile in layer:
            if current_tile.GetField('NAME') in tile_name_field:
                break
        if not current_tile:
            print("Tile "+str(tile_name_field)+" does not exist")
            return intersect_raster

        poly = ogr.Geometry(ogr.wkbPolygon)
        tile_footprint = current_tile.GetGeometryRef()


        for image in self.raw_raster_list:
            print(image.get_manifest())
            print(image.get_images_list())
            if len(image.get_images_list())==0:
                print("Problem with : "+image.get_manifest())
                print("Remove the raw data for this SAFE file")
                sys.exit(-1)

            date_safe=os.path.basename(image.get_images_list()[0])[14:14+8]

            if date_safe in date_exist:
                continue
            manifest = image.get_manifest()
            nw_coord, ne_coord, se_coord, sw_coord = get_origin(manifest)

            poly = ogr.Geometry(ogr.wkbPolygon)
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(nw_coord[1], nw_coord[0], 0)
            ring.AddPoint(ne_coord[1], ne_coord[0], 0)
            ring.AddPoint(se_coord[1], se_coord[0], 0)
            ring.AddPoint(sw_coord[1], sw_coord[0], 0)
            ring.AddPoint(nw_coord[1], nw_coord[0], 0)
            poly.AddGeometry(ring)

            intersection = poly.Intersection(tile_footprint)
            if intersection.GetArea() != 0:
                area_polygon = tile_footprint.GetGeometryRef(0)
                points = area_polygon.GetPoints()
                intersect_raster.append((image, [(point[0], point[1]) for point\
                                                 in points[:-1]]))

        return intersect_raster

    def get_mgrs_tile_geometry_by_name(self, mgrs_tile_name):
        """
        This method returns the MGRS tile geometry
        as OGRGeometry given its identifier

        Args:
          mgrs_tile_name: MGRS tile identifier

        Returns:
          The MGRS tile geometry as OGRGeometry or raise ValueError
        """
        driver = ogr.GetDriverByName("ESRI Shapefile")
        mgrs_ds = driver.Open(self.cfg.output_grid, 0)
        mgrs_layer = mgrs_ds.GetLayer()

        for mgrs_tile in mgrs_layer:
            if mgrs_tile.GetField('NAME') == mgrs_tile_name:
                return mgrs_tile.GetGeometryRef().Clone()
        raise ValueError("MGRS tile does not exist", mgrs_tile_name)


    def check_srtm_coverage(self, tiles_to_process):
        """
        Given a set of MGRS tiles to process, this method
        returns the needed SRTM tiles and the corresponding coverage.

        Args:
          tile_to_process: The list of MGRS tiles identifiers to process

        Return:
          A list of tuples (SRTM tile id, coverage of MGRS tiles).
          Coverage range is [0,1]

        """
        driver = ogr.GetDriverByName("ESRI Shapefile")
        srtm_ds = driver.Open(self.cfg.SRTMShapefile, 0)
        srtm_layer = srtm_ds.GetLayer()

        needed_srtm_tiles = {}

        for tile in tiles_to_process:
            print("Check SRTM tile for ",tile)

            srtm_tiles = []
            mgrs_footprint = self.get_mgrs_tile_geometry_by_name(tile)
            area = mgrs_footprint.GetArea()
            srtm_layer.ResetReading()
            for srtm_tile in srtm_layer:
                srtm_footprint = srtm_tile.GetGeometryRef()
                intersection = mgrs_footprint.Intersection(srtm_footprint)
                if intersection.GetArea() > 0:
                    coverage = intersection.GetArea()/area
                    srtm_tiles.append((srtm_tile.GetField('FILE'), coverage))
            needed_srtm_tiles[tile] = srtm_tiles
        return needed_srtm_tiles

    def record_processed_filenames(self):
        """ Record the list of processed filenames (DEPRECATED)"""
        with open(os.path.join(self.cfg.output_preprocess,\
                               "processed_filenames.txt"), "a") as in_file:
            for fic in self.processed_filenames:
                in_file.write(fic+"\n")

    def get_processed_filenames(self,cfg):
        """ Read back the list of processed filenames (DEPRECATED)"""
        try:
            with open(os.path.join(self.cfg.output_preprocess,\
                                   "processed_filenames.txt"), "r") as in_file:
                return in_file.read().splitlines()
        except (IOError, OSError):
            return []

    def get_raster_list(self):
        """
        Get the list of raw S1 product rasters

        Returns:
          the list of raw rasters
        """

        return self.raw_raster_list
