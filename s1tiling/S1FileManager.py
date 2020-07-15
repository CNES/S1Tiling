#!/usr/bin/env python3
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
from s1tiling.Utils import get_origin, list_files, list_dirs
from s1tiling.S1DateAcquisition import S1DateAcquisition
import tempfile
import fnmatch, glob
import sys
import logging

logger = logging.getLogger('s1tiling')

class Layer(object):
    """
    Thin wrapper that requests GDL Layers and keep a living reference to intermediary objects.
    """
    def __init__(self, grid):
        self.__grid        = grid
        self.__driver      = ogr.GetDriverByName("ESRI Shapefile")
        self.__data_source = self.__driver.Open(self.__grid, 0)
        self.__layer       = self.__data_source.GetLayer()

    def __iter__(self):
        return self.__layer.__iter__()

    def ResetReading(self):
        return self.__layer.ResetReading()

    def find_tile_named(self, tile_name_field):
        for tile in self.__layer:
            if tile.GetField('NAME') in tile_name_field:
                return tile
        return None

def download(raw_directory, pepscommand, lonmin, lonmax, latmin, latmax, tile_data, tile_name):
    """
    Process with the call to peps_download
    """
    from subprocess import Popen
    import time
    command = pepscommand\
            +" --lonmin "+str(lonmin)+" --lonmax "+str(lonmax)\
            +" --latmin "+str(latmin)+" --latmax "+str(latmax)\
            +" -w "+raw_directory\
            +" --tiledata "+tile_data
    logger.debug('Download with %s', command)
    status = -1
    while status != 0:
        if tile_name: # <=> self.cfg.cluster is True
            pid = Popen(command, stdout=open(tile_name,"a"),      stderr=None,shell=True)
        else:
            pid = Popen(command, stdout=open("/dev/stdout", 'w'), stderr=open("/dev/stderr", 'w'),shell=True)
        while pid.poll() is None:
            unzip_images(raw_directory)
            time.sleep(20)
        status = pid.poll()

def unzip_images(raw_directory):
    """This method handles unzipping of product archives"""
    import zipfile
    for file_it in list_files(raw_directory, '*.zip'):
        logger.debug("unzipping %s",file_it.name)
        try:
            with zipfile.ZipFile(file_it.path, 'r') as zip_ref:
                zip_ref.extractall(raw_directory)
        except  zipfile.BadZipfile:
            logger.warning("%s is corrupted. This file will be removed", file_it.path)
        try:
            os.remove(file_it.path)
        except:
            pass

def filter_images_or_ortho(kind, all_images):
    pattern = "*"+kind+"*-???.tiff"
    ortho_pattern = "*"+kind+"*-???_OrthoReady.tiff"
    # fnmatch cannot be used with patterns like 'dir/*.foo'
    # => As the directory as been filtered with glob(), just work without the directory part
    images = fnmatch.filter(all_images, pattern)
    logger.debug("  * %s images: %s", kind, images)
    if not images:
        images = [f.replace("_OrthoReady.tiff",".tiff") for f in fnmatch.filter(all_images, ortho_pattern)]
        logger.debug("    %s images from Ortho: %s", kind, images)
    return images


class S1FileManager(object):
    """ Class to manage processed files (downloads, checks) """
    def __init__(self,cfg):

        self.cfg=cfg
        self.raw_raster_list = []
        self.nb_images = 0

        self.__tmpsrtmdir = None

        self.tiff_pattern = "measurement/*.tiff"
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
            self.roi_by_tiles       = None

            try:
                self.roi_by_tiles = self.cfg.ROI_by_tiles
            except cfg.NoOptionError:
                try:
                    self.roi_by_coordinates = cfg.ROI_by_coordinates.split()
                except cfg.NoOptionError:
                    logger.critical("No ROI defined in the config file")
                    exit(-1)

        try:
            os.makedirs(self.cfg.raw_directory)
        except os.error:
            pass

    def __enter__(self):
        """
        Turn the S1FileManager into a context manager, context acquisition function
        """
        return self
    def __exit__(self, type, value, traceback):
        """
        Turn the S1FileManager into a context manager, cleanup function
        """
        if self.__tmpsrtmdir:
            logger.debug('Cleaning temporary SRTM diretory (%s)', self.__tmpsrtmdir)
            self.__tmpsrtmdir.cleanup()
            self.__tmpsrtmdir = None
        return False

    def tmpsrtmdir(self, srtm_tiles):
        """
        Generate the temporary directory for SRTM tiles on the fly
        And populate it with symbolic link to the actual SRTM tiles
        """
        if not self.__tmpsrtmdir:
            # copy all needed SRTM file in a temp directory for orthorectification processing
            self.__tmpsrtmdir = tempfile.TemporaryDirectory(dir=self.cfg.tmpdir)
            logger.debug('Create temporary SRTM diretory (%s) for needed tiles %s', self.__tmpsrtmdir, srtm_tiles)
            assert(os.path.isdir(self.__tmpsrtmdir.name))
            for srtm_tile in srtm_tiles:
                logger.debug('ln -s %s  <-- %s',
                        os.path.join(self.cfg.srtm,          srtm_tile),
                        os.path.join(self.__tmpsrtmdir.name, srtm_tile))
                os.symlink(
                        os.path.join(self.cfg.srtm,          srtm_tile),
                        os.path.join(self.__tmpsrtmdir.name, srtm_tile))
        return self.__tmpsrtmdir.name

    def keep_X_latest_S1_files(self, threshold):
        safeFileList = sorted(glob.glob(os.path.join(self.cfg.raw_directory,"*")), key=os.path.getctime)
        if len(safeFileList) > threshold:
            for f in safeFileList[:len(safeFileList)-threshold]:
                logger.debug("Remove : ",os.path.basename(f))
                shutil.rmtree(f, ignore_errors=True)
            self.get_s1_img()

    def download_images(self,tiles=None):
        """ This method downloads the required images if pepsdownload is True"""
        import numpy as np
        if not self.cfg.pepsdownload:
            logger.info("Using images already downloaded, as per configuration request")
            return

        if self.roi_by_tiles is not None:
            if tiles:
                tiles_list=tiles
            elif "ALL" in self.roi_by_tiles:
                tiles_list = self.cfg.tiles_list
            else:
                tiles_list = self.roi_by_tiles
            logger.debug("Tiles requested to download: %s", tiles_list)

            layer = Layer(self.cfg.output_grid)
            for current_tile in layer:
                if current_tile.GetField('NAME') in tiles_list:
                    tile_footprint = current_tile.GetGeometryRef().GetGeometryRef(0)
                    latmin = np.min([p[1] for p in tile_footprint.GetPoints()])
                    latmax = np.max([p[1] for p in tile_footprint.GetPoints()])
                    lonmin = np.min([p[0] for p in tile_footprint.GetPoints()])
                    lonmax = np.max([p[0] for p in tile_footprint.GetPoints()])
                    download(self.cfg.raw_directory, self.pepscommand,
                            lonmin, lonmax, latmin, latmax,
                            os.path.join(self.cfg.output_preprocess,tiles_list),
                            current_tile.GetField('NAME')+".txt" if self.cfg.cluster else None)
        else: # roi_by_tiles is None
            download(self.cfg.raw_directory, self.pepscommand,
                    self.roi_by_coordinates[0], self.roi_by_coordinates[2],
                    self.roi_by_coordinates[1], self.roi_by_coordinates[3],
                    os.path.join(self.cfg.output_preprocess,current_tile),
                    current_tile.GetField('NAME')+".txt" if self.cfg.cluster else None)
        unzip_images(self.cfg.raw_directory)
        self.get_s1_img()

    def get_s1_img(self):
        """
        This method updates the list of S1 images available
        (from analysis of raw_directory)

        Returns:
           the list of S1 images available as instances
           of S1DateAcquisition class
        """

        self.raw_raster_list=[]
        if os.path.exists(self.cfg.raw_directory) == False:
            os.makedirs(self.cfg.raw_directory)
            return
        content = list_dirs(self.cfg.raw_directory)

        for current_content in content:
            safe_dir = current_content.path
            manifest = os.path.join(safe_dir, self.manifest_pattern)
            acquisition = S1DateAcquisition(manifest, [])
            all_tiffs = glob.glob(os.path.join(safe_dir, self.tiff_pattern))
            logger.debug("# Safe dir: %s", safe_dir)
            logger.debug("  all tiffs: %s", list(all_tiffs))

            vv_images = filter_images_or_ortho('vv', all_tiffs)
            vh_images = filter_images_or_ortho('vh', all_tiffs)
            hv_images = filter_images_or_ortho('hv', all_tiffs)
            hh_images = filter_images_or_ortho('hh', all_tiffs)

            for image in vv_images + vh_images + hv_images + hh_images:
                if image not in self.processed_filenames:
                    acquisition.add_image(image)
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
        layer = Layer(self.cfg.output_grid)

        for current_tile in layer:
            #logger.debug("%s", current_tile.GetField('NAME'))
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

        layer = Layer(self.cfg.output_grid)

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
        logger.debug('Test intersections of %s', tile_name_field)
        # TODO: don't abort if there is only vv or vh
        # => move to another dependency analysis policy
        date_exist=[os.path.basename(f)[21:21+8] for f in glob.glob(os.path.join(self.cfg.output_preprocess,tile_name_field,"s1?_*.tif"))]
        intersect_raster = []

        layer = Layer(self.cfg.output_grid)
        current_tile = layer.find_tile_named(tile_name_field)
        if not current_tile:
            logger.info("Tile %s does not exist", tile_name_field)
            return intersect_raster

        poly = ogr.Geometry(ogr.wkbPolygon)
        tile_footprint = current_tile.GetGeometryRef()

        for image in self.raw_raster_list:
            logger.debug('- Manifest: %s', image.get_manifest())
            logger.debug('  Image list: %s', image.get_images_list())
            if len(image.get_images_list())==0:
                logger.critical("Problem with %s",image.get_manifest())
                logger.critical("Please remove the raw data for this SAFE file")
                sys.exit(-1)

            date_safe=os.path.basename(image.get_images_list()[0])[14:14+8]

            if date_safe in date_exist:
                logger.debug('  -> Safe date (%s) found in %s => Ignore %s', date_safe, date_exist, image.get_images_list())
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
            logger.debug('   -> Test intersection: requested: %s  VS tile: %s --> %s', ring, tile_footprint, intersection)
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
        mgrs_layer = Layer(self.cfg.output_grid)

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
        srtm_layer = Layer(self.cfg.SRTMShapefile)

        needed_srtm_tiles = {}

        for tile in tiles_to_process:
            logger.debug("Check SRTM tile for %s",tile)

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
        logger.info("SRTM ok")
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
