#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2020 (c) CESBIO. All rights reserved.
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

""" This module contains the S1FileManager class"""

import os
import ogr
from libs.Utils import get_origin, list_files, list_dirs
from libs.S1DateAcquisition import S1DateAcquisition
import fnmatch
import glob
import logging
import re
import shutil
import sys
import tempfile

from eodag.api.core import EODataAccessGateway
from eodag.utils.logging import setup_logging
setup_logging(verbose=1)

logger = logging.getLogger('s1tiling')


class Layer:
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


def product_property(prod, key, default=None):
    res = prod.properties.get(key, default)
    return res


def unzip_images(raw_directory):
    """This method handles unzipping of product archives"""
    import zipfile
    for file_it in list_files(raw_directory, '*.zip'):
        logger.debug("unzipping %s", file_it.name)
        try:
            with zipfile.ZipFile(file_it.path, 'r') as zip_ref:
                zip_ref.extractall(raw_directory)
        except zipfile.BadZipfile:
            logger.warning("%s is corrupted. This file will be removed", file_it.path)
        try:
            os.remove(file_it.path)
        except OSError:
            pass


def does_final_product_need_to_be_generated_for(product, tile_name, polarizations, s2images):
    logger.debug('Searching %s in %s', product, s2images)
    # id=S1A_IW_GRDH_1SDV_20200108T044150_20200108T044215_030704_038506_C7F5,
    prod_re = re.compile(r'(S1.)_IW_...._...._(\d{8})T\d{6}.*')
    sat, start = prod_re.match(product.as_dict()['id']).groups()
    for pol in polarizations:
        # s1a_{tilename}_{polarization}_DES_007_20200108txxxxxx.tif
        pat = '%s_%s_%s_*_%stxxxxxx.tif' % (sat.lower(), tile_name, pol, start)
        found = fnmatch.filter(s2images, pat)
        logger.debug('searching w/ %s ==> Found: %s', pat, found)
        if not found:
            return True
    return False


def filter_images_or_ortho(kind, all_images):
    pattern = "*" + kind + "*-???.tiff"
    ortho_pattern = "*" + kind + "*-???_OrthoReady.tiff"
    # fnmatch cannot be used with patterns like 'dir/*.foo'
    # => As the directory as been filtered with glob(), just work without the directory part
    images = fnmatch.filter(all_images, pattern)
    logger.debug("  * %s images: %s", kind, images)
    if not images:
        images = [f.replace("_OrthoReady.tiff", ".tiff")
                for f in fnmatch.filter(all_images, ortho_pattern)]
        logger.debug("    %s images from Ortho: %s", kind, images)
    return images


class S1FileManager:
    """ Class to manage processed files (downloads, checks) """
    def __init__(self, cfg):

        self.cfg              = cfg
        self.raw_raster_list  = []
        self.nb_images        = 0

        self.__tmpsrtmdir     = None

        self.tiff_pattern     = "measurement/*.tiff"
        self.vh_pattern       = "measurement/*vh*-???.tiff"
        self.vv_pattern       = "measurement/*vv*-???.tiff"
        self.hh_pattern       = "measurement/*hh*-???.tiff"
        self.hv_pattern       = "measurement/*hv*-???.tiff"
        self.manifest_pattern = "manifest.safe"

        self._ensure_workspaces_exist()
        self.processed_filenames = self.get_processed_filenames(cfg)
        self._update_s1_img_list()

        if self.cfg.download:
            self.fd = cfg.first_date
            self.ld = cfg.last_date
            logger.debug('Using %s EODAG configuration file', self.cfg.eodagConfig or 'user default')
            self._dag = EODataAccessGateway(self.cfg.eodagConfig)
            # TODO: update once eodag directly offers "DL directory setting" feature v1.7? +?
            dest_dir = os.path.abspath(self.cfg.raw_directory)
            logger.debug('Override EODAG output directory to %s', dest_dir)
            for provider in self._dag.providers_config.keys():
                if hasattr(self._dag.providers_config[provider], 'download'):
                    self._dag.providers_config[provider].download.update({'outputs_prefix': dest_dir})
                    logger.debug(' - for %s', provider)
                else:
                    logger.debug(' - NOT for %s', provider)

            self.roi_by_coordinates = None
            self.roi_by_tiles       = None

            try:
                self.roi_by_tiles = self.cfg.ROI_by_tiles
            except cfg.NoOptionError:
                try:
                    self.roi_by_coordinates = cfg.ROI_by_coordinates.split()
                except cfg.NoOptionError:
                    logger.critical("No ROI defined in the config file")
                    sys.exit(-1)

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

    def _ensure_workspaces_exist(self):
        """
        Makes sure the directories used for :
        - raw data
        - output data
        - and temporary data
        all exist
        """
        for p in [self.cfg.raw_directory, self.cfg.tmpdir, self.cfg.output_preprocess]:
            if not os.path.exists(p):
                os.makedirs(p, exist_ok=True)

    def ensure_tile_workspaces_exist(self, tile_name):
        """
        Makes sure the directories used for :
        - output data/{tile}
        - and temporary data/S2/{tile}
        all exist
        """
        working_directory = os.path.join(self.cfg.tmpdir, 'S2', tile_name)
        os.makedirs(working_directory, exist_ok=True)

        out_dir = os.path.join(self.cfg.output_preprocess, tile_name)
        os.makedirs(out_dir, exist_ok=True)
        return working_directory, out_dir

    def tmpsrtmdir(self, srtm_tiles):
        """
        Generate the temporary directory for SRTM tiles on the fly
        And populate it with symbolic link to the actual SRTM tiles
        """
        if not self.__tmpsrtmdir:
            # copy all needed SRTM file in a temp directory for orthorectification processing
            self.__tmpsrtmdir = tempfile.TemporaryDirectory(dir=self.cfg.tmpdir)
            logger.debug('Create temporary SRTM diretory (%s) for needed tiles %s', self.__tmpsrtmdir, srtm_tiles)
            assert os.path.isdir(self.__tmpsrtmdir.name)
            for srtm_tile in srtm_tiles:
                logger.debug('ln -s %s  <-- %s',
                        os.path.join(self.cfg.srtm,          srtm_tile),
                        os.path.join(self.__tmpsrtmdir.name, srtm_tile))
                os.symlink(
                        os.path.join(self.cfg.srtm,          srtm_tile),
                        os.path.join(self.__tmpsrtmdir.name, srtm_tile))
        return self.__tmpsrtmdir.name

    def keep_X_latest_S1_files(self, threshold):
        safeFileList = sorted(glob.glob(os.path.join(self.cfg.raw_directory, "*")), key=os.path.getctime)
        if len(safeFileList) > threshold:
            for f in safeFileList[ : len(safeFileList) - threshold]:
                logger.debug("Remove old SAFE: %s", os.path.basename(f))
                shutil.rmtree(f, ignore_errors=True)
            self._update_s1_img_list()

    def _download(self, dag: EODataAccessGateway,
            lonmin, lonmax, latmin, latmax,
            first_date, last_date,
            tile_out_dir, tile_name, polarization):
        """
        Process with the call to eodag download.
        :param raw_directory:
        """
        product_type = 'S1_SAR_GRD'
        extent = {
                'lonmin': lonmin,
                'lonmax': lonmax,
                'latmin': latmin,
                'latmax': latmax
                }
        products, estimated_total_nbr_of_results = dag.search(
                productType=product_type,
                start=first_date, end=last_date,
                box=extent,
                # If we have eodag v1.6, we try to filter product during the search request
                polarizationMode=polarization,
                sensorMode="IW"
                )
        logger.info("%s remote S1 products found: %s", len(products), products)
        ##for p in products:
        ##    logger.debug("%s --> %s -- %s", p, p.provider, p.properties)

        # First only keep "IW" sensor products with the expected polarisation
        # -> This filter is required with eodag < v1.6, it's redundant w/ v1.6+
        products = [p for p in products
                if (    product_property(p, "sensorMode",       "") == "IW"
                    and product_property(p, "polarizationMode", "") == polarization)
                ]
        logger.debug("%s remote S1 product(s) found and filtered (IW && %s): %s", len(products), polarization, products)
        if not products:  # no need to continue
            return []

        # Filter out products that either:
        # - already exist in the "cache"
        # logger.debug('Check products against the cache: %s', self.product_list)
        products = [p for p in products
                if not p.as_dict()['id'] in self.product_list
                ]
        logger.debug("%s remote S1 product(s) are not found in the cache: %s", len(products), products)
        if not products:  # no need to continue
            return []
        # - or for which we found matching dates
        #   Beware: a matching VV while the VH doesn't exist and is present in the
        #   remote product shall trigger the download of the product.
        #   TODO: We should actually inject the expected filenames into the task graph
        #   generator in order to download what is stricly necessary and nothing more
        polarizations = polarization.lower().split(' ')
        s2images_pat = 's1?_%s_*.tif' % (tile_name, )
        logger.debug('search %s for %s', s2images_pat, polarizations)
        s2images = glob.glob1(tile_out_dir, s2images_pat)
        products = [p for p in products
                if does_final_product_need_to_be_generated_for(p, tile_name, polarizations, s2images)
                ]

        # And finally download all!
        # TODO: register downloading into Dask
        logger.info("%s remote S1 product(s) will be downloaded: %s", len(products), products)
        if not products:  # no need to continue
            # Actually, in that special case we could almost detect there is nothing to do
            return []
        paths = dag.download_all(
                products[:],  # pass a copy because eodag modifies the list
                )
        # paths returns the list of .SAFE directories
        logger.info("Remote S1 products saved into %s", paths)
        # And clean temporary files
        for product in products:
            file = os.path.join(self.cfg.raw_directory, product.as_dict()['id']) + '.zip'
            try:
                logger.debug('Removing downloaded ZIP: %s', file)
                os.remove(file)
            except OSError:
                pass
        return paths

    def download_images(self, tiles=None):
        """ This method downloads the required images if download is True"""
        import numpy as np
        if not self.cfg.download:
            logger.info("Using images already downloaded, as per configuration request")
            return

        if self.roi_by_tiles is not None:
            if tiles:
                tiles_list = tiles
            elif "ALL" in self.roi_by_tiles:
                tiles_list = self.cfg.tiles_list
            else:
                tiles_list = self.roi_by_tiles
            logger.debug("Tiles requested to download: %s", tiles_list)

            layer = Layer(self.cfg.output_grid)
            for current_tile in layer:
                tile_name = current_tile.GetField('NAME')
                if tile_name in tiles_list:
                    tile_footprint = current_tile.GetGeometryRef().GetGeometryRef(0)
                    latmin = np.min([p[1] for p in tile_footprint.GetPoints()])
                    latmax = np.max([p[1] for p in tile_footprint.GetPoints()])
                    lonmin = np.min([p[0] for p in tile_footprint.GetPoints()])
                    lonmax = np.max([p[0] for p in tile_footprint.GetPoints()])
                    self._download(self._dag,
                            lonmin, lonmax, latmin, latmax,
                            self.fd, self.ld,
                            os.path.join(self.cfg.output_preprocess, tiles_list),
                            tile_name,
                            # tile_name+".txt" if self.cfg.cluster else None,
                            self.cfg.polarisation)
        else:  # roi_by_tiles is None
            # TODO: BUG: there is no current_tile/tile_name set in that case
            self._download(self._dag,
                    self.roi_by_coordinates[0], self.roi_by_coordinates[2],
                    self.roi_by_coordinates[1], self.roi_by_coordinates[3],
                    self.fd, self.ld,
                    os.path.join(self.cfg.output_preprocess, current_tile),
                    tile_name,
                    # tile_name+".txt" if self.cfg.cluster else None,
                    self.cfg.polarisation)
        self._update_s1_img_list()

    def _update_s1_img_list(self):
        """
        This method updates the list of S1 images available
        (from analysis of raw_directory)

        Returns:
           the list of S1 images available as instances
           of S1DateAcquisition class
        """

        self.raw_raster_list = []
        self.product_list    = []
        content = list_dirs(self.cfg.raw_directory)

        for current_content in content:
            # EODAG save SAFEs into {rawdir}/{prod}/{prod}.SAFE
            safe_dir = os.path.join(current_content.path, os.path.basename(current_content.path) + '.SAFE')
            if not os.path.isdir(safe_dir):
                continue

            self.product_list += [os.path.basename(current_content.path)]
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
            # logger.debug("%s", current_tile.GetField('NAME'))
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

        # Loop on images
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
                if intersection.GetArea() / tile_footprint.GetArea()\
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
        # date_exist = [os.path.basename(f)[21:21+8]
        #         for f in glob.glob(os.path.join(self.cfg.output_preprocess, tile_name_field, "s1?_*.tif"))]
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
            if len(image.get_images_list()) == 0:
                logger.critical("Problem with %s", image.get_manifest())
                logger.critical("Please remove the raw data for %s SAFE file", image.get_manifest())
                sys.exit(-1)

            # date_safe=os.path.basename(image.get_images_list()[0])[14:14+8]

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
                intersect_raster.append((image, [(point[0], point[1]) for point
                                                 in points[:-1]]))

        return intersect_raster

    def _get_mgrs_tile_geometry_by_name(self, mgrs_tile_name):
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
            logger.debug("Check SRTM tile for %s", tile)

            srtm_tiles = []
            mgrs_footprint = self._get_mgrs_tile_geometry_by_name(tile)
            area = mgrs_footprint.GetArea()
            srtm_layer.ResetReading()
            for srtm_tile in srtm_layer:
                srtm_footprint = srtm_tile.GetGeometryRef()
                intersection = mgrs_footprint.Intersection(srtm_footprint)
                if intersection.GetArea() > 0:
                    coverage = intersection.GetArea() / area
                    srtm_tiles.append((srtm_tile.GetField('FILE'), coverage))
            needed_srtm_tiles[tile] = srtm_tiles
        logger.info("SRTM ok")
        return needed_srtm_tiles

    def record_processed_filenames(self):
        """ Record the list of processed filenames (DEPRECATED)"""
        with open(os.path.join(self.cfg.output_preprocess,
                               "processed_filenames.txt"), "a") as in_file:
            for fic in self.processed_filenames:
                in_file.write(fic + "\n")

    def get_processed_filenames(self, cfg):
        """ Read back the list of processed filenames (DEPRECATED)"""
        try:
            with open(os.path.join(self.cfg.output_preprocess,
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
