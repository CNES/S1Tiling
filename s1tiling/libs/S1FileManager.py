#!/usr/bin/env python3
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
# - Luc HERMITTE (CSGROUP)
#
# =========================================================================

""" This module contains the S1FileManager class"""

from enum import Enum
import fnmatch
from functools import partial
import glob
import logging
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

from eodag.api.core import EODataAccessGateway
from eodag.utils.logging import setup_logging
from eodag.utils import get_geometry_from_various
try:
    from shapely.errors import TopologicalError
except ImportError:
    from shapely.geos import TopologicalError

import numpy as np

from s1tiling.libs import exits
from .Utils import get_shape, list_dirs, Layer, extract_product_start_time
from .S1DateAcquisition import S1DateAcquisition
from .otbpipeline import mp_worker_config

setup_logging(verbose=1)

logger = logging.getLogger('s1tiling')


class WorkspaceKinds(Enum):
    """
    Enum used to list the kinds of "workspaces" needed.
    A workspace is a directory where products will be stored.

    :todo: Use a more flexible and OCP (Open-Close Principle) compliant solution.
        Indeed At this moment, only two kinds of workspaces are supported.
    """
    TILE = 1
    LIA  = 2


def product_property(prod, key, default=None):
    """
    Returns the required (EODAG) product property, or default in the property isn't found.
    """
    res = prod.properties.get(key, default)
    return res


def product_cover(product, geometry):
    """
    Compute the coverage of the intersection of the product and the target geometry
    relativelly to the target geometry.
    Return a percentile in the range [0..100].

    This function has been extracted and adapted from
    :func:`eodag.plugins.crunch.filter_overlap.FilterOverlap.proceed`, which is
    under the Apache Licence 2.0.

    Unlike the original function, the actual filtering is done differenty and we
    only need the computed coverage. Also, we are not interrested in the
    coverage of the intersection relativelly to the input product.
    """
    search_geom = get_geometry_from_various(geometry=geometry)
    if product.search_intersection:
        intersection = product.search_intersection
        product_geometry = product.geometry
    elif product.geometry.is_valid:
        product_geometry = product.geometry
        intersection = search_geom.intersection(product_geometry)
    else:
        logger.debug(
                "Trying our best to deal with invalid geometry on product: %r",
                product,
                )
        product_geometry = product.geometry.buffer(0)
        try:
            intersection = search_geom.intersection(product_geometry)
        except TopologicalError:
            logger.debug(
                    "Product geometry still invalid. Force its acceptance"
                    )
        return 100

    ipos = (intersection.area / search_geom.area) * 100
    return ipos


# def unzip_images(raw_directory):
#     """This method handles unzipping of product archives"""
#     for file_it in list_files(raw_directory, '*.zip'):
#         logger.debug("unzipping %s", file_it.name)
#         try:
#             with zipfile.ZipFile(file_it.path, 'r') as zip_ref:
#                 zip_ref.extractall(raw_directory)
#         except zipfile.BadZipfile:
#             logger.warning("%s is corrupted. This file will be removed", file_it.path)
#         try:
#             os.remove(file_it.path)
#         except OSError:
#             pass


def does_final_product_need_to_be_generated_for(product, tile_name, polarizations, s2images):
    """
    Tells whether finals products associated to a tile needs to be generated.

    :param product:       S1 images that are available for download through EODAG
    :param tile_name:     Name of the S2 tile
    :param polarizations: Requested polarizations as per configuration.
    :param s2images:      List of already globbed S2 images files

    Searchs in `s2images` whether all the expected product filenames for the given S2 tile name
    and the requested polarizations exists.
    """
    logger.debug('Searching %s in %s', product, s2images)
    # e.g. id=S1A_IW_GRDH_1SDV_20200108T044150_20200108T044215_030704_038506_C7F5,
    prod_re = re.compile(r'(S1.)_IW_...._...._(\d{8})T\d{6}.*')
    sat, start = prod_re.match(product.as_dict()['id']).groups()
    for pol in polarizations:
        # e.g. s1a_{tilename}_{polarization}_DES_007_20200108txxxxxx.tif
        pat          = f'{sat.lower()}_{tile_name}_{pol}_*_{start}t??????.tif'
        pat_filtered = f'{sat.lower()}_{tile_name}_vh_*_{start}t??????_filtered.tif'
        found = fnmatch.filter(s2images, pat) or fnmatch.filter(s2images, pat_filtered)
        logger.debug('searching w/ %s and %s ==> Found: %s', pat, pat_filtered, found)
        if not found:
            return True
    return False


def filter_images_or_ortho(kind, all_images):
    """
    Analyses the existing orthorectified image files, or the ortho ready ones, for the input
    raster provided.
    This will be used to register the image raster to transform.
    """
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


def filter_images_providing_enough_cover_by_pair(products, target_cover, target_geometry, ident=None):
    """
    Associate products of the same date and orbit into pairs (at most),
    to compute the total coverage of the target zone.
    If the total coverage is inferior to the target coverage, the products
    are filtered out.
    """
    if not products or not target_cover:
        return products
    if not ident:
        ident = lambda name: name
    prod_re = re.compile(r'S1._IW_...._...._(\d{8})T\d{6}_\d{8}T\d{6}.*')
    kept_products = []
    date_grouped_products = {}
    logger.debug('Checking coverage for each product')
    for p in products:
        id    = ident(p)
        date  = prod_re.match(id).groups()[0]
        cover = product_cover(p, target_geometry)
        ron   = product_property(p, 'relativeOrbitNumber')
        dron  = f'{date}#{ron:03}'
        logger.debug('* @ %s, %s%% coverage for %s', dron, round(cover, 2), id)
        if dron not in date_grouped_products:
            date_grouped_products[dron] = {}
        date_grouped_products[dron].update({cover : p})

    logger.debug('Checking coverage for each date (and # relative orbit number)')
    for dron, cov_prod in date_grouped_products.items():
        covers         = cov_prod.keys()
        cov_sum        = round(sum(covers), 2)
        str_cov_to_sum = '+'.join((str(round(c, 2)) for c in covers))
        logger.debug('* @ %s -> %s%% = %s', dron, cov_sum, str_cov_to_sum)
        if cov_sum < target_cover:
            logger.warning('Reject products @ %s for insufficient coverage: %s=%s%% < %s%% %s',
                    dron, str_cov_to_sum, cov_sum, target_cover,
                    [ident(p) for p in cov_prod.values()])
        else:
            kept_products.extend(cov_prod.values())
    return kept_products


def discard_small_redundant(products, ident=None):
    """
    Sometimes there are several S1 product with the same start date, but a different end-date.
    Let's discard the smallest products
    """
    if not products:
        return products
    if not ident:
        ident = lambda name: name
    prod_re = re.compile(r'S1._IW_...._...._(\d{8}T\d{6})_(\d{8}T\d{6}).*')

    ordered_products = sorted(products, key=lambda p: ident(p))
    # logger.debug("all products before clean: %s", ordered_products)
    res = [ordered_products[0]]
    last, _ = prod_re.match(ident(res[0])).groups()
    for product in ordered_products[1:]:
        start, __unused = prod_re.match(ident(product)).groups()
        if last == start:
            # We can suppose the new end date to be >
            # => let's replace
            logger.warning('Discarding %s that is smallest than %s', res[-1], product)
            res[-1] = product
        else:
            res.append(product)
            last = start
    return res


def _download_and_extract_one_product(dag, raw_directory, product):
    """
    Takes care of downloading exactly one remote product and unzipping it,
    if required.

    Some products are already unzipped on the fly by eodag.
    """
    logging.info("Starting download of %s...", product)
    ok_msg = f"Successful download (and extraction) of {product}"  # because eodag'll clear product
    file = os.path.join(raw_directory, product.as_dict()['id']) + '.zip'
    try:
        path = dag.download(
                product,       # EODAG will clear this variable
                extract=True,  # Let's eodag do the job
                wait=1,        # Wait time in minutes between two download tries
                timeout=2      # Maximum time in mins before stop retrying to download (default=20’)
                )
        logging.debug(ok_msg)
        if os.path.exists(file) :
            try:
                logger.debug('Removing downloaded ZIP: %s', file)
                os.remove(file)
            except OSError:
                pass
    except BaseException:  # pylint: disable=broad-except
        logging.error('Failed to download (and extract) %s', product)
        path = None

    return path


def _parallel_download_and_extraction_of_products(
        dag, raw_directory, products, nb_procs, tile_name):
    """
    Takes care of downloading exactly all remote products and unzipping them,
    if required, in parallel.
    """
    paths = []
    log_queue = multiprocessing.Queue()
    log_queue_listener = logging.handlers.QueueListener(log_queue)
    dl_work = partial(_download_and_extract_one_product, dag, raw_directory)
    with multiprocessing.Pool(nb_procs, mp_worker_config, [log_queue]) as pool:
        log_queue_listener.start()
        try:
            for count, result in enumerate(pool.imap_unordered(dl_work, products), 1):
                logger.info("%s correctly downloaded", result)
                logger.info(' --> Downloading products for %s... %s%%', tile_name, count * 100. / len(products))
                paths.append(result)
        finally:
            pool.close()
            pool.join()
            log_queue_listener.stop()  # no context manager for QueueListener unfortunately

    # paths returns the list of .SAFE directories
    return paths


class S1FileManager:
    """ Class to manage processed files (downloads, checks) """
    def __init__(self, cfg):
        self.cfg              = cfg
        self.raw_raster_list  = []
        self.nb_images        = 0

        self.__tmpsrtmdir     = None
        self.__caching_option = cfg.cache_srtm_by
        assert self.__caching_option in ['copy', 'symlink']

        self.tiff_pattern     = "measurement/*.tiff"
        # self.vh_pattern       = "measurement/*vh*-???.tiff"
        # self.vv_pattern       = "measurement/*vv*-???.tiff"
        # self.hh_pattern       = "measurement/*hh*-???.tiff"
        # self.hv_pattern       = "measurement/*hv*-???.tiff"
        self.manifest_pattern = "manifest.safe"

        self._ensure_workspaces_exist()
        self.processed_filenames = self.get_processed_filenames()

        self.first_date = cfg.first_date
        self.last_date  = cfg.last_date
        self._update_s1_img_list()
        if self.cfg.download:
            logger.debug('Using %s EODAG configuration file', self.cfg.eodagConfig or 'user default')
            self._dag = EODataAccessGateway(self.cfg.eodagConfig)
            # TODO: update once eodag directly offers "DL directory setting" feature v1.7? +?
            dest_dir = os.path.abspath(self.cfg.raw_directory)
            logger.debug('Override EODAG output directory to %s', dest_dir)
            for provider in self._dag.providers_config.keys():
                if hasattr(self._dag.providers_config[provider], 'download'):
                    self._dag.providers_config[provider].download.update(
                            {'outputs_prefix': dest_dir})
                    logger.debug(' - for %s', provider)
                else:
                    logger.debug(' - NOT for %s', provider)

            self.roi_by_tiles = self.cfg.ROI_by_tiles

    def __enter__(self):
        """
        Turn the S1FileManager into a context manager, context acquisition function
        """
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
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
        for path in [self.cfg.raw_directory, self.cfg.tmpdir, self.cfg.output_preprocess]:
            if not os.path.isdir(path):
                os.makedirs(path, exist_ok=True)

    def ensure_tile_workspaces_exist(self, tile_name, required_workspaces):
        """
        Makes sure the directories used for :
        - output data/{tile},
        - temporary data/S2/{tile}
        - and LIA data (if required)
        all exist
        """
        working_directory = os.path.join(self.cfg.tmpdir, 'S2', tile_name)
        os.makedirs(working_directory, exist_ok=True)

        if WorkspaceKinds.TILE in required_workspaces:
            out_dir = os.path.join(self.cfg.output_preprocess, tile_name)
            os.makedirs(out_dir, exist_ok=True)

        # if self.cfg.calibration_type == 'normlim':
        if WorkspaceKinds.LIA in required_workspaces:
            lia_directory = self.cfg.lia_directory
            os.makedirs(lia_directory, exist_ok=True)

    def tmpsrtmdir(self, srtm_tiles_id, srtm_suffix='.hgt'):
        """
        Generate the temporary directory for SRTM tiles on the fly,
        and either populate it with symbolic links to the actual SRTM
        tiles, or copies of the actual SRTM tiles.
        """
        assert self.__caching_option in ['copy', 'symlink']
        if not self.__tmpsrtmdir:
            # copy all needed SRTM file in a temp directory for orthorectification processing
            self.__tmpsrtmdir = tempfile.TemporaryDirectory(dir=self.cfg.tmpdir)
            logger.debug('Create temporary SRTM diretory (%s) for needed tiles %s', self.__tmpsrtmdir.name, srtm_tiles_id)
            assert Path(self.__tmpsrtmdir.name).is_dir()
            for srtm_tile in srtm_tiles_id:
                srtm_tile_filepath=Path(self.cfg.srtm, srtm_tile + srtm_suffix)
                srtm_tile_filelink=Path(self.__tmpsrtmdir.name, srtm_tile + srtm_suffix)
                if self.__caching_option == 'symlink':
                    logger.debug('- ln -s %s <-- %s', srtm_tile_filepath, srtm_tile_filelink)
                    srtm_tile_filelink.symlink_to(srtm_tile_filepath)
                else:
                    logger.debug('- cp %s <-- %s', srtm_tile_filepath, srtm_tile_filelink)
                    shutil.copy2(srtm_tile_filepath, srtm_tile_filelink)

        return self.__tmpsrtmdir.name

    def keep_X_latest_S1_files(self, threshold):
        """
        Makes sure there is no more than `threshold`  S1 SAFEs in the raw directory.
        Oldest ones will be removed.
        """
        safefile_list = sorted(
                glob.glob(os.path.join(self.cfg.raw_directory, "*")),
                key=os.path.getctime)
        if len(safefile_list) > threshold:
            for safe in safefile_list[ : len(safefile_list) - threshold]:
                logger.debug("Remove old SAFE: %s", os.path.basename(safe))
                shutil.rmtree(safe, ignore_errors=True)
            self._update_s1_img_list()

    def _download(self, dag: EODataAccessGateway,
            lonmin, lonmax, latmin, latmax,
            first_date, last_date,
            tile_out_dir, tile_name,
            orbit_direction, relative_orbit_list, polarization, cover,
            searched_items_per_page,dryrun):
        """
        Process with the call to eodag download.
        """
        product_type = 'S1_SAR_GRD'
        extent = {
                'lonmin': lonmin,
                'lonmax': lonmax,
                'latmin': latmin,
                'latmax': latmax
                }
        products = []
        page = 1
        k_dir_assoc = { 'ASC': 'ascending', 'DES': 'descending' }
        while True:
            assert (not orbit_direction) or (orbit_direction in ['ASC', 'DES'])
            assert polarization in ['VV VH', 'VV', 'VH', 'HH HV', 'HH', 'HV']
            # In case only 'VV' or 'VH' is requested, we still need to
            # request 'VV VH' to the data provider through eodag.
            dag_polarization_param = 'VV VH' if polarization in ['VV VH', 'VV', 'VH'] else 'HH HV'
            dag_orbit_dir_param    = k_dir_assoc.get(orbit_direction, None)  # None => all
            dag_orbit_list_param   = relative_orbit_list[0] if len(relative_orbit_list) == 1 else None
            page_products, _ = dag.search(
                    page=page, items_per_page=searched_items_per_page,
                    productType=product_type,
                    start=first_date, end=last_date,
                    box=extent,
                    # If we have eodag v1.6, we try to filter product during the search request
                    polarizationMode=dag_polarization_param,
                    sensorMode="IW",
                    orbitDirection=dag_orbit_dir_param,        # None => all
                    relativeOrbitNumber=dag_orbit_list_param,  # List doesn't work. Single number yes!
                    )
            logger.info("%s remote S1 products returned in page %s: %s", len(page_products), page, page_products)
            products += page_products
            page += 1
            if len(page_products) < searched_items_per_page:
                break
        logger.debug("%s remote S1 products found: %s", len(products), products)
        ##for p in products:
        ##    logger.debug("%s --> %s -- %s", p, p.provider, p.properties)

        # Filter relative_orbits -- if it could not be done earier in the search() request.
        if len(relative_orbit_list) > 1:
            filtered_products = []
            for rel_orbit in relative_orbit_list:
                filtered_products.extend(products.filter_property(relativeOrbitNumber=rel_orbit))
            products = filtered_products

        # Final log
        extra_filter_log1 = ''
        if dag_orbit_dir_param:
            extra_filter_log1 = f'{dag_orbit_dir_param} '
        extra_filter_log2 = ''
        if len(relative_orbit_list) > 0:
            if len(relative_orbit_list) > 1:
                extra_filter_log2 = 's'
            extra_filter_log2 += ' ' + ', '.join([str(i) for i in relative_orbit_list])
        extra_filter_log = ''
        if extra_filter_log1 or extra_filter_log2:
            extra_filter_log = f' && {extra_filter_log1}orbit{extra_filter_log2}'
        logger.info("%s remote S1 product(s) found and filtered (IW && %s%s): %s", len(products), polarization, extra_filter_log, products)

        if not products:  # no need to continue
            return []

        # Filter out products that either:
        # - are overlapped by bigger ones
        #   Sometimes there are several S1 product with the same start
        #   date, but a different end-date.  Let's discard the
        #   smallest products
        products = discard_small_redundant(products, ident=lambda p: p.as_dict()['id'])
        logger.debug("%s remote S1 product(s) left after discarding smallest redundant ones: %s", len(products), products)
        # Filter cover
        if cover:
            products = filter_images_providing_enough_cover_by_pair(
                    products, cover, extent, ident=lambda p: p.as_dict()['id'])
            # products = products.filter_overlap(
            #         minimum_overlap=cover, geometry=extent)
            logger.debug("%s remote S1 product(s) found and filtered (cover >= %s): %s", len(products), cover, products)


        # - already exist in the "cache"
        # logger.debug('Check products against the cache: %s', self.product_list)
        products = [p for p in products
                if not p.as_dict()['id'] in self.product_list
                ]
        logger.debug("%s remote S1 product(s) are not yet in the cache: %s", len(products), products)
        if not products:  # no need to continue
            return []
        # - or for which we found matching dates
        #   Beware: a matching VV while the VH doesn't exist and is present in the
        #   remote product shall trigger the download of the product.
        #   TODO: We should actually inject the expected filenames into the task graph
        #   generator in order to download what is stricly necessary and nothing more
        polarizations = polarization.lower().split(' ')
        s2images_pat = f's1?_{tile_name}_*.tif'
        logger.debug('search %s for %s', s2images_pat, polarizations)
        s2images = glob.glob1(tile_out_dir, s2images_pat) + glob.glob1(os.path.join(tile_out_dir, "filtered"), s2images_pat)
        products = [p for p in products
                if does_final_product_need_to_be_generated_for(
                    p, tile_name, polarizations, s2images)
                ]

        # And finally download all!
        # TODO: register downloading into Dask
        logger.info("%s remote S1 product(s) will be downloaded", len(products))
        for p in products:
            logger.info('- %s: %s %s, [%s]', p,
                    product_property(p, "orbitDirection", ""),
                    product_property(p, "relativeOrbitNumber", ""),
                    product_property(p, "startTimeFromAscendingNode", ""),
                    )
        if not products:  # no need to continue
            # Actually, in that special case we could almost detect there is nothing to do
            return []
        if dryrun:
            paths = [p.as_dict()['id'] for p in products] # TODO: return real name
            logger.info("Remote S1 products would have been saved into %s", paths)
            return paths

        paths = _parallel_download_and_extraction_of_products(
                dag, self.cfg.raw_directory, products, self.cfg.nb_download_processes,
                tile_name)
        logger.info("Remote S1 products saved into %s", paths)
        return paths

    def download_images(self, searched_items_per_page, dryrun=False, tiles=None):
        """ This method downloads the required images if download is True"""
        if not self.cfg.download:
            logger.info("Using images already downloaded, as per configuration request")
            return

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
                        self.first_date, self.last_date,
                        os.path.join(self.cfg.output_preprocess, tiles_list),
                        tile_name,
                        orbit_direction=self.cfg.orbit_direction,
                        relative_orbit_list=self.cfg.relative_orbit_list,
                        polarization=self.cfg.polarisation,
                        cover=self.cfg.tile_to_product_overlap_ratio,
                        searched_items_per_page=searched_items_per_page,
                        dryrun=dryrun)
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
        content = list_dirs(self.cfg.raw_directory, 'S1*_IW_GRD*')  # ignore of .download on the-fly
        content = [d for d in content if self.is_product_in_time_range(d.path)]
        content = discard_small_redundant(content, ident=lambda d: d.name)

        for current_content in content:
            # EODAG save SAFEs into {rawdir}/{prod}/{prod}.SAFE
            logger.debug('current_content: %s', current_content)
            safe_dir = os.path.join(
                    current_content.path,
                    os.path.basename(current_content.path) + '.SAFE')
            if not os.path.isdir(safe_dir):
                continue

            self.product_list += [os.path.basename(current_content.path)]
            manifest = os.path.join(safe_dir, self.manifest_pattern)
            acquisition = S1DateAcquisition(manifest, [])
            all_tiffs = glob.glob(os.path.join(safe_dir, self.tiff_pattern))
            logger.debug("# Safe dir: %s", safe_dir)
            logger.debug("  all tiffs: %s", list(all_tiffs))

            vv_images = filter_images_or_ortho('vv', all_tiffs) if self.cfg.polarisation in ['VV', 'VV VH'] else []
            vh_images = filter_images_or_ortho('vh', all_tiffs) if self.cfg.polarisation in ['VH', 'VV VH'] else []
            hv_images = filter_images_or_ortho('hv', all_tiffs) if self.cfg.polarisation in ['HV', 'HH HV'] else []
            hh_images = filter_images_or_ortho('hh', all_tiffs) if self.cfg.polarisation in ['HH', 'HH HV'] else []

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
            if current_tile.GetField('NAME') == tile_name_field:
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
        for image in self.get_raster_list():
            manifest = image.get_manifest()
            poly = get_shape(manifest)

            for current_tile in layer:
                tile_footprint = current_tile.GetGeometryRef()
                intersection = poly.Intersection(tile_footprint)
                if intersection.GetArea() / tile_footprint.GetArea()\
                   > self.cfg.tile_to_product_overlap_ratio:
                    tile_name = current_tile.GetField('NAME')
                    if tile_name not in tiles:
                        tiles.append(tile_name)
        return tiles

    def is_product_in_time_range(self, product):
        """
        Returns whether the product name is within time range [first_date, last_date]
        """
        start_time = extract_product_start_time(product)
        if not start_time:
            return False
        start = '{YYYY}-{MM}-{DD}'.format_map(start_time)
        is_in_range = self.first_date <= start <= self.last_date
        logger.debug('  %s %s /// %s == %s <= %s <= %s', 'KEEP' if is_in_range else 'DISCARD',
                os.path.basename(product), is_in_range, self.first_date, start, self.last_date)
        return is_in_range

    def get_s1_intersect_by_tile(self, tile_name_field):
        """
        This method returns the list of S1 product intersecting a given MGRS tile

        Args:
          tile_name_field: The MGRS tile identifier

        Returns:
          A list of tuple (image as instance of
          S1DateAcquisition class, [corners]) for S1 products
          intersecting the given tile
        """
        logger.debug('Test intersections of %s', tile_name_field)
        # date_exist = [os.path.basename(f)[21:21+8]
        #    for f in glob.glob(os.path.join(self.cfg.output_preprocess, tile_name_field, "s1?_*.tif"))]
        intersect_raster = []

        layer = Layer(self.cfg.output_grid)
        current_tile = layer.find_tile_named(tile_name_field)
        if not current_tile:
            logger.info("Tile %s does not exist", tile_name_field)
            return intersect_raster

        tile_footprint = current_tile.GetGeometryRef()

        for image in self.get_raster_list():
            logger.debug('- Manifest: %s', image.get_manifest())
            logger.debug('  Image list: %s', image.get_images_list())
            if len(image.get_images_list()) == 0:
                logger.critical("Problem with %s", image.get_manifest())
                logger.critical("Please remove the raw data for %s SAFE file", image.get_manifest())
                sys.exit(exits.CORRUPTED_DATA_SAFE)

            manifest = image.get_manifest()
            poly = get_shape(manifest)

            intersection = poly.Intersection(tile_footprint)
            logger.debug('   -> Test intersection: requested: %s  VS tile: %s --> %s', poly, tile_footprint, intersection)
            if intersection.GetArea() != 0:
                area_polygon = tile_footprint.GetGeometryRef(0)
                points = area_polygon.GetPoints()
                # intersect_raster.append((image, [(point[0], point[1]) for point in points[:-1]]))
                intersect_raster.append( {
                    'raster': image,
                    'tile_origin': [(point[0], point[1]) for point in points[:-1]],
                    'tile_coverage': intersection.GetArea() / tile_footprint.GetArea()
                    })

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
        srtm_layer = Layer(self.cfg.srtm_db_filepath, driver_name='GPKG')

        needed_srtm_tiles = {}

        for tile in tiles_to_process:
            logger.debug("Check SRTM tile for %s", tile)

            srtm_tiles = []
            mgrs_footprint = self._get_mgrs_tile_geometry_by_name(tile)
            area = mgrs_footprint.GetArea()
            srtm_layer.reset_reading()
            for srtm_tile in srtm_layer:
                srtm_footprint = srtm_tile.GetGeometryRef()
                intersection = mgrs_footprint.Intersection(srtm_footprint)
                if intersection.GetArea() > 0:
                    coverage = intersection.GetArea() / area
                    srtm_tiles.append((srtm_tile.GetField('id'), coverage))
            needed_srtm_tiles[tile] = srtm_tiles
        logger.info("SRTM ok")
        return needed_srtm_tiles

    def record_processed_filenames(self):
        """ Record the list of processed filenames (DEPRECATED)"""
        with open(os.path.join(self.cfg.output_preprocess,
                               "processed_filenames.txt"), "a") as in_file:
            for fic in self.processed_filenames:
                in_file.write(fic + "\n")

    def get_processed_filenames(self):
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
