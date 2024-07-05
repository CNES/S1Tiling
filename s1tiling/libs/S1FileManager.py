#!/usr/bin/env python3
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
# - Luc HERMITTE (CSGROUP)
#
# =========================================================================

""" This module contains the S1FileManager class"""

from enum import Enum
import fnmatch
from functools import partial
import glob
import logging
import logging.handlers
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable, Dict, List, Optional, Tuple, Union

from osgeo import ogr
from requests.exceptions     import ReadTimeout
from eodag.api.core          import EODataAccessGateway
from eodag.api.product       import EOProduct
from eodag.api.search_result import SearchResult
from eodag.utils             import get_geometry_from_various
from eodag.utils.exceptions  import NotAvailableError
from eodag.utils.logging     import setup_logging
try:
    from shapely.errors import TopologicalError
except ImportError:
    from shapely.geos   import TopologicalError

import numpy as np

from s1tiling.libs      import exceptions
from .Utils             import (
    Layer,
    extract_product_start_time, find_dem_intersecting_poly, get_mgrs_tile_geometry_by_name,
    get_orbit_direction, get_relative_orbit, get_shape, list_dirs,
)
from .S1DateAcquisition import S1DateAcquisition
from .configuration     import (
        Configuration, dname_fmt_lia_product, dname_fmt_mask, dname_fmt_tiled, dname_fmt_filtered, fname_fmt_concatenation, fname_fmt_filtered
)
from .otbpipeline       import mp_worker_config
from .outcome           import DownloadOutcome

setup_logging(verbose=1)

logger = logging.getLogger('s1tiling.filemanager')


# Default configuration value for people using S1Tiling API functions s1_process, and s1_process_lia.
EODAG_DEFAULT_DOWNLOAD_WAIT         = 2   #: If download fails, wait time in minutes between two download tries
EODAG_DEFAULT_DOWNLOAD_TIMEOUT      = 20  #: If download fails, maximum time in minutes before stop retrying to download
EODAG_DEFAULT_SEARCH_MAX_RETRIES    = 5   #: If search fails on timeout, number of retries attempted
EODAG_DEFAULT_SEARCH_ITEMS_PER_PAGE = 20  #: Number of items returns by each page search


class WorkspaceKinds(Enum):
    """
    Enum used to list the kinds of "workspaces" needed.
    A workspace is a directory where products will be stored.

    :todo: Use a more flexible and OCP (Open-Close Principle) compliant solution.
        Indeed At this moment, only two kinds of workspaces are supported.
    """
    TILE   = 1
    LIA    = 2
    FILTER = 3
    MASK   = 4


def product_property(prod: EOProduct, key: str, default=None):
    """
    Returns the required (EODAG) product property, or default in the property isn't found.
    """
    res = prod.properties.get(key, default)
    return res


def product_cover(product: EOProduct, geometry: Dict[str, float]) -> float:
    """
    Compute the coverage of the intersection of the product and the target geometry
    relativelly to the target geometry.
    Return a percentage in the range [0..100].

    This function has been extracted and adapted from
    :func:`eodag.plugins.crunch.filter_overlap.FilterOverlap.proceed`, which is
    under the Apache Licence 2.0.

    Unlike the original function, the actual filtering is done differenty and we
    only need the computed coverage. Also, we are not interrested in the
    coverage of the intersection relativelly to the input product.
    """
    search_geom = get_geometry_from_various(geometry=geometry)
    assert search_geom, "Let's suppose eodag returns a geometry"
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
            logger.debug("Product geometry still invalid. Force its acceptance")
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


def does_final_product_need_to_be_generated_for(  # pylint: disable=too-many-locals
    product:       EOProduct,
    tile_name:     str,
    polarizations: List[str],
    cfg:           Configuration,
    s2images:      List[str]
) -> bool:
    """
    Tells whether finals products associated to a tile needs to be generated.

    :param product:       S1 images that are available for download through EODAG
    :param tile_name:     Name of the S2 tile
    :param polarizations: Requested polarizations as per configuration.
    :param s2images:      List of already globbed S2 images files

    Searchs in `s2images` whether all the expected product filenames for the given S2 tile name
    and the requested polarizations exists.
    """
    logger.debug('>  Searching whether %s final products have already been generated (in polarizations: %s)',
                 product, polarizations)
    if len(s2images) == 0:
        return True
    # e.g. id=S1A_IW_GRDH_1SDV_20200108T044150_20200108T044215_030704_038506_C7F5,
    prod_re = re.compile(r'(S1.)_IW_...._...._(\d{8})T\d{6}.*')
    pid = product.as_dict()['id']
    match = prod_re.match(pid)
    if not match:
        raise AssertionError(f"Unexpected name for S1 Product: {pid} doesn't match expected pattern")
    sat, start = match.groups()
    keys = {
        'flying_unit_code'  : sat.lower(),
        'tile_name'         : tile_name,
        'acquisition_stamp' : f'{start}t??????',
        'orbit_direction'   : '*',
        'orbit'             : '*',
        'calibration_type'  : cfg.calibration_type,
    }
    fname_fmt_4concatenation = fname_fmt_concatenation(cfg)
    fname_fmt_4filtered      = fname_fmt_filtered(cfg)
    for polarisation in polarizations:
        # e.g. s1a_{tilename}_{polarization}_DES_007_20200108txxxxxx.tif
        # We should use the `Processing.fname_fmt.concatenation` option
        pat          = fname_fmt_4concatenation.format(**keys, polarisation=polarisation)
        pat_filtered = fname_fmt_4filtered.format(**keys, polarisation=polarisation)
        found_s2 = fnmatch.filter(s2images, pat)
        found_filt = fnmatch.filter(s2images, pat_filtered)
        found = found_s2 or found_filt
        logger.debug('   searching w/ %s and %s ==> Found: %s', pat, pat_filtered, found)
        if not found:
            return True
        # FIXME:
        # - if found_s2 and not found_filt => we have everything that is needed
        # - if found_filt and not found_s2 => we have prevent the required S1 products from being downloaded
        #                                     if the S2 product is required
    return False


def filter_images_or_ortho(kind, all_images: List[str]) -> List[str]:
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


def _filter_images_providing_enough_cover_by_pair(  # pylint: disable=too-many-locals
    products:     Union[List[EOProduct], List[Dict]],  # EOProduct or content_info
    target_cover: float,
    ident:        Callable[[Union[EOProduct, Dict]], str],
    get_cover:    Callable[[Union[EOProduct, Dict]], float],
    get_orbit:    Callable[[Union[EOProduct, Dict]], int],
) -> Union[List[EOProduct], List[Dict]]:
    """
    Associate products of the same date and orbit into pairs (at most),
    to compute the total coverage of the target zone.
    If the total coverage is inferior to the target coverage, the products
    are filtered out.

    This function can be used on product information returned by EODAG as well
    as product information extracted from existing files. It's acheived thanks
    to the `ident`, `get_cover` and `get_orbit` variation points.
    """
    if not products or not target_cover:
        return products
    prod_re = re.compile(r'S1._IW_...._...._(\d{8})T\d{6}_\d{8}T\d{6}.*')
    kept_products : Union[List[EOProduct], List[Dict]] = []
    date_grouped_products : Dict[str, Dict[float, EOProduct]] = {}
    logger.debug('Checking coverage for each product')
    for p in products:
        pid   = ident(p)
        match = prod_re.match(pid)
        assert match
        date  = match.groups()[0]
        cover = get_cover(p)
        ron   = get_orbit(p)
        dron  = f'{date}#{ron:03}'
        logger.debug('* @ %s, %s%% coverage for %s', dron, round(cover, 2), pid)
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


def _keep_products_with_enough_coverage(
    content_info: List[Dict],
    target_cover: float,
    current_tile: ogr.Feature,
) -> List[Dict]:
    """
    Helper function that filters the products (/pairs of products) that provide
    enough coverage.
    It's meant to be used on product information extracted from existing files.
    """
    tile_footprint = current_tile.GetGeometryRef()
    area_polygon = tile_footprint.GetGeometryRef(0)
    points = area_polygon.GetPoints()
    origin = [(point[0], point[1]) for point in points[:-1]]
    content_info_with_intersection = []
    for ci in content_info:
        # p        = ci['product']
        # safe_dir = ci['safe_dir']
        if 'product_shape' not in ci:
            manifest = ci['manifest']
            poly = get_shape(manifest)
            ci['product_shape'] = poly
        else:
            poly = ci['product_shape']
        intersection = poly.Intersection(tile_footprint)
        ci['coverage']    = intersection.GetArea() / tile_footprint.GetArea() * 100
        ci['tile_origin'] = origin
        logger.debug('%s -> %s %% (inter %s / tile %s)',
                     ci['product'].name, ci['coverage'], intersection.GetArea(), tile_footprint.GetArea())
        if ci['coverage']:
            # If no intersection at all => we ignore!
            content_info_with_intersection.append(ci)

    return _filter_images_providing_enough_cover_by_pair(
            content_info_with_intersection, target_cover,
            ident=lambda ci: ci['product'].name,
            get_cover=lambda ci: ci['coverage'],
            get_orbit=lambda ci: ci['relative_orbit'],
    )


def _discard_small_redundant(
    products: Union[List[EOProduct], List[os.DirEntry]],
    ident:    Callable[[Union[EOProduct, os.DirEntry]], str],
) -> Union[List[EOProduct], List[os.DirEntry]]:
    """
    Sometimes there are several S1 product with the same start date, but a different end-date.
    Let's discard the smallest products
    """
    if not products:
        return products
    assert ident is not None, "Please call with a ident parameter!"
    prod_re = re.compile(r'S1._IW_...._...._(\d{8}T\d{6})_(\d{8}T\d{6}).*')

    ordered_products = sorted(products, key=ident)
    # logger.debug("all products before clean: %s", ordered_products)
    res = [ordered_products[0]]
    match = prod_re.match(ident(res[0]))
    assert match
    last, _ = match.groups()
    for product in ordered_products[1:]:
        match = prod_re.match(ident(product))
        assert match
        start, _ = match.groups()
        if last == start:
            # We can suppose the new end date to be >
            # => let's replace
            logger.warning('Discarding %s that is smallest than %s', res[-1], product)
            res[-1] = product
        else:
            res.append(product)
            last = start
    return res


def _keep_requested_orbits(
    content_info:           List[Dict],
    rq_orbit_direction:     Optional[str],
    rq_relative_orbit_list: List[int],
) -> List[Dict]:
    """
    Takes care of discarding products that don't match the requested orbit
    specification.

    Note: Beware that specifications could be contradictory and end up
    discarding everything.
    """
    if not rq_orbit_direction and not rq_relative_orbit_list:
        return content_info
    kept_products = []
    for ci in content_info:
        p         = ci['product']
        direction = ci['orbit_direction']
        orbit     = ci['relative_orbit']
        # logger.debug('CHECK orbit: %s / %s / %s', p, safe_dir, manifest)

        if rq_orbit_direction:
            if direction != rq_orbit_direction:
                logger.debug('Discard %s as its direction (%s) differs from the requested %s',
                        p.name, direction, rq_orbit_direction)
                continue
        if rq_relative_orbit_list:
            if orbit not in rq_relative_orbit_list:
                logger.debug('Discard %s as its orbit (%s) differs from the requested ones %s',
                        p.name, orbit, rq_relative_orbit_list)
                continue
        kept_products.append(ci)
    return kept_products


def _keep_requested_platforms(
    content_info: List[Dict],
    rq_platform_list: List[str]
) -> List[Dict]:
    """
    Takes care of discarding products that don't match the requested platform specification.

    Note: Beware that specifications could be contradictory and end up discarding everything.
    """
    if not rq_platform_list:
        return content_info
    kept_products = []
    for ci in content_info:
        p        = ci['product']
        platform = ci['platform']
        logger.debug('CHECK platform: %s / %s', p, platform)

        if rq_platform_list:
            if platform not in rq_platform_list:
                logger.debug('Discard %s as its platform (%s) differs from the requested ones %s',
                        p.name, platform, rq_platform_list)
                continue
        kept_products.append(ci)
    return kept_products


def _download_and_extract_one_product(
    dag:           EODataAccessGateway,
    raw_directory: str,
    dl_wait:       int,
    dl_timeout:    int,
    product:       EOProduct
) -> DownloadOutcome[str, EOProduct]:
    """
    Takes care of downloading exactly one remote product and unzipping it,
    if required.

    Some products are already unzipped on the fly by eodag.
    """
    logging.info("Starting download of %s...", product)
    ok_msg = f"Successful download (and extraction) of {product}"  # because eodag'll clear product
    prod_id = product.as_dict()['id']
    zip_file = os.path.join(raw_directory, prod_id) + '.zip'
    path: DownloadOutcome[str, EOProduct]
    try:
        path = DownloadOutcome(
                dag.download(
                    product,            # EODAG will clear this variable
                    extract=True,       # Let's eodag do the job
                    wait=dl_wait,       # Wait time in minutes between two download tries
                    timeout=dl_timeout  # Maximum time in mins before stop retrying to download (default=20’)
                ),
                product)
        logging.debug(ok_msg)
        if os.path.exists(zip_file) :
            try:
                logger.debug('Removing downloaded ZIP: %s', zip_file)
                os.remove(zip_file)
            except OSError:
                pass
        # eodag may say the product is correctly downloaded while it failed to do so
        # => let's do a quick sanity check
        manifest = os.path.join(raw_directory, prod_id, f'{prod_id}.SAFE', 'manifest.safe')
        if not os.path.exists(manifest):
            logger.error('Actually download of %s failed, the expected manifest could not be found in the product (%s)', prod_id, manifest)
            e = exceptions.CorruptedDataSAFEError(prod_id, f"no manifest file named {manifest!r} found")
            path = DownloadOutcome(e, product)
    except BaseException as e:  # pylint: disable=broad-except
        logger.warning('%s', e)  # EODAG error message is good and precise enough, just use it!
        # logger.error('Product is %s', product_property(product, 'storageStatus', 'online?'))
        logger.debug('Exception type is: %s', e.__class__.__name__)
        ## ERROR - Product is OFFLINE
        ## ERROR - Exception type is: NotAvailableError
        # logger.error('======================')
        # logger.exception(e)
        ## Traceback (most recent call last):
        ##   File "s1tiling/libs/S1FileManager.py", line 350, in _download_and_extract_one_product
        ##     path = DownloadOutcome(dag.download(
        ##   File "site-packages/eodag/api/core.py", line 1487, in download
        ##     path = product.download(
        ##   File "site-packages/eodag/api/product/_product.py", line 288, in download
        ##     fs_path = self.downloader.download(
        ##   File "site-packages/eodag/plugins/download/http.py", line 269, in download
        ##     raise NotAvailableError(
        ## eodag.utils.exceptions.NotAvailableError: S1A_IW_GRDH_1SDV_20200401T044214_20200401T044239_031929_03AFBC_0C9E
        ##                                           is not available (OFFLINE) and could not be downloaded, timeout reached

        path = DownloadOutcome(e, product)

    return path


def _parallel_download_and_extraction_of_products(  # pylint: disable=too-many-arguments, too-many-locals
    dag:           EODataAccessGateway,
    raw_directory: str,
    products:      List[EOProduct],
    nb_procs:      int,
    tile_name:     str,
    dl_wait:       int,
    dl_timeout:    int,
) -> List[DownloadOutcome]:
    """
    Takes care of downloading exactly all remote products and unzipping them,
    if required, in parallel.

    Returns :class:`DownloadOutcome` of :class:`EOProduct` or Exception.
    """
    nb_products = len(products)
    paths : List[DownloadOutcome] = []
    log_queue : multiprocessing.Queue = multiprocessing.Queue()
    log_queue_listener = logging.handlers.QueueListener(log_queue)
    dl_work = partial(_download_and_extract_one_product, dag, raw_directory, dl_wait, dl_timeout)
    with multiprocessing.Pool(nb_procs, mp_worker_config, [log_queue]) as pool:
        log_queue_listener.start()
        try:
            # In case timeout happens, we try again if and only if we have been able to
            # download other products after the timeout.
            # -> IOW, downloading instability justifies trying again.
            # /> On the contrary, on a complete network failure, we should not try again and again...
            while len(products) > 0:
                products_in_timeout : List[DownloadOutcome] = []
                nb_successes_since_timeout = 0
                for count, result in enumerate(pool.imap_unordered(dl_work, products), 1):
                    # logger.debug('DL -> %s', result)
                    if result:
                        logger.info("%s correctly downloaded", result.value())
                        logger.info(' --> Downloading products for %s... %s%%', tile_name, count * 100. / nb_products)
                        paths.append(result)
                        if len(products_in_timeout) > 0:
                            nb_successes_since_timeout += 1
                    else:
                        logger.warning("Cannot download %s: %s", result.related_product(), result.error())
                        # TODO: make it possible to detect missing products in the analysis
                        if isinstance(result.error(), ReadTimeout):
                            products_in_timeout.append(result)
                        else:
                            paths.append(result)
                products = []
                if nb_successes_since_timeout > nb_procs:
                    products = [r.related_product() for r in products_in_timeout]
                    logger.info("Attempting again to download %s products on timeout...", len(products))
                elif len(products_in_timeout) > 0:
                    paths.extend(products_in_timeout)
        finally:
            pool.close()
            pool.join()
            log_queue_listener.stop()  # no context manager for QueueListener unfortunately

    # paths returns the list of .SAFE directories
    return paths


class S1FileManager:
    """
    Class to manage processed files (downloads, checks)

    In a first step, all S1 products are found and filtered according to their
    date, and their orbit.

    Then, this list of all known products is filtered according to the target S2
    tile to retain only the S1 products that provide enough coverage.

    Eventually, the S1 products are scanned for the raster images of
    polarisation compatible with the requested one(s).
    """

    tiff_pattern     = "measurement/*.tiff"
    manifest_pattern = "manifest.safe"

    def __init__(self, cfg: Configuration) -> None:
        # Configuration
        self.cfg              = cfg
        self.__searched_items_per_page = getattr(cfg, 'searched_items_per_page', EODAG_DEFAULT_SEARCH_ITEMS_PER_PAGE)
        self.__nb_max_search_retries   = getattr(cfg, 'nb_max_search_retries',   EODAG_DEFAULT_SEARCH_MAX_RETRIES)
        self.__dl_wait                 = getattr(cfg, 'dl_wait',                 EODAG_DEFAULT_DOWNLOAD_WAIT)
        self.__dl_timeout              = getattr(cfg, 'dl_timeout',              EODAG_DEFAULT_DOWNLOAD_TIMEOUT)

        self.raw_raster_list  : List[S1DateAcquisition] = []
        self.nb_images        = 0

        # Failures related to download (e.g. missing products)
        self.__search_failures                = 0
        self.__download_failures              : List[DownloadOutcome]            = []
        self.__failed_S1_downloads_by_S2_uid  : Dict[str, List[DownloadOutcome]] = {}  # by S2 unique id: date + rel_orbit
        self.__skipped_S2_products            : List[str]                        = []

        self.__tmpdemdir      : Optional[tempfile.TemporaryDirectory] = None
        self.__caching_option = cfg.cache_dem_by
        assert self.__caching_option in ['copy', 'symlink']

        self._ensure_workspaces_exist()
        self.processed_filenames = self.get_processed_filenames()

        self.first_date = cfg.first_date
        self.last_date  = cfg.last_date
        self._refresh_s1_product_list()
        if self.cfg.download:
            logger.debug('Using %s EODAG configuration file', self.cfg.eodag_config or 'user default')
            self._dag = EODataAccessGateway(self.cfg.eodag_config)
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

            self.roi_by_tiles = self.cfg.roi_by_tiles

    def __enter__(self) -> "S1FileManager":
        """
        Turn the S1FileManager into a context manager, context acquisition function
        """
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """
        Turn the S1FileManager into a context manager, cleanup function
        """
        if self.__tmpdemdir:
            logger.debug('Cleaning temporary DEM diretory (%s)', self.__tmpdemdir)
            self.__tmpdemdir.cleanup()
            self.__tmpdemdir = None
        return False

    def get_skipped_S2_products(self) -> List[str]:
        """
        List of S2 products whose production will be skipped because of a
        download failure of a S1 product.
        """
        return self.__skipped_S2_products

    def get_search_failures(self) -> int:
        """Returns the number of times querying matching products failed"""
        return self.__search_failures

    def get_download_failures(self) -> List[DownloadOutcome]:
        """
        Returns the list of download failures as a list of :class:DownloadOutcome`
        """
        return self.__download_failures

    def get_download_timeouts(self) -> List[DownloadOutcome]:
        """
        Returns the list of download timeours as a list of :class:DownloadOutcome`
        """
        return list(filter(lambda f: isinstance(f.error(), NotAvailableError), self.__download_failures))

    def _ensure_workspaces_exist(self) -> None:
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

    def ensure_tile_workspaces_exist(self, tile_name: str, required_workspaces: List[WorkspaceKinds]) -> None:
        """
        Makes sure the directories used for :
        - output data/{tile},
        - temporary data/S2/{tile}
        - and LIA data (if required)
        all exist
        """
        directories = {
                'out_dir': self.cfg.output_preprocess,
                'tmp_dir': self.cfg.tmpdir,
                'lia_dir': self.cfg.lia_directory,
        }

        working_directory = os.path.join(self.cfg.tmpdir, 'S2', tile_name)
        os.makedirs(working_directory, exist_ok=True)

        if WorkspaceKinds.TILE in required_workspaces:
            wdir = dname_fmt_tiled(self.cfg).format(**directories, tile_name=tile_name)
            os.makedirs(wdir, exist_ok=True)

        if WorkspaceKinds.MASK in required_workspaces:
            wdir = dname_fmt_mask(self.cfg).format(**directories, tile_name=tile_name)
            os.makedirs(wdir, exist_ok=True)

        if WorkspaceKinds.FILTER in required_workspaces:
            wdir = dname_fmt_filtered(self.cfg).format(**directories, tile_name=tile_name)
            os.makedirs(wdir, exist_ok=True)

        # if self.cfg.calibration_type == 'normlim':
        if WorkspaceKinds.LIA in required_workspaces:
            wdir = dname_fmt_lia_product(self.cfg).format(**directories, tile_name=tile_name)
            os.makedirs(wdir, exist_ok=True)

    def tmpdemdir(self, dem_tile_infos: Dict, dem_filename_format: str, geoid_file: str) -> str:
        """
        Generate the temporary directory for DEM tiles on the fly,
        and either populate it with symbolic links to the actual DEM
        tiles, or copies of the actual DEM tiles.
        """
        assert self.__caching_option in ['copy', 'symlink']
        if not self.__tmpdemdir:
            # copy all needed DEM & geoid files in a temp directory for orthorectification processing
            self.__tmpdemdir = tempfile.TemporaryDirectory(dir=self.cfg.tmpdir)
            logger.debug('Create temporary DEM directory (%s) for needed tiles %s', self.__tmpdemdir.name, list(dem_tile_infos.keys()))
            assert Path(self.__tmpdemdir.name).is_dir()
            def do_symlink(src: Union[Path, str], dst: Path):
                logger.debug('- ln -s %s <-- %s', src, dst)
                dst.symlink_to(src)
            def do_copy(src: Union[Path, str], dst: Path):
                logger.debug('- cp %s --> %s', src, dst)
                shutil.copy2(src, dst)
            do_localize = do_symlink if self.__caching_option == 'symlink' else do_copy

            for _, dem_tile_info in dem_tile_infos.items():
                dem_file          = dem_filename_format.format_map(dem_tile_info)
                dem_tile_filepath = Path(self.cfg.dem, dem_file)
                dem_tile_filelink = Path(self.__tmpdemdir.name, os.path.basename(dem_file))  # for copernicus dem
                dem_tile_filelink.parent.mkdir(parents=True, exist_ok=True)
                do_localize(dem_tile_filepath, dem_tile_filelink)
            # + copy/link geoid
            geoid_filelink = Path(self.cfg.tmpdir, 'geoid', os.path.basename(geoid_file))
            if not geoid_filelink.exists():
                geoid_filelink.parent.mkdir(parents=True, exist_ok=True)
                do_localize(geoid_file, geoid_filelink)
                # in case there is an associated file like (egm96.grd.hdr), copy/symlink it as well
                if os.path.isfile(with_hdr := f"{geoid_file}.hdr"):
                    do_localize(with_hdr, geoid_filelink.with_suffix(geoid_filelink.suffix+'.hdr'))

        return self.__tmpdemdir.name

    def keep_X_latest_S1_files(self, threshold: int, tile_name: str) -> None:
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
            self._refresh_s1_product_list()  # TODO: decremental update
            self._update_s1_img_list_for(tile_name)

    def _search_products(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        dag:                            EODataAccessGateway,
        extent:                         Dict[str, float],
        first_date:                     str,
        last_date:                      str,
        platform_list:                  List[str],
        orbit_direction:                Optional[str],
        relative_orbit_list:            List[int],
        polarization:                   str,
        dryrun:                         bool,
    ) -> SearchResult:
        """
        Process with the call to eodag search.
        """
        product_type = 'S1_SAR_GRD'
        products = SearchResult(None)
        page = 1
        k_dir_assoc = { 'ASC': 'ascending', 'DES': 'descending' }
        assert (not orbit_direction) or (orbit_direction in ['ASC', 'DES'])
        assert polarization in ['VV VH', 'VV', 'VH', 'HH HV', 'HH', 'HV']
        # In case only 'VV' or 'VH' is requested, we still need to
        # request 'VV VH' to the data provider through eodag.
        dag_polarization_param  = 'VV VH' if polarization in ['VV VH', 'VV', 'VH'] else 'HH HV'
        dag_orbit_dir_param     = k_dir_assoc.get(orbit_direction, None)  # None => all
        dag_orbit_list_param    = relative_orbit_list[0] if len(relative_orbit_list) == 1 else None
        dag_platform_list_param = platform_list[0] if len(platform_list) == 1 else None
        while True:  # While we haven't analysed all search result pages
            timeout : Optional[ReadTimeout] = None
            for _ in range(self.__nb_max_search_retries):
                # Manual workaround https://github.com/CS-SI/eodag/issues/908
                try:
                    page_products, _ = dag.search(
                            page=page, items_per_page=self.__searched_items_per_page,
                            productType=product_type,
                            raise_errors=True,
                            start=first_date, end=last_date,
                            box=extent,
                            # If we have eodag v1.6+, we try to filter product during the search request
                            polarizationMode=dag_polarization_param,
                            sensorMode="IW",
                            orbitDirection=dag_orbit_dir_param,        # None => all
                            relativeOrbitNumber=dag_orbit_list_param,  # List doesn't work. Single number yes!
                            platformSerialIdentifier=dag_platform_list_param,
                    )
                    logger.info("%s remote S1 products returned in page %s: %s", len(page_products), page, page_products)
                    products.extend(page_products)
                    page += 1
                    break  # no need to try again
                except ReadTimeout as e:
                    timeout = e
            else:
                assert isinstance(timeout, ReadTimeout)
                raise RuntimeError(f"Product search has timeout'd {self.__nb_max_search_retries} times") from timeout

            if len(page_products) < self.__searched_items_per_page:
                break
        logger.debug("%s remote S1 products found: %s", len(products), products)
        ## for p in products:
        ##     logger.debug("%s --> %s -- %s", p, p.provider, p.properties)

        # Filter relative_orbits -- if it could not be done earlier in the search() request.
        if len(relative_orbit_list) > 1:
            filtered_products = SearchResult(None)
            for rel_orbit in relative_orbit_list:
                filtered_products.extend(products.filter_property(relativeOrbitNumber=rel_orbit))
            products = filtered_products

        # Filter platform -- if it could not be done earlier in the search() request.
        if len(platform_list) > 1:
            filtered_products = SearchResult(None)
            for platform in platform_list:
                filtered_products.extend(products.filter_property(platformSerialIdentifier=platform))
            products = filtered_products

        # Final log
        orbit_filter_log1 = ''
        if dag_orbit_dir_param:
            orbit_filter_log1 = f'{dag_orbit_dir_param} '
        orbit_filter_log2 = ''
        if len(relative_orbit_list) > 0:
            if len(relative_orbit_list) > 1:
                orbit_filter_log2 = 's'
            orbit_filter_log2 += ' ' + ', '.join([str(i) for i in relative_orbit_list])
        orbit_filter_log = ''
        if orbit_filter_log1 or orbit_filter_log2:
            orbit_filter_log = f'{orbit_filter_log1}orbit{orbit_filter_log2}'
        extra_filters = ['IW', polarization]
        if platform_list:
            extra_filters.append('|'.join(platform_list))
        if orbit_filter_log:
            extra_filters.append(orbit_filter_log)
        logger.info("%s remote S1 product(s) found and filtered (%s): %s", len(products),
                    " && ".join(extra_filters), products)

        return products

    def _filter_products(  # pylint: disable=too-many-arguments
        self,
        products:     List[EOProduct],
        extent:       Dict[str, float],
        tile_out_dir: str,
        tile_name:    str,
        polarization: str,
        cover:        float
    ) -> List[EOProduct]:
        """
        Filter products to download according to their polarization and coverage
        """
        if not products:  # no need to continue
            return []

        # Filter out products that either:
        # - are overlapped by bigger ones
        #   Sometimes there are several S1 product with the same start date, but a different end-date.
        #   Let's discard the smallest products
        def ident(p: EOProduct) -> str:
            # assert isinstance(p, EOProduct), f"Expecting a EOProduct, got: {type(p)!r}"
            return p.as_dict()['id']

        products = _discard_small_redundant(products, ident=ident)
        logger.debug("%s remote S1 product(s) left after discarding smallest redundant ones: %s", len(products), products)
        # Filter cover
        if cover:
            products = _filter_images_providing_enough_cover_by_pair(
                    products, cover,
                    ident=ident,
                    get_cover=lambda p: product_cover(p, extent),
                    get_orbit=lambda p: product_property(p, 'relativeOrbitNumber')
                    )
            # products = products.filter_overlap(
            #         minimum_overlap=cover, geometry=extent)
            logger.debug("%s remote S1 product(s) found and filtered (cover >= %s): %s", len(products), cover, products)

        # - already exist in the "cache"
        # logger.debug('Check products against the cache: %s', self.product_list)
        # self._refresh_s1_product_list()  # No need: as it has been done at startup, and after download
                                           # And let's suppose nobody deletd files
                                           # manually!
        products = list(filter(lambda p: ident(p) not in self._product_list, products))
        # products = [p for p in products if ident(p) not in self._product_list]
        # logger.debug('Products cache: %s', self._product_list.keys())
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
        logger.debug('Search %s for %s on disk in %s(/filtered)/%s', s2images_pat, polarizations, tile_out_dir, tile_name)

        def glob1(pat, *paths) -> List[str]:
            pathname = glob.escape(os.path.join(*paths))
            return [os.path.basename(p) for p in glob.glob(os.path.join(pathname, pat))]
        dname_options = { 'tile_name': tile_name, 'out_dir': tile_out_dir, }
        s2images = glob1(
                s2images_pat, dname_fmt_tiled(self.cfg).format_map(dname_options)
        ) + glob1(
                s2images_pat, dname_fmt_filtered(self.cfg).format_map(dname_options)
        )
        logger.debug(' => S2 products found on %s: %s', tile_name, s2images)
        products = [p for p in products
                if does_final_product_need_to_be_generated_for(
                    p, tile_name, polarizations, self.cfg, s2images)
                ]
        return products

    def _download(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        dag:                     EODataAccessGateway,
        lonmin:                  float,
        lonmax:                  float,
        latmin:                  float,
        latmax:                  float,
        first_date:              str,
        last_date:               str,
        tile_out_dir:            str,
        tile_name:               str,
        platform_list:           List[str],
        orbit_direction:         Optional[str],
        relative_orbit_list:     List[int],
        polarization:            str,
        cover:                   float,
        dryrun:                  bool,
    ) -> List[DownloadOutcome]:
        """
        Process with the call to eodag search + filter + download.

        :rtype: :class:`DownloadOutcome` of :class:`EOProduct` or Exception.
        :raises RuntimeError: If the search fails
        """
        extent = {
                'lonmin': lonmin,
                'lonmax': lonmax,
                'latmin': latmin,
                'latmax': latmax
                }
        try:
            sproducts = self._search_products(
                    dag, extent,
                    first_date, last_date, platform_list, orbit_direction, relative_orbit_list,
                    polarization, dryrun)
        except Exception as e:
            self.__search_failures += 1
            raise RuntimeError(f"Cannot request products for tile {tile_name} on data provider: {e}") from e

        products = self._filter_products(list(sproducts), extent, tile_out_dir, tile_name, polarization, cover)

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
            paths = [p.as_dict()['id'] for p in products]  # TODO: return real name
            logger.info("Remote S1 products would have been saved into %s", paths)
            return paths

        paths = _parallel_download_and_extraction_of_products(
                dag, self.cfg.raw_directory, products, self.cfg.nb_download_processes,
                tile_name,
                self.__dl_wait, self.__dl_timeout)
        logger.info("Remote S1 products saved into %s", [p.value() for p in paths if p.has_value()])
        return paths

    def download_images(
        self,
        dryrun: bool                = False,
        tiles:  Optional[List[str]] = None
    ) -> None:
        """ This method downloads the required images if download is True"""
        if not self.cfg.download:
            logger.info("Using images already downloaded, as per configuration request")
            return

        # TODO: Fix the logic behind these tests and the fonction interface/calls
        # -> i.e. download_images is always called with tiles=[one_tile_name]
        if tiles:
            tile_list = tiles
        elif "ALL" in self.roi_by_tiles:
            tile_list = self.cfg.tile_list
        else:
            tile_list = re.split(r'\s*,\s*', self.roi_by_tiles)
        logger.debug("Tiles requested to download: %s", tile_list)

        self.__failed_S1_downloads_by_S2_uid = {}  # Needs to be reset for each tile!
        downloaded_products: List[DownloadOutcome] = []
        layer = Layer(self.cfg.output_grid)  # TODO: This could be cached
        for current_tile in layer:
            name = current_tile.GetField('NAME')
            if not isinstance(name, str):
                raise AssertionError(f"Invalid data type for current tile NAME field: {name}. `str` was expected")
            tile_name : str = name
            if tile_name in tile_list:
                tile_footprint = current_tile.GetGeometryRef().GetGeometryRef(0)
                latmin = np.min([p[1] for p in tile_footprint.GetPoints()])
                latmax = np.max([p[1] for p in tile_footprint.GetPoints()])
                lonmin = np.min([p[0] for p in tile_footprint.GetPoints()])
                lonmax = np.max([p[0] for p in tile_footprint.GetPoints()])
                downloaded_products += self._download(self._dag,
                        lonmin, lonmax, latmin, latmax,
                        self.first_date, self.last_date,
                        tile_out_dir=self.cfg.output_preprocess,
                        tile_name=tile_name,
                        platform_list=self.cfg.platform_list,
                        orbit_direction=self.cfg.orbit_direction,
                        relative_orbit_list=self.cfg.relative_orbit_list,
                        polarization=self.cfg.polarisation,
                        cover=self.cfg.tile_to_product_overlap_ratio,
                        dryrun=dryrun)
        if downloaded_products:
            failed_products: List[DownloadOutcome] = list(filter(lambda p: not p, downloaded_products))
            if failed_products:
                self._analyse_download_failures(failed_products)
            success_products = [p.value() for p in filter(lambda p: p.has_value(), downloaded_products)]
            self._refresh_s1_product_list(success_products)  # incremental update

    def _analyse_download_failures(self, failed_products: List[DownloadOutcome]) -> None:
        """
        Record the download failures and mark S2 products that cannot be generated.
        """
        logger.warning('Some products could not be downloaded. Analysing donwload failures...')
        for fp in failed_products:
            logger.warning('* %s', fp.error())
            prod: EOProduct = fp.related_product()
            start_time = extract_product_start_time(prod.as_dict()['id'])
            day   = '{YYYY}{MM}{DD}'.format_map(start_time) if start_time else "????????"
            orbit = product_property(prod, 'relativeOrbitNumber')
            key = f'{day}#{orbit}'
            if key in self.__failed_S1_downloads_by_S2_uid:
                self.__failed_S1_downloads_by_S2_uid[key].append(fp)
            else:
                self.__failed_S1_downloads_by_S2_uid[key] = [fp]
            logger.debug('Register product to ignore: %s --> %s', key, self.__failed_S1_downloads_by_S2_uid[key])
        self.__download_failures.extend(failed_products)

    def _refresh_s1_product_list(self, new_products: Optional[List[EOProduct]] = None) -> None:
        """
        Scan all the available products and filter them according to:
        - platform requirements
        - orbit requirements
        - date requirements

        Todo: optimize the workflow:
            - remove product (from keep_X_latest_S1_files()
        """
        content = list_dirs(self.cfg.raw_directory, 'S1*_IW_GRD*')  # ignore of .download on the-fly
        logger.debug('%s local products found on disk', len(content))
        # Filter with new product only
        if new_products:
            logger.debug('new products:')
            for new_product in new_products:
                logger.debug('-> %s', new_product)
            # content is DirEntry
            # NEW is str!! Always
            # logger.debug('content[0]: %s -> %s', type(content[0]), content[0])
            # logger.debug('NEW[0]: %s -> %s', type(new_products[0]), new_products[0])
            # logger.debug('dirs found: %s', content)
            # If the directory appear directly
            content0 = content

            def in_new_products(d: os.DirEntry) -> bool:
                return d.path in new_products
            content = list(filter(in_new_products, content0))
            # Or if the directory appear with an indirection: e.g. {prod}/{prod}.SAFE
            # content += list(filter(lambda d: d.path in (p.parent for p in new_products), content0))
            parent_dirs = [os.path.dirname(p) for p in new_products]
            content += list(filter(lambda d: d.path in parent_dirs, content0))

            logger.debug('dirs found & filtered: %s', content)  # List(DirEntry)
            logger.debug("products DL'ed: %s", new_products)    # List(str)
            if len(content) != len(new_products):
                logger.warning(f'Not all new products are found in {self.cfg.raw_directory}: {new_products}. Some products downloaded may be corrupted.')
        else:
            self._product_list = {}
            self._products_info = []

        # Filter by date specification
        content = [d for d in content if self.is_product_in_time_range(d.name)]
        logger.debug('%s local products remaining in the specified time range', len(content))
        # Discard incomplete products (when the complete products are there)

        def ident(d: os.DirEntry) -> str:
            # assert isinstance(d, os.DirEntry), f"Expecting a DirEntry, got: {type(d)!r}"
            return d.name
        content = _discard_small_redundant(content, ident=ident)
        logger.debug('%s local products remaining after discarding incomplete and redundant products', len(content))

        # Build tuples of {product_dir, safe_dir, manifest_path,
        # orbit_direction, relative_orbit}
        products_info = [ {
            'product':  p,
            # EODAG saves SAFEs into {rawdir}/{prod}/{prod}.SAFE
            'safe_dir': os.path.join(p.path, p.name + '.SAFE'),
            } for p in content]
        products_info = list(filter(lambda ci: os.path.isdir(ci['safe_dir']), products_info))
        # TODO: filter corrupted products (e.g. .zip files that couldn't be correctly unzipped (because of a previous disk saturation for instance)

        for ci in products_info:
            manifest = os.path.join(ci['safe_dir'], self.manifest_pattern)
            ci['manifest']        = manifest
            ci['orbit_direction'] = get_orbit_direction(manifest)
            ci['relative_orbit']  = get_relative_orbit(manifest)
            ci['platform']        = ci['product'].name[:3]

        # Filter by orbit specification
        if self.cfg.orbit_direction or self.cfg.relative_orbit_list:
            products_info = _keep_requested_orbits(products_info,
                    self.cfg.orbit_direction, self.cfg.relative_orbit_list)
            logger.debug('%s local products remaining after filtering requested orbits', len(products_info))

        # Filter by platform specification
        if self.cfg.platform_list:
            products_info = _keep_requested_platforms(products_info, self.cfg.platform_list)
            logger.debug('%s local products remaining after filtering requested platforms (%s)',
                         len(products_info), ", ".join(self.cfg.platform_list))

        # Final log + extend "global" products_info with newly analysed ones
        if products_info:
            logger.debug('%s time, platform and orbit compatible products found on disk:', len(products_info))
            for ci in products_info:
                current_content = ci['product']
                logger.debug('* %s', current_content.name)
                self._product_list[current_content.name] = current_content
            self._products_info.extend(products_info)
        else:
            logger.warning('No time and orbit compatible products found on disk!')

    def _filter_complete_dowloads_by_pair(  # pylint: disable=too-many-locals
            self, tile_name: str, s1_products_info: List[Dict]
    ) -> List[Dict]:
        keys = {
            'tile_name'         : tile_name,
            'calibration_type'  : self.cfg.calibration_type,
        }
        fname_fmt_4concatenation = fname_fmt_concatenation(self.cfg)
        k_dir_assoc = { 'ascending': 'ASC', 'descending': 'DES' }
        ident     : Callable[[Dict], str] = lambda ci: ci['product'].name
        get_orbit : Callable[[Dict], int] = lambda ci: ci['relative_orbit']
        get_direc : Callable[[Dict], str] = lambda ci: k_dir_assoc.get(ci['orbit_direction'], ci['orbit_direction'])
        prod_re = re.compile(r'(S1.)_IW_...._...._(\d{8})T\d{6}_\d{8}T\d{6}.*')

        # We need to report every S2 product that could not be generated,
        # even if we have no S1 product associated. We cannot use s1_products_info
        # list for that purpose as it only contains S1 products that have been
        # successfully downloaded. => we iterate over the download blacklist
        for failure, missing in self.__failed_S1_downloads_by_S2_uid.items():
            # Reference missing product for the orbit + date
            # (we suppose there won't be a mix of S1A + S1B for the same pair)
            ref_missing_S1_product = missing[0].related_product()
            eo_ron  = product_property(ref_missing_S1_product, 'relativeOrbitNumber')
            eo_dir  = product_property(ref_missing_S1_product, 'orbitDirection')
            eo_dir  = k_dir_assoc.get(eo_dir, eo_dir)
            eo_id   = ref_missing_S1_product.as_dict()['id']
            match   = prod_re.match(eo_id)
            eo_date = match.groups()[1] if match else "????????"
            # Generate the reference name of S2 products that can't be produced
            keys['orbit_direction']   = eo_dir
            keys['orbit']             = f'{eo_ron:03}'  # 3 digits, pad w/ zeros
            match                     = prod_re.match(eo_id)
            keys['flying_unit_code']  = match.groups()[0].lower() if match else "S1?"
            keys['acquisition_stamp'] = f'{eo_date}txxxxxx'
            keys['polarisation']      = '*'
            s2_product_name = fname_fmt_4concatenation.format_map(keys)
            keeps   = []  # Workaround to filter out the current list.
            for ci in s1_products_info:
                pid   = ident(ci)
                match = prod_re.match(pid)
                date  = match.groups()[1] if match else "????????"
                ron   = get_orbit(ci)
                logger.debug('Check if the ignore-key %s matches the key (%s) of the paired S1 product %s', f'{date}#{ron}', failure, pid)
                if f'{date}#{ron}' == failure:
                    assert eo_date == date
                    assert eo_ron  == ron
                    assert eo_dir  == get_direc(ci), f"EO product: {eo_id} doesn't match product on disk: {pid}"
                    logger.debug('%s will be ignored to produce %s because: %s', ci, s2_product_name, missing)
                    # At most this could happen once as s1 products go by pairs,
                    # and thus a DL failure may be associated to zero or one DL success.
                    # assert len(self.__download_failures[failure]) == 1
                    # dont-break  # We don't actually need to continue except to... keep non "failure" products...
                else:
                    keeps.append(ci)
            s1_products_info = keeps
            logger.warning("Don't generate %s, because %s", s2_product_name, missing)
            self.__skipped_S2_products.append(
                    f'Download failure: {s2_product_name} cannot be produced because of the following issues with the inputs: {missing}')
        return s1_products_info

    def _filter_products_with_enough_coverage(self, tile_name: str, products_info: List[Dict]) -> List[Dict]:
        """
        Filter products (/pairs of products) that provide enough coverage for
        the requested tile.

        This function is meant to be a mock entry-point.
        """
        layer = Layer(self.cfg.output_grid)
        current_tile = layer.find_tile_named(tile_name)
        if not current_tile:
            logger.info("Tile %s does not exist", tile_name)
            return []
        products_info = _keep_products_with_enough_coverage(
                products_info, self.cfg.tile_to_product_overlap_ratio, current_tile)
        return products_info

    def _update_s1_img_list_for(  # pylint: disable=too-many-locals
            self, tile_name: str
    ) -> None:
        """
        This method updates the list of S1 images available
        (from analysis of raw_directory), and it keeps only the images (/pairs
        or images) that provide enough coverage.

        Returns:
           the list of S1 images available as instances
           of S1DateAcquisition class
        """
        self.raw_raster_list = []

        # Filter products not associated to offline/timeout-ed products
        # [p.properties["storageStatus"] for p in search_results]
        products_info = self._filter_complete_dowloads_by_pair(tile_name, self._products_info)
        logger.debug('%s products remaining after clearing out download failures: %s', len(products_info), products_info)

        # Filter products with enough coverage of the tile
        products_info = self._filter_products_with_enough_coverage(tile_name, products_info)

        # Finally, search for the files with the requested polarities only
        for ci in products_info:
            current_content = ci['product']
            safe_dir        = ci['safe_dir']
            manifest        = ci['manifest']
            logger.debug('current_content: %s', current_content)

            # self._product_list[current_content.name] = current_content
            acquisition = S1DateAcquisition(manifest, [], ci)
            all_tiffs = glob.glob(os.path.join(safe_dir, self.tiff_pattern))
            logger.debug("# Safe dir: %s", safe_dir)
            logger.debug("  all tiffs: %s", list(all_tiffs))

            l_vv, vv_images = self._filter_images_or_ortho_according_to_conf('vv', all_tiffs)
            l_vh, vh_images = self._filter_images_or_ortho_according_to_conf('vh', all_tiffs)
            l_hv, hv_images = self._filter_images_or_ortho_according_to_conf('hv', all_tiffs)
            l_hh, hh_images = self._filter_images_or_ortho_according_to_conf('hh', all_tiffs)

            for image in vv_images + vh_images + hv_images + hh_images:
                if image not in self.processed_filenames:
                    acquisition.add_image(image)
                    self.nb_images += 1
            if l_vv + l_vh + l_hv + l_hh == 0:
                # There is not a single file that would have been compatible
                # with what is expected
                logger.warning("Product associated to %s is corrupted: no VV, VH, HV or HH file found", manifest)
                raise exceptions.CorruptedDataSAFEError(current_content.name, f"no image files in {safe_dir!r} found")

            self.raw_raster_list.append(acquisition)

    def _filter_images_or_ortho_according_to_conf(self, polarisation: str, all_tiffs: List[str]) -> Tuple[int, List[str]]:
        """
        Helper function that returns the images compatible with the required
        polarisation, and if that polarisation has been requested in the
        configuration file.

        It also returns the number of file that would have been compatibles,
        independently of the requested polarisation. This will permit to control
        the SAFE contains what it's expected to contain.
        """
        k_polarisation_associations = {
            'vv' : ['VV', 'VV VH'],
            'vh' : ['VH', 'VV VH'],
            'hv' : ['HV', 'HH HV'],
            'hh' : ['HH', 'HH HV'],
        }
        all_images = filter_images_or_ortho(polarisation, all_tiffs)
        pol_images = all_images if self.cfg.polarisation in k_polarisation_associations[polarisation] else []
        return len(all_images), pol_images

    def tile_exists(self, tile_name_field: str) -> bool:
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
            if current_tile.GetField('NAME') == tile_name_field:
                return True
        return False

    def get_tiles_covered_by_products(self) -> List[str]:
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

    def is_product_in_time_range(self, product : str) -> bool:
        """
        Returns whether the product name is within time range [first_date, last_date]
        """
        assert '/' not in product, f"Expecting a basename for {product}"
        start_time = extract_product_start_time(product)
        if not start_time:
            return False
        start = '{YYYY}-{MM}-{DD}'.format_map(start_time)
        is_in_range = self.first_date <= start <= self.last_date
        logger.debug('  %s %s /// %s == %s <= %s <= %s', 'KEEP' if is_in_range else 'DISCARD',
                product, is_in_range, self.first_date, start, self.last_date)
        return is_in_range

    def get_s1_intersect_by_tile(self, tile_name_field: str) -> List[Dict]:
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

        # Get all the images that cover enough of the requested tile (the
        # coverage may be obtained with 2 concatenated images)
        self._update_s1_img_list_for(tile_name_field)

        for image in self.get_raster_list():
            logger.debug('- Manifest: %s', image.get_manifest())
            logger.debug('  Image list: %s', image.get_images_list())
            assert len(image.get_images_list()) > 0
            intersect_raster.append( {
                'raster'         : image,
                'tile_origin'    : image.product_info['tile_origin'],
                'tile_coverage'  : image.product_info['coverage'],
                # 'orbit_direction': get_orbit_direction(manifest),
                # 'orbit'          : '{:0>3d}'.format(get_relative_orbit(manifest)),
                })

        return intersect_raster

    def check_dem_coverage(self, tiles_to_process: List[str]) -> Dict[str, Dict]:
        """
        Given a set of MGRS tiles to process, this method
        returns the needed DEM tiles and the corresponding coverage.

        Args:
          tile_to_process: The list of MGRS tiles identifiers to process

        Return:
          A list of tuples (DEM tile id, coverage of MGRS tiles).
          Coverage range is [0,1]
        """
        dem_layer  = Layer(self.cfg.dem_db_filepath)
        mgrs_layer = Layer(self.cfg.output_grid)

        needed_dem_tiles = {}

        for tile in tiles_to_process:
            logger.debug("Check DEM tiles for %s", tile)
            mgrs_footprint = get_mgrs_tile_geometry_by_name(tile, mgrs_layer)
            logger.debug("%s original %s footprint is %s", tile, mgrs_footprint.GetSpatialReference().GetName(), mgrs_footprint)
            dem_tiles = find_dem_intersecting_poly(
                    mgrs_footprint, dem_layer, self.cfg.dem_field_ids, self.cfg.dem_main_field_id)
            needed_dem_tiles[tile] = dem_tiles
            logger.info("S2 tile %s is covered by %s DEM tiles", tile, len(dem_tiles))
        logger.info("DEM ok")
        return needed_dem_tiles

    def get_processed_filenames(self) -> List[str]:
        """ Read back the list of processed filenames (DEPRECATED)"""
        try:
            with open(
                    os.path.join(self.cfg.output_preprocess, "processed_filenames.txt"),
                    "r",
                    encoding="utf-8")as in_file:
                return in_file.read().splitlines()
        except (IOError, OSError):
            return []

    def get_raster_list(self) -> List[S1DateAcquisition]:
        """
        Get the list of raw S1 product rasters

        Returns:
          the list of raw rasters
        """
        return self.raw_raster_list
