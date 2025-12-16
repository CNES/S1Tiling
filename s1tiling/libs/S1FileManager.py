#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2025 (c) CNES.
#   Copyright 2022-2024 (c) CS GROUP France.
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

""" This module contains the S1FileManager class"""

from collections.abc import Iterable, Sequence
import fnmatch
import glob
import logging
import os
import re
import shutil
from types import ModuleType
from typing import Dict, List, Optional, Protocol, Tuple

from osgeo import ogr
from requests.exceptions     import ReadTimeout
from eodag.api.core          import EODataAccessGateway
from eodag.api.product       import EOProduct
from eodag.api.search_result import SearchResult
from eodag.utils.exceptions  import NotAvailableError, TimeOutError
from eodag.utils.logging     import setup_logging


from .                   import exceptions
from .utils.eodag        import (
    EODAG_DEFAULT_DOWNLOAD_TIMEOUT,
    EODAG_DEFAULT_DOWNLOAD_WAIT,
    EODAG_DEFAULT_SEARCH_ITEMS_PER_PAGE,
    EODAG_DEFAULT_SEARCH_MAX_RETRIES,
    download_and_extract_products,
)
from .Utils              import (
    Layer,
    extract_product_start_time,
    get_shape,
    regex_escape_dot,
    regex_join,
)
from .S1DateAcquisition  import S1DateAcquisition
# from .orbit._conversions import ORBIT_CONVERTERS
from .outcome            import S1DownloadOutcome
from .s1.filters         import (
    discard_small_redundant,
    filter_image_groups_providing_enough_cover_by_pair,
    filter_images_providing_enough_cover_by_pair,
    find_paired_products,
    keep_requested_orbits,
    keep_requested_platforms,
)
from .s1.product         import EOProductInformation, FileProductInformation, product_property
from .utils.timer        import timethis
from .utils.formatters   import ResilientFormatter
from .utils.layer        import footprint2extent
from .utils.path         import AnyPath, list_dirs, list_files

setup_logging(verbose=1)

logger = logging.getLogger('s1tiling.filemanager')


class S1FileManagerConfiguration(Protocol):
    """
    Specialized protocol for configuration information related :class:`S1FileManager` configuration data.

    Can be seen an a ISP compliant concept for Configuration object regarding S1 file managing.
    """
    first_date                   : str
    last_date                    : str
    download                     : bool
    roi_by_tiles                 : str
    raw_directory                : str
    tmpdir                       : str
    output_preprocess            : str
    extra_directories            : Dict[str, AnyPath]
    nb_download_processes        : int
    tile_list                    : List[str]
    output_grid                  : str
    platform_list                : List[str]
    orbit_direction              : Optional[str]
    relative_orbit_list          : List[int]
    polarisation                 : str
    tile_to_product_overlap_ratio: int
    calibration_type             : str
    fname_fmt                    : Dict
    dname_fmt                    : Dict


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

def iterate_on_filename_formats(
    fname_formats: Sequence[str],
    polarizations: Sequence[str],
    fname_options: Dict[str, str],
    *,
    build_a_regex: bool,
) -> Iterable[str]:
    """
    Given filename options, yields a regex for all the possible output filename formats.

    The filename options depend on output target (tile name, orbit direction...), and on
    metadata extracted from the name of the input product.
    """
    assert len(fname_formats) > 0, "No output filename formats have been registered!"
    any_char = '.*' if build_a_regex else '*'
    for fname_fmt in fname_formats:
        if build_a_regex:
            fname_fmt = regex_escape_dot(fname_fmt)
        if "polarisation" in fname_fmt:
            # Special case for polarisation: a same input may be used for several outputs
            for polarisation in polarizations:
                # logger.debug('yielding format from %s <-- %s', fname_fmt, fname_options)
                yield ResilientFormatter(any_char).format(fname_fmt, polarisation=polarisation, **fname_options)
        else:
            # logger.debug('yielding format from %s <-- %s', fname_fmt, fname_options)
            yield ResilientFormatter(any_char).format(fname_fmt, **fname_options)


def is_there_a_final_product_that_needs_to_be_generated_for_this_input(  # pylint: disable=too-many-arguments, too-many-locals
    input_product:            EOProductInformation,
    *,
    tile_name:                str,
    polarizations:            Sequence[str],
    calibration_type:         str,
    fname_formats:            Sequence[str],
    existing_output_products: List[str],
) -> bool:
    """
    Tells whether any of the theorical output products could depend on a specific input proudct.

    :param str       input_product:            The speficic input product considered
    :param str       tile_name:                Target tile name considered
    :param list[str] polarizations:            Target polarizations considered
    :param str       calibration_type:         Target calibration type considered
    :param list[str] fname_formats:            Filename formats of all possible output products
    :param list[str] existing_output_products: List of all the pre-existing output products already
                                               detected on disk.
    """
    pid = input_product.identifier
    logger.debug('>  Searching whether %r final products have already been generated (in polarizations: %s)',
                 pid, polarizations)
    if len(existing_output_products) == 0:
        logger.debug('  -> No known output => keep every possible input')
        return True
    ## # e.g. id=S1A_IW_GRDH_1SDV_20200108T044150_20200108T044215_030704_038506_C7F5,
    ## prod_re = re.compile(r'(S1.)_IW_...._...._(\d{8})T\d{6}_\d{8}T\d{6}_(\d{6})_.*')
    ## match = prod_re.match(pid)
    ## if not match:
    ##     raise AssertionError(f"Unexpected name for S1 Product: {pid!r} doesn't match expected pattern")
    ## sat, start, absolute_orbit = match.groups()
    ## if sat.upper() not in ORBIT_CONVERTERS.keys():
    ##     logger.warning("Platform %r is not supported by S1Tiling. Only %s are supported. Relative orbit will be ignored",
    ##                    sat, ORBIT_CONVERTERS.keys(), pid)
    ##     relative_orbit = r'\d{3}'
    ## else:
    ##     relative_orbit = ORBIT_CONVERTERS[sat.upper()].to_relative(int(absolute_orbit))
    sat            = input_product.platform
    start          = input_product.start_date
    # If the input remote product is known to be paired (according to the search result), we know to
    # search a txxxxxx date. Any date is accepted otherwise.
    time           = "xxxxxx" if input_product.is_appaired else '......'
    absolute_orbit = input_product.absolute_orbit
    relative_orbit = input_product.relative_orbit
    fname_options = {
        # --[ Keys coming from the request, to speed-up pattern matching
        'tile_name'         : tile_name,
        # There are 2 orbit directions: the requested one (if any), and the one from the remote file.
        # The one from the remote file matches the search criteria => we use it
        'orbit_direction'   : input_product.orbit_direction.short,
        'calibration_type'  : calibration_type,
        # --[ Keys coming from the reference input product, to match the related output products
        'flying_unit_code'  : sat.lower(),
        'acquisition_stamp' : f'{start}t{time}',
        'absolute_orbit'    : f"{absolute_orbit:06}",
        'orbit'             : f"{relative_orbit:03}",
    }

    # TODO: handle case where filtered output only need non-filtered output
    for fname_pattern in iterate_on_filename_formats(fname_formats, polarizations, fname_options, build_a_regex=True):
        logger.debug("   - check if there are actual outputs matching %r", fname_pattern)
        output_re = re.compile(fname_pattern, re.IGNORECASE)
        for op in existing_output_products:
            logger.debug("      -> %r ? -> %s  || w/ %r", op, re.match(output_re, op), output_re)
        if not any((re.match(output_re, op) for op in existing_output_products)):  # <=> none
            logger.debug("    => Can't find any related output => False")
            return True

    logger.debug("   => All possible outputs have been found for %r = True", pid)
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


def _filter_s1_images_required_for_expected_s2_product(  # pylint: disable=too-many-arguments, too-many-locals
    *,
    s1_products:         Sequence[EOProductInformation],
    tile_out_dir:        AnyPath,
    tile_name:           str,
    gamma_area_dir:      AnyPath,
    polarization:        str,
    orbit_direction:     Optional[str],
    relative_orbit_list: List[int],
    calibration_type:    str,
    name_formats:        List[Tuple[str, str]],  # Zip list of dname_fmt + fname_fmt
) -> Sequence[EOProductInformation]:
    # 1. First: list local output products matching the production criteria

    #   Beware: a matching VV while the VH doesn't exist and is present in the
    #   remote product shall trigger the download of the product.
    #   TODO: We should actually inject the expected filenames into the task graph
    #   generator in order to download what is stricly necessary and nothing more

    polarizations = polarization.lower().split(' ')
    logger.debug('Filter %s products for %s on disk in %s', tile_name, polarizations, tile_out_dir)
    def re_glob1(pat: str, *paths) -> List[str]:
        logger.debug('  Search %r on disk in %r', pat, "/".join(paths))
        re_glob = re.compile(f"^{pat}$", re.IGNORECASE)
        pathname = os.path.join(*paths)
        if os.path.isdir(pathname):
            return [p.name for p in list_files(pathname, re_glob)]
        else:
            logger.debug("Note: directory %r doesn't exist", pathname)
            return []

    dname_options = { 'tile_name': tile_name, 'out_dir': tile_out_dir, 'gamma_area_dir': gamma_area_dir, }
    fname_options = {
        'flying_unit_code': 'S1.',  # TODO: inject filter here!
        'tile_name'       : tile_name,
        'orbit_direction' : orbit_direction or '.*',  # Optional[str]
        'orbit'           : regex_join(relative_orbit_list, lambda o: f"{o:03}") if relative_orbit_list else '.*', # List[int]
        'calibration_type': calibration_type,  # needs to be case incensitive b/c of NormLim/GammaNaughtRTC
        'polarisation'    : f"({polarization.replace(' ', '|')})"
    }

    # First rough filter on possible products fname where we replace: tile_name, orbit, orbit dir
    existing_output_products : List[str] = []
    for dname_fmt, fname_fmt in name_formats:
        # NB replace unknown keys with ".*"
        output_product_pat = ResilientFormatter('.*').format(
            regex_escape_dot(fname_fmt),  # Need to escape dots, but not in keys!!!
            **fname_options,
        )
        existing_output_products.extend(
            re_glob1(output_product_pat, dname_fmt.format_map(dname_options))
        )
    logger.debug(' => related output products found on %s: %s', tile_name, existing_output_products)

    # Second exact filter taking into account anything else that may matter: polarization, orbit
    # number...
    logger.debug('Analyse %s products for %s on disk in %s', tile_name, polarizations, tile_out_dir)

    # 2. Then: filter out input products for which an output product has been found
    # Beware of the case an input is required for several outputs!
    s1_products = [
        # The idea is this time to extract S1 info from the inputs, and generate a new pattern that
        # will be used to filter inputs found
        # it will be mostly about the input dates, whether they are found in the output name
        # (some output have no dates (LIA, RTC...))
        # It's about what we can find in the input name:
        # - sat!
        # - start_date!
        # - absolute orbit?
        p for p in s1_products
        if is_there_a_final_product_that_needs_to_be_generated_for_this_input(
            input_product=p,
            tile_name=tile_name,
            polarizations=polarizations,
            calibration_type=calibration_type,
            fname_formats=[fname_fmt for _, fname_fmt in name_formats],
            existing_output_products=existing_output_products,
        )
    ]
    return s1_products


# @timethis("_keep_products_with_enough_coverage")  # This is fast enough
def _keep_products_with_enough_coverage(
    content_info: Sequence[FileProductInformation],
    target_cover: float,
    current_tile: ogr.Feature,
) -> Sequence[FileProductInformation]:
    """
    Helper function that filters the products (/pairs of products) that provide
    enough coverage.
    It's meant to be used on product information extracted from existing files.
    """
    tile_footprint = current_tile.GetGeometryRef()
    area_polygon = tile_footprint.GetGeometryRef(0)
    points = area_polygon.GetPoints()
    origin = [(point[0], point[1]) for point in points[:-1]]
    logger.debug("Analyse coverage in %s\n\t\torigin -> %s", content_info, origin)
    content_info_with_intersection = []
    for ci in content_info:
        ci.set_tile_origin(origin)
        cover = ci.compute_relative_cover_of(tile_footprint)
        if cover:
            # If no intersection at all => we ignore!
            content_info_with_intersection.append(ci)

    return filter_images_providing_enough_cover_by_pair(
            content_info_with_intersection,
            target_cover,
            get_cover=lambda ci: ci.get_current_tile_coverage() or 0
    )


def sanatize_S1_product(
    raw_directory: str,
    product:       EOProduct,
    logger_:       logging.Logger|ModuleType,
) -> Optional[Exception]:
    """
    Sanitize check for downloaded S1 products

    :return: an exception instance if an error has been detected, None otherwise
    """
    # eodag2 product naming scheme
    prod_id = product.as_dict()['id']
    manifest = os.path.join(raw_directory, prod_id, f'{prod_id}.SAFE', 'manifest.safe')
    if not os.path.exists(manifest):
        # eodag3 product naming scheme
        manifest = os.path.join(raw_directory, prod_id, 'manifest.safe')
        if not os.path.exists(manifest):
            logger_.error('  Actually download of %s failed, the expected manifest could not be found in the product (%s)', prod_id, manifest)
            e = exceptions.CorruptedDataSAFEError(prod_id, f"no manifest file named {manifest!r} found")
            return e
    return None


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

    def __init__(self, cfg: S1FileManagerConfiguration, dag: Optional[EODataAccessGateway]) -> None:
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
        self.__download_failures              : List[S1DownloadOutcome]            = []
        self.__failed_S1_downloads_by_S2_uid  : Dict[str, List[S1DownloadOutcome]] = {}  # by S2 unique id: date + rel_orbit
        self.__skipped_S2_products            : List[str]                          = []

        self._ensure_workspaces_exist()
        self.processed_filenames = self.get_processed_filenames()

        self.first_date = cfg.first_date
        self.last_date  = cfg.last_date
        self._refresh_s1_product_list()
        self._dag       = dag
        assert self.cfg.download == (self._dag is not None), f"EODAG object {dag=} expected when downloading is required {self.cfg.download=}"
        if self.cfg.download:
            self.roi_by_tiles = self.cfg.roi_by_tiles

    @property
    def dag(self) -> Optional[EODataAccessGateway]:
        """
        Return the internal instance of :class:`EODataAccessGateway`, or None if download is inhibited.
        """
        return self._dag

    @property
    def download_is_enabled(self) -> bool:
        """
        Returns whether download is enabled
        """
        assert self.cfg.download == (self._dag is not None), f"EODAG object {self._dag=} expected when downloading is required {self.cfg.download=}"
        return self._dag is not None

    def get_skipped_S2_products(self) -> List[str]:
        """
        List of S2 products whose production will be skipped because of a
        download failure of a S1 product.
        """
        return self.__skipped_S2_products

    def get_search_failures(self) -> int:
        """Returns the number of times querying matching products failed"""
        return self.__search_failures

    def get_download_failures(self) -> List[S1DownloadOutcome]:
        """
        Returns the list of download failures as a list of :class:S1DownloadOutcome`
        """
        return self.__download_failures

    def get_download_timeouts(self) -> List[S1DownloadOutcome]:
        """
        Returns the list of download timeours as a list of :class:S1DownloadOutcome`
        """
        return list(filter(lambda f: isinstance(f.error(), (NotAvailableError, TimeOutError)), self.__download_failures))

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

    def keep_X_latest_S1_files(
        self, threshold:     int,
        tile_name:           str,
        output_name_formats: List[Tuple[str, str]],
    ) -> None:
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
            self._update_s1_img_list_for(tile_name, output_name_formats)

    def _search_products(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        *,
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
        products = SearchResult([])
        page = 1
        k_dir_assoc = { 'ASC': 'ascending', 'DES': 'descending' }
        assert (not orbit_direction) or (orbit_direction in ['ASC', 'DES'])
        assert polarization in ['VV VH', 'VV', 'VH', 'HH HV', 'HH', 'HV']
        # In case only 'VV' or 'VH' is requested, we still need to
        # request 'VV VH' to the data provider through eodag.
        dag_polarization_param  = 'VV+VH' if polarization in ['VV VH', 'VV', 'VH'] else 'HH+HV'
        dag_orbit_dir_param     = k_dir_assoc.get(orbit_direction or "", None)  # None => all ; <<or "">> used to silence mypy
        dag_orbit_list_param    = relative_orbit_list[0] if len(relative_orbit_list) == 1 else None
        dag_platform_list_param = platform_list[0] if len(platform_list) == 1 else None
        while True:  # While we haven't analysed all search result pages
            timeout : Optional[ReadTimeout] = None
            for _ in range(self.__nb_max_search_retries):
                # Manual workaround https://github.com/CS-SI/eodag/issues/908
                try:
                    page_products = dag.search(
                            page=page, items_per_page=self.__searched_items_per_page,
                            productType=product_type,
                            raise_errors=True,
                            start=first_date, end=last_date,
                            box=extent,
                            # If we have eodag v1.6+, we try to filter product during the search request
                            polarizationChannels=dag_polarization_param,
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
            filtered_products = SearchResult([])
            for rel_orbit in relative_orbit_list:
                filtered_products.extend(products.filter_property(relativeOrbitNumber=rel_orbit))
            products = filtered_products

        # Filter platform -- if it could not be done earlier in the search() request.
        if len(platform_list) > 1:
            filtered_products = SearchResult([])
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

    def _filter_products_to_download(  # pylint: disable=too-many-arguments
        self,
        *,
        products:            Sequence[EOProductInformation],
        extent:              Dict[str, float],
        tile_out_dir:        AnyPath,
        gamma_area_dir:      AnyPath,
        tile_name:           str,
        polarization:        str,
        cover:               float,
        output_name_formats: List[Tuple[str, str]],
    ) -> Sequence[EOProductInformation]:
        """
        Filter products to download according to their:
        - polarization,
        - coverage,
        - presence in the cache (disk),
        - and whether we do need them to generate the expected product.
        """
        if not products:  # no need to continue
            return []

        # Filter out products that either:
        # - are overlapped by bigger ones
        #   Sometimes there are several S1 product with the same start date, but a different end-date.
        #   Let's discard the smallest products
        products = discard_small_redundant(products)
        logger.info ("%s remote S1 product(s) left after discarding smallest redundant products", len(products))
        logger.debug(" => %s", [f"{p}" for p in products])

        # Find pairs of S1 products
        product_groups = find_paired_products(products)

        # Filter cover
        if cover:
            products = filter_image_groups_providing_enough_cover_by_pair(
                product_groups,
                cover,
                get_cover=lambda p: p.compute_relative_cover_of(extent)
            )
            logger.info ("%s remote S1 product(s) found and filtered (cover >= %s)", len(products), cover)
            logger.debug(" => %s", [f"{p}" for p in products])
        else:
            logger.debug("No coverage check is performed")

        # - already exist in the "cache"
        # logger.debug('Check products against the cache: %s', self.product_list)
        # self._refresh_s1_product_list()  # No need: as it has been done at startup, and after download
                                           # And let's suppose nobody has deleted files manually!
        products = list(filter(lambda p: p.identifier not in self._product_list, products))
        # logger.debug('Products cache: %s', self._product_list.keys())
        logger.info ("%s remote S1 product(s) are not yet in the (local disk) cache", len(products))
        logger.debug(" => %s", [f"{p}" for p in products])
        if not products:  # no need to continue
            return []
        # - or for which we found matching dates

        products = _filter_s1_images_required_for_expected_s2_product(
            s1_products=products,
            tile_out_dir=tile_out_dir,
            tile_name=tile_name,
            gamma_area_dir=gamma_area_dir,
            polarization=polarization,
            orbit_direction=self.cfg.orbit_direction,
            relative_orbit_list=self.cfg.relative_orbit_list,
            calibration_type=self.cfg.calibration_type,
            name_formats=output_name_formats,
        )
        logger.info ("%s remote S1 product(s) for which some output products are missing", len(products))
        # logger.debug(" => %s", [ident(p) for p in products])
        logger.debug(" => %s", [f"{p}" for p in products])
        return products

    def _download(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        *,
        dag:                     EODataAccessGateway,
        extent:                  Dict[str, float],
        first_date:              str,
        last_date:               str,
        tile_out_dir:            AnyPath,
        gamma_area_dir:          AnyPath,
        tile_name:               str,
        platform_list:           List[str],
        orbit_direction:         Optional[str],
        relative_orbit_list:     List[int],
        polarization:            str,
        cover:                   float,
        output_name_formats:     List[Tuple[str, str]],
        dryrun:                  bool,
    ) -> List[S1DownloadOutcome]:
        """
        Process with the call to eodag search + filter + download.

        :rtype: :class:`S1DownloadOutcome` of :class:`EOProduct` or Exception.
        :raises RuntimeError: If the search fails
        """
        try:
            sproducts = self._search_products(
                dag=dag,
                extent=extent,
                first_date=first_date,
                last_date=last_date,
                platform_list=platform_list,
                orbit_direction=orbit_direction,
                relative_orbit_list=relative_orbit_list,
                polarization=polarization,
                dryrun=dryrun,
            )
        except Exception as e:
            self.__search_failures += 1
            raise RuntimeError(f"Cannot request products for tile {tile_name} on data provider: {e}") from e

        products = self._filter_products_to_download(
            products=[EOProductInformation(p) for p in sproducts],
            extent=extent,
            tile_out_dir=tile_out_dir,
            gamma_area_dir=gamma_area_dir,
            tile_name=tile_name,
            polarization=polarization,
            cover=cover,
            output_name_formats=output_name_formats,
        )

        # And finally download all!
        logger.info("%s remote S1 product(s) will be downloaded", len(products))
        for p in products:
            logger.info('- %s: %s %03d, [%s]', p,
                        p.orbit_direction.name,
                        p.relative_orbit,
                        p.start_time,
            )
        if not products:  # no need to continue
            # Actually, in that special case we could almost detect there is nothing to do
            return []
        if dryrun:
            paths = [S1DownloadOutcome(p.identifier, p) for p in products]  # TODO: return real name
            logger.info("Remote S1 products would have been saved into %s", paths)
            return paths

        eo_products = [p.product for p in products]
        paths = download_and_extract_products(
            dag=dag,
            output_dir=self.cfg.raw_directory,
            products=eo_products,
            nb_procs=self.cfg.nb_download_processes,
            context=f" for {tile_name}",
            dl_wait=self.__dl_wait,
            dl_timeout=self.__dl_timeout,
            sanatize=sanatize_S1_product,
        )
        logger.info("Remote S1 products saved into %s", [p.value() for p in paths if p.has_value()])
        logger.debug("Problems observed during DL: %s", [p.error() for p in paths if not p.has_value()])
        return paths

    @timethis("Downloading images related to {tiles}", logging.INFO)
    def download_images(
        self,
        output_name_formats: List[Tuple[str, str]],
        dryrun:              bool                  = False,
        tiles:               Optional[List[str]]   = None
    ) -> None:
        """ This method downloads the required images if download is True"""
        if not self.download_is_enabled:
            logger.info("Using images already downloaded, as per configuration request")
            return
        assert self._dag  # Silence pyright warning

        # TODO: Fix the logic behind these tests and the function interface/calls
        # -> i.e. download_images is always called with tiles=[one_tile_name]
        if tiles:
            tile_list = tiles
        elif "ALL" in self.roi_by_tiles:
            tile_list = self.cfg.tile_list
        else:
            tile_list = re.split(r'\s*,\s*', self.roi_by_tiles)
        logger.debug("Tiles requested to download: %s", tile_list)

        self.__failed_S1_downloads_by_S2_uid = {}  # Needs to be reset for each tile!
        downloaded_products: List[S1DownloadOutcome] = []
        layer = Layer(self.cfg.output_grid)  # TODO: This could be cached
        for current_tile in layer:
            name = current_tile.GetField('NAME')
            if not isinstance(name, str):
                raise AssertionError(f"Invalid data type for current tile NAME field: {name}. `str` was expected")
            tile_name : str = name
            if tile_name in tile_list:
                tile_footprint = current_tile.GetGeometryRef().GetGeometryRef(0)
                downloaded_products += self._download(
                    dag=self._dag,
                    extent=footprint2extent(tile_footprint),
                    first_date=self.first_date, last_date=self.last_date,
                    tile_out_dir=self.cfg.output_preprocess,
                    gamma_area_dir=self.cfg.extra_directories['gamma_area_dir'],
                    tile_name=tile_name,
                    platform_list=self.cfg.platform_list,
                    orbit_direction=self.cfg.orbit_direction,
                    relative_orbit_list=self.cfg.relative_orbit_list,
                    polarization=self.cfg.polarisation,
                    cover=self.cfg.tile_to_product_overlap_ratio,
                    output_name_formats=output_name_formats,
                    dryrun=dryrun)
        if downloaded_products:
            failed_products: List[S1DownloadOutcome] = list(filter(lambda p: not p, downloaded_products))
            if failed_products:
                self._analyse_download_failures(failed_products)
            success_products = [p.value() for p in filter(lambda p: p.has_value(), downloaded_products)]
            self._refresh_s1_product_list(success_products)  # incremental update

    def _analyse_download_failures(self, failed_products: List[S1DownloadOutcome]) -> None:
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
            logger.debug('  -> Register product to ignore: %s --> %s', key, self.__failed_S1_downloads_by_S2_uid[key])
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
            self._products_info : List[FileProductInformation] = []

        # Filter by date specification
        # logger.debug('  Checking product in time range: %s .. %s', self.first_date, self.last_date)
        # for d in content:
        #     logger.debug('  - %r -> %s', d.name, self.is_product_in_time_range(d.name))
        content = [d for d in content if self.is_product_in_time_range(d.name)]

        logger.debug('%s local products remaining in the specified time range', len(content))
        # Discard incomplete products (when the complete products are there)

        products_info = FileProductInformation.filter_valid_products(content)

        logger.debug('%s local products remaining after filtering valid manifests', len(products_info))
        # TODO: filter corrupted products (e.g. .zip files that couldn't be correctly unzipped (because of a previous disk saturation for instance)

        # Filter by orbit specification
        if self.cfg.orbit_direction or self.cfg.relative_orbit_list:
            products_info = keep_requested_orbits(
                products_info,
                self.cfg.orbit_direction,
                self.cfg.relative_orbit_list)
            logger.debug('%s local products remaining after filtering requested orbits', len(products_info))

        # Filter by platform specification
        if self.cfg.platform_list:
            products_info = keep_requested_platforms(products_info, self.cfg.platform_list)
            logger.debug('%s local products remaining after filtering requested platforms (%s)',
                         len(products_info), ", ".join(self.cfg.platform_list))

        # Final log + extend "global" products_info with newly analysed ones
        if products_info:
            logger.debug('%s time, platform and orbit compatible products found on disk:', len(products_info))
            for ci in products_info:
                current_content = ci.product
                logger.debug('* %s', current_content.name)
                self._product_list[current_content.name] = current_content
            self._products_info.extend(products_info)
        else:
            logger.warning('No time and orbit compatible products found on disk!')

    def _filter_complete_dowloads_by_pair(  # pylint: disable=too-many-locals
        self,
        tile_name:           str,
        s1_products_info:    Sequence[FileProductInformation],
        output_name_formats: List[Tuple[str, str]],
    ) -> Sequence[FileProductInformation]:
        fname_formats=[fname_fmt for _, fname_fmt in output_name_formats]
        keys = {
            'tile_name'         : tile_name,
            'calibration_type'  : self.cfg.calibration_type,
        }
        assert output_name_formats, "Empty output filename formats"
        k_dir_assoc = {
            'ascending' : 'ASC',  # some eo providers return ascending/descending
            'descending': 'DES',
            'Ascending' : 'ASC',  # some eo providers return Ascending/Descending
            'Descending': 'DES',
            'ASCENDING' : 'ASC',  # some eo providers return ASCENDING/DESCENDING
            'DESCENDING': 'DES',
        }
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
            assert eo_ron, (
                f"Product information misses 'relativeOrbitNumber', only {ref_missing_S1_product.properties.keys()} are available, and {ref_missing_S1_product.properties['orbitNumber']=}"
            )
            eo_dir  = product_property(ref_missing_S1_product, 'orbitDirection', '')
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
            # TODO: Find a way to not report error when we have enough S1 products for γ-areas
            #       In those cases, any valid pair is enough. Missing pairs can be disregarded
            keeps   = []  # Workaround to filter out the current list.
            s2_product_names = list (iterate_on_filename_formats(fname_formats, ('*'), keys, build_a_regex=False))
            for ci in s1_products_info:
                pid   = ci.identifier
                match = prod_re.match(pid)
                date  = match.groups()[1] if match else "????????"
                ron   = ci.relative_orbit
                logger.debug('* Check if the ignore-key %s matches the key (%s) of the paired S1 product %s', f'{date}#{ron}', failure, pid)
                if f'{date}#{ron}' == failure:
                    assert eo_date == date
                    assert eo_ron  == ron
                    assert eo_dir  == ci.orbit_direction.short, f"EO product: {eo_id} doesn't match product on disk: {pid}"
                    logger.debug('  -> %s will be ignored to produce %s because: %s', ci, ' and '.join(s2_product_names), missing)
                    # At most this could happen once as s1 products go by pairs,
                    # and thus a DL failure may be associated to zero or one DL success.
                    # assert len(self.__download_failures[failure]) == 1
                    # dont-break  # We don't actually need to continue except to... keep non "failure" products...
                else:
                    keeps.append(ci)
            s1_products_info = keeps
            for s2_product_name in s2_product_names:
                logger.warning("Don't generate %s, because %s", s2_product_name, missing)
                self.__skipped_S2_products.append(
                        f'Download failure: {s2_product_name!r} cannot be produced because of the following issues with the inputs: {missing}')
        return s1_products_info

    @timethis("_filter_products_with_enough_coverage({tile_name})")
    def _filter_products_with_enough_coverage(
        self,
        tile_name: str,
        products_info: Sequence[FileProductInformation],
    ) -> Sequence[FileProductInformation]:
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
        logger.debug('calling _keep_products_with_enough_coverage(tgt=%s)', self.cfg.tile_to_product_overlap_ratio)
        products_info = _keep_products_with_enough_coverage(
                products_info, self.cfg.tile_to_product_overlap_ratio, current_tile)
        return products_info

    def _update_s1_img_list_for(  # pylint: disable=too-many-locals
            self,
            tile_name:           str,
            output_name_formats: List[Tuple[str, str]],
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
        products_info = self._filter_complete_dowloads_by_pair(tile_name, self._products_info, output_name_formats)
        logger.debug('%s products remaining after clearing out download failures: %s', len(products_info), products_info)

        # Filter products with enough coverage of the tile
        products_info = self._filter_products_with_enough_coverage(tile_name, products_info)

        # Finally, search for the files with the requested polarities only
        for ci in products_info:
            current_content = ci.product
            safe_dir        = ci.safe_dir
            manifest        = ci.manifest
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

            all_images = vv_images + vh_images + hv_images + hh_images
            for image in all_images:
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

    def get_tiles_covered_by_products(self) -> List[str]:
        """
        This method returns the list of MGRS tiles covered
        by available S1 products.

        Returns:
           The sorted list of MGRS tiles identifiers covered by product as string
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
        return sorted(tiles)

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

    @timethis("Intersecting raster list w/ {tile_name_field}", logging.INFO)
    def get_s1_intersect_by_tile(
        self,
        tile_name_field:     str,
        output_name_formats: List[Tuple[str, str]],
    ) -> List[Dict]:
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
        self._update_s1_img_list_for(tile_name_field, output_name_formats)

        for image in self.get_raster_list():
            logger.debug('- Manifest: %s', image.get_manifest())
            logger.debug('  Image list: %s', image.get_images_list())
            if len(image.get_images_list()) == 0:
                logger.debug("Skipping product %s: no matching image in the requested polarities has been found", image.product_info.product.name)
                continue
            assert image.product_info.get_current_tile_coverage() is not None
            intersect_raster.append( {
                'raster'         : image,
                'tile_origin'    : image.product_info.get_tile_origin(),
                'tile_coverage'  : image.product_info.get_current_tile_coverage(),
            })

        return intersect_raster

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
