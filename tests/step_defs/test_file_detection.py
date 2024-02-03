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
# Authors: Thierry KOLECK (CNES)
#          Luc HERMITTE (CS Group)
#
# =========================================================================

import fnmatch
import logging
import os
# from pathlib import Path
from typing import Dict, List, Set
from eodag.api.search_result import SearchResult

import shapely

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from tests.mock_otb  import isdir, glob, dirname
from tests.mock_data import FileDB
# import s1tiling.libs.Utils
from s1tiling.libs.S1FileManager     import S1FileManager
from s1tiling.libs.outcome           import DownloadOutcome

from eodag.utils.exceptions import (
    # AuthenticationError,
    NotAvailableError,
)

# ======================================================================
# Scenarios
scenarios(
        '../features/test_file_detection.feature',
        '../features/test_product_downloading.feature',
        '../features/test_offline_products.feature',
        )

# ======================================================================
# Test Data

TMPDIR = 'TMP'
INPUT  = 'INPUT'
OUTPUT = 'OUTPUT'
LIADIR = 'LIADIR'
TILE   = '33NWB'

file_db = FileDB(INPUT, TMPDIR, OUTPUT, LIADIR, TILE, 'unused', 'unused')

def safe_dir(idx) -> str:
    return file_db.safe_dir(idx)

def input_file(idx, polarity) -> str:
    return file_db.input_file(idx, polarity)

def input_file_vv(idx) -> str:
    return file_db.input_file(idx, 'vv')

def input_file_vh(idx) -> str:
    return file_db.input_file(idx, 'vh')

# ======================================================================
# Mocks

class Configuration():
    def __init__(self, inputdir, tmpdir, outputdir, *argv) -> None:
        """
        constructor
        """
        self.nb_products_to_download = 2

        self.first_date              = '2020-01-01'
        self.last_date               = '2020-01-10'
        self.polarisation            = None
        self.download                = False
        self.raw_directory           = inputdir
        self.tmpdir                  = tmpdir
        self.output_preprocess       = outputdir
        self.cache_dem_by            = 'symlink'
        self.fname_fmt               = {}
        self.platform_list           = []
        self.orbit_direction         = None
        self.relative_orbit_list     = []
        self.calibration_type        = 'sigma'
        self.nb_download_processes   = 1
        self.fname_fmt               = {
                'concatenation' : '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_{calibration_type}.tif',
                # 'concatenation' : '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}.tif',
                'filtered' : 'filtered/{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_{calibration_type}.tif'
                }
        self.fname_fmt_concatenation = self.fname_fmt['concatenation']
        self.fname_fmt_filtered      = self.fname_fmt['filtered']

class MockDirEntry:
    def __init__(self, pathname) -> None:
        """
        constructor
        """
        self.path = pathname
        # `name`: relative to scandir...
        self.name = os.path.relpath(pathname, INPUT)
        self.parent = os.path.dirname(pathname)

    def __repr__(self):
        return self.path


def list_dirs(dir, pat, known_dirs) -> List[MockDirEntry]:
    logging.debug('mock.list_dirs(%s, %s) ---> %s', dir, pat, known_dirs)
    return [MockDirEntry(kd) for kd in sorted(set(known_dirs))]


@pytest.fixture
def known_files() -> List[str]:
    kf = []
    return kf

@pytest.fixture
def known_dirs() -> Set[str]:
    kd = set()
    return kd

@pytest.fixture
def image_list():
    rl = []
    return rl

@pytest.fixture
def downloads():
    dn = []
    return dn

@pytest.fixture
def configuration() -> Configuration:
    cfg = Configuration(INPUT, TMPDIR, OUTPUT)
    return cfg

# ======================================================================
# Given steps

def _mock_S1Tiling_functions(mocker, known_files, known_dirs) -> None:
    # for k in known_files:
        # logging.debug(' - %s', k)
    known_dirs.update([INPUT, TMPDIR, OUTPUT])
    known_dirs.update([dirname(fn, 2) for fn in known_files])
    mocker.patch('os.path.isdir', lambda f: isdir(f, known_dirs))
    mocker.patch('glob.glob',     lambda pat : glob(pat, sorted(set(known_files))))
    # Utils.list_dirs has been imported in S1FileManager. This is the one that needs patching!
    # It's used to filter the product paths => don't register every possible known directory
    known_dirs_4_list_dir = sorted(set([dirname(fn, 3) for fn in known_files]))
    mocker.patch('s1tiling.libs.S1FileManager.list_dirs', lambda dir, pat : list_dirs(dir, pat, known_dirs_4_list_dir))
    # Utils.get_orbit_direction has been imported in S1FileManager. This is the one that needs patching!
    mocker.patch('s1tiling.libs.S1FileManager.get_orbit_direction', lambda manifest : 'DES')
    mocker.patch('s1tiling.libs.S1FileManager.get_relative_orbit',  lambda manifest : 7)
    mocker.patch('s1tiling.libs.S1FileManager.S1FileManager._filter_products_with_enough_coverage', lambda slf, tile, pi: slf._products_info)


def _declare_known_S1_files(known_files, patterns) -> None:
    # logging.debug('_declare_known_files(%s)', patterns)
    # all_files = [input_file(idx) for idx in range(len(FILES))]
    all_files = file_db.all_vvvh_files()
    # logging.debug('All files:')
    # for a in all_files:
    #     logging.debug(' - %s', Path(*Path(a).parts[-3:]))
    assert file_db.input_file(0, 'vv') != file_db.input_file(0, 'vh')
    assert all_files[0] != all_files[1]
    files = []
    for pattern in patterns:
        files += [fn for fn in all_files if fnmatch.fnmatch(fn, '*'+pattern+'*')]
    known_files.extend(files)
    logging.debug('Mocking w/ S1: %s', patterns)
    for file in files:
        logging.debug('--> %s', file)


@given('No S1 files are known')
def given_no_S1_files_are_known(mocker, known_files, known_dirs) -> None:
    _mock_S1Tiling_functions(mocker, known_files, known_dirs)

@given('All S1 files are known')
def given_all_S1_files_are_known(mocker, known_files, known_dirs) -> None:
    _declare_known_S1_files(known_files, ['vv', 'vh'])
    _mock_S1Tiling_functions(mocker, known_files, known_dirs)

@given('All S1 VV files are known')
def given_all_S1_VV_files_are_known(mocker, known_files, known_dirs) -> None:
    _declare_known_S1_files(known_files, ['vv'])
    _mock_S1Tiling_functions(mocker, known_files, known_dirs)

@given('All S1 VH files are known')
def given_all_S1_VH_files_are_known(mocker, known_files, known_dirs) -> None:
    _declare_known_S1_files(known_files, ['vh'])
    _mock_S1Tiling_functions(mocker, known_files, known_dirs)


# ----------------------------------------------------------------------
# Given / download scenarios

def polygon2extent(polygon) -> Dict[str, float]:
    extent = {
            'lonmin': min(a[0] for a in polygon),
            'lonmax': max(a[0] for a in polygon),
            'latmin': min(a[1] for a in polygon),
            'latmax': max(a[1] for a in polygon),
            }
    return extent

def extent2box(extent):
    coords = (
            float(extent['lonmin']),
            float(extent['latmin']),
            float(extent['lonmax']),
            float(extent['latmax']),
            )
    return shapely.geometry.box(*coords)


class MockEOProduct:
    def __init__(self, product_id) -> None:
        self._id = file_db.product_name(product_id)
        self.is_valid = True
        # TODO: geometry is not correctly set
        product_poly     = file_db.FILES[product_id]['polygon']
        product_geometry = extent2box(polygon2extent(product_poly))
        self.geometry            = shapely.geometry.shape(product_geometry)
        self.search_intersection = shapely.geometry.shape(product_geometry)
        self.properties = {
                'id'                 : self._id,
                'orbitDirection'     : file_db.get_orbit_direction(product_id),
                'relativeOrbitNumber': file_db.get_relative_orbit(product_id),
                }
        logging.debug('EOProduct(#%s) -> %s %s#%s %s', product_id, self._id,
                self.properties['orbitDirection'],
                self.properties['relativeOrbitNumber'],
                self.geometry)
        self._expected_path = MockDirEntry(f'{INPUT}/{self._id}/{self._id}.SAFE')

    def __repr__(self) -> str:
        return "EOProduct(%s) -> %s#%s" % (self._id,
                self.properties['orbitDirection'],
                self.properties['relativeOrbitNumber'],
                )
    def as_dict(self) -> Dict:
        return self.properties

#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@given('Request on 8th jan')
def given_requets_on_8th_jan(configuration) -> None:
    logging.debug('Request on 8th jan')
    configuration.first_date              = file_db.CONCATS[0]['first_date']
    configuration.last_date               = file_db.CONCATS[0]['last_date']
    configuration.nb_products_to_download = 2

@given('Request on all dates')
def given_requets_on_all_dates(configuration) -> None:
    logging.debug('Request on all dates')
    configuration.first_date              = file_db.CONCATS[0]['first_date']
    configuration.last_date               = file_db.CONCATS[-1]['last_date']
    configuration.nb_products_to_download = len(file_db.FILES)

@given('Request on VV')
def given_requets_on_VV(configuration) -> None:
    logging.debug('Request on VV')
    configuration.polarisation = 'VV'

@given('Request on VH')
def given_requets_on_VH(configuration) -> None:
    logging.debug('Request on VH')
    configuration.polarisation = 'VH'

@given('Request for _beta')
def given_requets_for_beta(configuration) -> None:
    logging.debug('Request for _beta')
    configuration.calibration_type = 'beta'

@given('Request with default fname_fmt_concatenation')
def given_requets_for_beta_with_default_fname_fmt_concatenation(configuration) -> None:
    logging.debug('Request with default fname_fmt_concatenation')
    configuration.fname_fmt['concatenation'] = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}.tif'
    configuration.fname_fmt_concatenation = configuration.fname_fmt['concatenation']

#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def _declare_known_products_for_download(mocker, product_ids) -> None:
    def mock_search_products(slf, dag,
            extent, first_date, last_date, platform_list, orbit_direction,
            relative_orbit_list, polarization, dryrun) -> SearchResult:
        return SearchResult([MockEOProduct(p) for p in product_ids])

    mocker.patch('s1tiling.libs.S1FileManager.S1FileManager._search_products',
            lambda slf, dag, extent_33NWB, first_date, last_date,
            platform_list, orbit_direction, relative_orbit_list, polarization,
            dryrun
            : mock_search_products(slf, dag, extent_33NWB, first_date, last_date,
                platform_list, orbit_direction, relative_orbit_list, polarization,
                dryrun))

@given('All products are available for download')
def given_all_products_are_available_for_download(mocker, configuration) -> None:
    _declare_known_products_for_download(mocker, range(configuration.nb_products_to_download))


#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def _declare_known_S2_files(mocker, known_files, patterns) -> None:
    nb_products = file_db.nb_S2_products
    all_S2 = [file_db.concatfile_from_two(idx, '', pol) for idx in range(nb_products) for pol in ['vh', 'vv']]
    files = []
    for pattern in patterns:
        files += [fn for fn in all_S2 if fnmatch.fnmatch(fn, '*'+pattern+'*')]
    logging.debug('Mocking w/ S2: %s --> %s', patterns, files)
    for k in files:
        logging.debug(' - %s', k)
    known_files.extend(files)

@given('All S2 files are known')
def given_all_S2_files_are_known(mocker, known_files) -> None:
    _declare_known_S2_files(mocker, known_files, ['vv', 'vh'])

@given('All S2 VV files are known')
def given_all_S2_VV_files_are_known(mocker, known_files) -> None:
    _declare_known_S2_files(mocker, known_files, ['vv'])

@given('All S2 VH files are known')
def given_all_S2_VH_files_are_known(mocker, known_files) -> None:
    _declare_known_S2_files(mocker, known_files, ['vh'])

@given('No S2 files are known')
def given_no_S2_files_are_known() -> None:
    pass

# ======================================================================
# When steps

def _search(configuration, image_list, polarisation) -> None:
    configuration.polarisation = polarisation
    manager = S1FileManager(configuration)
    manager._refresh_s1_product_list()
    manager._update_s1_img_list_for('33NWB')
    logging.debug('_search(%s) --> += %s', polarisation, manager.get_raster_list())
    for p in manager.get_raster_list():
        # logging.debug(" * %s", p.get_manifest())
        for im in p.get_images_list():
            # logging.debug("   -> %s", im)
            image_list.append(im)

@when('VV-VH files are searched')
def when_searching_VV_VH(configuration, image_list, mocker) -> None:
    _search(configuration, image_list, 'VV VH')

@when('VV files are searched')
def when_searching_VV(configuration, image_list, mocker) -> None:
    _search(configuration, image_list, 'VV')

@when('VH files are searched')
def when_searching_VH(configuration, image_list, mocker) -> None:
    _search(configuration, image_list, 'VH')

# ----------------------------------------------------------------------
# When / download scenarios

def mock_download_one_product(dag, raw_directory, dl_wait, dl_timeout, product) -> DownloadOutcome:
    logging.debug('mock: download1 -> %s', product)
    return DownloadOutcome(product, product)

@when('Searching which S1 files to download')
def when_searching_which_S1_to_download(configuration, mocker, downloads) -> None:
    # mocker.patch('s1tiling.libs.S1FileManager._search_products',
    #         lambda dag, lonmin, lonmax, latmin, latmax, first_date, last_date,
    #         orbit_direction, relative_orbit_list, polarization,
    #         searched_items_per_page
    #         : downloads)
    mocker.patch('s1tiling.libs.S1FileManager._download_and_extract_one_product',
            mock_download_one_product)

    default_polarisation = 'VV VH'
    configuration.polarisation = configuration.polarisation or default_polarisation
    manager = S1FileManager(configuration)
    manager._refresh_s1_product_list()

    origin_33NWB = file_db.tile_origins('33NWB')
    extent_33NWB = polygon2extent(origin_33NWB)
    manager._update_s1_img_list_for('33NWB')
    # logging.debug('_search(%s) --> += %s', polarisation, manager.get_raster_list())
    paths = manager._download(None,
            lonmin=extent_33NWB['lonmin'], lonmax=extent_33NWB['lonmax'],
            latmin=extent_33NWB['latmin'], latmax=extent_33NWB['latmax'],
            first_date=file_db.start_time(0), last_date=file_db.start_time(file_db.nb_S1_products-1),
            tile_out_dir=OUTPUT, tile_name='33NWB',
            platform_list=configuration.platform_list, orbit_direction=None, relative_orbit_list=[],
            polarization=configuration.polarisation,
            cover=10, dryrun=False)
    downloads.extend(paths)


# ======================================================================
# Then steps

@then('No (other) files are found')
def then_no_other_files_are_found(image_list) -> None:
    assert len(image_list) == 0

@then('VV files are found')
def then_VV_files_are_found(image_list) -> None:
    assert len(image_list) >= 2
    for i in [0, 1]:
        assert input_file_vv(i) in image_list
        image_list.remove(input_file_vv(i))

@then('VH files are found')
def then_VH_files_are_found(image_list) -> None:
    assert len(image_list) >= 2
    for i in [0, 1]:
        assert input_file_vh(i) in image_list
        image_list.remove(input_file_vh(i))

# ----------------------------------------------------------------------
# Then / download scenarios

@then('None are requested for download')
def then_none_are_requested_for_download(downloads) -> None:
    assert len(downloads) == 0

@then('All are requested for download')
def then_all_are_requested_for_download(downloads, configuration) -> None:
    assert len(downloads) == configuration.nb_products_to_download


# ######################################################################
# Test download failures

@pytest.fixture
def dl_successes():
    l = []
    return l

@pytest.fixture
def dl_failures():
    l = []
    return l

@pytest.fixture
def dl_kepts():
    l = []
    return l

@given(parsers.parse('S1 product {idx} has been downloaded'))
def given_S1_product_idx_has_been_downloaded(dl_successes, known_files, known_dirs, mocker, idx) -> None:
    product = MockEOProduct(int(idx))
    dl_successes.append(product)
    _declare_known_S1_files(known_files, [product.as_dict()['id']])

@given(parsers.parse('S1 product {idx} download has timed-out'))
def given_S1_product_idx_has_timed_out(dl_failures, mocker, idx) -> None:
    missing_product = MockEOProduct(int(idx))
    failed = DownloadOutcome(
            NotAvailableError(
                f"{missing_product._id} is not available (OFFLINE) and could not be downloaded, timeout reached"),
            missing_product)
    dl_failures.append(failed)


@when('Filtering products to use')
def when_filtering_products_to_use(configuration, dl_successes, dl_failures, dl_kepts, mocker, known_files, known_dirs) -> None:
    _mock_S1Tiling_functions(mocker, known_files, known_dirs)
    manager = S1FileManager(configuration)
    # `manager._products_info` is filled-up during manager construction
    # from the scanned (mocked) directories
    assert len(manager._products_info) == len(dl_successes), f'\nFound on disk: {[p["product"] for p in manager._products_info]},\nDownloading: {dl_successes}'
    if dl_failures:
        manager._analyse_download_failures(dl_failures)
    assert len(dl_kepts) == 0
    dl_kepts.extend(
            manager._filter_complete_dowloads_by_pair(TILE, manager._products_info)
            )
    assert dl_kepts is not manager._products_info
    logging.debug('Keeping: %s/%s', len(dl_kepts), len(manager._products_info))
    for k in dl_kepts:
        logging.debug(' -> %s', k)

@then('All S2 products will be generated')
def then_all_S2_products_will_be_generated(dl_successes, dl_failures, dl_kepts) -> None:
    assert len(dl_kepts) == len(dl_successes), f'Keeping {dl_kepts} instead of {dl_successes}'
    assert len(dl_failures) == 0, f'There should be no failures. Found: {dl_failures}'

@then('No S2 product will be generated')
def then_no_S2_product_will_be_generated(dl_successes, dl_failures, dl_kepts) -> None:
    assert len(dl_kepts) == 0, f'Keeping {dl_kepts} instead of nothing'

@then(parsers.parse('{nb} S2 product(s) will be generated'))
def then_nb_S2_products_will_be_generated(dl_successes, dl_failures, dl_kepts, nb) -> None:
    assert len(dl_kepts) == 2*int(nb), f'Keeping {[p["product"] for p in dl_kepts]} instead of {nb}'
    # assert len(dl_failures) == 0, f'There should be no failures. Found: {dl_failures}'

@then(parsers.parse('S2 product n° {idx} will be generated'))
def then_S2_product_idx_will_be_generated(dl_successes, dl_failures, dl_kepts, idx) -> None:
    idx = int(idx)
    kept_product_names = [str(p['product']) for p in dl_kepts]
    logging.debug('Keeping: %s', kept_product_names)
    for i in range(2*idx, 2*idx+2):
        prod = '%s/%s' % (INPUT, file_db.product_name(i))
        logging.debug('...checking #%s: %s', i, prod)
        assert prod in kept_product_names

