#!/usr/bin/env python
# -*- coding: utf-8 -*-
import fnmatch
import logging
import os
from pathlib import Path

import shapely

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from tests.mock_otb  import isfile, isdir, glob, dirname
from tests.mock_data import FileDB
import s1tiling.libs.Utils
from s1tiling.libs.S1FileManager     import S1FileManager
from s1tiling.libs.S1DateAcquisition import S1DateAcquisition
from s1tiling.libs.outcome           import Outcome

# ======================================================================
# Scenarios
scenarios('../features/test_file_detection.feature', '../features/test_product_downloading.feature')

# ======================================================================
# Test Data

TMPDIR = 'TMP'
INPUT  = 'INPUT'
OUTPUT = 'OUTPUT'
LIADIR = 'LIADIR'
TILE   = '33NWB'

file_db = FileDB(INPUT, TMPDIR, OUTPUT, LIADIR, TILE, 'unused', 'unused')

def safe_dir(idx):
    return file_db.safe_dir(idx)

def input_file(idx, polarity):
    return file_db.input_file(idx, polarity)

def input_file_vv(idx):
    return file_db.input_file(idx, 'vv')

def input_file_vh(idx):
    return file_db.input_file(idx, 'vh')

# ======================================================================
# Mocks

class Configuration():
    def __init__(self, inputdir, tmpdir, outputdir, *argv):
        """
        constructor
        """
        self.nb_products_to_download = 2

        self.first_date              = '2020-01-01'
        self.last_date               = '2020-01-10'
        self.download                = False
        self.raw_directory           = inputdir
        self.tmpdir                  = tmpdir
        self.output_preprocess       = outputdir
        self.cache_srtm_by           = 'symlink'
        self.fname_fmt               = {}
        self.orbit_direction         = None
        self.relative_orbit_list     = []
        self.calibration_type        = 'sigma'
        self.nb_download_processes   = 1
        self.fname_fmt               = {
                'concatenation' : '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_{calibration_type}.tif',
                'filtered' : 'filtered/{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_{calibration_type}.tif'
                }
        self.fname_fmt_concatenation = self.fname_fmt['concatenation']
        self.fname_fmt_filtered      = self.fname_fmt['filtered']

class MockDirEntry:
    def __init__(self, pathname):
        """
        constructor
        """
        self.path = pathname
        # `name`: relative to scandir...
        self.name = os.path.relpath(pathname, INPUT)


def list_dirs(dir, pat, known_dirs):
    logging.debug('mock.list_dirs(%s, %s) ---> %s', dir, pat, known_dirs)
    return [MockDirEntry(kd) for kd in known_dirs]


@pytest.fixture
def known_files():
    kf = []
    return kf

@pytest.fixture
def known_dirs():
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
def configuration(mocker):
    cfg = Configuration(INPUT, TMPDIR, OUTPUT)
    return cfg

# ======================================================================
# Given steps

def _mock_S1Tiling_functions(mocker, known_files, known_dirs):
    # for k in known_files:
        # logging.debug(' - %s', k)
    known_dirs.update([INPUT, TMPDIR, OUTPUT])
    known_dirs.update([dirname(fn, 2) for fn in known_files])
    mocker.patch('os.path.isdir', lambda f: isdir(f, known_dirs))
    mocker.patch('glob.glob',  lambda pat : glob(pat, known_files))
    # Utils.list_dirs has been imported in S1FileManager. This is the one that needs patching!
    # It's used to filter the product paths => don't register every possible known directory
    known_dirs_4_list_dir = [dirname(fn, 3) for fn in known_files]
    mocker.patch('s1tiling.libs.S1FileManager.list_dirs', lambda dir, pat : list_dirs(dir, pat, known_dirs_4_list_dir))
    # Utils.get_orbit_direction has been imported in S1FileManager. This is the one that needs patching!
    mocker.patch('s1tiling.libs.S1FileManager.get_orbit_direction', lambda manifest : 'DES')
    mocker.patch('s1tiling.libs.S1FileManager.get_relative_orbit', lambda manifest : 7)
    mocker.patch('s1tiling.libs.S1FileManager.S1FileManager._filter_products_with_enough_coverage', lambda slf, pi: slf._products_info)


def _declare_known_S1_files(mocker, known_files, known_dirs, patterns):
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
    logging.debug('Mocking w/ S1: %s --> %s', patterns, files)


@given('No S1 files are known')
def given_no_S1_files_are_known(mocker, known_files, known_dirs):
    # _declare_known_S1_files(mocker, known_files, known_dirs, ['vv', 'vh'])
    _mock_S1Tiling_functions(mocker, known_files, known_dirs)

@given('All S1 files are known')
def given_all_S1_files_are_known(mocker, known_files, known_dirs):
    _declare_known_S1_files(mocker, known_files, known_dirs, ['vv', 'vh'])
    _mock_S1Tiling_functions(mocker, known_files, known_dirs)

@given('All S1 VV files are known')
def given_all_S1_VV_files_are_known(mocker, known_files, known_dirs):
    _declare_known_S1_files(mocker, known_files, known_dirs, ['vv'])
    _mock_S1Tiling_functions(mocker, known_files, known_dirs)

@given('All S1 VH files are known')
def given_all_S1_VH_files_are_known(mocker, known_files, known_dirs):
    _declare_known_S1_files(mocker, known_files, known_dirs, ['vh'])
    _mock_S1Tiling_functions(mocker, known_files, known_dirs)


# ----------------------------------------------------------------------
# Given / download scenarios

def polygon2extent(polygon):
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
    def __init__(self, product_id):
        self._id = file_db.product_name(product_id)
        self.is_valid = True
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
    def __repr__(self):
        return "EOProduct(%s) -> %s#%s" % (self._id,
                self.properties['orbitDirection'],
                self.properties['relativeOrbitNumber'],
                )
    def as_dict(self):
        return self.properties

@given('Request on 8th jan')
def given_requets_on_8th_jan(configuration):
    logging.debug('Request on 8th jan')
    configuration.first_date              = file_db.CONCATS[0]['first_date']
    configuration.last_date               = file_db.CONCATS[0]['last_date']
    configuration.nb_products_to_download = 2

@given('Request on all dates')
def given_requets_on_all_dates(configuration):
    logging.debug('Request on all dates')
    configuration.first_date              = file_db.CONCATS[0]['first_date']
    configuration.last_date               = file_db.CONCATS[-1]['last_date']
    configuration.nb_products_to_download = len(file_db.FILES)

def _declare_known_products_for_download(mocker, product_ids):
    def mock_search_products(slf, dag,
            extent, first_date, last_date, orbit_direction, relative_orbit_list,
            polarization, searched_items_per_page,dryrun):
        return [MockEOProduct(p) for p in product_ids]

    mocker.patch('s1tiling.libs.S1FileManager.S1FileManager._search_products',
            lambda slf, dag, extent_33NWB, first_date, last_date,
            orbit_direction, relative_orbit_list, polarization,
            searched_items_per_page,dryrun
            : mock_search_products(slf, dag, extent_33NWB, first_date, last_date,
                orbit_direction, relative_orbit_list, polarization,
                searched_items_per_page,dryrun))

@given('All products are available for download')
def given_all_products_are_available_for_download(mocker, configuration):
    _declare_known_products_for_download(mocker, range(configuration.nb_products_to_download))

def _declare_known_S2_files(mocker, known_files, known_dirs, patterns):
    nb_products = file_db.nb_S2_products
    all_S2 = [file_db.concatfile_from_two(idx, '', pol) for idx in range(nb_products) for pol in ['vh', 'vv']]
    files = []
    for pattern in patterns:
        files += [fn for fn in all_S2 if fnmatch.fnmatch(fn, '*'+pattern+'*')]
    logging.debug('Mocking w/ S2: %s --> %s', patterns, files)
    for k in files:
        logging.debug(' - %s', k)
    known_files.extend(all_S2)

@given('All S2 files are known')
def given_all_S2_files_are_known(mocker, known_files, known_dirs):
    _declare_known_S2_files(mocker, known_files, known_dirs, ['vv', 'vh'])

@given('No S2 files are known')
def given_no_S2_files_are_known(mocker, known_files, known_dirs):
    # _declare_known_S2_files(mocker, known_files, known_dirs, ['vv', 'vh'])
    pass

# ======================================================================
# When steps

def _search(configuration, image_list, polarisation):
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
def when_searching_VV_VH(configuration, image_list, mocker):
    _search(configuration, image_list, 'VV VH')

@when('VV files are searched')
def when_searching_VV(configuration, image_list, mocker):
    _search(configuration, image_list, 'VV')

@when('VH files are searched')
def when_searching_VH(configuration, image_list, mocker):
    _search(configuration, image_list, 'VH')

# ----------------------------------------------------------------------
# When / download scenarios

def mock_download_one_product(dag, raw_directory, product):
    logging.debug('mock: download1 -> %s', product)
    return Outcome(product)

@when('Searching which S1 files to download')
def when_searching_which_S1_to_download(configuration, image_list, mocker, downloads):
    # mocker.patch('s1tiling.libs.S1FileManager._search_products',
    #         lambda dag, lonmin, lonmax, latmin, latmax, first_date, last_date,
    #         orbit_direction, relative_orbit_list, polarization,
    #         searched_items_per_page,dryrun
    #         : downloads)
    mocker.patch('s1tiling.libs.S1FileManager._download_and_extract_one_product',
            mock_download_one_product)

    polarisation = 'VV VH'
    configuration.polarisation = polarisation
    manager = S1FileManager(configuration)
    manager._refresh_s1_product_list()

    origin_33NWB = file_db.tile_origins('33NWB')
    extent_33NWB = polygon2extent(origin_33NWB)
    manager._update_s1_img_list_for('33NWB')
    # logging.debug('_search(%s) --> += %s', polarisation, manager.get_raster_list())
    paths = manager._download(None,
            extent_33NWB['lonmin'], extent_33NWB['lonmax'], extent_33NWB['latmin'], extent_33NWB['latmax'],
            file_db.start_time(0), file_db.start_time(file_db.nb_S1_products-1),
            OUTPUT+'/33NWB', '33NWB', None, [], polarisation, 10, 42, False)
    downloads.extend(paths)


# ======================================================================
# Then steps

@then('No (other) files are found')
def then_no_other_files_are_found(image_list):
    assert len(image_list) == 0

@then('VV files are found')
def then_VV_files_are_found(image_list):
    assert len(image_list) >= 2
    for i in [0, 1]:
        assert input_file_vv(i) in image_list
        image_list.remove(input_file_vv(i))

@then('VH files are found')
def then_VH_files_are_found(image_list):
    assert len(image_list) >= 2
    for i in [0, 1]:
        assert input_file_vh(i) in image_list
        image_list.remove(input_file_vh(i))

# ----------------------------------------------------------------------
# Then / download scenarios

@then('None are requested for download')
def then_none_are_requested_for_download(downloads):
    assert len(downloads) == 0

@then('All are requested for download')
def then_all_are_requested_for_download(downloads, configuration):
    assert len(downloads) == configuration.nb_products_to_download

