#!/usr/bin/env python
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
# Authors: Thierry KOLECK (CNES)
#          Luc HERMITTE (CS Group)
#
# =========================================================================

from datetime import datetime, timedelta
import fnmatch
import logging
import os
# from pathlib import Path
from typing import Callable, Dict, List, Sequence, Set, Tuple
from eodag.api.search_result import SearchResult

from eof.products import re
from shapely import geometry

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from tests.mock_otb  import isdir, isfile, glob, dirname
from tests.mock_data import FileDB
# import s1tiling.libs.Utils
from s1tiling.libs.S1FileManager import S1FileManager
from s1tiling.libs.api           import main_output_name_formats
from s1tiling.libs.configuration import (
    _split_option,
    dname_fmt_filtered,
    dname_fmt_gamma_area_product,
    dname_fmt_tiled,
    fname_fmt_concatenation,
    fname_fmt_filtered,
    fname_fmt_gamma_area_product,
)
from s1tiling.libs.utils.layer   import polygon2extent
from s1tiling.libs.outcome       import S1DownloadOutcome
from s1tiling.libs.s1.product    import FileProductInformation

from eodag.utils.exceptions import (
    # AuthenticationError,
    NotAvailableError,
)

def to_datetime(s: str) -> datetime:
    return datetime.strptime(s, '%Y:%m:%d %H:%M:%S')

# ======================================================================
# Scenarios
scenarios(
        '../features/test_file_detection.feature',
        '../features/test_product_downloading.feature',
        '../features/test_product_downloading2.feature',
        '../features/test_offline_products.feature',
        )

# ======================================================================
# Test Data

TMPDIR        = 'TMP'
INPUT         = 'INPUT'
OUTPUT        = 'OUTPUT'
EOFDIR        = 'EOFDIR'
LIADIR        = 'LIADIR'
GAMMA_AREADIR = 'GAMMA_AREADIR'
TILE          = '33NWB'

file_db = FileDB(INPUT, EOFDIR, TMPDIR, OUTPUT, LIADIR, GAMMA_AREADIR, TILE, 'unused', 'unused')

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

# Various naming policies
ORTHORECTIFICATION_NAMING = {
    # Use "_beta" in mocked tests
    'with_calibration': '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_time}_{calibration_type}.tif',

    # Theia fname_fmt: S1A_L1ORT_47PNR_VH_SIG_DES_135_20230112T122356
    'theia' : '{flying_unit_code!u}_L1ORT_{tile_name}_{polarisation!u}_{calibration_type!u:.3}_{orbit_direction}_{orbit}_{acquisition_time}.tif',
}

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
        self.extra_directories       : Dict[str, str] = {}
        self.cache_dem_by            = 'symlink'
        self.platform_list           : List[str] = []
        self.orbit_direction         = None
        self.relative_orbit_list     : List[int] = []
        self.calibration_type        = 'sigma'
        self.nb_download_processes   = 1
        self.fname_fmt               = {
                # Non standard filename format for concatenation
                'concatenation' : '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_{calibration_type}.tif',
                'filtered' : '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_{calibration_type}_filtered.tif'
        }
        self.dname_fmt               : Dict[str, str] = {}
        self.creation_options        : Dict[str, str] = {}
        self.disable_streaming       : Dict[str, bool] = {}
        self.filter                  = ''


class MockDirEntry:
    def __init__(self, pathname) -> None:
        """
        constructor
        """
        self.path = pathname
        # `name`: relative to scandir...
        self.name = os.path.basename(pathname)
        self.parent = os.path.dirname(pathname)

    def __repr__(self):
        return self.path

    def is_dir(self) -> bool:
        return os.path.isdir(self.path)


def list_dirs(dir, pattern, known_dirs) -> List[MockDirEntry]:
    logging.debug('mock.list_dirs(%r, %r) ---> %r', dir, pattern, known_dirs)
    if not pattern:
        filt = lambda _   : True
    elif isinstance(pattern, re.Pattern):
        filt = lambda path: re.match(f"{dir}/{pattern}", path)
    else:
        filt = lambda path: fnmatch.fnmatch(path, f"{dir}/{pattern}")
    return [MockDirEntry(kd) for kd in sorted(set(known_dirs)) if filt(kd)]


def list_files(dir, pattern, known_files) -> List[MockDirEntry]:
    if not pattern:
        filt = lambda _   : True
    elif isinstance(pattern, re.Pattern):
        filt = lambda path: re.match(pattern, path.name)
    else:
        filt = lambda path: fnmatch.fnmatch(path.name, pattern)
    dir_entries = [MockDirEntry(kd) for kd in known_files]
    res = [de for de in dir_entries if filt(de)]
    logging.debug('mock.list_files(%r, %r) ---> %s\n\t--> %s', dir, pattern, known_files, res)
    return res


@pytest.fixture
def known_files() -> List[str]:
    kf : List[str] = []
    return kf

@pytest.fixture
def known_dirs() -> Set[str]:
    kd : Set[str] = set()
    return kd

@pytest.fixture
def image_list():
    rl = []
    return rl

@pytest.fixture
def downloads() -> list[S1DownloadOutcome]:
    dn = []
    return dn

@pytest.fixture
def naming_policy() -> str:
    return 'with_calibration'


@pytest.fixture
def configuration() -> Configuration:
    cfg = Configuration(INPUT, TMPDIR, OUTPUT)
    return cfg

# ======================================================================
# Given steps

@given(
    parsers.parse('{policy} naming policy'),
    target_fixture='naming_policy',
)
def given_naming_policy(policy) -> str:
    assert policy in FileDB.CONCATENATION_NAMING
    return policy


def _output_name_formats(configuration) -> List[Tuple[str,str]]:
    if configuration.calibration_type == 'gamma_area':
        return [(dname_fmt_gamma_area_product(configuration), fname_fmt_gamma_area_product(configuration))]
    else:
        return main_output_name_formats(configuration)


def _mock_S1Tiling_functions(mocker, known_files, known_dirs) -> None:
    # for k in known_files:
        # logging.debug(' - %s', k)
    known_dirs.update([INPUT, TMPDIR, OUTPUT])
    known_dirs.update([d for fn in known_files if (d := dirname(fn, 2))])
    mocker.patch('os.path.isfile', lambda f: isfile(f, known_files))
    mocker.patch('os.path.isdir',  lambda f: isdir(f, known_dirs))
    mocker.patch('glob.glob',      lambda pat : glob(pat, sorted(set(known_files))))
    # Utils.list_dirs has been imported in S1FileManager. This is the one that needs patching!
    # It's used to filter the product paths => don't register every possible known directory
    known_dirs_4_list_dir = sorted(set([d for fn in known_files if (d := dirname(fn, 3))]))
    mocker.patch('s1tiling.libs.S1FileManager.list_dirs', lambda dir, pat : list_dirs(dir, pat, known_dirs_4_list_dir))
    mocker.patch('s1tiling.libs.S1FileManager.list_files', lambda dir, pat : list_files(dir, pat, known_files))
    # Utils.get_orbit_direction has been imported in S1FileManager. This is the one that needs patching!
    mocker.patch('s1tiling.libs.Utils.get_orbit_direction', lambda manifest : 'DES')
    mocker.patch('s1tiling.libs.Utils.get_relative_orbit',  lambda manifest : 7)
    mocker.patch('s1tiling.libs.S1FileManager.S1FileManager._filter_products_with_enough_coverage', lambda slf, tile, pi: slf._products_info)
    mocker.patch('s1tiling.libs.Utils.get_orbit_information',  lambda manifest : file_db.get_orbit_information(manifest))


def _declare_known_S1_files(known_files, patterns: list[str], all_manifests: bool = True) -> None:
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
        files.extend([fn for fn in all_files if fnmatch.fnmatch(fn, f'*{pattern}*')])
    if all_manifests:
        # all_manifests = True is likelly to be used with vv/vh patterns
        files.extend(file_db.all_manifests())
    else:
        # all_manifests = False is likelly to be used with product id patterns
        for pattern in patterns:
            files.extend([fn for fn in file_db.all_manifests() if fnmatch.fnmatch(fn, f'*{pattern}*')])

    known_files.extend(files)
    logging.debug('Mocking w/ S1 local files: %r', patterns)
    for file in files:
        logging.debug('--> %r', file)


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

def extent2box(extent):
    coords = (
        float(extent['lonmin']),
        float(extent['latmin']),
        float(extent['lonmax']),
        float(extent['latmax']),
    )
    return geometry.box(*coords)


class MockEOProduct:
    def __init__(self, product_id) -> None:
        self._id = file_db.product_name(product_id)
        self.is_valid = True
        # TODO: geometry is not correctly set
        product_poly     = file_db.FILES[product_id]['polygon']
        product_geometry = extent2box(polygon2extent(product_poly))
        self.geometry            = geometry.shape(product_geometry)
        self.search_intersection = geometry.shape(product_geometry)
        k_obt_dir = {'ASC': 'ascending', 'DES': 'descending'}
        self.properties = {
            'id'                              : self._id,
            'orbitDirection'                  : k_obt_dir[file_db.get_orbit_direction(product_id)],
            'relativeOrbitNumber'             : file_db.get_relative_orbit(product_id),
            'orbitNumber'                     : file_db.get_absolute_orbit(product_id),
            'platformSerialIdentifier'        : 'S1A',
            'startTimeFromAscendingNode'      : file_db.get_start_time(product_id),
            'completionTimeFromAscendingNode' : file_db.get_stop_time(product_id),
            'polarizationMode'                : 'VV-VH',
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
    def __eq__(self, other) -> bool:
        return self._id == other._id
    def __hash__(self):
        return hash(self._id)

#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@given('Request on 8th jan')
def given_requests_on_8th_jan(configuration) -> None:
    logging.debug('Request on 8th jan')
    configuration.first_date              = (to_datetime(file_db.CONCATS[0]['start_time']) - timedelta(1)).strftime('%Y-%m-%d')
    configuration.last_date               = (to_datetime(file_db.CONCATS[0]['start_time']) + timedelta(1)).strftime('%Y-%m-%d')
    logging.debug("searching in %s .. %s", configuration.first_date, configuration.last_date)
    configuration.nb_products_to_download = 2

@given('Request on all dates')
def given_requests_on_all_dates(configuration) -> None:
    logging.debug('Request on all dates')
    configuration.first_date              = (to_datetime(file_db.CONCATS[0]['start_time']) - timedelta(1)).strftime('%Y-%m-%d')
    configuration.last_date               = (to_datetime(file_db.CONCATS[-1]['start_time']) + timedelta(1)).strftime('%Y-%m-%d')
    logging.debug("searching in %s .. %s", configuration.first_date, configuration.last_date)
    configuration.nb_products_to_download = len(file_db.FILES)

@given('Request on VV')
def given_requests_on_VV(configuration) -> None:
    logging.debug('Request on VV')
    configuration.polarisation = 'VV'

@given('Request on VH')
def given_requests_on_VH(configuration) -> None:
    logging.debug('Request on VH')
    configuration.polarisation = 'VH'

@given('Request for _beta')
def given_requests_for_beta(configuration) -> None:
    logging.debug('Request for _beta')
    configuration.calibration_type = 'beta'

@given('Request with default fname_fmt_concatenation')
def given_requests_for_beta_with_default_fname_fmt_concatenation(configuration) -> None:
    logging.debug('Request with default fname_fmt_concatenation')
    configuration.fname_fmt['concatenation'] = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}.tif'

#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def _declare_known_products_for_download_from_ids(mocker, product_ids: Sequence[int]) -> None:
    def mock_search_products(slf, dag,
            extent, first_date, last_date, platform_list, orbit_direction,
            relative_orbit_list, polarization, dryrun) -> SearchResult:
        return SearchResult([MockEOProduct(p) for p in product_ids])

    mocker.patch('s1tiling.libs.S1FileManager.S1FileManager._search_products',
            lambda slf, dag, extent, first_date, last_date,
            platform_list, orbit_direction, relative_orbit_list, polarization,
            dryrun
            : mock_search_products(slf, dag, extent, first_date, last_date,
                platform_list, orbit_direction, relative_orbit_list, polarization,
                dryrun))

def _declare_known_products_for_download_from_names(mocker, product_ids: Sequence[str]) -> None:
    def mock_search_products(slf, dag,
            extent, first_date, last_date, platform_list, orbit_direction,
            relative_orbit_list, polarization, dryrun) -> SearchResult:
        return SearchResult([MockEOProduct(file_db._find_image(p)) for p in product_ids])

    mocker.patch('s1tiling.libs.S1FileManager.S1FileManager._search_products',
            lambda slf, dag, extent, first_date, last_date,
            platform_list, orbit_direction, relative_orbit_list, polarization,
            dryrun
            : mock_search_products(slf, dag, extent, first_date, last_date,
                platform_list, orbit_direction, relative_orbit_list, polarization,
                dryrun))

@given('All products are available for download')
def given_all_products_are_available_for_download(mocker, configuration) -> None:
    _declare_known_products_for_download_from_ids(mocker, range(configuration.nb_products_to_download))


#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def _declare_known_S2_files(known_files, patterns, known_dirs, naming_policy) -> None:
    nb_products = file_db.nb_S2_products
    all_S2 = [file_db.concatfile_from_two(idx, '', naming_policy=naming_policy, polarity=pol) for idx in range(nb_products) for pol in ['vh', 'vv']]
    files = []
    for pattern in patterns:
        files += [fn for fn in all_S2 if fnmatch.fnmatch(fn, '*'+pattern+'*')]
    logging.debug('Mocking w/ S2: %s --> %s', patterns, files)
    # logging.debug('all S2: %r', all_S2)
    for k in files:
        logging.debug(' - %s', k)
    known_files.extend(files)
    assert file_db.s2_product_dir()
    known_dirs.add(file_db.s2_product_dir())

@given('All S2 files are known')
def given_all_S2_files_are_known(known_files, known_dirs, naming_policy) -> None:
    _declare_known_S2_files(known_files, ['vv', 'vh'], known_dirs, naming_policy)

@given('All S2 VV files are known')
def given_all_S2_VV_files_are_known(known_files, known_dirs, naming_policy) -> None:
    _declare_known_S2_files(known_files, ['vv'], known_dirs, naming_policy)

@given('All S2 VH files are known')
def given_all_S2_VH_files_are_known(known_files, known_dirs, naming_policy) -> None:
    _declare_known_S2_files(known_files, ['vh'], known_dirs, naming_policy)

@given('No S2 files are known')
def given_no_S2_files_are_known(known_dirs) -> None:
    assert file_db.s2_product_dir()
    known_dirs.add(file_db.s2_product_dir())
    pass

#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def _declare_known_filtered_S2_files(known_files, patterns, known_dirs, naming_policy, /, extra=None, outdir=None) -> None:
    nb_products = file_db.nb_S2_products
    params = {
            'tmp'        : '',
            'extra'      : extra or '_filtered',
            'calibration': '_sigma',
            'dir'        : outdir or f'{file_db.outputdir}/filtered/33NWB',
            # 'outdir'     : file_db.outputdir,
    }
    all_S2 = [
            file_db.filtered_from_two(
                idx=idx, naming_policy=naming_policy, polarity=pol, **params,
            ) for idx in range(nb_products) for pol in ['vh', 'vv']]
    files = []
    for pattern in patterns:
        files += [fn for fn in all_S2 if fnmatch.fnmatch(fn, '*'+pattern+'*')]
    logging.debug('Mocking w/ S2/filtered: %s --> %s', patterns, files)
    for k in files:
        logging.debug(' - %s', k)
    known_files.extend(files)
    assert params['dir']
    known_dirs.add(params['dir'])

@given('No filtered S2 files are known')
def given_no_filtered_S2_files_are_known() -> None:
    pass

@given('All filtered S2 files are known under the default fname_fmt')
def given_all_filteredS2_files_are_known_default_fname_fmt(known_files, known_dirs, naming_policy) -> None:
    _declare_known_filtered_S2_files(known_files, ['vv', 'vh'], known_dirs, naming_policy)

@given('All filtered S2 files are known with a different fname_fmt')
def given_all_filteredS2_files_are_known_different_fname_fmt(known_files, known_dirs, naming_policy) -> None:
    _declare_known_filtered_S2_files(known_files, ['vv', 'vh'], known_dirs, naming_policy, extra='.FILTERED')

@given("fname_fmt.filtered has the default value")
def given_a_fname_fmt_filtered_has_the_default_value(configuration) -> None:
    configuration.filter = 'something'
    pass

@given("fname_fmt.filtered has a different value")
def given_a_fname_fmt_filtered_has_a_different_value(configuration) -> None:
    fname_fmt = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_{calibration_type}.FILTERED.tif'
    configuration.fname_fmt['filtered'] = fname_fmt
    configuration.filter = 'something'


@given('All filtered S2 files are known in the default dname_fmt')
def given_all_filteredS2_files_are_known_default_dname_fmt(known_files, known_dirs, naming_policy) -> None:
    _declare_known_filtered_S2_files(known_files, ['vv', 'vh'], known_dirs, naming_policy)

@given('All filtered S2 files are known in a different dname_fmt')
def given_all_filteredS2_files_are_known_different_dname_fmt(known_files, known_dirs, naming_policy) -> None:
    _declare_known_filtered_S2_files(known_files, ['vv', 'vh'], known_dirs, naming_policy, outdir=f'{file_db.outputdir}/33NWB/filters')

@given("dname_fmt.filtered has the default value")
def given_a_dname_fmt_filtered_has_the_default_value() -> None:
    pass

@given("dname_fmt.filtered has a different value")
def given_a_dname_fmt_filtered_has_a_different_value(configuration) -> None:
    dname_fmt = '{out_dir}/{tile_name}/filters'
    configuration.dname_fmt['filtered'] = dname_fmt

# ======================================================================
# When steps

def _search(configuration, image_list, polarisation) -> None:
    configuration.polarisation = polarisation
    manager = S1FileManager(configuration, None)
    output_name_formats = _output_name_formats(configuration)
    manager._refresh_s1_product_list()
    manager._update_s1_img_list_for('33NWB', output_name_formats)
    logging.debug('_search(%s) --> += %s', polarisation, manager.get_raster_list())
    for p in manager.get_raster_list():
        # logging.debug(" * %s", p.get_manifest())
        for im in p.get_images_list():
            # logging.debug("   -> %s", im)
            image_list.append(im)

@when('VV-VH files are searched')
def when_searching_VV_VH(configuration, image_list) -> None:
    _search(configuration, image_list, 'VV VH')

@when('VV files are searched')
def when_searching_VV(configuration, image_list) -> None:
    _search(configuration, image_list, 'VV')

@when('VH files are searched')
def when_searching_VH(configuration, image_list) -> None:
    _search(configuration, image_list, 'VH')

# ----------------------------------------------------------------------
# When / download scenarios

def mock_download_one_product(dag, raw_directory, dl_wait, dl_timeout, logging, product) -> S1DownloadOutcome:
    logging.debug('mock: download1 -> %s', product)
    return S1DownloadOutcome(product, product)

@when('Searching which S1 files to download', target_fixture='downloads')
def when_searching_which_S1_to_download(configuration, mocker) -> list:
    mocker.patch(
        's1tiling.libs.utils.eodag._download_and_extract_one_product',
        mock_download_one_product)

    default_polarisation = 'VV VH'
    configuration.polarisation = configuration.polarisation or default_polarisation
    manager = S1FileManager(configuration, None)
    manager._refresh_s1_product_list()

    origin_33NWB = file_db.tile_origins('33NWB')
    extent_33NWB = polygon2extent(origin_33NWB)
    output_name_formats=[(dname_fmt_tiled(configuration), fname_fmt_concatenation(configuration))]
    if configuration.filter:
        output_name_formats=[(dname_fmt_filtered(configuration), fname_fmt_filtered(configuration))]
    # output_name_formats = _output_name_formats(configuration)
    manager._update_s1_img_list_for('33NWB', output_name_formats)

    # logging.debug('_search(%s) --> += %s', polarisation, manager.get_raster_list())
    paths = manager._download(
        dag=None,
        extent=extent_33NWB,
        first_date=file_db.start_time(0), last_date=file_db.start_time(file_db.nb_S1_products-1),
        tile_out_dir=OUTPUT,
        gamma_area_dir="UNUSED",
        tile_name='33NWB',
        platform_list=configuration.platform_list, orbit_direction=None, relative_orbit_list=[],
        polarization=configuration.polarisation,
        cover=10,
        output_name_formats=output_name_formats,
        dryrun=False,
    )
    return paths


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
def dl_failures() -> List[S1DownloadOutcome]:
    l = []
    return l

@pytest.fixture
def dl_kepts() -> Sequence[FileProductInformation]:
    l = []
    return l

@pytest.fixture
def dl_skip() -> List[str]:
    l = []
    return l

@given(parsers.parse('S1 product {idx} has been downloaded'))
def given_S1_product_idx_has_been_downloaded(dl_successes, known_files, known_dirs, idx) -> None:
    product = MockEOProduct(int(idx))
    dl_successes.append(product)
    _declare_known_S1_files(known_files, [product.as_dict()['id']])

@given(parsers.parse('S1 product {idx} download has timed-out'))
def given_S1_product_idx_has_timed_out(dl_failures, mocker, idx) -> None:
    missing_product = MockEOProduct(int(idx))
    failed = S1DownloadOutcome(
            NotAvailableError(
                f"{missing_product._id} is not available (OFFLINE) and could not be downloaded, timeout reached"),
            missing_product)
    dl_failures.append(failed)


@when('Filtering products to use')
def when_filtering_products_to_use(
    configuration, dl_successes, dl_failures, dl_kepts, dl_skip, mocker, known_files, known_dirs
) -> None:
    _mock_S1Tiling_functions(mocker, known_files, known_dirs)
    output_name_formats = _output_name_formats(configuration)
    manager = S1FileManager(configuration, None)
    # `manager._products_info` is filled-up during manager construction
    # from the scanned (mocked) directories
    assert len(manager._products_info) == len(dl_successes), f'\nFound on disk: {[p.product for p in manager._products_info]},\nDownloading: {dl_successes}'
    if dl_failures:
        manager._analyse_download_failures(dl_failures)
    assert len(dl_kepts) == 0
    dl_kepts.extend(
        manager._filter_complete_dowloads_by_pair(TILE, manager._products_info, output_name_formats)
    )
    assert dl_kepts is not manager._products_info
    logging.debug('Keeping: %s/%s', len(dl_kepts), len(manager._products_info))
    for k in dl_kepts:
        logging.debug(' -> %s', k)
    dl_skip.extend(manager.get_skipped_S2_products())
    logging.debug('Skipping: %s outputs', len(dl_skip))
    for k in dl_skip:
        logging.debug(' -> %s', k)

@then('All S2 products will be generated')
def then_all_S2_products_will_be_generated(dl_successes, dl_failures, dl_kepts) -> None:
    assert len(dl_kepts) == len(dl_successes), f'Keeping {dl_kepts} instead of {dl_successes}'
    assert len(dl_failures) == 0, f'There should be no failures. Found: {dl_failures}'

@then('No S2 product will be generated')
def then_no_S2_product_will_be_generated(dl_kepts: Sequence[FileProductInformation]) -> None:
    assert len(dl_kepts) == 0, f'Keeping {dl_kepts} instead of nothing'

@then(parsers.parse('{nb} S2 product(s) will be generated'))
def then_nb_S2_products_will_be_generated(dl_kepts: Sequence[FileProductInformation], nb) -> None:
    assert len(dl_kepts) == 2*int(nb), f'Keeping {[p.product for p in dl_kepts]} instead of {nb}'
    # assert len(dl_failures) == 0, f'There should be no failures. Found: {dl_failures}'

@then(parsers.parse('S2 product n° {idx} will be generated'))
def then_S2_product_idx_will_be_generated(dl_kepts: Sequence[FileProductInformation], idx: int) -> None:
    idx = int(idx)
    kept_product_names = [str(p.product) for p in dl_kepts]
    logging.debug('Keeping: %s', kept_product_names)
    for i in range(2*idx, 2*idx+2):
        s1_input = '%s/%s' % (INPUT, file_db.product_name(i))
        logging.debug('...checking kept #%s: %s', i, s1_input)
        assert s1_input in kept_product_names

def _then_xx_product_idx_will_be_discarded(
    product_name_generator: Callable,
    dl_skip: List[str],
    idx: int,
) -> None:
    idx = int(idx)
    logging.debug('Failures:')
    for skip in dl_skip:
        logging.debug("-> %s", skip)
    skipped_product_names = [p for p in dl_skip]
    logging.debug('Discarding:')
    for skipped in skipped_product_names:
        logging.debug('-> %s', skipped)

    s1_inputs = [file_db.product_name(i) for i in range(2*idx, 2*idx+2)]
    prod = product_name_generator(idx)
    logging.debug('...checking discarded #%s: %s && %s', idx, prod, ' | '.join(s1_inputs))
    for skipped in skipped_product_names:
        if os.path.basename(prod) in skipped and (
            s1_inputs[0] in skipped or s1_inputs[1] in skipped
        ):
            break
    else:
        assert False, f"{os.path.basename(prod)} not in {skipped_product_names}"

@then(parsers.parse('S2 product n° {idx} will be discarded'))
def then_S2_product_idx_will_be_discarded(dl_skip: List[str], idx: int, configuration, naming_policy) -> None:
    calib = K_CALIBRATION_TO_SUFFIX[configuration.calibration_type]
    product_name_generator = lambda idx : file_db.concatfile_from_two(idx, tmp=False, naming_policy=naming_policy, polarity='*', calibration=calib)
    _then_xx_product_idx_will_be_discarded(product_name_generator, dl_skip, idx)

@then(parsers.parse('Gamma Area S2 product n° {idx} will be discarded'))
def then_gamma_area_S2_product_idx_will_be_discarded(dl_skip: List[str], idx: int) -> None:
    # Note: the error tested and reported is partly incorrect.
    # Yes, the γ area product cannot be generated from the pair of S1 input
    # but... it may be generated from a different pair
    product_name_generator = lambda idx : file_db.gamma_area_on_s2(tmp=False)
    _then_xx_product_idx_will_be_discarded(product_name_generator, dl_skip, idx)


# ######################################################################
# Test download for γ°RTC

# ======================================================================
# @given's

def value_to_list(value : str) -> list[str]:
    value = value.strip()
    return _split_option(value)


# ----------------------------------------------------------------------
# Product IDs:

# -----[ Input: S1 product IDs
@pytest.fixture
def s1_products() -> dict[str,str]:
    return {}


def _decode_product_ids(datatable):
    products = {}
    for ident, product in datatable[1:]:
        products[ident] = product
    return products

@given("the S1 products:", target_fixture="s1_products")
def given_s1_product_ids(datatable) -> dict[str,str]:
    return _decode_product_ids(datatable)


# -----[ Output: S2 product IDs
@pytest.fixture
def s2_products() -> dict[str,list[str]]:
    return {}


@given("the S2 products:", target_fixture="s2_products")
def given_s2_product_ids(datatable) -> dict[str,list[str]]:
    products = {}
    for ident, product in datatable[1:]:
        products[ident] = _split_option(product)
    return products


# -----[ Output: S2 γ area map product IDs
@pytest.fixture
def gamma_area_products() -> dict[str,str]:
    return {}


@given("the gamma areas:", target_fixture="gamma_area_products")
def given_gamma_area_product_ids(datatable) -> dict[str,str]:
    return _decode_product_ids(datatable)


# ----------------------------------------------------------------------
# Known products:

def _get_products(ids: list[str], reference_products: dict[str, str]) -> list[str]:
    products = []
    for product_id in ids:
        products.append(reference_products[product_id])
    return products


# -----[ S1 input remote products
@pytest.fixture
def known_remote_s1() -> list[str]:
    return []


@given(
    parsers.re("The following S1 products are available for download: (?P<remote_s1>.*?)"),
    target_fixture="known_remote_s1",
    converters={'remote_s1': value_to_list},
)
def given_remote_s1_product_list(
    mocker,
    s1_products: dict[str,str],
    remote_s1  : list[str],
) -> list[str]:
    known_remote_s1 = _get_products(remote_s1, s1_products)
    logging.debug("known remote S1: %r", known_remote_s1)
    _declare_known_products_for_download_from_names(mocker, known_remote_s1)
    return known_remote_s1


# -----[ S1 input products on local disk
@pytest.fixture
def known_local_s1() -> list[str]:
    return []


@given(
    parsers.re("The following S1 products are on disk: (?P<local_s1>.*?)"),
    target_fixture="known_local_s1",
    converters={'local_s1': value_to_list},
)
def given_local_s1_product_list(
    s1_products: dict[str,str],
    local_s1   : list[str],
) -> list[str]:
    # res = []
    # for product_id in local_s1:
    #     res.append(s1_products[product_id])
    res = _get_products(local_s1, s1_products)
    logging.debug("known local S1: %r", res)
    return res


# -----[ S2 output products on local disk
@pytest.fixture
def known_local_s2() -> list[str]:
    return []


@given(
    parsers.re("The following S2 products are on disk: (?P<local_s2>.*?)"),
    target_fixture="known_local_s2",
    converters={'local_s2': value_to_list},
)
def given_local_s2_product_list(
    s2_products        : dict[str,list[str]],
    gamma_area_products: dict[str, str],
    local_s2           : list[str],
) -> list[str]:
    res = []
    for product_id in local_s2:
        if product_id in s2_products:
            res.extend(s2_products[product_id])
        elif product_id in gamma_area_products:
            res.append(gamma_area_products[product_id])
        else:
            raise AssertionError(f"Invalid S2 local product {product_id!r}. It's not a valid S2 product name, nor a valid γ area map name")

    logging.debug("known local S2: %r", res)
    return res


# -----[ Requested Time range
@given("Requested time range is deduced from known remote S1 products")
def given_retroactive_set_time_range_from_inputs(
    configuration,
    known_remote_s1,
) -> None:
    actual_products = [
        file_db._find_image(product) for product in known_remote_s1
    ]
    first_start_time = min([to_datetime(file_db.FILES[idx]['start_time']) for idx in actual_products])
    last_stop_time   = max([to_datetime(file_db.FILES[idx]['stop_time'])  for idx in actual_products])
    configuration.first_date = (first_start_time - timedelta(1)).strftime('%Y-%m-%d')
    configuration.last_date  = (last_stop_time   + timedelta(1)).strftime('%Y-%m-%d')
    logging.debug("Request time range forged to %s .. %s (from known remote S1 products)", configuration.first_date, configuration.last_date)


# ----------------------------------------------------------------------
# Scenarios:

@given(
    parsers.re("We (?P<calibration>.*?) calibrate"),
)
def given_calibration_scenario(calibration, configuration):
    configuration.calibration_type = calibration


@given(
    parsers.re("We compute gamma area"),
)
def given_gamma_area_scenario(configuration):
    # Hijacked parameter to indicate which scenario is used: here the production of γ area maps
    configuration.calibration_type = "gamma_area"
    pass


# ======================================================================
# @when's

K_CALIBRATION_TO_SUFFIX = {
    'sigma'           : '_sigma',
    'gamma_naught_rtc': '_GammaNaughtRTC',
    'gamma_area'      : '_GammaNaughtRTC',  # not a real calibration...
}


def _declare_known_S2_files_from_ids(
    known_files,
    patterns,
    known_dirs,
    configuration,
    naming_policy,
) -> None:
    files = []
    calib = K_CALIBRATION_TO_SUFFIX[configuration.calibration_type]

    nb_products = file_db.nb_S2_products
    all_S2 = [
        file_db.concatfile_from_two(idx, '', naming_policy=naming_policy, polarity=pol, calibration=calib) for idx in range(nb_products) for pol in ['vh', 'vv']
    ] + [
        file_db.concatfile_from_one(idx, '', naming_policy=naming_policy, polarity=pol, calibration=calib) for idx in range(nb_products*2) for pol in ['vh', 'vv']
    ] + [
        file_db.selectedGAMMA_AREAfile()
    ]
    for pattern in patterns:
        files += [fn for fn in all_S2 if fnmatch.fnmatch(fn, f'*{pattern}*')]
    logging.debug('Mocking w/ S2: %s --> %s', patterns, files)
    logging.debug('all S2: %r', all_S2)
    for k in files:
        logging.debug(' - %s', k)
    known_files.extend(files)
    assert file_db.s2_product_dir()
    known_dirs.add(file_db.s2_product_dir())
    known_dirs.add(file_db.gamma_area_dir())


@when('Searching which S1 files to download II', target_fixture='downloads')
def when_searching_which_S1_to_download2(
    configuration,
    known_local_s1,
    known_local_s2,
    mocker,
    known_files,
    known_dirs,
    naming_policy,
) -> list:
    default_polarisation = 'VV VH'
    configuration.polarisation = configuration.polarisation or default_polarisation

    def list_mocked_nodes(node_list: list, what: str):
        logging.debug("* %s:", what)
        for node in node_list:
            logging.debug("  - %r", node)
    _declare_known_S1_files(known_files, [file for file in known_local_s1], all_manifests=False)
    _declare_known_S2_files_from_ids(
        known_files,
        [file for file in known_local_s2],
        known_dirs,
        configuration,
        naming_policy=naming_policy)
    _mock_S1Tiling_functions(mocker, known_files, known_dirs)
    list_mocked_nodes(known_dirs, "known dirs")
    list_mocked_nodes(known_files, "known files")


    mocker.patch(
        's1tiling.libs.utils.eodag._download_and_extract_one_product',
        mock_download_one_product)

    manager = S1FileManager(configuration, None)
    manager._refresh_s1_product_list()

    origin_33NWB = file_db.tile_origins('33NWB')
    extent_33NWB = polygon2extent(origin_33NWB)

    output_name_formats = _output_name_formats(configuration)

    manager._update_s1_img_list_for('33NWB', output_name_formats)
    # logging.debug('_search(%s) --> += %s', polarisation, manager.get_raster_list())
    paths = manager._download(
        dag=None,
        extent=extent_33NWB,
        first_date=file_db.start_time(0), last_date=file_db.start_time(file_db.nb_S1_products-1),
        tile_out_dir=OUTPUT,
        gamma_area_dir=GAMMA_AREADIR,
        tile_name=TILE,
        platform_list=configuration.platform_list,
        orbit_direction=None,
        relative_orbit_list=[],
        polarization=configuration.polarisation,
        cover=10,
        output_name_formats=output_name_formats,
        dryrun=False,
    )
    return paths


# ======================================================================
# @then's

@then(
    parsers.re("The following S1 products will be downloaded: (?P<dl_s1>.*?)"),
    converters={'dl_s1': value_to_list},
)
def then_download_these_s1_products(dl_s1, downloads: list[S1DownloadOutcome], s1_products: list[str]):
    logging.debug("Expecting to DL: %r", dl_s1)
    new_s1 = []
    for outcome in downloads:
        assert outcome
        new_s1.append(outcome.value())
    dl_expected = {MockEOProduct(file_db._find_image(s1_products[exp_id])) for exp_id in dl_s1}
    assert dl_expected == set(new_s1)
