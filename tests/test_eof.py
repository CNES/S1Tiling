#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2025 (c) CNES.
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

# Notes:
# In order to generate/update the VCR cassettes, run the test once. Not all tests will pass.
# Run the test another time and it should be perfect


# All the tests for test_eof have been written with the following EOF files.
# * Disk
#     around date | orbit range
#   - 2020-12-05  : 120..136
#   - 2023-10-18  : 164..005
#   - 2023-11-07  : 106..121
#   - 2023-11-08  : 120..136
#   - 2023-11-17  : 076..092
#   - 2023-11-18  : 091..107
# * Cassettes
#     around date | orbit range
#   - 2019-12-31  : 062..078
#   - 2020-01-01  : 076..092
#   - 2020-01-02  : 091..107
#   - 2020-01-03  : 106..121

from collections.abc import Generator
from contextlib import suppress
from datetime import datetime, timedelta
from math import prod
from urllib.parse import parse_qs, urlencode, urlparse
from dateutil.parser import parse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import pytest
from pytest_recording._vcr import use_cassette
from _pytest.fixtures import SubRequest
import pprint
from unittest import mock

from eodag.api.core import EODataAccessGateway

from s1tiling.libs.orbit._providers   import EodagProvider
from s1tiling.libs.orbit._manager     import EOFFileManager
from s1tiling.libs.orbit._conversions import ORBIT_CONVERTERS
from s1tiling.libs.orbit._file        import (
    SentinelOrbitFile,
    extract_min_max_abs_orbit_numbers,
    filter_eof_files_according_to_orbit_and_mission,
    filter_intersecting_eof_file_list,
    filter_uniq_eofs,
    glob_eof_files,
    keep_one_eof_per_orbit,
    orbit_range,
)
from s1tiling.libs.utils.path import AnyPath

logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("vcr").setLevel(logging.WARNING)
logging.getLogger("eodag").setLevel(logging.WARNING)
logging.getLogger("http.cookiejar").setLevel(logging.WARNING)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
K_1D       = timedelta(days=1)

# =====[ VCR Cassettes configuration
def filter_response(response):
    """Scrub various secrets from ASF and Copernicus Dataspace"""
    # Scrub SET-COOKIE from ASF responses
    response['headers'].pop('SET-COOKIE', None)
    # Scrub set-cookie from Copernicus Dataspace responses
    response['headers'].pop('set-cookie', None)
    # Scrub access_token and refresh_token from Copernicus Dataspace
    if "body" in response and "string" in response["body"]:
        body_string = response["body"]["string"]
        with suppress(json.decoder.JSONDecodeError):
            decoded_body = json.loads(body_string)
            for key in ("access_token", "refresh_token"):
                if key in decoded_body:
                    decoded_body[key] = f"REDACTED_{key}"
            response["body"]["string"] = bytes(json.dumps(decoded_body), 'utf8')
    if "Location" in response["headers"]:
        location_string = response["headers"]["Location"][0]
        with suppress(json.decoder.JSONDecodeError):
            parsed_location = urlparse(location_string)
            query = parse_qs(parsed_location.query)
            query["token"] = ["REDACTED_token"]
            url = parsed_location._replace(query=urlencode(query, doseq=True)).geturl()
            response["headers"]["Location"] = [url]
    return response


@pytest.fixture(scope="module")
def vcr_config():
    """
    Tweak the cassette recorder to remove secrets from queries and responses
    """
    return {
        "filter_headers"             : ["authorization", "Cookie"],
        "filter_query_parameters"    : ["username", "password", "totp", "token"],
        "filter_post_data_parameters": ["username", "password", "totp"],
        "before_record_response"     : [filter_response],
        "allow_playback_repeats"     : True,
    }


@pytest.fixture(scope="module")  # type: ignore
def vcr_cassette_dir(request: SubRequest) -> str:
    """Override vcr_cassette_dir to use cassettes from $BASELINEDIR

    For example each test module could have test function with the same names:
      - test_users.py:test_create
      - test_profiles.py:test_create
    """
    baseline = request.config.getoption("--baselinedir")
    assert isinstance(baseline, (str, Path))
    assert os.path.exists(baseline)

    module = request.node.fspath  # current test file
    return os.path.join(baseline, "cassettes", module.purebasename)


def is_replaying_cop_access_token(vcr_cassette_dir: str) -> bool:
    # Let's assume that replaying <=> the token file exists.
    cop_access_token_k7 = os.path.join(vcr_cassette_dir, "cop_access_token.yaml")
    logging.debug(f"REPLAY? {cop_access_token_k7=!r} {os.path.exists(cop_access_token_k7)=}")
    return os.path.exists(cop_access_token_k7)


# @pytest.fixture(scope="module")
@pytest.fixture
def eodag_provider(
    # request: SubRequest,
    vcr_cassette_dir: str,
    record_mode: str,
    vcr_config: dict,
    pytestconfig: pytest.Config,
    module_mocker,
    eodag_whitelist,
) -> EodagProvider:
    # Hook used to generate cassettes with Copernicus when 2FA is used
    # In that case set $EODAG__COP_DATASPACE__AUTH__CREDENTIALS__TOTP before calling pytest, e.g.:
    # $> EODAG__COP_DATASPACE__AUTH__CREDENTIALS__TOTP=999999 pytest  -vvv  --log-cli-level=DEBUG -o log_cli=true --capture=no --durations=0 --record-mode=once  eof/tests/test_eof.py 2>&1  | less -R

    # Since eodag v3, access token are validated, and working around the extra security in cleansed
    # cassettes is quite difficult and annoying => Let's mock the validation function instead, but
    # only in replay cases.
    # We can assume that when the cassette file exists, then the access token request is already
    # stored.
    if is_replaying_cop_access_token(vcr_cassette_dir):
        def no_op(slf):
            logging.info("Replaying K7, disabling cop_access_token verification!")
            slf.access_token = "REDACTED_access_token"
            return slf.access_token
        module_mocker.patch("eodag.plugins.authentication.keycloak.KeycloakOIDCPasswordAuth._get_access_token", no_op)
    with use_cassette('cop_access_token', vcr_cassette_dir, record_mode, [], vcr_config, pytestconfig):
        dag = EODataAccessGateway()
        provider = EodagProvider(dag)
        return provider


# =====[ Global Fixtures
@pytest.fixture
def eodag_config(request, vcr_cassette_dir: str) -> Optional[str]:
    # This fixture permits to configure the returned result for eodag_config name
    config_file = getattr(request, 'param', None)
    if config_file:
        logging.debug("Using specified eodag.yml: %r", config_file)
        return config_file
    if is_replaying_cop_access_token(vcr_cassette_dir):
        logging.debug("Using redacted eodag.yml file for replay: %r", DUMMY_EODAG)
        return DUMMY_EODAG
    logging.debug("Using actual eodag.yml file, if any")
    return None  # Using actual configuration file: this mode will be use to regenerate cassettes


@pytest.fixture
def eodag_whitelist(eodag_config: Optional[str]) -> Generator:
    # This fixture sets EODAG_PROVIDERS_WHITELIST appropriately for the duration of the test
    # It's almost redundant to eodag_config. Unfortunately, we cannot say "use only the providers
    # declared in the eodag.yml config file", hence this "duplication".
    if eodag_config == NO_EODAG:
        whitelist = ""
    else:
        whitelist = "cop_dataspace"
    with mock.patch.dict(os.environ, {"EODAG_PROVIDERS_WHITELIST": whitelist}):
        yield


@pytest.fixture
def dag(eodag_config: Optional[str], eodag_whitelist, request, record_mode: str) -> EODataAccessGateway:
    # logging.debug("dag(%s)", eodag_config)
    res = EODataAccessGateway(eodag_config)
    logging.debug("==========[ %s ]==================================================", request.node.originalname)
    logging.debug("=> Exact test            = %s", request.node.name)
    # logging.debug("=> record_mode           = %s", record_mode)
    # logging.debug("=> dag                   = %s <-- %s", res, eodag_config)
    provider_config = getattr(dag, 'providers_config', {})
    if provider_config:
        logging.debug("=> username?             = %s", getattr(providers_config["cop_dataspace"].auth, 'credentials', {})).get("username", "???")
    else:
        logging.debug("=> cop_dataspace UNKONWN")
    # logging.debug("=> dag._plugins_manager = %s", res._plugins_manager)

    # Clean any cached token information
    # | I'm not sure why/how several distinct (they have different ids) instances of
    # | plugins_manager.get_auth_plugin('cop_dataspace') may share a same token_info instance
    # | (they all have the same id)
    # search_plugins = res._plugins_manager.get_search_plugins(provider='cop_dataspace')
    # if search_plugins:
    #     logging.debug("=> token                 = %s", res._plugins_manager.get_auth_plugin(next(search_plugins)).token_info)
    #     res._plugins_manager.get_auth_plugin(next(search_plugins)).token_info = {}
    return res


# =====[ Direct tests on internal providers
DT1 = datetime(2020, 1, 1)   # 00:00:00
DT2 = datetime(2020, 1, 2, 23, 59, 59)   # 00:00:00
EXPECTED_NB = 4  # 1th, 2nd + 2 extra days before and after


# @pytest.mark.vcr
@pytest.mark.vcr(
    "cop_access_token.yaml", "test_cop_dataspace_eodag.yaml",
)
def test_cop_dataspace_eodag(
    dag: EODataAccessGateway,
    tmp_path_factory,
    eodag_provider: EodagProvider,
):
    assert eodag_provider.is_configured(dag)
    eofs = eodag_provider.search(DT1, DT2, ("S1A",))
    dest = tmp_path_factory.mktemp("s1tiling-cdse")
    files = eodag_provider.download(list(eofs), dest)
    assert len(files) == EXPECTED_NB

    # Register as well the request used in test_manager_eof_retrieval_edge_tests
    eofs = eodag_provider.search(DT1 - 2 * K_1D, DT2 - K_1D, ("S1A",))
    files = eodag_provider.download(list(eofs), dest)
    assert len(files) == EXPECTED_NB + 1


# =====[ Tests through public interface
class MockConfiguration:
    def __init__(
        self,
        first_date    : str,
        last_date     : str,
        eof_directory : AnyPath,
        platform_list : List[str],
        eodag_config  : Optional[str],
    ):
        self.first_date    = first_date
        self.last_date     = last_date
        self.platform_list = platform_list
        self.eodag_config  = eodag_config
        self.download      = True
        self.extra_directories = { 'eof_dir': eof_directory}


def make_configuration(
    tmp_path_factory,
    eodag_config,
    start: str = '2020-01-01',
    stop: str = '2020-01-02',
) -> MockConfiguration:
    logging.debug("configuration(%s, %s)", tmp_path_factory, eodag_config)
    assert tmp_path_factory
    return MockConfiguration(
        start,
        stop,
        tmp_path_factory.mktemp('config'),
        ["S1A"],
        eodag_config,
    )

@pytest.fixture
def configuration(
    tmp_path_factory,
    eodag_config,
) -> MockConfiguration:
    return make_configuration(tmp_path_factory, eodag_config)

# cassettes names needs to be filenames; relative filenames are OK; => extension are required!!
NO_EODAG    = os.path.join(DATA_DIR, 'dummy-empty-eodag.yml')
DUMMY_EODAG = os.path.join(DATA_DIR, 'dummy-redacted-eodag.yml')

logging.debug("eodag config file: %s", NO_EODAG)
assert os.path.isfile(NO_EODAG)


@pytest.mark.vcr(
    "cop_access_token.yaml", "test_cop_dataspace_eodag.yaml",
)
@pytest.mark.parametrize(
    "eodag_config",
    [
        (None),
    ],
    indirect=["eodag_config"],
)
def test_manager_with_provider(eodag_config, configuration, dag, baseline_dir, eodag_provider):
    assert os.path.exists(baseline_dir)
    assert os.path.exists(os.path.join(baseline_dir, 'cassettes'))
    logging.debug('test_manager_with_provider(%s)', eodag_config)
    # dummy-empty => Copernicus not configured
    # Otherwise, we expect the eodag.yaml config file of testing-user is configured for Copernicus Dataspace.
    is_configured_for_dataspace = eodag_config != NO_EODAG

    assert is_configured_for_dataspace == EodagProvider.is_configured(dag)
    assert is_configured_for_dataspace
    assert eodag_provider.is_configured(dag)

    manager = EOFFileManager(configuration, dag)
    res = manager.do_download_eof_files()
    assert len(res) == EXPECTED_NB


# @pytest.mark.vcr(
#     "cop_access_token.yaml",
# )
@pytest.mark.parametrize(
    "eodag_config",
    [
        (NO_EODAG),
    ],
    indirect=["eodag_config"],
)
def test_manager_no_provider(eodag_config, eodag_whitelist, configuration, dag):
    logging.debug('test_manager_no_provider(%s)', eodag_config)
    # dummy-empty => Copernicus not configured
    # Otherwise, we expect the eodag.yaml config file of testing-user is configured for Copernicus Dataspace.
    assert os.getenv('EODAG_PROVIDERS_WHITELIST', None) == ""

    assert not EodagProvider.is_configured(dag)

    manager = EOFFileManager(configuration, dag)
    res = manager.do_download_eof_files()
    assert len(res) == 1
    assert not res[0].has_value()


# =====[ Tests orbit conversions
def test_orbit_conversions():
    s1a_converter = ORBIT_CONVERTERS["S1A"]

    assert s1a_converter.to_relative(30632) == 110
    assert s1a_converter.to_relative(30704) == 7
    assert s1a_converter.to_relative(51107) == 110

    assert s1a_converter.closest_absolute(30632, 109) == 30806
    assert s1a_converter.closest_absolute(30632, 110) == 30632
    assert s1a_converter.closest_absolute(30704,   7) == 30704
    assert s1a_converter.closest_absolute(51107, 110) == 51107

    assert s1a_converter.closest_absolute(30631, 110) == 30632
    assert s1a_converter.closest_absolute(30806, 110) == 30807
    assert s1a_converter.closest_absolute(30633, 110) == 30632 + 175


# =====[ Test XML analyse of EOF file
def eof_id_to_file(dirname: Path, eof_id: str) -> Path:
    return dirname / f"S1A_OPER_AUX_POEORB_OPOD_{eof_id}.EOF"


@pytest.mark.parametrize(
    "eof_id,expected_abs_min,expected_abs_max,expected_rel_min,expected_rel_max",
    [
        ('20231107T080717_V20231017T225942_20231019T005942', 50811, 50827, 164,   5),
        ('20231127T070702_V20231106T225942_20231108T005942', 51103, 51118, 106, 121),
        ('20231128T070717_V20231107T225942_20231109T005942', 51117, 51133, 120, 136),
        ('20231207T070724_V20231116T225942_20231118T005942', 51248, 51264,  76,  92),
        ('20231208T070704_V20231117T225942_20231119T005942', 51263, 51279,  91, 107),
    ],
)
def test_min_max_orbits(
    eof_id: str,
    expected_abs_min: int, expected_abs_max: int,
    expected_rel_min: int, expected_rel_max: int,
    baseline_dir: Path
):
    full_path = eof_id_to_file(baseline_dir / "eofs", eof_id)
    abs_min, abs_max = extract_min_max_abs_orbit_numbers(full_path)
    assert abs_min == expected_abs_min
    assert abs_max == expected_abs_max

    sof = SentinelOrbitFile(full_path)
    assert sof.mission         == "S1A"
    assert sof.first_abs_orbit == expected_abs_min
    assert sof.last_abs_orbit  == expected_abs_max
    assert sof.first_rel_orbit == expected_rel_min
    assert sof.last_rel_orbit  == expected_rel_max


# =====[ Test manager search_for
@pytest.fixture
def tmp_eof_dir(tmp_path_factory) -> Path:
    dest_dir = tmp_path_factory.mktemp("s1tiling-out_eofs")
    eof_files = glob_eof_files(dest_dir)
    assert len(eof_files) == 0
    return dest_dir


@pytest.fixture
def eof_baseline_dir(baseline_dir: Path) -> Path:
    # return Path('/home/lhermitt/dev/S1tiling/tests/20200306-NR/test-montagne/run8.1.2/out/_EOF')
    return baseline_dir / "eofs"


def prepare_tmp_eof_dir_from_files(eof_files: List[SentinelOrbitFile], tmp_eof_dir):
    for eof_product in eof_files:
        eof_file = Path(eof_product.filename)
        eof_name = eof_file.name
        dest     = tmp_eof_dir / eof_name
        dest.symlink_to(eof_file)


def prepare_tmp_eof_dir_from_ids(eof_baseline_dir, tmp_eof_dir, eof_ids):
    for eof_id in eof_ids:
        eof_file = eof_id_to_file(eof_baseline_dir, eof_id)
        dest     = eof_id_to_file(tmp_eof_dir, eof_id)
        dest.symlink_to(eof_file)


@pytest.mark.parametrize(
    "eof_ids",
    [[
        '20231107T080717_V20231017T225942_20231019T005942',
        '20231127T070702_V20231106T225942_20231108T005942',
        '20231128T070717_V20231107T225942_20231109T005942',
        '20231207T070724_V20231116T225942_20231118T005942',
        '20231208T070704_V20231117T225942_20231119T005942',
    ]],
)
def test_manager_dir_analysis_simple_filter(
    eof_ids         : List[str],
    eof_baseline_dir: Path,
    tmp_eof_dir     : Path,
):
    assert len(eof_ids) == 5
    orig_eof_files = glob_eof_files(eof_baseline_dir)
    assert len(orig_eof_files) == 6

    eof_files = glob_eof_files(tmp_eof_dir)
    assert len(eof_files) == 0

    prepare_tmp_eof_dir_from_ids(eof_baseline_dir, tmp_eof_dir, eof_ids)

    eof_files = glob_eof_files(tmp_eof_dir)
    assert len(eof_files) == len(eof_ids)

    dt1 = datetime(2020, 1, 1)   # 00:00:00
    dt2 = datetime(2020, 1, 2, 23, 59, 59)   # 00:00:00
    eof_files_in_range = filter_intersecting_eof_file_list(eof_files, dt1, dt2)
    assert len(eof_files_in_range) == 0

    dt1 = datetime(2020, 1, 1)   # 00:00:00
    dt2 = datetime(2023, 10, 30, 23, 59, 59)   # 00:00:00
    eof_files_in_range = filter_intersecting_eof_file_list(eof_files, dt1, dt2)
    assert len(eof_files_in_range) == 1
    assert eof_files_in_range[0].filename == eof_files[0].filename
    for eof_file in eof_files_in_range:
        for orbit in orbit_range(eof_file):
            filtered_eofs = filter_eof_files_according_to_orbit_and_mission(eof_files, [orbit], 0)
            assert len(filtered_eofs) == 1

            assert eof_file in [eof[orbit] for eof in filtered_eofs if orbit in eof], (
                f"{eof_file.first_rel_orbit} <= {orbit} <= {eof_file.last_rel_orbit} failed for {eof_file}"
            )

    dt1 = datetime(2023, 11, 1)   # 00:00:00
    dt2 = datetime(2023, 11, 10, 23, 59, 59)   # 00:00:00
    eof_files_in_range = filter_intersecting_eof_file_list(eof_files, dt1, dt2)
    assert len(eof_files_in_range) == 2
    assert eof_files_in_range[0].filename == eof_files[1].filename
    assert eof_files_in_range[1].filename == eof_files[2].filename
    for eof_file in eof_files_in_range:
        for orbit in orbit_range(eof_file):
            filtered_eofs = filter_eof_files_according_to_orbit_and_mission(eof_files, [orbit], 0)

            assert eof_file in [eof[orbit] for eof in filtered_eofs if orbit in eof], (
                f"{eof_file.first_rel_orbit} <= {orbit} <= {eof_file.last_rel_orbit} failed for {eof_file}"
            )
            uniq_eofs = keep_one_eof_per_orbit(filtered_eofs, dt1, dt2, ("S1A", "S1B"))
            assert len(uniq_eofs) == 1


@pytest.mark.parametrize(
    "eof_ids",
    [[
        '20231107T080717_V20231017T225942_20231019T005942',
        '20231127T070702_V20231106T225942_20231108T005942',
        '20231128T070717_V20231107T225942_20231109T005942',
        '20231207T070724_V20231116T225942_20231118T005942',
        '20231208T070704_V20231117T225942_20231119T005942',
        '20210318T180016_V20201204T225942_20201206T005942',
    ]],
)
def test_manager_dir_analysis_actual_filter(
    eof_ids         : List[str],
    eof_baseline_dir: Path,
    tmp_eof_dir     : Path,
):
    assert len(eof_ids) == 6
    orig_eof_files = glob_eof_files(eof_baseline_dir)
    # assert len(orig_eof_files) == 6

    eof_files = glob_eof_files(tmp_eof_dir)
    logging.debug("All baseline/EOF:\n%s", pprint.pformat(orig_eof_files))
    assert len(eof_files) == 0

    prepare_tmp_eof_dir_from_ids(eof_baseline_dir, tmp_eof_dir, eof_ids)
    # prepare_tmp_eof_dir_from_files(orig_eof_files, tmp_eof_dir)

    eof_files = glob_eof_files(tmp_eof_dir)
    logging.debug("All tmp/EOF:\n%s", pprint.pformat(eof_files))
    assert len(eof_files) == len(eof_ids)

    # Regarding EOF files in the baseline:
    # - 001 appears once
    # - 106 & 107 appear at the edge of the baseline EOF files, and twice
    # - 110 appears once
    # - 130 appears twice: @ 2020-12-05 and 2023-11-08
    orbits = [107, 110, 130, 1]
    # orbits = range(1, 175)
    missions = ('S1A', 'S1B')
    obt_filtered_eofs = filter_eof_files_according_to_orbit_and_mission(eof_files, orbits, 0)
    logging.debug("All EOF containing %s:\n%s", orbits, pprint.pformat(obt_filtered_eofs))
    assert len(obt_filtered_eofs) == 6

    ## Results will be found, but not in the time range
    dt1 = datetime(2020, 1, 1)   # 00:00:00
    dt2 = datetime(2020, 1, 2, 23, 59, 59)   # 00:00:00
    eof_files_in_range = keep_one_eof_per_orbit(obt_filtered_eofs, dt1, dt2, missions)
    logging.debug("EOF files for %s .. %s:\n%s", dt1, dt2, pprint.pformat(eof_files_in_range))
    assert len(eof_files_in_range) == 4
    assert set(eof_files_in_range.keys()).issuperset(orbits)
    for obt in eof_files_in_range:
        assert not eof_files_in_range[obt].does_intersect(dt1, dt2)
    fully_filtered_eofs, missing_eofs = filter_uniq_eofs(eof_files, dt1, dt2, orbits, missions)
    assert eof_files_in_range == fully_filtered_eofs
    assert not missing_eofs

    ## Results will be found, but not all in requested time range
    dt1 = datetime(2020, 1, 1)   # 00:00:00
    dt2 = datetime(2023, 10, 30, 23, 59, 59)   # 00:00:00
    eof_files_in_range = keep_one_eof_per_orbit(obt_filtered_eofs, dt1, dt2, missions)
    logging.debug("EOF files for %s .. %s:\n%s", dt1, dt2, pprint.pformat(eof_files_in_range))
    assert len(eof_files_in_range) == 4
    assert set(eof_files_in_range.keys()).issuperset(orbits)
    assert     eof_files_in_range[1].does_intersect(dt1, dt2)
    assert     eof_files_in_range[130].does_intersect(dt1, dt2)  # take the older in range
    assert not eof_files_in_range[110].does_intersect(dt1, dt2)
    assert not eof_files_in_range[107].does_intersect(dt1, dt2)
    fully_filtered_eofs, missing_eofs = filter_uniq_eofs(eof_files, dt1, dt2, orbits, missions)
    assert eof_files_in_range == fully_filtered_eofs
    assert not missing_eofs

    ## Results will be found, but not all in requested time range
    dt1 = datetime(2023, 11, 1)   # 00:00:00
    dt2 = datetime(2023, 11, 10, 23, 59, 59)   # 00:00:00
    eof_files_in_range = keep_one_eof_per_orbit(obt_filtered_eofs, dt1, dt2, missions)
    logging.debug("EOF files for %s .. %s:\n%s", dt1, dt2, pprint.pformat(eof_files_in_range))
    assert len(eof_files_in_range) == 4
    assert set(eof_files_in_range.keys()).issuperset(orbits)
    assert not eof_files_in_range[1].does_intersect(dt1, dt2)
    assert     eof_files_in_range[130].does_intersect(dt1, dt2)  # take the newer in range
    assert     eof_files_in_range[110].does_intersect(dt1, dt2)
    assert     eof_files_in_range[107].does_intersect(dt1, dt2)
    fully_filtered_eofs, missing_eofs = filter_uniq_eofs(eof_files, dt1, dt2, orbits, missions)
    assert eof_files_in_range == fully_filtered_eofs
    assert not missing_eofs


def test_filter_orbits_on_the_periphery(
    baseline_dir: Path,
):
    # When two EOF files follow each others, they should share two orbits.
    # > Makes sure we obtain only one, and the right one when requesting orbits on the periphery
    eof_ids = [
        '20231127T070702_V20231106T225942_20231108T005942',
        '20231128T070717_V20231107T225942_20231109T005942',
    ]
    eof_baseline_dir = baseline_dir / "eofs"
    eof_files = [SentinelOrbitFile(eof_id_to_file(eof_baseline_dir, eof_id)) for eof_id in eof_ids]

    assert len(eof_files) == 2

    # 1. Make sure the files contain what is required to test the algorithms
    assert eof_files[0].mission == eof_files[-1].mission, "They are from the same mission"
    N = eof_files[0].nb_orbits_in_mission

    # eof#1: [... 120, 121]
    # eof#2:     [120, 121, ...]
    ultimate_rel_obt_of_1st_eof    = eof_files[0].last_rel_orbit
    penultimate_rel_obt_of_1st_eof = (ultimate_rel_obt_of_1st_eof - 1 - 1) % N + 1

    first_rel_obt_of_2nd_eof = eof_files[-1].first_rel_orbit
    second_rel_obt_of_2nd_eof = (first_rel_obt_of_2nd_eof - 1 + 1) % N + 1

    assert penultimate_rel_obt_of_1st_eof == first_rel_obt_of_2nd_eof
    assert ultimate_rel_obt_of_1st_eof    == second_rel_obt_of_2nd_eof

    # 2. Do test filter_eof_files_according_to_orbit_and_mission with offset
    files1 = filter_eof_files_according_to_orbit_and_mission(eof_files, [first_rel_obt_of_2nd_eof], -1)
    assert len(files1) == 1
    assert files1[0][first_rel_obt_of_2nd_eof] is eof_files[0]

    files2 = filter_eof_files_according_to_orbit_and_mission(eof_files, [second_rel_obt_of_2nd_eof], -1)
    assert len(files2) == 1
    assert files2[0][second_rel_obt_of_2nd_eof] is eof_files[1]


@pytest.mark.vcr(
    "cop_access_token.yaml",
)
@pytest.mark.parametrize(
    "eof_ids",
    [[
        '20231107T080717_V20231017T225942_20231019T005942',
        '20231127T070702_V20231106T225942_20231108T005942',
        '20231128T070717_V20231107T225942_20231109T005942',
        '20231207T070724_V20231116T225942_20231118T005942',
        '20231208T070704_V20231117T225942_20231119T005942',
    ]],
)
def test_manager_analysis_of_cache(
    eof_ids         : List[str],
    eof_baseline_dir: Path,
    tmp_eof_dir     : Path,
    dag,
):
    assert len(eof_ids) == 5
    orig_eof_files = glob_eof_files(eof_baseline_dir)
    assert len(orig_eof_files) == 6

    eof_files = glob_eof_files(tmp_eof_dir)
    assert len(eof_files) == 0

    prepare_tmp_eof_dir_from_ids(eof_baseline_dir, tmp_eof_dir, eof_ids)

    eof_files = glob_eof_files(tmp_eof_dir)
    assert len(eof_files) == len(eof_ids)

    # Wider range, but we're looking for files already in cache
    cfg = MockConfiguration(
        "2023-11-01", "2023-11-10",
        tmp_eof_dir,
        [],
        None,
    )
    eof_manager = EOFFileManager(cfg, dag)

    obt_file_expectations = [
        (118, 1),
        (120, 1),
        (122, 2),
        (130, 2),
    ]
    for obt, file_id in obt_file_expectations:
        files_found = eof_manager.search_for([obt])
        assert len(files_found) == 1
        assert files_found[0].has_value()
        obt_found, eof_found = list(files_found[0].value().items())[0]
        # file_found    : AnyPath = files[0].value()
        file_found    : AnyPath = eof_found.filename
        file_expected : AnyPath = eof_files[file_id].filename
        logging.debug(f"{type(file_found)=}    ; {file_found=!r}")
        logging.debug(f"{type(file_expected)=} ; {file_expected=!r}")
        assert file_found == file_expected, f"Orbit {obt} not found in #{file_id} -> {files_found[0]!r}"
        assert obt_found == obt



@pytest.mark.vcr(
    "cop_access_token.yaml", "test_cop_dataspace_eodag.yaml",
)
@pytest.mark.parametrize(
    "eof_ids",
    [[
        '20210315T155112_V20191230T225942_20200101T005942',
        '20210316T161714_V20191231T225942_20200102T005942',
        '20210316T184157_V20200101T225942_20200103T005942',
        '20210316T190114_V20200102T225942_20200104T005942',
    ]],
)
def test_manager_eof_retrieval(
    eof_ids         : List[str],
    configuration,
    dag,
):
    assert configuration.eodag_config != NO_EODAG
    assert len(eof_ids) == 4

    tmp_eof_dir = configuration.extra_directories['eof_dir']
    eof_files = glob_eof_files(tmp_eof_dir)
    assert len(eof_files) == 0, "Cache dir should be empty when test starts"

    eof_manager = EOFFileManager(configuration, dag)

    obt_file_expectations = [
        ( 65, 0),
        ( 76, 0),
        ( 78, 1),
        ( 91, 1),
        ( 92, 2),
        (106, 2),
        (107, 3),
        (110, 3),
        (121, None)
    ]
    for obt, file_id in obt_file_expectations:
        files_found = eof_manager.search_for([obt])
        logging.debug("Files found for obt %s => %s", obt, files_found)
        assert len(files_found) <= 2
        if file_id is not None:
            assert len(files_found) >= 1
            assert files_found[0].has_value()
            obt_found, eof_found = list(files_found[0].value().items())[0]
            # file_found    : AnyPath = files_found[0].value()
            file_found    : AnyPath = eof_found.filename
            file_expected : AnyPath = SentinelOrbitFile(eof_id_to_file(tmp_eof_dir, eof_ids[file_id])).filename
            logging.debug(f"{file_found=}")
            logging.debug(f"{type(file_found)=}    ; {file_found=!r}")
            logging.debug(f"{type(file_expected)=} ; {file_expected=!r}")
            assert str(file_found) == str(file_expected), f"Orbit {obt} not found in #{file_id} -> {files_found[0]!r}"
        else:
            assert not files_found[0].has_value()

    eof_files = [SentinelOrbitFile(eof_id_to_file(tmp_eof_dir, eof_id)) for eof_id in eof_ids]
    assert eof_files == glob_eof_files(tmp_eof_dir), "They should have been downloaded eventually"


@pytest.mark.vcr(
    "cop_access_token.yaml", "test_cop_dataspace_eodag.yaml",
)
@pytest.mark.parametrize(
    "eodag_config,obt_list,start,stop,expected_found,expected_in_range",
    [
        # Checks on disk
        # - 80 & 100 are covered once
        (NO_EODAG, [80],      "2023-11-17", "2023-11-19", True, True),
        (NO_EODAG, [80, 100], "2023-11-17", "2023-11-19", True, True),
        # - 130 is covered twice on disk
        (NO_EODAG, [130],     "2023-11-07", "2023-11-09", True, True),
        (NO_EODAG, [130],     "2020-12-05", "2020-12-06", True, True),

        # Stuff on disk, not in time range
        (NO_EODAG, [4],       "2023-11-16", "2023-11-16", True, False),
        (None,     [4],       "2023-11-16", "2023-11-16", True, False),

        # Pure download
        (NO_EODAG, [70],      "2019-12-30", "2020-01-01", False, None),  # no provider, no DL
        (None,     [70],      "2019-12-30", "2020-01-01", True, True),  # do DL

        # Stuff on disk, on cassette, but found on disk
        (NO_EODAG, [80],      "2019-12-30", "2020-01-01", True, False),  # on disk, not in range
        (None,     [80],      "2019-12-30", "2020-01-01", True, False),  # on disk, not in range
        (None,     [70, 80],  "2019-12-30", "2020-01-01", True, True),  # become in range thanks to DL

        # Orbits not in the time ranges required
        # - EOF in time range on disk
        (NO_EODAG, [150],      "2023-11-17", "2023-11-19", False, None),
        (None,     [150],      "2023-11-17", "2023-11-19", False, None), # don't try to DL; would need some mocking to know whether download attempts were made...
        # - EOF not in time range on disk, but K7 yes
        (None,     [150],      "2019-12-30", "2020-01-01", False, None),  # do DL, but mocking required to test..
        # TODO: test hybrid sitution with stuff found, and stuff can cannot be found...
    ],
    indirect=["eodag_config"],
)
def test_manager_eof_retrieval_edge_tests(
    eof_baseline_dir: Path,
    dag,
    tmp_path_factory,
    # test fixture-parameters
    eodag_config,
    obt_list: List[int],
    start: str,
    stop: str,
    expected_found: bool,
    expected_in_range: Optional[bool],
):
    # Checks on disk
    logging.debug(f"Testing EDGE {obt_list=} {start=} {stop=}")

    config = make_configuration(tmp_path_factory, eodag_config, start, stop)
    eof_manager = EOFFileManager(config, dag)

    baseline_eof_files = glob_eof_files(eof_baseline_dir)
    tmp_eof_dir = config.extra_directories['eof_dir']
    prepare_tmp_eof_dir_from_files(baseline_eof_files, tmp_eof_dir)

    start_time = parse(start)
    stop_time = parse(stop)
    # obt_list = [80]
    search_results = eof_manager.search_for(obt_list, first_date=start_time, last_date=stop_time)
    assert len(search_results) == len(obt_list), f"Expected {len(obt_list)} EOF in {start}..{stop} for {obt_list}, got: {search_results}"
    for result in search_results:
        if expected_found:
            assert result
            result_info = result.value()
            assert len(result_info.keys()) == 1
            obt = list(result_info.keys())[0]
            assert obt in obt_list
            obt_list.remove(obt)
            product = result_info[obt]
            assert product.has_relative_orbit(obt)
            if expected_in_range:
                assert product.start_time <= stop_time + K_1D
                assert start_time         <= product.stop_time
            else:
                assert not (product.start_time <= stop_time + K_1D and
                            start_time         <= product.stop_time)
        else:
            assert not result
