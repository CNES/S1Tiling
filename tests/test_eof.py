#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import List

import pytest
from pytest_recording._vcr import use_cassette
from _pytest.fixtures import SubRequest

from eodag.api.core import EODataAccessGateway

from s1tiling.libs.orbit._providers   import ASFProvider, DataspaceProvider
from s1tiling.libs.orbit._manager     import EOFFileManager, ProviderKind
from s1tiling.libs.orbit._conversions import ORBIT_CONVERTERS
from s1tiling.libs.orbit._file        import (
        SentinelOrbitFile,
        extract_min_max_abs_orbit_numbers,
        filter_intersecting_eof_files,
        filter_eof_files_containing_orbit,
        glob_eof_files,
        orbit_range,
)

logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("vcr").setLevel(logging.WARNING)
logging.getLogger("sentineleof").setLevel(logging.WARNING)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')

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
        try:
            decoded_body = json.loads(body_string)
            for key in ("access_token", "refresh_token"):
                if key in decoded_body:
                    decoded_body[key] = f"REDACTED_{key}"
            response["body"]["string"] = bytes(json.dumps(decoded_body), 'utf8')
        except json.decoder.JSONDecodeError:
            pass
    return response


@pytest.fixture(scope="module")
def vcr_config():
    """
    Tweak the cassette recorder to remove secrets from queries and responses
    """
    return {
            "filter_headers"             : ["authorization", "Cookie"],
            "filter_query_parameters"    : ["username", "password", "totp"],
            "filter_post_data_parameters": ["username", "password", "totp"],
            "before_record_response"     : [filter_response],
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


@pytest.fixture(scope="module")
def cop_access_token(
        # request: SubRequest,
        vcr_cassette_dir: str,
        record_mode: str,
        vcr_config: dict,
        pytestconfig: pytest.Config
):
    # Hook used to generate cassettes with Copernicus when 2FA is used
    # In that case set $EOF_CDSE_2FA_TOKEN and $NETRC before calling pytest, e.g.:
    # $> EOF_CDSE_2FA_TOKEN=999999 NETRC=~/.config/.netrc pytest  -vvv  --log-cli-level=DEBUG -o log_cli=true --capture=no --durations=0 --record-mode=once  eof/tests/test_eof.py 2>&1  | less -R
    with use_cassette('cop_access_token', vcr_cassette_dir, record_mode, [], vcr_config, pytestconfig):
        dag = EODataAccessGateway()
        provider = DataspaceProvider(dag)
        token = provider.get_token()
        assert token, "Invalid (empty) copernicus datasapce access token"
        return token


# =====[ Global Fixtures
@pytest.fixture
def eodag_config(request):
    # This fixture permits to configure the returned result for eodag_config name
    return getattr(request, 'param', None)

@pytest.fixture
def dag(eodag_config):
    # logging.debug("dag(%s)", eodag_config)
    res = EODataAccessGateway(eodag_config)
    # logging.debug("=> dag                  = %s", res)
    # logging.debug("=> dag._plugins_manager = %s", res._plugins_manager)

    # Clean any cached token information
    # | I'm not sure why/how several distinct (they have different ids) instances of
    # | plugins_manager.get_auth_plugin('cop_dataspace') may share a same token_info instance
    # | (they all have the same id)
    res._plugins_manager.get_auth_plugin('cop_dataspace').token_info = {}
    return res


# =====[ Direct tests on internal providers
DT1 = datetime(2020, 1, 1)   # 00:00:00
DT2 = datetime(2020, 1, 2, 23, 59, 59)   # 00:00:00
EXPECTED_NB = 4  # 1th, 2nd + 2 extra days before and after


@pytest.mark.vcr
def test_cop_dataspace(dag, tmp_path_factory, cop_access_token):
    # dag.providers_config["cop_dataspace"].auth.credentials["totp"] = '999999'
    # EODAG__COP_DATASPACE__AUTH__CREDENTIALS__TOTP
    provider = DataspaceProvider(dag, access_token=cop_access_token)
    eofs = provider.search(DT1, DT2, ("S1A",))
    dest = tmp_path_factory.mktemp("s1tiling-cdse")
    files = provider.download(eofs, dest)
    assert len(files) == EXPECTED_NB


@pytest.mark.vcr
def test_earthdata(tmp_path_factory, baseline_dir):
    assert os.path.exists(baseline_dir)
    assert os.path.exists(os.path.join(baseline_dir, 'cassettes'))
    provider = ASFProvider(cache_dir=baseline_dir)
    eofs = provider.search(DT1, DT2, ("S1A",))
    dest = tmp_path_factory.mktemp("s1tiling-asf")
    files = provider.download(eofs, dest)
    assert len(files) == EXPECTED_NB


# =====[ Tests through public interface
class MockConfiguration:
    def __init__(self, first_date, last_date, eof_directory, platform_list, eodag_config):
        self.first_date    = first_date
        self.last_date     = last_date
        self.eof_directory = eof_directory
        self.platform_list = platform_list
        self.eodag_config  = eodag_config
        self.download      = True


@pytest.fixture
def configuration(tmp_path_factory, eodag_config):
    logging.debug("configuration(%s, %s)", tmp_path_factory, eodag_config)
    assert tmp_path_factory
    return MockConfiguration(
            '2020-01-01',
            '2020-01-02',
            tmp_path_factory.mktemp('config'),
            ["S1A"],
            eodag_config,
    )


# cassettes names needs to be filenames; relative filenames are OK; => extension are required!!
DUMMY_EODAG = os.path.join(DATA_DIR, 'dummy-empty-eodag.yml')
DUMMY_NETRC = os.path.join(DATA_DIR, 'dummy-empty-netrc')


@pytest.mark.vcr(
        "cop_access_token.yaml", "test_cop_dataspace.yaml", "test_earthdata.yaml",
)
@pytest.mark.parametrize(
        "eodag_config,netrc",
        [
            (None,        None),
            (None,        DUMMY_NETRC),
            (DUMMY_EODAG, None),
        ],
        indirect=["eodag_config"],
)
def test_manager_with_provider(eodag_config, netrc, configuration, dag, baseline_dir):
    assert os.path.exists(baseline_dir)
    assert os.path.exists(os.path.join(baseline_dir, 'cassettes'))
    logging.debug('test_manager_with_provider(%s, %s)', eodag_config, netrc)
    # dummy-empty => Copernicus not configured
    # Otherwise, we expect the eodag.yaml config file of testing-user is configured for Copernicus Dataspace.
    is_configured_for_dataspace = eodag_config is None

    assert is_configured_for_dataspace == DataspaceProvider.is_configured(dag)

    with pytest.MonkeyPatch.context() as mp:
        if netrc is not None:
            mp.setenv('NETRC', netrc)
        assert (not netrc) or os.getenv('NETRC') == netrc
        is_configured_for_earthdata = netrc is None
        assert is_configured_for_earthdata == ASFProvider.is_configured(dag)
        assert is_configured_for_dataspace or is_configured_for_earthdata

        manager = EOFFileManager(configuration, dag)
        manager.add_extra_build_option(ProviderKind.EARTHDATA, cache_dir=baseline_dir)
        res = manager.download_eof()
        assert len(res) == EXPECTED_NB


@pytest.mark.parametrize(
        "eodag_config,netrc",
        [
            (DUMMY_EODAG, DUMMY_NETRC),
        ],
        indirect=["eodag_config"],
)
def test_manager_no_provider(eodag_config, netrc, configuration, dag):
    logging.debug('test_manager_no_provider(%s, %s)', eodag_config, netrc)
    # dummy-empty => Copernicus not configured
    # Otherwise, we expect the eodag.yaml config file of testing-user is configured for Copernicus Dataspace.

    assert not DataspaceProvider.is_configured(dag)

    with pytest.MonkeyPatch.context() as mp:
        if netrc is not None:
            mp.setenv('NETRC', netrc)
        assert os.getenv('NETRC') == netrc
        assert not ASFProvider.is_configured(dag)
        manager = EOFFileManager(configuration, dag)
        res = manager.download_eof()
        assert len(res) == 1
        assert not res[0].has_value()


# =====[ Tests orbit conversions
def test_orbit_conversions():
    s1a_converter = ORBIT_CONVERTERS["S1A"]

    assert s1a_converter.to_relative(30632) == 110
    assert s1a_converter.to_relative(30704) == 7
    assert s1a_converter.to_relative(51107) == 110

    assert s1a_converter.closest_absolute(30632, 110) == 30632
    assert s1a_converter.closest_absolute(30704,   7) == 30704
    assert s1a_converter.closest_absolute(51107, 110) == 51107

    assert s1a_converter.closest_absolute(30631, 110) == 30632
    assert s1a_converter.closest_absolute(30806, 110) == 30807
    # assert s1a_converter.closest_absolute(30633, 110) == 30632 + 175


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
def test_manager_dir_analysis(
        eof_ids     : List[str],
        baseline_dir: Path,
        tmp_path_factory,
):
    assert len(eof_ids) == 5
    eof_baseline_dir = baseline_dir / "eofs"
    orig_eof_files = glob_eof_files(eof_baseline_dir)
    assert len(orig_eof_files) == 5

    dest_dir = tmp_path_factory.mktemp("s1tiling-out_eofs")
    eof_files = glob_eof_files(dest_dir)
    assert len(eof_files) == 0

    for eof_id in eof_ids:
        eof_file = eof_id_to_file(eof_baseline_dir, eof_id)
        dest     = eof_id_to_file(dest_dir, eof_id)
        dest.symlink_to(eof_file)

    eof_files = glob_eof_files(dest_dir)
    assert len(eof_files) == len(eof_ids)

    dt1 = datetime(2020, 1, 1)   # 00:00:00
    dt2 = datetime(2020, 1, 2, 23, 59, 59)   # 00:00:00
    eof_files_in_range = filter_intersecting_eof_files(eof_files, dt1, dt2)
    assert len(eof_files_in_range) == 0

    dt1 = datetime(2020, 1, 1)   # 00:00:00
    dt2 = datetime(2023, 10, 30, 23, 59, 59)   # 00:00:00
    eof_files_in_range = filter_intersecting_eof_files(eof_files, dt1, dt2)
    assert len(eof_files_in_range) == 1
    assert eof_files_in_range[0].filename == eof_files[0].filename
    for eof_file in eof_files_in_range:
        for orbit in orbit_range(eof_file):
            assert eof_file in filter_eof_files_containing_orbit(eof_files_in_range, orbit), (
                    f"{eof_file.first_rel_orbit} <= {orbit} <= {eof_file.last_rel_orbit} failed for {eof_file}"
            )

    dt1 = datetime(2023, 11, 1)   # 00:00:00
    dt2 = datetime(2023, 11, 10, 23, 59, 59)   # 00:00:00
    eof_files_in_range = filter_intersecting_eof_files(eof_files, dt1, dt2)
    assert len(eof_files_in_range) == 2
    assert eof_files_in_range[0].filename == eof_files[1].filename
    assert eof_files_in_range[1].filename == eof_files[2].filename
    for eof_file in eof_files_in_range:
        for orbit in orbit_range(eof_file):
            assert eof_file in filter_eof_files_containing_orbit(eof_files_in_range, orbit), (
                    f"{eof_file.first_rel_orbit} <= {orbit} <= {eof_file.last_rel_orbit} failed for {eof_file}"
            )

    

