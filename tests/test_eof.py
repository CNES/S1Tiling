#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import json
import logging
import os
from pathlib import Path

import pytest
from pytest_recording._vcr import use_cassette
from _pytest.fixtures import SubRequest

from eodag.api.core import EODataAccessGateway

from s1tiling.libs.orbit._providers import ASFProvider, DataspaceProvider
from s1tiling.libs.orbit._manager import EOFFileManager, ProviderKind

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
