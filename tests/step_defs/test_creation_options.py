#!/usr/bin/env python
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

"""BDD tests for creation_options"""

from typing import Any, Dict
import logging
import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from s1tiling.libs.configuration import (
    CreationOptionConfiguration,
    PIXEL_TYPES,
    _analyse_creation_option,
    _extended_filename,
    _split_option,
    pixel_type,
)

KEY = "dummy_key"

# ======================================================================
# Scenarios
scenarios(
    '../features/test_creation_options.feature',
)

# ======================================================================
# Fixtures

@pytest.fixture
def defaults() -> Dict:
    res : Dict[str, Any] = {}
    return res

## def _analyse_creation_option(
##     creation_options: Dict,
##     s_cos           : str,
##     key             : str,
##     throw           : Callable[[str], None],
## ) -> None:
##     assert s_cos, "Creation option string shall not be empty"
##     # logging.debug(" creation_options.%s = %s", key, s_cos)
##     # Default value is defined in associated StepFactories
##     l_cos = _split_option(s_cos)
##     cos = {}
##     if l_cos[0] in PIXEL_TYPES:
##         cos['pixel_type'] = l_cos[0]  # OTB_pixel_type
##         cos['gdal_options'] = l_cos[1:]
##     else:
##         cos['gdal_options'] = l_cos[0:]
##     for co in cos['gdal_options']:
##         KEY_PATTERN = re.compile(r'[A-Z_0-9]+=')
##         if not KEY_PATTERN.match(co):
##             # The only validation used is UPPERCASE=value
##             # We don't check against a list that may change over time. In that case the error will be caught later.
##             throw(f"{co} is not a valid GDAL creation option for {key}. Expected syntax is `<OPTIONNAME>=<value>`")
##
##     creation_options[key] = cos


class Configuration:
    def __init__(self):
        self.creation_options  = {}
        self.disable_streaming = {}

    def analyse_creation_option(self, s_cos: str, key: str) -> None:
        def _fail(msg):
            raise Exception(msg)

        if s_cos:
            _analyse_creation_option(
                self.creation_options,
                s_cos,
                key,
                _fail,
            )


@pytest.fixture
def configuration() -> CreationOptionConfiguration:
    res = Configuration()
    return res


@pytest.fixture
def the_option() -> str:
    res = ""
    return res


# ======================================================================
# Given steps

# ----------------------------------------------------------------------
# given steps on pixel_type
@given(parsers.parse("Default pixel type is {default_pixel_type}"))
def given_default_pixel_type(defaults, default_pixel_type : str) -> None:
    if default_pixel_type != "unset":
        defaults['pixel_type'] = default_pixel_type


# ----------------------------------------------------------------------
# given steps on extended_filename
@given(parsers.parse("Default extended filename is {default_extended_filename}"))
def given_default_extended_filename(defaults, default_extended_filename : str) -> None:
    if default_extended_filename != "unset":
        defaults['extended_filename'] = _split_option(default_extended_filename)


# ----------------------------------------------------------------------
# given steps on options set
@given(
    parsers.parse("Creation option is {option}"),
    target_fixture="the_option"
)
def given_creation_option(option: str) -> str:
    if option != "unset":
        return option
    return ""


# ======================================================================
# When steps


@when("Analysing creation option")
def when_analysing_creation_option(configuration, the_option) -> None:
    logging.debug("Configure with %r", the_option)
    configuration.analyse_creation_option(the_option, KEY)
    logging.debug("Deduced options: %r", configuration.creation_options.get(KEY, {}))


# ======================================================================
# Then steps

@then(parsers.parse("Pixel type is {expectation}"))
def then_pixel_type_is___(configuration, defaults, expectation) -> None:
    default_pixel_type = defaults.get("pixel_type", None)
    if expectation == "not specified":
        assert pixel_type(configuration, KEY, default_pixel_type) is None
    else:
        assert pixel_type(configuration, KEY, default_pixel_type) == PIXEL_TYPES[expectation]


@then(parsers.parse("Filename is {expectation}"))
def then_filename_is____extended(configuration, defaults, expectation) -> None:
    default_extended_filename = defaults.get("extended_filename", ())
    if expectation == "not extended":
        assert _extended_filename(configuration, KEY, default_extended_filename) == ""
    else:
        assert _extended_filename(configuration, KEY, default_extended_filename) == expectation
