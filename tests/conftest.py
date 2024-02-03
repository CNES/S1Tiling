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

import os
import pathlib
import argparse
from pathlib import Path

import pytest

# - ${S1TILING_TEST_DATA_OUTPUT}
# - ${S1TILING_TEST_DATA_INPUT}
# - ${S1TILING_TEST_SRTM}
# - ${S1TILING_TEST_TMPDIR}
# - ${S1TILING_TEST_DOWNLOAD}
# - ${S1TILING_TEST_RAM}

def dir_path(path) -> Path:
    if os.path.isdir(path):
        return pathlib.Path(path)
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid directory")

def pytest_addoption(parser) -> None:
    crt_dir = pathlib.Path(__file__).parent.absolute()
    src_dir = crt_dir.parent.absolute()
    test_dir = (crt_dir.parent.parent / "tests").absolute()

    parser.addoption("--baselinedir", action="store",      default=crt_dir/'baseline',                 type=dir_path, help="Directory where the baseline is")
    parser.addoption("--outputdir",   action="store",      default=crt_dir/'output',                   type=dir_path, help="Directory where the S2 products will be generated. Don't forget to clean it eventually.")
    parser.addoption("--liadir",      action="store",      default=crt_dir/'LIAs',                     type=dir_path, help="Directory where the LIA products will be generated. Don't forget to clean it eventually.")
    parser.addoption("--tmpdir",      action="store",      default=crt_dir/'tmp',                      type=dir_path, help="Directory where the temporary files will be generated. Don't forget to clean it eventually.")
    parser.addoption("--demdir",     action="store",      default=os.getenv('SRTM_DIR', '$SRTM_DIR'), type=dir_path, help="Directory where DEM files are - default: $SRTM_DIR")
    parser.addoption("--ram",         action="store",      default='4096'                            , type=int     , help="Available RAM allocated to each OTB process")
    parser.addoption("--download",    action="store_true", default=False, help="Download the input files with eodag instead of using the compressed ones from the baseline. If true, raw S1 products will be downloaded into {tmpdir}/inputs")
    parser.addoption("--watch_ram",   action="store_true", default=False, help="Watch memory usage")

def pytest_generate_tests(metafunc) -> None:
    # print("metafunc ->", metafunc.function)
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_list = ['baselinedir', 'demdir', 'download', 'outputdir', 'tmpdir', 'liadir', 'watch_ram', 'ram']
    for option in option_list:
        value = getattr(metafunc.config.option, option)
        # print("%s ===> %s // %s" % (option, value, option in metafunc.fixturenames))
        # value = metafunc.config.option.baselinedir
        if option in metafunc.fixturenames and value is not None:
            metafunc.parametrize(option, [value])
    global the_baseline
    the_baseline = metafunc.config.option.baselinedir

crt_dir = pathlib.Path(__file__).parent.absolute()
the_baseline = crt_dir/'baseline'

@pytest.fixture
def baseline_dir():
    # pytest_generate_tests doesn't work to expose fixtures to pytest-bdd
    # Hence this dirty workaround. pytest_generate_tests sets the global
    # the_baseline that is returned then by this fixture...
    # ~> https://github.com/pytest-dev/pytest-bdd/issues/620
    global the_baseline
    return the_baseline
