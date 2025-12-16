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

import subprocess
import sys

from setuptools import setup


def request_gdal_version() -> str:
    try:
        r = subprocess.run(['gdal-config', '--version'], stdout=subprocess.PIPE )
        version = r.stdout.decode('utf-8').strip('\n')
        print("GDAL %s detected on the system, using 'gdal==%s'" % (version, version))
        return version
    except Exception:  # pylint: disable=broad-except
        return '3.9.0'

extra_packages = []
if sys.version_info < (3,11,0):
    extra_packages.append("typing_extensions")


# Hybrid dependencies configuration in order to have a dynamic version detection for GDAL python
# bindings.
# See https://johnscolaro.xyz/blog/dynamically-specify-dependencies-with-pyproject-toml
setup(
    install_requires=[
        "click",
        "dask[distributed]>=2022.8.1",
        "eodag>=3.9.1,<4",
        "gdal=="+request_gdal_version(),
        "graphviz",
        "lxml",     # already used by eodag actually
        "numpy<2",
        "objgraph", # leaks
        # "packaging", # version
        "portion",  # intervals
        "pympler", # leaks
        "pyyaml>=5.1",
        "rtree",
        # Any way to require OTB ?
    ] + extra_packages,
)
