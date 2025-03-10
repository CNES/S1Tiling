#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =========================================================================
#   Program:   S1LIAMap
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

"""
S1Tiling Command Line Interface

Usage: S1LIAMap [OPTIONS] CONFIGFILE

  Generate Incidence Angle map(s) to the WGS84 Ellipsoid.

  It performs the following steps:
   1- Download EOF files on the selected time range and relative orbits
   2- Compute the selected LIA map(s)

  Parameters have to be set by the user in the S1Processor.cfg file

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.
  --dryrun    Display the processing shall would be realized, but none is done.
  --debug-otb Investigation mode were OTB Applications are directly used without Dask
              in order to run them through gdb for instance.
  --graphs    Generate task graphs showing the processing flow that need to be done.
"""

from __future__ import absolute_import, print_function, unicode_literals

from typing import NoReturn

import click

from s1tiling.libs.cli import cli_main
from s1tiling.libs.api import s1_process_lia
from s1tiling.__meta__ import __version__, __pages__


# ======================================================================
# S1LIAMap
@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog=f"""\b
    This tools is part of S1Tiling {__version__}. See also: S1IAMap, S1GammaAreaMap, S1Processor

    \b
    Check out our docs at {__pages__} for more details.
    Copyright 2017-2025 (c) CNES.
    Copyright 2022-2024 (c) CS GROUP France.
    """
)
@click.version_option()
@click.option(
        "--trace-errors",
        is_flag=True,
        help="Display error full traceback, if any",
)
@click.option(
        "--dryrun",
        is_flag=True,
        help="Display the processing shall would be realized, but none is done.")
@click.option(
        "--debug-otb",
        is_flag=True,
        help="Investigation mode were OTB Applications are directly used without Dask in order to run them through gdb for instance.")
@click.option(
        "--debug-caches",
        is_flag=True,
        help="Investigation mode were intermediary cached files are not purged.")
@click.option(
        "--watch-ram",
        is_flag=True,
        help="Trigger investigation mode for watching memory usage")
@click.option(
        "--graphs", "debug_tasks",
        is_flag=True,
        help="Generate SVG images showing task graphs of the processing flows")
@click.argument('config_filename', type=click.Path(exists=True))
def run_lia(
        config_filename,
        **kwargs  # All click parameters that'll directly be forwarded to s1_process_lia
) -> NoReturn:
    """
    Generates maps of Local Incidence Angles for Sentinel-1 orbits over S2 MGRS tiles.

    These maps can be used for NORMLIM σ° calibration.
    """
    cli_main(
        s1_process_lia,
        config_filename,
        **kwargs
    )


# ======================================================================
if __name__ == '__main__':  # Required for Dask: https://github.com/dask/distributed/issues/2422
    run_lia()  # pylint: disable=no-value-for-parameter
