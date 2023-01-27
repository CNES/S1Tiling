#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2023 (c) CNES. All rights reserved.
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

"""
S1Tiling Command Line Interface

Usage: S1Processor [OPTIONS] CONFIGFILE

  On demand Ortho-rectification of Sentinel-1 data on Sentinel-2 grid.

  It performs the following steps:
   1- Download S1 images from S1 data provider (through eodag)
   2- Calibrate the S1 images to gamma0
   3- Orthorectify S1 images and cut their on geometric tiles
   4- Concatenate images from the same orbit on the same tile
   5- Build mask files

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

import logging
from pathlib import Path
import sys

import click
from s1tiling.libs.api import s1_process, s1_process_lia
from s1tiling.libs.exits import translate_exception_into_exit_code

logger = None
# logger = logging.getLogger('s1tiling')

# ======================================================================
def cli_execute(processing, *args, **kwargs):
    """
    Factorize code common to all S1Tiling CLI entry points (exception
    translation into exit codes...)
    """
    situation = processing(*args, **kwargs)
    try:
        # situation = processing(*args, **kwargs)
        # logger.debug('nominal exit: %s', situation.code)
        return situation.code
    except BaseException as e:
        # Logger object won't always exist at this time (like in configuration
        # errors) hence we may use click report mechanism instead.
        if logger:
            logger.critical(e)
            # logger.exception(e)
        else:
            click.echo(f"Error: {e}", err=True)
        return translate_exception_into_exit_code(e)


# ======================================================================
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option()
@click.option(
        "--cache-before-ortho/--no-cache-before-ortho",
        is_flag=True,
        default=False,
        help="""Force to store Calibration|Cutting result on disk before orthorectorectification.

        BEWARE, this option will produce temporary files that you'll need to explicitely delete.""")
@click.option(
        "--searched_items_per_page",
        default=20,
        help="Number of products simultaneously requested by eodag"
        )
@click.option(
        "--eodag_download_timeout",
        default=20,
        help="If download fails, maximum time in mins before stop retrying to download (default: 20 mins)"
        )
@click.option(
        "--eodag_download_wait",
        default=2,
        help="If download fails, wait time in minutes between two download tries (default: 2 mins)"
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
def run( searched_items_per_page, dryrun, debug_caches, debug_otb, watch_ram,
         debug_tasks, cache_before_ortho, config_filename,
         eodag_download_wait, eodag_download_timeout):
    """
    This function is used as entry point to create console scripts with setuptools.
    """
    sys.exit(
            cli_execute(s1_process, config_filename,
                        dl_wait=eodag_download_wait, dl_timeout=eodag_download_timeout,
                        searched_items_per_page=searched_items_per_page,
                        dryrun=dryrun,
                        debug_otb=debug_otb,
                        debug_caches=debug_caches,
                        watch_ram=watch_ram,
                        debug_tasks=debug_tasks,
                        cache_before_ortho=cache_before_ortho))

# ======================================================================
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option()
@click.option(
        "--searched_items_per_page",
        default=20,
        help="Number of products simultaneously requested by eodag"
        )
@click.option(
        "--eodag_download_timeout",
        default=20,
        help="If download fails, maximum time in mins before stop retrying to download"
        )
@click.option(
        "--eodag_download_wait",
        default=2,
        help="If download fails, wait time in minutes between two download tries"
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
def run_lia( searched_items_per_page, dryrun, debug_otb, debug_caches, watch_ram,
         debug_tasks, config_filename, eodag_download_wait, eodag_download_timeout):
    """
    This function is used as entry point to create console scripts with setuptools.
    """
    sys.exit(
            cli_execute(s1_process_lia, config_filename,
                        dl_wait=eodag_download_wait, dl_timeout=eodag_download_timeout,
                        searched_items_per_page=searched_items_per_page,
                        dryrun=dryrun,
                        debug_otb=debug_otb,
                        debug_caches=debug_caches,
                        watch_ram=watch_ram,
                        debug_tasks=debug_tasks))

# ======================================================================
if __name__ == '__main__':  # Required for Dask: https://github.com/dask/distributed/issues/2422
    run()
