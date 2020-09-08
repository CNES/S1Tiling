#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2020 (c) CESBIO. All rights reserved.
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
This module contains a script to build temporal series of S1 images by tiles
It performs the following steps:
  1- Download S1 images from S1 data provider (through eodag)
  2- Calibrate the S1 images to gamma0
  3- Orthorectify S1 images and cut their on geometric tiles
  4- Concatenate images from the same orbit on the same tile
  5- Build mask files
  6- Filter images by using a multiimage filter

 Parameters have to be set by the user in the S1Processor.cfg file
"""

import logging
import os
import sys
# import dask.distributed
from dask.distributed import Client, LocalCluster

from libs.S1FileManager import S1FileManager
# from libs import S1FilteringProcessor
from libs import Utils
from libs.configuration import Configuration

from libs.otbpipeline import FirstStep, PipelineDescriptionSequence
from libs.otbwrappers import AnalyseBorders, Calibrate, CutBorders, OrthoRectify, Concatenate, BuildBorderMask, SmoothBorderMask

DRYRUN      = False  # Global to see what would be executed, without producing anything.
DEBUG_OTB   = False  # Global to run the pipeline through gdb and debug OTB applications.
DEBUG_TASKS = False  # Global to generate a SVG image for each task graph.

# Graphs
# import dask
if DEBUG_TASKS:
    from libs.vis import SimpleComputationGraph

logger = None
# logger = logging.getLogger('s1tiling')


def remove_files(files):
    """
    Removes the files from the disk
    """
    logger.debug("Remove %s", files)
    for file_it in files:
        if os.path.exists(file_it):
            os.remove(file_it)


def extract_tiles_to_process(cfg, s1_file_manager):
    """
    Deduce from the configuration all the tiles that need to be processed.
    """
    tiles_to_process = []

    all_requested = False

    for tile in cfg.tile_list:
        if tile == "ALL":
            all_requested = True
            break
        elif True:  # s1_file_manager.tile_exists(tile):
            tiles_to_process.append(tile)
        else:
            logger.info("Tile %s does not exist, skipping ...", tile)
    logger.info('Requested tiles: %s', cfg.tile_list)

    # We can not require both to process all tiles covered by downloaded products
    # and and download all tiles

    if all_requested:
        if cfg.download and "ALL" in cfg.roi_by_tiles:
            logger.critical("Can not request to download ROI_by_tiles : ALL if Tiles : ALL."
                    + " Use ROI_by_coordinates or deactivate download instead")
            sys.exit(1)
        else:
            tiles_to_process = s1_file_manager.get_tiles_covered_by_products()
            logger.info("All tiles for which more than %s%% of the surface is covered by products will be produced: %s",
                    100 * cfg.TileToProductOverlapRatio, tiles_to_process)

    return tiles_to_process


def check_tiles_to_process(tiles_to_process, s1_file_manager):
    """
    Search the SRTM tiles required to process the tiles to process.
    """
    needed_srtm_tiles = []
    tiles_to_process_checked = []  # TODO: don't they exactly match tiles_to_process?

    # Analyse SRTM coverage for MGRS tiles to be processed
    srtm_tiles_check = s1_file_manager.check_srtm_coverage(tiles_to_process)

    # For each MGRS tile to process
    for tile in tiles_to_process:
        logger.info("Check SRTM coverage for %s", tile)
        # Get SRTM tiles coverage statistics
        srtm_tiles = srtm_tiles_check[tile]
        current_coverage = 0
        current_needed_srtm_tiles = []
        # Compute global coverage
        for (srtm_tile, coverage) in srtm_tiles:
            current_needed_srtm_tiles.append(srtm_tile)
            current_coverage += coverage
        # If SRTM coverage of MGRS tile is enough, process it
        needed_srtm_tiles += current_needed_srtm_tiles
        tiles_to_process_checked.append(tile)
        if current_coverage < 1.:
            logger.warning("Tile %s has insuficient SRTM coverage (%s%%)",
                    tile, 100 * current_coverage)

    # Remove duplicates
    needed_srtm_tiles = list(set(needed_srtm_tiles))
    return tiles_to_process_checked, needed_srtm_tiles


def check_srtm_tiles(cfg, srtm_tiles):
    """
    Check the SRTM tiles exist on disk.
    """
    res = True
    for srtm_tile in srtm_tiles:
        tile_path = os.path.join(cfg.srtm, srtm_tile)
        if not os.path.exists(tile_path):
            res = False
            logger.critical("%s is missing!", tile_path)
    return res


def setup_worker_logs(config, dask_worker):
    """
    Set-up the logger on Dask Worker.
    """
    d_logger = logging.getLogger('distributed.worker')
    r_logger = logging.getLogger()
    old_handlers = d_logger.handlers[:]

    for _, cfg in config['handlers'].items():
        if 'filename' in cfg and '%' in cfg['filename']:
            cfg['filename'] = cfg['filename'] % ('worker-' + str(dask_worker.name),)

    logging.config.dictConfig(config)
    # Restore old dask.distributed handlers, and inject them in root handler as well
    for hdlr in old_handlers:
        d_logger.addHandler(hdlr)
        r_logger.addHandler(hdlr)  # <-- this way we send s1tiling messages to dask channel

    # From now on, redirect stdout/stderr messages to s1tiling
    Utils.RedirectStdToLogger(logging.getLogger('s1tiling'))


def process_one_tile(
        tile_name, tile_idx, tiles_nb,
        s1_file_manager, pipelines, client,
        debug_otb=False, dryrun=False):
    """
    Process one S2 tile.

    I.E. run the OTB pipeline on all the S1 images that match the S2 tile.
    """
    s1_file_manager.ensure_tile_workspaces_exist(tile_name)

    logger.info("Processing tile %s (%s/%s)", tile_name, tile_idx + 1, tiles_nb)

    s1_file_manager.keep_X_latest_S1_files(1000)

    with Utils.ExecutionTimer("Downloading images related to " + tile_name, True):
        s1_file_manager.download_images(tiles=tile_name)

    with Utils.ExecutionTimer("Intersecting raster list w/ " + tile_name, True):
        intersect_raster_list = s1_file_manager.get_s1_intersect_by_tile(tile_name)

    if len(intersect_raster_list) == 0:
        logger.info("No intersection with tile %s", tile_name)
        return []

    dsk, required_products = pipelines.generate_tasks(tile_name, intersect_raster_list,
            debug_otb=debug_otb, dryrun=dryrun)
    logger.debug('Summary of tasks related to S1 -> S2 transformations of %s', tile_name)
    results = []
    if debug_otb:
        for product, how in reversed(dsk):
            logger.debug('- task: %s <-- %s', product, how)
        logger.info('Executing tasks one after the other for %s (debugging OTB)', tile_name)
        for product, how in reversed(dsk):
            logger.info('- execute: %s <-- %s', product, how)
            if not issubclass(type(how), FirstStep):
                results += [how[0](*list(how)[1:])]
    else:
        for product, how in dsk.items():
            logger.debug('- task: %s <-- %s', product, how)

        if DEBUG_TASKS:
            SimpleComputationGraph().simple_graph(
                    dsk,
                    filename='tasks-%s-%s.svg' % (tile_idx + 1, tile_name))
        logger.info('Start S1 -> S2 transformations for %s', tile_name)
        results = client.get(dsk, required_products)
    return results


# Main code
def main(config_filename):
    """
    S1Processor main() function.
    """
    config = Configuration(config_filename)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(config.OTBThreads)
    global logger
    logger = logging.getLogger('s1tiling')
    with S1FileManager(config) as s1_file_manager:
        tiles_to_process = extract_tiles_to_process(config, s1_file_manager)
        if len(tiles_to_process) == 0:
            logger.critical("No existing tiles found, exiting ...")
            sys.exit(1)

        tiles_to_process_checked, needed_srtm_tiles = check_tiles_to_process(tiles_to_process, s1_file_manager)

        logger.info("%s images to process on %s tiles",
                s1_file_manager.nb_images, tiles_to_process_checked)

        if len(tiles_to_process_checked) == 0:
            logger.critical("No tiles to process, exiting ...")
            sys.exit(1)

        logger.info("Required SRTM tiles: %s", needed_srtm_tiles)

        if not check_srtm_tiles(config, needed_srtm_tiles):
            logger.critical("Some SRTM tiles are missing, exiting ...")
            sys.exit(1)

        if not os.path.exists(config.GeoidFile):
            logger.critical("Geoid file does not exists (%s), exiting ...", config.GeoidFile)
            sys.exit(1)

        # Prepare directories where to store temporary files
        # These directories won't be cleaned up automatically
        S1_tmp_dir = os.path.join(config.tmpdir, 'S1')
        os.makedirs(S1_tmp_dir, exist_ok=True)

        config.tmp_srtm_dir = s1_file_manager.tmpsrtmdir(needed_srtm_tiles)

        pipelines = PipelineDescriptionSequence(config)
        pipelines.register_pipeline([AnalyseBorders, Calibrate, CutBorders], 'PrepareForOrtho', product_required=False)
        pipelines.register_pipeline([OrthoRectify],                          'OrthoRectify',    product_required=False)
        pipelines.register_pipeline([Concatenate],                                              product_required=True)
        if config.mask_cond:
            pipelines.register_pipeline([BuildBorderMask, SmoothBorderMask], 'GenerateMask',    product_required=True)

        # filtering_processor = S1FilteringProcessor.S1FilteringProcessor(config)

        if not DEBUG_OTB:
            cluster = LocalCluster(threads_per_worker=1, processes=True, n_workers=config.nb_procs, silence_logs=False)
            client = Client(cluster)
            client.register_worker_callbacks(lambda dask_worker: setup_worker_logs(config.log_config, dask_worker))

        results = []
        for idx, tile_it in enumerate(tiles_to_process_checked):
            with Utils.ExecutionTimer("Processing of tile " + tile_it, True):
                res = process_one_tile(
                        tile_it, idx, len(tiles_to_process_checked),
                        s1_file_manager, pipelines, client,
                        debug_otb=DEBUG_OTB, dryrun=DRYRUN)
                results += res

        logger.info('Execution report:')
        if results:
            for res in results:
                logger.info(' - %s', res)
        else:
            logger.info(' -> Nothing has been executed')

if __name__ == '__main__':  # Required for Dask: https://github.com/dask/distributed/issues/2422
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " config.cfg")
        sys.exit(1)
    main(sys.argv[1])
