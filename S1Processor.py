#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright (c) CESBIO. All rights reserved.
#
#   See LICENSE for details.
#
#   This software is distributed WITHOUT ANY WARRANTY; without even
#   the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#   PURPOSE.  See the above copyright notices for more information.
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
  1- Download S1 images from PEPS server
  2- Calibrate the S1 images to gamma0
  3- Orthorectify S1 images and cut their on geometric tiles
  4- Concatenate images from the same orbit on the same tile
  5- Build mask files
  6- Filter images by using a multiimage filter

 Parameters have to be set by the user in the S1Processor.cfg file
"""

import os
import pathlib
import sys
import glob
import shutil
import gdal, rasterio
from rasterio.windows import Window
import logging
from s1tiling import S1FileManager
from s1tiling import S1FilteringProcessor
from s1tiling import Utils
from s1tiling.configuration import Configuration

from s1tiling.otbpipeline import FirstStep, PipelineDescriptionSequence
from s1tiling.otbwrappers import AnalyseBorders, Calibrate, CutBorders, OrthoRectify, Concatenate, BuildBorderMask, SmoothBorderMask

import dask.distributed
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar

dryrun = False
DEBUG_OTB = False # Global that permits to run the pipeline through gdb and debug OTB applications.

def remove_files(files):
    """
    Removes the files from the disk
    """
    logger.debug("Remove %s", files)
    return
    for file_it in files:
        if os.path.exists(file_it):
            os.remove(file_it)


def extract_tiles_to_process(cfg, s1_file_manager):
    TILES_TO_PROCESS = []

    ALL_REQUESTED = False

    for tile_it in cfg.tile_list:
        if tile_it == "ALL":
            ALL_REQUESTED = True
            break
        elif True:  #s1_file_manager.tile_exists(tile_it):
            TILES_TO_PROCESS.append(tile_it)
        else:
            logger.info("Tile %s does not exist, skipping ...", tile_it)
    logger.info('Requested tiles: %s', cfg.tile_list)

    # We can not require both to process all tiles covered by downloaded products
    # and and download all tiles

    if ALL_REQUESTED:
        if cfg.pepsdownload and "ALL" in cfg.roi_by_tiles:
            logger.critical("Can not request to download ROI_by_tiles : ALL if Tiles : ALL."\
                +" Use ROI_by_coordinates or deactivate download instead")
            sys.exit(1)
        else:
            TILES_TO_PROCESS = s1_file_manager.get_tiles_covered_by_products()
            logger.info("All tiles for which more than "\
                +str(100*cfg.TileToProductOverlapRatio)\
                +"% of the surface is covered by products will be produced: "\
                +str(TILES_TO_PROCESS))

    return TILES_TO_PROCESS


def check_tiles_to_process(TILES_TO_PROCESS, s1_file_manager):
    NEEDED_SRTM_TILES = []
    TILES_TO_PROCESS_CHECKED = []

    # Analyse SRTM coverage for MGRS tiles to be processed
    SRTM_TILES_CHECK = s1_file_manager.check_srtm_coverage(TILES_TO_PROCESS)

    # For each MGRS tile to process
    for tile_it in TILES_TO_PROCESS:
        logger.info("Check SRTM coverage for %s",tile_it)
        # Get SRTM tiles coverage statistics
        srtm_tiles = SRTM_TILES_CHECK[tile_it]
        current_coverage = 0
        current_NEEDED_SRTM_TILES = []
        # Compute global coverage
        for (srtm_tile, coverage) in srtm_tiles:
            current_NEEDED_SRTM_TILES.append(srtm_tile)
            current_coverage += coverage
        # If SRTM coverage of MGRS tile is enough, process it
        NEEDED_SRTM_TILES += current_NEEDED_SRTM_TILES
        TILES_TO_PROCESS_CHECKED.append(tile_it)
        if current_coverage < 1.:
            logger.warning("Tile %s has insuficient SRTM coverage (%s%%)",
                    tile_it, 100*current_coverage)

    # Remove duplicates
    NEEDED_SRTM_TILES = list(set(NEEDED_SRTM_TILES))
    return TILES_TO_PROCESS_CHECKED, NEEDED_SRTM_TILES


def check_srtm_tiles(cfg, srtm_tiles):
    res = True
    for srtm_tile in NEEDED_SRTM_TILES:
        tile_path = os.path.join(cfg.srtm, srtm_tile)
        if not os.path.exists(tile_path):
            res = False
            logger.critical(tile_path+" is missing!")
    return res


def setup_worker_logs(config, dask_worker):
    d_logger = logging.getLogger('distributed.worker')
    r_logger = logging.getLogger()
    old_handlers = d_logger.handlers[:]

    for h, cfg in config['handlers'].items():
        if 'filename' in cfg and '%' in cfg['filename']:
            cfg['filename'] = cfg['filename'] % ('worker-'+str(dask_worker.name),)

    logging.config.dictConfig(config)
    # Restore old dask.distributed handlers, and inject them in root handler as well
    for hdlr in old_handlers:
        d_logger.addHandler(hdlr)
        r_logger.addHandler(hdlr) # <-- this way we send s1tiling messages to dask channel


def process_one_tile(
        tile_name, tile_idx, tiles_nb,
        config, s1_file_manager, pipelines,
        debug_otb=False, dryrun=False
        ):
    working_directory = os.path.join(config.tmpdir, 'S2', tile_name)
    os.makedirs(working_directory, exist_ok=True)

    out_dir = os.path.join(config.output_preprocess, tile_name)
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Processing tile %s (%s/%s)", tile_name, idx+1, tiles_nb)

    s1_file_manager.keep_X_latest_S1_files(1000)

    with Utils.ExecutionTimer("Downloading images related to "+tile_name, True) as t:
        s1_file_manager.download_images(tiles=tile_name)

    with Utils.ExecutionTimer("Intersecting raster list w/ "+tile_name, True) as t:
        intersect_raster_list = s1_file_manager.get_s1_intersect_by_tile(tile_name)

    if len(intersect_raster_list) == 0:
        logger.info("No intersection with tile %s", tile_name)
        return

    dsk, required_products = pipelines.generate_tasks(tile_name, intersect_raster_list, debug_otb=debug_otb)
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

        logger.info('Start S1 -> S2 transformations for %s', tile_name)
        results = client.get(dsk, required_products)
    return results


# Main code
if __name__ == '__main__': # Required for Dask: https://github.com/dask/distributed/issues/2422
    if len(sys.argv) != 2:
        print("Usage: "+sys.argv[0]+" config.cfg")
        sys.exit(1)

    CFG = sys.argv[1]
    Cg_Cfg=Configuration(CFG)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(Cg_Cfg.OTBThreads)
    logger = logging.getLogger('s1tiling')
    with S1FileManager.S1FileManager(Cg_Cfg) as S1_FILE_MANAGER:
        TILES_TO_PROCESS = extract_tiles_to_process(Cg_Cfg, S1_FILE_MANAGER)
        if len(TILES_TO_PROCESS) == 0:
            logger.critical("No existing tiles found, exiting ...")
            sys.exit(1)

        TILES_TO_PROCESS_CHECKED, NEEDED_SRTM_TILES = check_tiles_to_process(TILES_TO_PROCESS, S1_FILE_MANAGER)

        logger.info("%s images to process on %s tiles",
                S1_FILE_MANAGER.nb_images, TILES_TO_PROCESS_CHECKED)

        if len(TILES_TO_PROCESS_CHECKED) == 0:
            logger.critical("No tiles to process, exiting ...")
            sys.exit(1)

        logger.info("Required SRTM tiles: %s", NEEDED_SRTM_TILES)

        if not check_srtm_tiles(Cg_Cfg, NEEDED_SRTM_TILES):
            logger.critical("Some SRTM tiles are missing, exiting ...")
            sys.exit(1)

        if not os.path.exists(Cg_Cfg.GeoidFile):
            logger.critical("Geoid file does not exists (%s), exiting ...", Cg_Cfg.GeoidFile)
            sys.exit(1)

        # Prepare directories where to store temporary files
        # These directories won't be cleaned up automatically
        S1_tmp_dir = os.path.join(Cg_Cfg.tmpdir, 'S1')
        os.makedirs(S1_tmp_dir, exist_ok=True)

        Cg_Cfg.tmp_srtm_dir = S1_FILE_MANAGER.tmpsrtmdir(NEEDED_SRTM_TILES)

        pipelines = PipelineDescriptionSequence(Cg_Cfg)
        pipelines.register_pipeline([AnalyseBorders, Calibrate, CutBorders], 'PrepareForOrtho', product_required=False)
        pipelines.register_pipeline([OrthoRectify],                          'OrthoRectify',    product_required=False)
        pipelines.register_pipeline([Concatenate],                                              product_required=True)
        pipelines.register_pipeline([BuildBorderMask, SmoothBorderMask],     'GenerateMask',    product_required=True)

        filteringProcessor=S1FilteringProcessor.S1FilteringProcessor(Cg_Cfg)

        cluster = LocalCluster(threads_per_worker=1, processes=True, n_workers=Cg_Cfg.nb_procs, silence_logs=False)
        client = Client(cluster)
        client.register_worker_callbacks(lambda dask_worker: setup_worker_logs(Cg_Cfg.log_config, dask_worker))

        results = []
        for idx, tile_it in enumerate(TILES_TO_PROCESS_CHECKED):
            with Utils.ExecutionTimer("Processing of tile "+tile_it, True) as t:
                results += process_one_tile(
                        tile_it, idx, len(TILES_TO_PROCESS_CHECKED),
                        Cg_Cfg, S1_FILE_MANAGER, pipelines,
                        debug_otb=DEBUG_OTB)

            """
            if Cg_Cfg.filtering_activated:
                with Utils.ExecutionTimer("MultiTemp Filter", True) as t:
                    filteringProcessor.process(tile_it)
            """

        logger.info('Execution report:')
        if results:
            for r in results:
                logger.info(' - %s', r)
        else:
            logger.info(' -> Nothing has been executed')

