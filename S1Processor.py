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

from s1tiling.otbpipeline import Processing, FirstStep, Store, PipelineDescriptionSequence
from s1tiling.otbwrappers import AnalyseBorders, Calibrate, CutBorders, OrthoRectify, Concatenate, BuildBorderMask, SmoothBorderMask

import dask.distributed
from dask.distributed import Client, LocalCluster

# dryrun=True
dryrun = False
DEBUG_OTB = False

def remove_files(files):
    """
    Removes the files from the disk
    """
    logging.debug("Remove %s", files)
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
            logging.info("Tile %s does not exist, skipping ...", tile_it)
    logging.info('Requested tiles: %s', cfg.tile_list)

    # We can not require both to process all tiles covered by downloaded products
    # and and download all tiles

    if ALL_REQUESTED:
        if cfg.pepsdownload and "ALL" in cfg.roi_by_tiles:
            logging.critical("Can not request to download ROI_by_tiles : ALL if Tiles : ALL."\
                +" Use ROI_by_coordinates or deactivate download instead")
            sys.exit(1)
        else:
            TILES_TO_PROCESS = s1_file_manager.get_tiles_covered_by_products()
            logging.info("All tiles for which more than "\
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
        logging.info("Check SRTM coverage for %s",tile_it)
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
            logging.warning("Tile %s has insuficient SRTM coverage (%s%%)",
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
            logging.critical(tile_path+" is missing!")
    return res

# Main code
import dask.core, dask.order
if __name__ == '__main__': # Required for Dask: https://github.com/dask/distributed/issues/2422
    if len(sys.argv) != 2:
        print("Usage: "+sys.argv[0]+" config.cfg")
        sys.exit(1)

    CFG = sys.argv[1]
    Cg_Cfg=Configuration(CFG)
    with S1FileManager.S1FileManager(Cg_Cfg) as S1_FILE_MANAGER:
        TILES_TO_PROCESS = extract_tiles_to_process(Cg_Cfg, S1_FILE_MANAGER)
        if len(TILES_TO_PROCESS) == 0:
            logging.critical("No existing tiles found, exiting ...")
            sys.exit(1)

        TILES_TO_PROCESS_CHECKED, NEEDED_SRTM_TILES = check_tiles_to_process(TILES_TO_PROCESS, S1_FILE_MANAGER)

        logging.info("%s images to process on %s tiles",
                S1_FILE_MANAGER.nb_images, TILES_TO_PROCESS_CHECKED)

        if len(TILES_TO_PROCESS_CHECKED) == 0:
            logging.critical("No tiles to process, exiting ...")
            sys.exit(1)

        logging.info("Required SRTM tiles: %s", NEEDED_SRTM_TILES)

        if not check_srtm_tiles(Cg_Cfg, NEEDED_SRTM_TILES):
            logging.critical("Some SRTM tiles are missing, exiting ...")
            sys.exit(1)

        if not os.path.exists(Cg_Cfg.GeoidFile):
            logging.critical("Geoid file does not exists (%s), exiting ...", Cg_Cfg.GeoidFile)
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

        cluster = LocalCluster(threads_per_worker=2, processes=True, n_workers=2)
        client = Client(cluster)

        for idx, tile_it in enumerate(TILES_TO_PROCESS_CHECKED):

            working_directory = os.path.join(Cg_Cfg.tmpdir, 'S2', tile_it)
            os.makedirs(working_directory, exist_ok=True)

            out_dir = os.path.join(Cg_Cfg.output_preprocess, tile_it)
            os.makedirs(out_dir, exist_ok=True)

            logging.info("Tile: "+tile_it+" ("+str(idx+1)+"/"+str(len(TILES_TO_PROCESS_CHECKED))+")")

            S1_FILE_MANAGER.keep_X_latest_S1_files(1000)

            with Utils.ExecutionTimer("Downloading tiles", True) as t:
                S1_FILE_MANAGER.download_images(tiles=tile_it)

            with Utils.ExecutionTimer("Intersecting raster list", True) as t:
                intersect_raster_list = S1_FILE_MANAGER.get_s1_intersect_by_tile(tile_it)

            if len(intersect_raster_list) == 0:
                logging.info("No intersection with tile %s",tile_it)
                continue

            dsk, final_products = pipelines.generate_tasks(tile_it, intersect_raster_list)
            for product, how in dsk.items():
                logging.debug('- task: %s <-- %s', product, how)
                if DEBUG_OTB:
                    if not issubclass(type(how), FirstStep):
                        how[0](*list(how)[1:])

            if not DEBUG_OTB:
                client.get(dsk, final_products)

            sys.exit(0)



            with Utils.ExecutionTimer("Calibration|Cut|Ortho", True) as t:
                process = Processing(Cg_Cfg)
                process.register_pipeline(
                        [AnalyseBorders, Calibrate, CutBorders, Store, OrthoRectify])
                        # [AnalyseBorders, Calibrate, CutBorders, OrthoRectify])
                inputs = []
                for raster, tile_origin in intersect_raster_list:
                    manifest = raster.get_manifest()
                    for image in raster.get_images_list():
                        start = FirstStep(tile_name=tile_it, tile_origin=tile_origin, manifest=manifest, basename=image, dryrun=dryrun)
                        inputs += [start]
                process.process(inputs)

            msg = "Concatenate"
            steps = [Concatenate]
            if Cg_Cfg.mask_cond:
                steps += [Store, BuildBorderMask, SmoothBorderMask]
                msg += "|Generate Border Mask"
            with Utils.ExecutionTimer(msg, True) as t:
                inputs = []
                image_list = [i.name for i in Utils.list_files(os.path.join(Cg_Cfg.tmpdir, 'S2', tile_it))
                        if (len(i.name) == 40 and "xxxxxx" not in i.name)]
                image_list.sort()

                while len(image_list) > 0:
                    image_sublist=[i for i in image_list if (image_list[0][:29] in i)]

                    if len(image_sublist) > 0:
                        images_to_concatenate=[os.path.join(Cg_Cfg.tmpdir, 'S2', tile_it,i) for i in image_sublist]
                        output_image = images_to_concatenate[0][:-10]+"xxxxxx"+images_to_concatenate[0][-4:]

                    start = FirstStep(tile_name=tile_it, basename=output_image, out_filename=images_to_concatenate, dryrun=dryrun)
                    inputs += [start]
                    for i in image_sublist:
                        image_list.remove(i)
                    logging.info("Concat %s --> %s", image_sublist, output_image)
                process = Processing(Cg_Cfg)
                process.register_pipeline(steps)
                process.process(inputs)

            """
            if Cg_Cfg.filtering_activated:
                with Utils.ExecutionTimer("MultiTemp Filter", True) as t:
                    filteringProcessor.process(tile_it)
            """

