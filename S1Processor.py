#!/usr/bin/env python
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

from s1tiling.otbpipeline import Processing, FirstStep, Store
from s1tiling.otbwrappers import AnalyseBorders, Calibrate, CutBorders, OrthoRectify, Concatenate, BuildBorderMask, SmoothBorderMask

# dryrun=True
dryrun = False

def remove_files(files):
    """
    Removes the files from the disk
    """
    logging.debug("Remove %s", files)
    return
    for file_it in files:
        if os.path.exists(file_it):
            os.remove(file_it)


# Main code

if len(sys.argv) != 2:
    print("Usage: "+sys.argv[0]+" config.cfg")
    sys.exit(1)

CFG = sys.argv[1]
Cg_Cfg=Configuration(CFG)
S1_FILE_MANAGER = S1FileManager.S1FileManager(Cg_Cfg)


TILES_TO_PROCESS = []

ALL_REQUESTED = False

for tile_it in Cg_Cfg.tile_list:
    logging.info('Requesting to process tile %s', tile_it)
    if tile_it == "ALL":
        ALL_REQUESTED = True
        break
    elif True:  #S1_FILE_MANAGER.tile_exists(tile_it):
        TILES_TO_PROCESS.append(tile_it)
    else:
        logging.info("Tile %s does not exist, skipping ...", tile_it)

# We can not require both to process all tiles covered by downloaded products
# and and download all tiles


if ALL_REQUESTED:
    if Cg_Cfg.pepsdownload and "ALL" in Cg_Cfg.roi_by_tiles:
        logging.critical("Can not request to download ROI_by_tiles : ALL if Tiles : ALL."\
            +" Use ROI_by_coordinates or deactivate download instead")
        sys.exit(1)
    else:
        TILES_TO_PROCESS = S1_FILE_MANAGER.get_tiles_covered_by_products()
        logging.info("All tiles for which more than "\
            +str(100*Cg_Cfg.TileToProductOverlapRatio)\
            +"% of the surface is covered by products will be produced: "\
            +str(TILES_TO_PROCESS))

if len(TILES_TO_PROCESS) == 0:
    logging.critical("No existing tiles found, exiting ...")
    sys.exit(1)

# Analyse SRTM coverage for MGRS tiles to be processed
SRTM_TILES_CHECK = S1_FILE_MANAGER.check_srtm_coverage(TILES_TO_PROCESS)

NEEDED_SRTM_TILES = []
TILES_TO_PROCESS_CHECKED = []
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
    if current_coverage >= 1.:
        NEEDED_SRTM_TILES += current_NEEDED_SRTM_TILES
        TILES_TO_PROCESS_CHECKED.append(tile_it)
    else:
        # Skip it
        logging.warning("Tile %s has insuficient SRTM coverage (%s%%)",
                tile_it, 100*current_coverage)
        NEEDED_SRTM_TILES += current_NEEDED_SRTM_TILES
        TILES_TO_PROCESS_CHECKED.append(tile_it)


# Remove duplicates
NEEDED_SRTM_TILES = list(set(NEEDED_SRTM_TILES))

logging.info("%s images to process on %s tiles",
        S1_FILE_MANAGER.nb_images, TILES_TO_PROCESS_CHECKED)

if len(TILES_TO_PROCESS_CHECKED) == 0:
    logging.critical("No tiles to process, exiting ...")
    sys.exit(1)

logging.info("Required SRTM tiles: %s", NEEDED_SRTM_TILES)

SRTM_OK = True

for srtm_tile in NEEDED_SRTM_TILES:
    tile_path = os.path.join(Cg_Cfg.srtm, srtm_tile)
    if not os.path.exists(tile_path):
        SRTM_OK = False
        logging.critical(tile_path+" is missing")

if not SRTM_OK:
    logging.critical("Some SRTM tiles are missing, exiting ...")
    sys.exit(1)

if not os.path.exists(Cg_Cfg.GeoidFile):
    logging.critical("Geoid file does not exists (%s), exiting ...", Cg_Cfg.GeoidFile)
    sys.exit(1)

# copy all needed SRTM file in a temp directory for orthorectification processing
# TODO: clean this tmp directory at the end!
for srtm_tile in NEEDED_SRTM_TILES:
    os.symlink(os.path.join(Cg_Cfg.srtm,srtm_tile),os.path.join(S1_FILE_MANAGER.tmpsrtmdir,srtm_tile))


filteringProcessor=S1FilteringProcessor.S1FilteringProcessor(Cg_Cfg)

for idx, tile_it in enumerate(TILES_TO_PROCESS_CHECKED):

    working_directory = os.path.join(Cg_Cfg.tmpdir, tile_it)
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)

    out_dir = os.path.join(Cg_Cfg.output_preprocess, tile_it)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logging.info("Tile: "+tile_it+" ("+str(idx+1)+"/"+str(len(TILES_TO_PROCESS_CHECKED))+")")

    # keep only the 500's newer files
    safeFileList=sorted(glob.glob(os.path.join(Cg_Cfg.raw_directory,"*")),key=os.path.getctime)
    if len(safeFileList)> 1000	:
        for f in safeFileList[:len(safeFileList)-1000]:
            logging.debug("Remove : ",os.path.basename(f))
            shutil.rmtree(f,ignore_errors=True)
        S1_FILE_MANAGER.get_s1_img()

    with Utils.ExecutionTimer("Downloading tiles", True) as t:
        S1_FILE_MANAGER.download_images(tiles=tile_it)

    with Utils.ExecutionTimer("Intersecting raster list", True) as t:
        intersect_raster_list = S1_FILE_MANAGER.get_s1_intersect_by_tile(tile_it)

    if len(intersect_raster_list) == 0:
        logging.info("No intersection with tile %s",tile_it)
        continue

    Cg_Cfg.tmp_srtm_dir = S1_FILE_MANAGER.tmpsrtmdir
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
        image_list = [i.name for i in Utils.list_files(os.path.join(Cg_Cfg.tmpdir, tile_it))
                if (len(i.name) == 40 and "xxxxxx" not in i.name)]
        image_list.sort()

        while len(image_list) > 1:
            image_sublist=[i for i in image_list if (image_list[0][:29] in i)]

            if len(image_sublist) >1 :
                images_to_concatenate=[os.path.join(Cg_Cfg.tmpdir, tile_it,i) for i in image_sublist]
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
