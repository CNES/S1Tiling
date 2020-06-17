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
# import numpy as np
# from PIL import Image
# from subprocess import Popen
# import multiprocessing
import gdal, rasterio
from rasterio.windows import Window
# import subprocess
# import datetime
import logging
# import logging.handlers
# from contextlib import redirect_stdout
from s1tiling import S1FileManager
from s1tiling import S1FilteringProcessor
from s1tiling import Utils
from s1tiling.configuration import Configuration

from s1tiling.otbpipeline import Processing, FirstStep, Store
from s1tiling.otbwrappers import AnalyseBorders, Calibrate, CutBorders, OrthoRectify, Concatenate, BuildBorderMask, SmoothBorderMask

def remove_files(files):
    """
    Removes the files from the disk
    """
    logging.debug("Remove %s", files)
    return
    for file_it in files:
        if os.path.exists(file_it):
            os.remove(file_it)


class Sentinel1PreProcess():
    """ This class handles the processing for Sentinel1 ortho-rectification """
    def __init__(self,cfg):
        try:
            os.remove("S1ProcessorErr.log.log")
            os.remove("S1ProcessorOut.log")
        except os.error:
            pass
        self.cfg=cfg

    def generate_border_mask(self, all_ortho):
        """
        This method generate the border mask files from the
        orthorectified images.

        Args:
          all_ortho: A list of ortho-rectified S1 images
          """
        cmd_bandmath = []
        cmd_morpho = []
        files_to_remove = []
        logging.info("Generate Mask ...")
        for current_ortho in all_ortho:
            if "vv" not in current_ortho:
                continue
            working_directory, basename = os.path.split(current_ortho)
            name_border_mask            = basename.replace(".tif", "_BorderMask.tif")
            name_border_mask_tmp        = basename.replace(".tif", "_BorderMask_TMP.tif")
            pathname_border_mask_tmp    = os.path.join(working_directory, name_border_mask_tmp)

            files_to_remove.append(pathname_border_mask_tmp)
            cmd_bandmath.append(['    Mask building of '+name_border_mask_tmp,
                'export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={};'.format(self.cfg.OTBThreads)+'otbcli_BandMath -ram '\
                +str(self.cfg.ram_per_process)\
                +' -il '+current_ortho\
                +' -out '+pathname_border_mask_tmp\
                +' uint8 -exp "im1b1==0?0:1"'])

            #due to threshold approximation

            cmd_morpho.append(['    Mask smoothing of '+name_border_mask,
                'export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={};'.format(self.cfg.OTBThreads)+"otbcli_BinaryMorphologicalOperation -ram "\
                +str(self.cfg.ram_per_process)+" -progress false -in "\
                +pathname_border_mask_tmp\
                +" -out "\
                +os.path.join(working_directory, name_border_mask)\
                +" uint8 -structype ball"\
                +" -structype.ball.xradius 5"\
                +" -structype.ball.yradius 5 -filter opening"])

        self.run_processing(cmd_bandmath, title="   Mask building")
        self.run_processing(cmd_morpho,   title="   Mask smoothing")
        remove_files(files_to_remove)
        logging.info("Generate Mask done")

    def run_processing(self, cmd_list, title=""):
        """
        This method executes a given command.
        Args:
          cmd_list: the command to run
          title: optional title
        """
        import time
        nb_cmd = len(cmd_list)

        with multiprocessing.Pool(self.cfg.nb_procs, worker_config, [self.cfg.log_queue]) as pool:
            self.cfg.log_queue_listener.start()
            for count, result in enumerate(pool.imap_unordered(execute_command, cmd_list), 1):
                logging.info("%s correctly finished", result)
                logging.info(' --> %s... %s%%', title, count*100./nb_cmd)

            pool.close()
            pool.join()
            self.cfg.log_queue_listener.stop()

        logging.info("%s done", title)


# Main code

if len(sys.argv) != 2:
    print("Usage: "+sys.argv[0]+" config.cfg")
    sys.exit(1)

CFG = sys.argv[1]
Cg_Cfg=Configuration(CFG)
S1_CHAIN = Sentinel1PreProcess(Cg_Cfg)
S1_FILE_MANAGER = S1FileManager.S1FileManager(Cg_Cfg)


TILES_TO_PROCESS = []

ALL_REQUESTED = False

for tile_it in Cg_Cfg.tiles_list:
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
                [AnalyseBorders, Calibrate, CutBorders, OrthoRectify])
        inputs = []
        for raster, tile_origin in intersect_raster_list:
            manifest = raster.get_manifest()
            for image in raster.get_images_list():
                start = FirstStep(tile_name=tile_it, tile_origin=tile_origin, manifest=manifest, basename=image)
                inputs += [start]
        # process.process(inputs)

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

            start = FirstStep(tile_name=tile_it, basename=output_image, out_filename=images_to_concatenate)
            inputs += [start]
            for i in image_sublist:
                image_list.remove(i)
            logging.info("Concat %s --> %s", image_sublist, output_image)
        process = Processing(Cg_Cfg)
        process.register_pipeline(steps)
        process.process(inputs)

    ##if Cg_Cfg.mask_cond:
    ##    with Utils.ExecutionTimer("Generate Border Mask", True) as t:
    ##        S1_CHAIN.generate_border_mask(raster_tiles_list)

    """
    if Cg_Cfg.filtering_activated:
        with Utils.ExecutionTimer("MultiTemp Filter", True) as t:
            filteringProcessor.process(tile_it)
    """
