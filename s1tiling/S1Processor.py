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
import os
from pathlib import Path
import sys

import click
from distributed.scheduler import KilledWorker
from dask.distributed import Client, LocalCluster

from s1tiling.libs.S1FileManager import S1FileManager, WorkspaceKinds
# from libs import S1FilteringProcessor
from s1tiling.libs import Utils
from s1tiling.libs.configuration import Configuration
from s1tiling.libs.otbpipeline import FirstStep, PipelineDescriptionSequence
from s1tiling.libs.otbwrappers import (
        ExtractSentinel1Metadata, AnalyseBorders, Calibrate, CorrectDenoising,
        CutBorders, OrthoRectify, Concatenate, BuildBorderMask,
        SmoothBorderMask, AgglomerateDEM, SARDEMProjection,
        SARCartesianMeanEstimation, ComputeNormals, ComputeLIA, filter_LIA,
        OrthoRectifyLIA, ConcatenateLIA, SelectBestCoverage,
        ApplyLIACalibration, SpatialDespeckle)
from s1tiling.libs import exits

# Graphs
from s1tiling.libs.vis import SimpleComputationGraph

# logger = None
logger = logging.getLogger('s1tiling.processor')

# Default configuration value for people using S1Tiling API functions s1_process, and s1_process_lia.
EODAG_DEFAULT_DOWNLOAD_WAIT    = 2   # If download fails, wait time in minutes between two download tries
EODAG_DEFAULT_DOWNLOAD_TIMEOUT = 20  # If download fails, maximum time in minutes before stop retrying to download


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

    logger.info('Requested tiles: %s', cfg.tile_list)

    all_requested = False
    tiles_to_process = []
    if cfg.tile_list[0] == "ALL":
        all_requested = True
    else:
        for tile in cfg.tile_list:
            if s1_file_manager.tile_exists(tile):
                tiles_to_process.append(tile)
            else:
                logger.warning("Tile %s does not exist, skipping ...", tile)

    # We can not require both to process all tiles covered by downloaded products
    # and and download all tiles

    if all_requested:
        if cfg.download and "ALL" in cfg.roi_by_tiles:
            logger.critical("Can not request to download 'ROI_by_tiles : ALL' if 'Tiles : ALL'."
                    + " Change either value or deactivate download instead")
            sys.exit(exits.CONFIG_ERROR)
        else:
            tiles_to_process = s1_file_manager.get_tiles_covered_by_products()
            logger.info("All tiles for which more than %s%% of the surface is covered by products will be produced: %s",
                    100 * cfg.tile_to_product_overlap_ratio, tiles_to_process)

    logger.info('The following tiles will be processed: %s', tiles_to_process)
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
        # Round coverage at 3 digits as tile footprint has a very limited precision
        current_coverage = round(current_coverage, 3)
        if current_coverage < 1.:
            logger.warning("Tile %s has insuficient SRTM coverage (%s%%)",
                    tile, 100 * current_coverage)
        else:
            logger.info("-> %s coverage = %s => OK", tile, current_coverage)

    # Remove duplicates
    needed_srtm_tiles = list(set(needed_srtm_tiles))
    return tiles_to_process_checked, needed_srtm_tiles


def check_srtm_tiles(cfg, srtm_tiles_id, srtm_suffix='.hgt'):
    """
    Check the SRTM tiles exist on disk.
    """
    res = True
    for srtm_tile in srtm_tiles_id:
        tile_path_hgt = Path(cfg.srtm, srtm_tile + srtm_suffix)
        if not tile_path_hgt.exists():
            res = False
            logger.critical("%s is missing!", tile_path_hgt)
    return res


def clean_logs(config, nb_workers):
    """
    Clean all the log files.
    Meant to be called once, at startup
    """
    filenames = []
    for _, cfg in config['handlers'].items():
        if 'filename' in cfg and '%' in cfg['filename']:
            pattern = cfg['filename'] % ('worker-%s',)
            filenames += [pattern%(w,) for w in range(nb_workers)]
    remove_files(filenames)


def setup_worker_logs(config, dask_worker):
    """
    Set-up the logger on Dask Worker.
    """
    d_logger = logging.getLogger('distributed.worker')
    r_logger = logging.getLogger()
    old_handlers = d_logger.handlers[:]

    for _, cfg in config['handlers'].items():
        if 'filename' in cfg and '{kind}' in cfg['filename']:
            cfg['mode']     = 'a'  # Make sure to not reset worker log file
            cfg['filename'] = cfg['filename'].format(kind=f"worker-{dask_worker.name}")

    logging.config.dictConfig(config)
    # Restore old dask.distributed handlers, and inject them in root handler as well
    for hdlr in old_handlers:
        d_logger.addHandler(hdlr)
        r_logger.addHandler(hdlr)  # <-- this way we send s1tiling messages to dask channel

    # From now on, redirect stdout/stderr messages to s1tiling
    Utils.RedirectStdToLogger(logging.getLogger('s1tiling'))


class DaskContext:
    """
    Custom context manager for :class:`dask.distributed.Client` +
    :class:`dask.distributed.LocalCluster` classes.
    """
    def __init__(self, config, debug_otb):
        self.__client    = None
        self.__cluster   = None
        self.__config    = config
        self.__debug_otb = debug_otb

    def __enter__(self):
        if not self.__debug_otb:
            clean_logs(self.__config.log_config, self.__config.nb_procs)
            self.__cluster = LocalCluster(
                    threads_per_worker=1, processes=True, n_workers=self.__config.nb_procs,
                    silence_logs=False)
            self.__client = Client(self.__cluster)
            # Work around: Cannot pickle local object in lambda...
            global the_config
            the_config = self.__config
            self.__client.register_worker_callbacks(
                    lambda dask_worker: setup_worker_logs(the_config.log_config, dask_worker))
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if self.__client:
            self.__client.close()
            self.__cluster.close()
        return False

    @property
    def client(self):
        """
        Return a :class:`dask.distributed.Client`
        """
        return self.__client


def _execute_tasks_debug(dsk, tile_name):
    """
    Execute the tasks directly, one after the other, without Dask layer.
    The objective is to be able to debug OTB applications.
    """
    tasks = list(Utils.tsort(dsk, dsk.keys(),
        lambda dasktask_data : [] if isinstance(dasktask_data, FirstStep) else dasktask_data[2])
        )
    logger.debug('%s tasks', len(tasks))
    for product in reversed(tasks):
        how = dsk[product]
        logger.debug('- task: %s <-- %s', product, how)
    logger.info('Executing tasks one after the other for %s (debugging OTB)', tile_name)
    results = []
    for product in reversed(tasks):
        how = dsk[product]
        logger.info('- execute: %s <-- %s', product, how)
        if not issubclass(type(how), FirstStep):
            results += [how[0](*list(how)[1:])]
    return results


def _execute_tasks_with_dask(dsk, tile_name, tile_idx, intersect_raster_list, required_products,
        client, pipelines, do_watch_ram, debug_tasks):
    """
    Execute the tasks in parallel through Dask.
    """
    for product, how in dsk.items():
        logger.debug('- task: %s <-- %s', product, how)

    if debug_tasks:
        SimpleComputationGraph().simple_graph(
                dsk, filename=f'tasks-{tile_idx+1}-{tile_name}.svg')
    logger.info('Start S1 -> S2 transformations for %s', tile_name)
    nb_tries = 2
    for run_attemp in range(1, nb_tries+1):
        try:
            results = client.get(dsk, required_products)
            return results
        except KilledWorker as e:
            logger.critical('%s', dir(e))
            logger.exception("Worker %s has been killed when processing %s on %s tile: (%s). Workers will be restarted: %s/%s",
                    e.last_worker.name, e.task, tile_name, e, run, nb_tries)
            # TODO: don't overwrite previous logs
            # And we'll need to use the synchronous=False parameter to be able to check
            # successful executions but then, how do we clean up futures and all??
            client.restart()
            # Update the list of remaining tasks
            if run_attemp < nb_tries:
                dsk, required_products = pipelines.generate_tasks(tile_name,
                        intersect_raster_list, do_watch_ram=do_watch_ram)
            else:
                raise
    return []


def process_one_tile(
        tile_name, tile_idx, tiles_nb,
        s1_file_manager, pipelines, client,
        required_workspaces, dl_wait, dl_timeout, searched_items_per_page,
        debug_otb=False, dryrun=False, do_watch_ram=False, debug_tasks=False):
    """
    Process one S2 tile.

    I.E. run the OTB pipeline on all the S1 images that match the S2 tile.
    """
    s1_file_manager.ensure_tile_workspaces_exist(tile_name, required_workspaces)

    logger.info("Processing tile %s (%s/%s)", tile_name, tile_idx + 1, tiles_nb)

    s1_file_manager.keep_X_latest_S1_files(1000, tile_name)

    try:
        with Utils.ExecutionTimer("Downloading images related to " + tile_name, True):
            s1_file_manager.download_images(tiles=tile_name,
                    dl_wait=dl_wait, dl_timeout=dl_timeout,
                    searched_items_per_page=searched_items_per_page, dryrun=dryrun)
            # download_images will have updated the list of know products
    except BaseException:  # pylint: disable=broad-except
        logger.exception('Cannot download S1 images associated to %s', tile_name)
        sys.exit(exits.DOWNLOAD_ERROR)

    with Utils.ExecutionTimer("Intersecting raster list w/ " + tile_name, True):
        intersect_raster_list = s1_file_manager.get_s1_intersect_by_tile(tile_name)
        logger.debug('%s products found to intersect %s: %s', len(intersect_raster_list), tile_name, intersect_raster_list)

    if len(intersect_raster_list) == 0:
        logger.info("No intersection with tile %s", tile_name)
        return []

    dsk, required_products = pipelines.generate_tasks(tile_name, intersect_raster_list,
            do_watch_ram=do_watch_ram)
    logger.debug('######################################################################')
    logger.debug('Summary of tasks related to S1 -> S2 transformations of %s', tile_name)
    if debug_otb:
        return _execute_tasks_debug(dsk, tile_name)
    else:
        return _execute_tasks_with_dask(dsk, tile_name, tile_idx, intersect_raster_list,
                required_products, client, pipelines, do_watch_ram, debug_tasks)


def read_config(config_opt):
    """
    The config_opt can be either the configuration filename or an already initialized configuration
    object
    """
    if isinstance(config_opt, str):
        return Configuration(config_opt)
    else:
        return config_opt


def do_process_with_pipeline(config_opt,
        pipeline_builder,
        dl_wait, dl_timeout,
        searched_items_per_page=20,
        dryrun=False,
        debug_caches=False,
        debug_otb=False,
        watch_ram=False,
        debug_tasks=False,
        ):
    """
    Internal function for executing pipelines.
    # TODO: parametrize tile loop, product download...
    """
    config = read_config(config_opt)

    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(config.OTBThreads)
    # For the OTB applications that don't receive the path as a parameter (like SARDEMProjection)
    # -> we set $OTB_GEOID_FILE
    os.environ["OTB_GEOID_FILE"] = config.GeoidFile
    with S1FileManager(config) as s1_file_manager:
        tiles_to_process = extract_tiles_to_process(config, s1_file_manager)
        if len(tiles_to_process) == 0:
            logger.critical("No existing tiles found, exiting ...")
            sys.exit(exits.NO_S2_TILE)

        tiles_to_process_checked, needed_srtm_tiles = check_tiles_to_process(
                tiles_to_process, s1_file_manager)

        logger.info("%s images to process on %s tiles",
                s1_file_manager.nb_images, tiles_to_process_checked)

        if len(tiles_to_process_checked) == 0:
            logger.critical("No tiles to process, exiting ...")
            sys.exit(exits.NO_S1_IMAGE)

        logger.info("Required SRTM tiles: %s", needed_srtm_tiles)

        if not check_srtm_tiles(config, needed_srtm_tiles):
            logger.critical("Some SRTM tiles are missing, exiting ...")
            sys.exit(exits.MISSING_SRTM)

        if not os.path.exists(config.GeoidFile):
            logger.critical("Geoid file does not exists (%s), exiting ...", config.GeoidFile)
            sys.exit(exits.MISSING_GEOID)

        # Prepare directories where to store temporary files
        # These directories won't be cleaned up automatically
        S1_tmp_dir = os.path.join(config.tmpdir, 'S1')
        os.makedirs(S1_tmp_dir, exist_ok=True)

        config.tmp_srtm_dir = s1_file_manager.tmpsrtmdir(needed_srtm_tiles)

        pipelines, required_workspaces = pipeline_builder(config, dryrun=dryrun, debug_caches=debug_caches)

        log_level = lambda res: logging.INFO if bool(res) else logging.WARNING
        with DaskContext(config, debug_otb) as dask_client:
            results = []
            for idx, tile_it in enumerate(tiles_to_process_checked):
                with Utils.ExecutionTimer("Processing of tile " + tile_it, True):
                    res = process_one_tile(
                            tile_it, idx, len(tiles_to_process_checked),
                            s1_file_manager, pipelines, dask_client.client,
                            required_workspaces,
                            dl_wait=dl_wait, dl_timeout=dl_timeout,
                            searched_items_per_page=searched_items_per_page,
                            debug_otb=debug_otb, dryrun=dryrun, do_watch_ram=watch_ram,
                            debug_tasks=debug_tasks)
                    results += res

            nb_errors_detected = sum(not bool(res) for res in results)

            skipped_for_download_failures = s1_file_manager.get_skipped_S2_products()
            results.extend([fp for fp in skipped_for_download_failures])

            logger.debug('#############################################################################')
            if nb_errors_detected + len(skipped_for_download_failures) > 0:
                logger.warning('Execution report: %s errors detected', nb_errors_detected + len(skipped_for_download_failures))
            else:
                logger.info('Execution report: no error detected')

            if results:
                for res in results:
                    logger.log(log_level(res), ' - %s', res)
            else:
                logger.info(' -> Nothing has been executed')

            download_failures = s1_file_manager.get_download_failures()
            download_timeouts = s1_file_manager.get_download_timeouts()
            return exits.Situation(
                    nb_computation_errors=nb_errors_detected,
                    nb_download_failures=len(download_failures),
                    nb_download_timeouts=len(download_timeouts)
                    )


def register_LIA_pipelines(pipelines: PipelineDescriptionSequence, produce_angles: bool):
    """
    Internal function that takes care to register all pipelines related to
    LIA map and sin(LIA) map.
    """
    dem = pipelines.register_pipeline([AgglomerateDEM], 'AgglomerateDEM',
            inputs={'insar': 'basename'})
    demproj = pipelines.register_pipeline([ExtractSentinel1Metadata, SARDEMProjection], 'SARDEMProjection', is_name_incremental=True,
            inputs={'insar': 'basename', 'indem': dem})
    xyz = pipelines.register_pipeline([SARCartesianMeanEstimation],                     'SARCartesianMeanEstimation',
            inputs={'insar': 'basename', 'indem': dem, 'indemproj': demproj})
    lia = pipelines.register_pipeline([ComputeNormals, ComputeLIA],                     'Normals|LIA', is_name_incremental=True,
            inputs={'xyz': xyz})

    # "inputs" parameter doesn't need to be specified in the following pipeline declarations
    # but we still use it for clarity!
    ortho           = pipelines.register_pipeline([filter_LIA('LIA'), OrthoRectifyLIA],        'OrthoLIA',  inputs={'in': lia}, is_name_incremental=True)
    concat          = pipelines.register_pipeline([ConcatenateLIA],                            'ConcatLIA', inputs={'in': ortho})
    pipelines.register_pipeline([SelectBestCoverage],                                          'SelectLIA', inputs={'in': concat}, product_required=produce_angles)

    ortho_sin       = pipelines.register_pipeline([filter_LIA('sin_LIA'), OrthoRectifyLIA],    'OrthoSinLIA',  inputs={'in': lia}, is_name_incremental=True)
    concat_sin      = pipelines.register_pipeline([ConcatenateLIA],                            'ConcatSinLIA', inputs={'in': ortho_sin})
    best_concat_sin = pipelines.register_pipeline([SelectBestCoverage],                        'SelectSinLIA', inputs={'in': concat_sin}, product_required=True)

    return best_concat_sin


def s1_process(config_opt,
        dl_wait=EODAG_DEFAULT_DOWNLOAD_WAIT,
        dl_timeout=EODAG_DEFAULT_DOWNLOAD_TIMEOUT,
        searched_items_per_page=20,
        dryrun=False,
        debug_otb=False,
        debug_caches=False,
        watch_ram=False,
        debug_tasks=False,
        cache_before_ortho=False):
    """
      On demand Ortho-rectification of Sentinel-1 data on Sentinel-2 grid.

      It performs the following steps:
      1. Download S1 images from S1 data provider (through eodag)
      2. Calibrate the S1 images to gamma0
      3. Orthorectify S1 images and cut their on geometric tiles
      4. Concatenate images from the same orbit on the same tile
      5. Build mask files

      Parameters have to be set by the user in the S1Processor.cfg file
    """
    def builder(config, dryrun, debug_caches):
        assert (not config.filter) or (config.keep_non_filtered_products or not config.mask_cond), \
                'Cannot purge non filtered products when mask are also produced!'

        chain_LIA_and_despeckle_inmemory    = config.filter and not config.keep_non_filtered_products
        chain_concat_and_despeckle_inmemory = False  # See issue #118

        pipelines = PipelineDescriptionSequence(config, dryrun=dryrun, debug_caches=debug_caches)

        # Calibration ... OrthoRectification
        calib_seq = [ExtractSentinel1Metadata, AnalyseBorders, Calibrate]
        if config.removethermalnoise:
            calib_seq += [CorrectDenoising]
        calib_seq += [CutBorders]

        if cache_before_ortho:
            pipelines.register_pipeline(calib_seq,      'PrepareForOrtho', product_required=False, is_name_incremental=True)
            pipelines.register_pipeline([OrthoRectify], 'OrthoRectify',    product_required=False)
        else:
            calib_seq += [OrthoRectify]
            pipelines.register_pipeline(calib_seq, 'FullOrtho', product_required=False, is_name_incremental=True)

        calibration_is_done_in_S1 = config.calibration_type in ['sigma', 'beta', 'gamma', 'dn']

        # Concatenation (... + Despeckle)  // not working yet, see issue #118
        concat_seq = [Concatenate]
        if chain_concat_and_despeckle_inmemory:
            concat_seq.append(SpatialDespeckle)
            need_to_keep_non_filtered_products = False
        else:
            need_to_keep_non_filtered_products = True

        concat_S2 = pipelines.register_pipeline(concat_seq, product_required=calibration_is_done_in_S1, is_name_incremental=True)
        last_product_S2 = concat_S2

        required_workspaces = [WorkspaceKinds.TILE]

        # LIA Calibration (...+ Despeckle)
        if config.calibration_type == 'normlim':
            apply_LIA_seq = [ApplyLIACalibration]
            if chain_LIA_and_despeckle_inmemory:
                apply_LIA_seq.append(SpatialDespeckle)
                need_to_keep_non_filtered_products = False
            else:
                need_to_keep_non_filtered_products = True

            concat_sin = register_LIA_pipelines(pipelines, produce_angles=config.produce_lia_map)
            apply_LIA = pipelines.register_pipeline(apply_LIA_seq, product_required=True,
                    inputs={'sin_LIA': concat_sin, 'concat_S2': concat_S2}, is_name_incremental=True)
            last_product_S2 = apply_LIA
            required_workspaces.append(WorkspaceKinds.LIA)

        # Masking
        if config.mask_cond:
            pipelines.register_pipeline([BuildBorderMask, SmoothBorderMask], 'GenerateMask',
                    product_required=True, inputs={'in': last_product_S2})

        # Despeckle in non-inmemory case
        if config.filter:
            # Use SpatialDespeckle, only if filter ∈ [lee, gammamap, frost, kuan]
            required_workspaces.append(WorkspaceKinds.FILTER)
            if need_to_keep_non_filtered_products:  # config.keep_non_filtered_products:
                # Define another pipeline if chaining cannot be done in memory
                pipelines.register_pipeline([SpatialDespeckle], product_required=True,
                        inputs={'in': last_product_S2})

        return pipelines, required_workspaces

    return do_process_with_pipeline(config_opt, builder,
            searched_items_per_page=searched_items_per_page,
            dl_wait=dl_wait, dl_timeout=dl_timeout,
            dryrun=dryrun,
            debug_otb=debug_otb,
            debug_caches=debug_caches,
            watch_ram=watch_ram,
            debug_tasks=debug_tasks,
            )


def s1_process_lia(config_opt,
        dl_wait=EODAG_DEFAULT_DOWNLOAD_WAIT,
        dl_timeout=EODAG_DEFAULT_DOWNLOAD_TIMEOUT,
        searched_items_per_page=20,
        dryrun=False,
        debug_otb=False,
        debug_caches=False,
        watch_ram=False,
        debug_tasks=False,
        ):
    """
      Generate Local Incidence Angle Maps on S2 geometry.

      1. Determine the S1 products to process
          Given a list of S2 tiles, we first determine the day that'll the best
          coverage of each S2 tile in terms of S1 products.

          In case there is no single day that gives the best coverage for all
          S2 tiles, we try to determine the best solution that minimizes the
          number of S1 products to download and process.
      2. Process these S1 products
    """
    def builder(config, dryrun, debug_caches):
        pipelines = PipelineDescriptionSequence(config, dryrun=dryrun, debug_caches=debug_caches)
        register_LIA_pipelines(pipelines, produce_angles=config.produce_lia_map)
        required_workspaces = [WorkspaceKinds.LIA]
        return pipelines, required_workspaces

    return do_process_with_pipeline(config_opt, builder,
            dl_wait=dl_wait, dl_timeout=dl_timeout,
            searched_items_per_page=searched_items_per_page,
            dryrun=dryrun,
            debug_caches=debug_caches,
            debug_otb=debug_otb,
            watch_ram=watch_ram,
            debug_tasks=debug_tasks,
            )

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

    Returns the number of tasks that could not be processed.
    """
    situation = s1_process( config_filename,
                dl_wait=eodag_download_wait, dl_timeout=eodag_download_timeout,
                searched_items_per_page=searched_items_per_page,
                dryrun=dryrun,
                debug_otb=debug_otb,
                debug_caches=debug_caches,
                watch_ram=watch_ram,
                debug_tasks=debug_tasks,
                cache_before_ortho=cache_before_ortho)
    sys.exit(situation.code)

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

    Returns the number of tasks that could not be processed.
    """
    situation = s1_process_lia( config_filename,
                dl_wait=eodag_download_wait, dl_timeout=eodag_download_timeout,
                searched_items_per_page=searched_items_per_page,
                dryrun=dryrun,
                debug_otb=debug_otb,
                debug_caches=debug_caches,
                watch_ram=watch_ram,
                debug_tasks=debug_tasks)
    sys.exit(situation.code)

# ======================================================================
if __name__ == '__main__':  # Required for Dask: https://github.com/dask/distributed/issues/2422
    run()
