#!/usr/bin/env python3
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

"""
Submodule that defines all API related functions and classes.
"""

import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

from distributed.scheduler import KilledWorker
from dask.distributed import Client, LocalCluster

from s1tiling.libs.vis import SimpleComputationGraph  # Graphs
from .S1FileManager import (
        S1FileManager, WorkspaceKinds, EODAG_DEFAULT_DOWNLOAD_WAIT, EODAG_DEFAULT_DOWNLOAD_TIMEOUT,
        EODAG_DEFAULT_SEARCH_MAX_RETRIES, EODAG_DEFAULT_SEARCH_ITEMS_PER_PAGE,
)
from . import exits
from . import exceptions
from . import Utils
from .configuration import Configuration
from .otbpipeline import FirstStep, PipelineDescription, PipelineDescriptionSequence, StepFactory, AbstractStep
from .otbwrappers import (
        # Main S1 -> S2 Step Factories
        ExtractSentinel1Metadata, AnalyseBorders, Calibrate, CorrectDenoising,
        CutBorders, OrthoRectify, Concatenate, BuildBorderMask, SmoothBorderMask,
        # LIA relate Step Factories
        AgglomerateDEMOnS2, ProjectDEMToS2Tile, ProjectGeoidToS2Tile,
        SumAllHeights, ComputeGroundAndSatPositionsOnDEM,
        ComputeLIAOnS2, filter_LIA, ComputeNormalsOnS2,
        ApplyLIACalibration,
        # Deprecated LIA related Step Factories
        AgglomerateDEMOnS1, SARDEMProjection, SARCartesianMeanEstimation,
        ComputeNormalsOnS1, OrthoRectifyLIA, ComputeLIAOnS1, ConcatenateLIA, SelectBestCoverage,
        # Filter Step Factories
        SpatialDespeckle)
from .outcome import Outcome


logger = logging.getLogger('s1tiling.api')


def remove_files(files: List[Union[str, Path]], what: str) -> None:
    """
    Removes the files from the disk
    """
    logger.debug("Remove %s: %s", what, files)
    for file_it in files:
        if os.path.exists(file_it):
            os.remove(file_it)


def extract_tiles_to_process(cfg: Configuration, s1_file_manager: S1FileManager) -> List[str]:
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
            # TODO: In order to avoid opening the Layer 42 times, Check all tiles at once
            if s1_file_manager.tile_exists(tile):
                tiles_to_process.append(tile)
            else:
                logger.warning("Tile %s does not exist, skipping ...", tile)

    # We can not require both to process all tiles covered by downloaded products
    # and and download all tiles

    if all_requested:
        # Check already done in the configuration object
        assert not (cfg.download and "ALL" in cfg.roi_by_tiles), \
            "Can not request to download 'ROI_by_tiles : ALL' if 'Tiles : ALL'. Change either value or deactivate download instead"
        tiles_to_process = s1_file_manager.get_tiles_covered_by_products()
        logger.info("All tiles for which more than %s%% of the surface is covered by products will be produced: %s",
                100 * cfg.tile_to_product_overlap_ratio, tiles_to_process)

    logger.info('The following tiles will be processed: %s', tiles_to_process)
    return tiles_to_process


def check_tiles_to_process(tiles_to_process: List[str], s1_file_manager: S1FileManager) -> Tuple[List[str], Dict, Dict[str, Dict]]:
    """
    Search the DEM tiles required to process the tiles to process.
    """
    needed_dem_tiles = {}
    tiles_to_process_checked = []  # TODO: don't they exactly match tiles_to_process?

    # Analyse DEM coverage for MGRS tiles to be processed
    dem_tiles_check = s1_file_manager.check_dem_coverage(tiles_to_process)

    # For each MGRS tile to process
    for tile in tiles_to_process:
        logger.info("Check DEM coverage for %s", tile)
        # Get DEM tiles coverage statistics
        dem_tiles = dem_tiles_check[tile]
        current_coverage = 0
        # Compute global coverage
        for _, dem_info in dem_tiles.items():
            current_coverage += dem_info['_coverage']
        needed_dem_tiles.update(dem_tiles)
        # If DEM coverage of MGRS tile is enough, process it
        tiles_to_process_checked.append(tile)
        # Round coverage at 3 digits as tile footprint has a very limited precision
        current_coverage = round(current_coverage, 3)
        if current_coverage < 1.:
            logger.warning("Tile %s has insufficient DEM coverage (%s%%)",
                    tile, 100 * current_coverage)
        else:
            logger.info("-> %s coverage = %s => OK", tile, current_coverage)

    # Remove duplicates
    return tiles_to_process_checked, needed_dem_tiles, dem_tiles_check


def check_dem_tiles(cfg: Configuration, dem_tile_infos: Dict) -> bool:
    """
    Check the DEM tiles exist on disk.
    """
    fmt = cfg.dem_filename_format
    res = True
    for _, dem_tile_info in dem_tile_infos.items():
        dem_filename = fmt.format_map(dem_tile_info)
        tile_path_hgt = Path(cfg.dem, dem_filename)
        # logger.debug('checking "%s" # "%s" =(%s)=> "%s"', cfg.dem, dem_filename, fmt, tile_path_hgt)
        if not tile_path_hgt.exists():
            res = False
            logger.critical("%s is missing!", tile_path_hgt)
    return res


def clean_logs(config: Dict, nb_workers: int) -> None:
    """
    Clean all the log files.
    Meant to be called once, at startup
    """
    filenames = []
    for _, cfg in config['handlers'].items():
        if 'filename' in cfg and '{kind}' in cfg['filename']:
            filenames += [cfg['filename'].format(kind=f"worker-{w}") for w in range(nb_workers)]
    remove_files(filenames, "logs")


def setup_worker_logs(config: Dict, dask_worker) -> None:
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


the_config : Configuration


class DaskContext:
    """
    Custom context manager for :class:`dask.distributed.Client` +
    :class:`dask.distributed.LocalCluster` classes.
    """
    def __init__(self, config: Configuration, debug_otb: bool) -> None:
        self.__client    : Optional[Client]       = None
        self.__cluster   : Optional[LocalCluster] = None
        self.__config    : Configuration          = config
        self.__debug_otb : bool                   = debug_otb

    def __enter__(self) -> "DaskContext":
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

    def __exit__(self, exception_type, exception_value, exception_traceback) -> Literal[False]:
        if self.__client:
            self.__client.close()
            assert self.__cluster, "client existence implies cluster existence"
            self.__cluster.close()
        return False

    @property
    def client(self) -> Optional[Client]:
        """
        Return a :class:`dask.distributed.Client`
        """
        return self.__client


def _how2str(how: Union[Tuple, AbstractStep]) -> str:
    """
    Make task definition from logger friendly
    """
    if isinstance(how, AbstractStep):
        return str(how)
    else:
        return f"Task(pipeline: {how[1]}; keys: {how[2]})"


def _execute_tasks_debug(dsk: Dict, tile_name: str) -> List:
    """
    Execute the tasks directly, one after the other, without Dask layer.
    The objective is to be able to debug OTB applications.
    """
    tasks = list(Utils.tsort(dsk, dsk.keys(),
        lambda dasktask_data : [] if isinstance(dasktask_data, FirstStep) else dasktask_data[2])
    )
    logger.debug('Debug execution of %s tasks', len(tasks))
    for product in reversed(tasks):
        how = dsk[product]
        logger.debug('- task: %s <-- %s', product, _how2str(how))
    logger.info('Executing tasks one after the other for %s (debugging OTB)', tile_name)
    results = []
    for product in reversed(tasks):
        how = dsk[product]
        logger.info('- execute: %s <-- %s', product, _how2str(how))
        if not issubclass(type(how), FirstStep):
            results += [how[0](*list(how)[1:])]
    return results


def _execute_tasks_with_dask(  # pylint: disable=too-many-arguments
    dsk:                   Dict[str, Union[Tuple, "FirstStep"]],
    tile_name:             str,
    tile_idx:              int,
    intersect_raster_list: List[Dict],
    required_products:     List[str],
    client:                Client,
    pipelines:             PipelineDescriptionSequence,
    do_watch_ram:          bool,
    debug_tasks:           bool
) -> List:
    """
    Execute the tasks in parallel through Dask.
    """
    if debug_tasks:
        SimpleComputationGraph().simple_graph(
                dsk, filename=f'tasks-{tile_idx+1}-{tile_name}.svg')
    logger.info('Start S1 -> S2 transformations for %s', tile_name)
    nb_tries = 2
    for run_attempt in range(1, nb_tries + 1):
        try:
            logger.debug("  Execute tasks, attempt #%s", run_attempt)
            results = client.get(dsk, required_products)
            return results
        except KilledWorker as e:
            logger.critical('%s', dir(e))
            logger.exception("Worker %s has been killed when processing %s on %s tile: (%s). Workers will be restarted: %s/%s",
                    e.last_worker.name, e.task, tile_name, e, run_attempt, nb_tries)
            # TODO: don't overwrite previous logs
            # And we'll need to use the synchronous=False parameter to be able to check
            # successful executions but then, how do we clean up futures and all??
            client.restart()
            # Update the list of remaining tasks
            if run_attempt < nb_tries:
                dsk, required_products = pipelines.generate_tasks(tile_name,
                        intersect_raster_list, do_watch_ram=do_watch_ram)
            else:
                raise
    return []


def process_one_tile(  # pylint: disable=too-many-arguments, too-many-locals
    tile_name:               str,
    tile_idx:                int,
    tiles_nb:                int,
    s1_file_manager:         S1FileManager,
    pipelines:               PipelineDescriptionSequence,
    client:                  Optional[Client],
    required_workspaces:     List[WorkspaceKinds],
    debug_otb:               bool = False,
    dryrun:                  bool = False,
    do_watch_ram:            bool = False,
    debug_tasks:             bool = False
) -> List:
    """
    Process one S2 tile.

    I.E. run the OTB pipeline on all the S1 images that match the S2 tile.
    """
    s1_file_manager.ensure_tile_workspaces_exist(tile_name, required_workspaces)

    logger.info("Processing tile %s (%s/%s)", tile_name, tile_idx + 1, tiles_nb)

    s1_file_manager.keep_X_latest_S1_files(1000, tile_name)

    try:
        with Utils.ExecutionTimer("Downloading images related to " + tile_name, True):
            s1_file_manager.download_images(tiles=[tile_name], dryrun=dryrun)
            # download_images will have updated the list of know products
    except RuntimeError as e:
        logger.warning('Cannot download S1 images associated to %s: %s', tile_name, e)
        return [Outcome(e)]

    except BaseException as e:
        logger.debug('Download error intercepted: %s', e)
        raise exceptions.DownloadS1FileError(tile_name)

    with Utils.ExecutionTimer("Intersecting raster list w/ " + tile_name, True):
        intersect_raster_list = s1_file_manager.get_s1_intersect_by_tile(tile_name)
        logger.debug('%s products found to intersect %s: %s', len(intersect_raster_list), tile_name, intersect_raster_list)

    if len(intersect_raster_list) == 0:
        logger.info("No intersection with tile %s", tile_name)
        return []

    dsk, required_products = pipelines.generate_tasks(tile_name, intersect_raster_list, do_watch_ram)
    logger.debug('######################################################################')
    logger.debug('Summary of %s tasks related to S1 -> S2 transformations of %s', len(dsk), tile_name)
    for product, how in dsk.items():
        logger.debug('- task: %s <-- %s', product, _how2str(how))

    if debug_otb:
        return _execute_tasks_debug(dsk, tile_name)
    else:
        assert client, "Dask client shall exist when not debugging calls to OTB applications"
        return _execute_tasks_with_dask(dsk, tile_name, tile_idx, intersect_raster_list,
                required_products, client, pipelines, do_watch_ram, debug_tasks)


def read_config(config_opt: Union[str, Configuration]) -> Configuration:
    """
    The config_opt can be either the configuration filename or an already initialized configuration
    object
    """
    if isinstance(config_opt, str):
        return Configuration(config_opt)
    else:
        return config_opt


def _extend_config(config: Configuration, extra_opts: Dict, overwrite: bool = False) -> Configuration:
    """
    Adds attributes to configuration object.

    .. todo:: Configuration object shall be closer to a dictionary to avoid these workarounds...
    """
    for k in extra_opts:
        if overwrite or not hasattr(config, k):
            setattr(config, k, extra_opts[k])
    return config


def do_process_with_pipeline(  # pylint: disable=too-many-arguments, too-many-locals
    config_opt             : Union[str, Configuration],
    pipeline_builder,
    dl_wait                : int  = EODAG_DEFAULT_DOWNLOAD_WAIT,
    dl_timeout             : int  = EODAG_DEFAULT_DOWNLOAD_TIMEOUT,
    searched_items_per_page: int  = EODAG_DEFAULT_SEARCH_ITEMS_PER_PAGE,
    nb_max_search_retries  : int  = EODAG_DEFAULT_SEARCH_MAX_RETRIES,
    dryrun                 : bool = False,
    debug_caches           : bool = False,
    debug_otb              : bool = False,
    watch_ram              : bool = False,
    debug_tasks            : bool = False,
) -> exits.Situation:
    """
    Internal function for executing pipelines.
    # TODO: parametrize tile loop, product download...
    """
    config: Configuration  = read_config(config_opt)
    extra_opts = {
            "dl_wait"                : dl_wait,
            "dl_timeout"             : dl_timeout,
            "searched_items_per_page": searched_items_per_page,
            "nb_max_search_retries"  : nb_max_search_retries,
    }
    _extend_config(config, extra_opts, overwrite=False)

    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(config.OTBThreads)
    # For the OTB applications that don't receive the path as a parameter (like SARDEMProjection)
    # -> we set $OTB_GEOID_FILE
    os.environ["OTB_GEOID_FILE"] = config.GeoidFile
    with S1FileManager(config) as s1_file_manager:
        tiles_to_process = extract_tiles_to_process(config, s1_file_manager)
        if len(tiles_to_process) == 0:
            raise exceptions.NoS2TileError()

        tiles_to_process_checked, needed_dem_tiles, dems_by_s2_tiles = check_tiles_to_process(
                tiles_to_process, s1_file_manager)

        logger.info("%s images to process on %s tiles",
                s1_file_manager.nb_images, tiles_to_process_checked)

        if len(tiles_to_process_checked) == 0:
            raise exceptions.NoS1ImageError()

        logger.info("Required DEM tiles: %s", list(needed_dem_tiles.keys()))

        if not check_dem_tiles(config, needed_dem_tiles):
            raise exceptions.MissingDEMError()

        if not os.path.exists(config.GeoidFile):
            raise exceptions.MissingGeoidError(config.GeoidFile)

        # Prepare directories where to store temporary files
        # These directories won't be cleaned up automatically
        S1_tmp_dir = os.path.join(config.tmpdir, 'S1')
        os.makedirs(S1_tmp_dir, exist_ok=True)

        config.tmp_dem_dir = s1_file_manager.tmpdemdir(
                needed_dem_tiles, config.dem_filename_format,
                config.GeoidFile)

        pipelines, required_workspaces = pipeline_builder(config, dryrun=dryrun, debug_caches=debug_caches)
        config.register_dems_related_to_S2_tiles(dems_by_s2_tiles)

        log_level : Callable[[Any], int] = lambda res: logging.INFO if bool(res) else logging.WARNING
        results = []
        with DaskContext(config, debug_otb) as dask_client:
            for idx, tile_it in enumerate(tiles_to_process_checked):
                with Utils.ExecutionTimer("Processing of tile " + tile_it, True):
                    res = process_one_tile(
                            tile_it, idx, len(tiles_to_process_checked),
                            s1_file_manager, pipelines, dask_client.client,
                            required_workspaces,
                            debug_otb=debug_otb, dryrun=dryrun, do_watch_ram=watch_ram,
                            debug_tasks=debug_tasks)
                    results += res

        nb_errors_detected = sum(not bool(res) for res in results)

        skipped_for_download_failures = s1_file_manager.get_skipped_S2_products()
        results.extend(skipped_for_download_failures)

        logger.debug('#############################################################################')
        nb_issues = nb_errors_detected + len(skipped_for_download_failures)
        if nb_issues > 0:
            logger.warning('Execution report: %s errors detected', nb_issues)
        else:
            logger.info('Execution report: no error detected')

        if results:
            for res in results:
                logger.log(log_level(res), ' - %s', res)
        else:
            logger.info(' -> Nothing has been executed')

        search_failures   = s1_file_manager.get_search_failures()
        download_failures = s1_file_manager.get_download_failures()
        download_timeouts = s1_file_manager.get_download_timeouts()
        return exits.Situation(
                nb_computation_errors=nb_errors_detected - search_failures,
                nb_search_failures=search_failures,
                nb_download_failures=len(download_failures),
                nb_download_timeouts=len(download_timeouts)
        )


def register_LIA_pipelines_v0(pipelines: PipelineDescriptionSequence, produce_angles: bool) -> PipelineDescription:
    """
    Internal function that takes care to register all pipelines related to
    LIA map and sin(LIA) map.
    """
    dem = pipelines.register_pipeline([AgglomerateDEMOnS1], 'AgglomerateDEM',
            inputs={'insar': 'basename'})

    demproj = pipelines.register_pipeline([ExtractSentinel1Metadata, SARDEMProjection], 'SARDEMProjection', is_name_incremental=True,
            inputs={'insar': 'basename', 'indem': dem})
    xyz = pipelines.register_pipeline([SARCartesianMeanEstimation],                     'SARCartesianMeanEstimation',
            inputs={'insar': 'basename', 'indem': dem, 'indemproj': demproj})
    lia = pipelines.register_pipeline([ComputeNormalsOnS1, ComputeLIAOnS1],                     'Normals|LIA', is_name_incremental=True,
            inputs={'xyz': xyz})

    # "inputs" parameter doesn't need to be specified in the following pipeline declarations
    # but we still use it for clarity!
    ortho_deg       = pipelines.register_pipeline(
            [filter_LIA('LIA'), OrthoRectifyLIA],
            'OrthoLIA',
            inputs={'in': lia},
            is_name_incremental=True)
    concat_deg      = pipelines.register_pipeline(
            [ConcatenateLIA],
            'ConcatLIA',
            inputs={'in': ortho_deg})
    pipelines.register_pipeline(
            [SelectBestCoverage],
            'SelectLIA',
            inputs={'in': concat_deg},
            product_required=produce_angles)

    ortho_sin       = pipelines.register_pipeline(
            [filter_LIA('sin_LIA'), OrthoRectifyLIA],
            'OrthoSinLIA',
            inputs={'in': lia},
            is_name_incremental=True)
    concat_sin      = pipelines.register_pipeline(
            [ConcatenateLIA],
            'ConcatSinLIA',
            inputs={'in': ortho_sin})
    best_concat_sin = pipelines.register_pipeline(
            [SelectBestCoverage],
            'SelectSinLIA',
            inputs={'in': concat_sin},
            product_required=True)

    return best_concat_sin


def register_LIA_pipelines(pipelines: PipelineDescriptionSequence, produce_angles: bool) -> PipelineDescription:
    """
    Internal function that takes care to register all pipelines related to
    LIA map and sin(LIA) map.
    """
    dem_vrt = pipelines.register_pipeline(
            [AgglomerateDEMOnS2], 'AgglomerateDEM',
            inputs={'tilename': 'tilename'},
    )

    s2_dem = pipelines.register_pipeline(
            [ProjectDEMToS2Tile], "ProjectDEMToS2Tile",
            is_name_incremental=True,
            inputs={"indem": dem_vrt}
    )

    s2_height = pipelines.register_pipeline(
            [ProjectGeoidToS2Tile, SumAllHeights], "GenerateHeightForS2Tile",
            is_name_incremental=True,
            inputs={"in_s2_dem": s2_dem},
    )

    # Notes:
    # * ComputeGroundAndSatPositionsOnDEM cannot be merged in memory with
    #   normals production AND LIA production: indeed the XYZ, and satposXYZ
    #   data needs to be reused several times, and in-memory pipeline can't
    #   support that (yet?)
    # * ExtractSentinel1Metadata needs to be in its own pipeline to make sure
    #   all meta are available later on to filter on the coverage.
    # * ComputeGroundAndSatPositionsOnDEM takes care of filtering on the
    #   coverage. We don't need any SelectBestS1onS2Coverage prior to this step.
    sar = pipelines.register_pipeline(
            [ExtractSentinel1Metadata],
            inputs={'inrawsar': 'basename'}
    )
    xyz = pipelines.register_pipeline(
            [ComputeGroundAndSatPositionsOnDEM],
            "ComputeGroundAndSatPositionsOnDEM",
            inputs={'insar': sar, 'inheight': s2_height},
    )

    # Always generate sin(LIA). If LIA° is requested, then it's also a
    # final/requested product.
    # produce_angles is ignored as there is no extra select_LIA step
    lia = pipelines.register_pipeline(
            [ComputeNormalsOnS2, ComputeLIAOnS2],
            'ComputeLIAOnS2',
            is_name_incremental=True,
            inputs={'xyz': xyz},
            product_required=True,
    )
    return lia


def s1_process(  # pylint: disable=too-many-arguments, too-many-locals
        config_opt              : Union[str, Configuration],
        dl_wait                 : int  = EODAG_DEFAULT_DOWNLOAD_WAIT,
        dl_timeout              : int  = EODAG_DEFAULT_DOWNLOAD_TIMEOUT,
        searched_items_per_page : int  = EODAG_DEFAULT_SEARCH_ITEMS_PER_PAGE,
        nb_max_search_retries   : int  = EODAG_DEFAULT_SEARCH_MAX_RETRIES,
        dryrun                  : bool = False,
        debug_otb               : bool = False,
        debug_caches            : bool = False,
        watch_ram               : bool = False,
        debug_tasks             : bool = False,
        cache_before_ortho      : bool = False,
        lia_process                    = None,
) -> exits.Situation:
    """
    Entry point to :ref:`S1Tiling classic scenario <scenario.S1Processor>` and
    :ref:`S1Tiling NORMLIM scenario <scenario.S1ProcessorLIA>` of on demand
    Ortho-rectification of Sentinel-1 data on Sentinel-2 grid for all
    calibration kinds.

    It performs the following steps:

    1. Download S1 images from S1 data provider (through eodag)
        This step may be ignored if ``config_opt`` *download* option is false;
    2. Calibrate the S1 images according to the *calibration* option from ``config_opt``;
    3. Orthorectify S1 images and cut their on geometric tiles;
    4. Concatenate images from the same orbit on the same tile;
    5. Build mask files;
    6. Despeckle final images.

    :param config_opt:
        Either a :ref:`request configuration file <request-config-file>` or a
        :class:`s1tiling.libs.configuration.Configuration` instance.
    :param dl_wait:
        Permits to override EODAG default wait time in minutes between two
        download tries.
    :param dl_timeout:
        Permits to override EODAG default maximum time in mins before stop
        retrying to download (default=20)
    :param searched_items_per_page:
        Tells how many items are to be returned by EODAG when searching for S1
        images.
    :param dryrun:
        Used for debugging: external (OTB/GDAL) application aren't executed.
    :param debug_otb:
        Used for debugging: Don't execute processing tasks in DASK workers but
        directly in order to be able to analyse OTB/external application
        through a debugger.
    :param debug_caches:
        Used for debugging: Don't delete the intermediary files but leave them
        behind.
    :param watch_ram:
        Used for debugging: Monitoring Python/Dask RAM consumption.
    :param debug_tasks:
        Generate SVG images showing task graphs of the processing flows
    :param cache_before_ortho:
        Cutting, calibration and orthorectification are chained in memory
        unless this option is true. In that case, :ref:`Cut and calibrated (aka
        "OrthoReady") files <orthoready-files>` are stored in :ref:`%(tmp)
        <paths.tmp>`:samp:`/S1/` directory.
        Do not forget to regularly clean up this space.

    :return:
        A *nominal* exit code depending of whether everything could have been
        downloaded and produced.
    :rtype: :class:`s1tiling.libs.exits.Situation`

    :exception Error: A variety of exceptions. See below (follow the link).
    """
    def builder(config: Configuration, dryrun: bool, debug_caches: bool) -> Tuple[PipelineDescriptionSequence, List[WorkspaceKinds]]:
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
        concat_seq : List[Type[StepFactory]] = [Concatenate]
        if chain_concat_and_despeckle_inmemory:
            concat_seq.append(SpatialDespeckle)
            need_to_keep_non_filtered_products = False
        else:
            need_to_keep_non_filtered_products = True

        concat_S2 = pipelines.register_pipeline(
                concat_seq,
                product_required=calibration_is_done_in_S1,
                is_name_incremental=True
        )
        last_product_S2 = concat_S2

        required_workspaces = [WorkspaceKinds.TILE]

        # LIA Calibration (...+ Despeckle)
        if config.calibration_type == 'normlim':
            apply_LIA_seq : List[Type[StepFactory]] = [ApplyLIACalibration]
            if chain_LIA_and_despeckle_inmemory:
                apply_LIA_seq.append(SpatialDespeckle)
                need_to_keep_non_filtered_products = False
            else:
                need_to_keep_non_filtered_products = True

            LIA_registration = lia_process or register_LIA_pipelines
            lias = LIA_registration(pipelines, config.produce_lia_map)

            # This steps helps forwarding sin(LIA) (only) to the next step
            # that corrects the β° with sin(LIA) map.
            sin_LIA = pipelines.register_pipeline(
                    [filter_LIA('sin_LIA')],
                    'SelectSinLIA',
                    is_name_incremental=True,
                    inputs={'in': lias},
            )
            # TODO: Merge filter_LIA in apply_LIA_seq!
            apply_LIA = pipelines.register_pipeline(apply_LIA_seq, product_required=True,
                    inputs={'sin_LIA': sin_LIA, 'concat_S2': concat_S2}, is_name_incremental=True)
            last_product_S2 = apply_LIA
            required_workspaces.append(WorkspaceKinds.LIA)

        # Masking
        if config.mask_cond:
            pipelines.register_pipeline([BuildBorderMask, SmoothBorderMask], 'GenerateMask',
                    product_required=True, inputs={'in': last_product_S2})
            required_workspaces.append(WorkspaceKinds.MASK)

        # Despeckle in non-inmemory case
        if config.filter:
            # Use SpatialDespeckle, only if filter ∈ [lee, gammamap, frost, kuan]
            required_workspaces.append(WorkspaceKinds.FILTER)
            if need_to_keep_non_filtered_products:  # config.keep_non_filtered_products:
                # Define another pipeline if chaining cannot be done in memory
                pipelines.register_pipeline([SpatialDespeckle], product_required=True,
                        inputs={'in': last_product_S2})

        return pipelines, required_workspaces

    return do_process_with_pipeline(
            config_opt, builder,
            dl_wait=dl_wait, dl_timeout=dl_timeout,
            searched_items_per_page=searched_items_per_page,
            nb_max_search_retries=nb_max_search_retries,
            dryrun=dryrun,
            debug_otb=debug_otb,
            debug_caches=debug_caches,
            watch_ram=watch_ram,
            debug_tasks=debug_tasks,
    )


def s1_process_lia_v0(  # pylint: disable=too-many-arguments
        config_opt             : Union[str, Configuration],
        dl_wait                : int  = EODAG_DEFAULT_DOWNLOAD_WAIT,
        dl_timeout             : int  = EODAG_DEFAULT_DOWNLOAD_TIMEOUT,
        searched_items_per_page: int  = EODAG_DEFAULT_SEARCH_ITEMS_PER_PAGE,
        nb_max_search_retries  : int  = EODAG_DEFAULT_SEARCH_MAX_RETRIES,
        dryrun                 : bool = False,
        debug_otb              : bool = False,
        debug_caches           : bool = False,
        watch_ram              : bool = False,
        debug_tasks            : bool = False,
) -> exits.Situation:
    """
    Entry point to :ref:`LIA Map production scenario <scenario.S1LIAMap>` that
    generates Local Incidence Angle Maps on S2 geometry.

    It performs the following steps:

    1. Determine the S1 products to process
        Given a list of S2 tiles, we first determine the day that'll the best
        coverage of each S2 tile in terms of S1 products.

        In case there is no single day that gives the best coverage for all
        S2 tiles, we try to determine the best solution that minimizes the
        number of S1 products to download and process.
    2. Process these S1 products

    :param config_opt:
        Either a :ref:`request configuration file <request-config-file>` or a
        :class:`s1tiling.libs.configuration.Configuration` instance.
    :param dl_wait:
        Permits to override EODAG default wait time in minutes between two
        download tries.
    :param dl_timeout:
        Permits to override EODAG default maximum time in mins before stop
        retrying to download (default=20)
    :param searched_items_per_page:
        Tells how many items are to be returned by EODAG when searching for S1
        images.
    :param dryrun:
        Used for debugging: external (OTB/GDAL) application aren't executed.
    :param debug_otb:
        Used for debugging: Don't execute processing tasks in DASK workers but
        directly in order to be able to analyse OTB/external application
        through a debugger.
    :param debug_caches:
        Used for debugging: Don't delete the intermediary files but leave them
        behind.
    :param watch_ram:
        Used for debugging: Monitoring Python/Dask RAM consumption.
    :param debug_tasks:
        Generate SVG images showing task graphs of the processing flows

    :return:
        A *nominal* exit code depending of whether everything could have been
        downloaded and produced.
    :rtype: :class:`s1tiling.libs.exits.Situation`

    :exception Error: A variety of exceptions. See below (follow the link).
    """
    def builder(config: Configuration, dryrun: bool, debug_caches: bool) -> Tuple[PipelineDescriptionSequence, List[WorkspaceKinds]]:
        pipelines = PipelineDescriptionSequence(config, dryrun=dryrun, debug_caches=debug_caches)
        register_LIA_pipelines_v0(pipelines, produce_angles=config.produce_lia_map)
        required_workspaces = [WorkspaceKinds.LIA]
        return pipelines, required_workspaces

    return do_process_with_pipeline(
            config_opt, builder,
            dl_wait=dl_wait, dl_timeout=dl_timeout,
            searched_items_per_page=searched_items_per_page,
            nb_max_search_retries=nb_max_search_retries,
            dryrun=dryrun,
            debug_caches=debug_caches,
            debug_otb=debug_otb,
            watch_ram=watch_ram,
            debug_tasks=debug_tasks,
    )


def s1_process_lia(  # pylint: disable=too-many-arguments
        config_opt             : Union[str, Configuration],
        dl_wait                : int  = EODAG_DEFAULT_DOWNLOAD_WAIT,
        dl_timeout             : int  = EODAG_DEFAULT_DOWNLOAD_TIMEOUT,
        searched_items_per_page: int  = EODAG_DEFAULT_SEARCH_ITEMS_PER_PAGE,
        nb_max_search_retries  : int  = EODAG_DEFAULT_SEARCH_MAX_RETRIES,
        dryrun                 : bool = False,
        debug_otb              : bool = False,
        debug_caches           : bool = False,
        watch_ram              : bool = False,
        debug_tasks            : bool = False,
) -> exits.Situation:
    """
    Entry point to :ref:`LIA Map production scenario <scenario.S1LIAMap>` that
    generates Local Incidence Angle Maps on S2 geometry.

    It performs the following steps:

    1. Determine the S1 products to process
        Given a list of S2 tiles, we first determine the day that'll the best
        coverage of each S2 tile in terms of S1 products.

        In case there is no single day that gives the best coverage for all
        S2 tiles, we try to determine the best solution that minimizes the
        number of S1 products to download and process.
    2. Process these S1 products

    :param config_opt:
        Either a :ref:`request configuration file <request-config-file>` or a
        :class:`s1tiling.libs.configuration.Configuration` instance.
    :param dl_wait:
        Permits to override EODAG default wait time in minutes between two
        download tries.
    :param dl_timeout:
        Permits to override EODAG default maximum time in mins before stop
        retrying to download (default=20)
    :param searched_items_per_page:
        Tells how many items are to be returned by EODAG when searching for S1
        images.
    :param dryrun:
        Used for debugging: external (OTB/GDAL) application aren't executed.
    :param debug_otb:
        Used for debugging: Don't execute processing tasks in DASK workers but
        directly in order to be able to analyse OTB/external application
        through a debugger.
    :param debug_caches:
        Used for debugging: Don't delete the intermediary files but leave them
        behind.
    :param watch_ram:
        Used for debugging: Monitoring Python/Dask RAM consumption.
    :param debug_tasks:
        Generate SVG images showing task graphs of the processing flows

    :return:
        A *nominal* exit code depending of whether everything could have been
        downloaded and produced.
    :rtype: :class:`s1tiling.libs.exits.Situation`

    :exception Error: A variety of exceptions. See below (follow the link).
    """
    def builder(config: Configuration, dryrun: bool, debug_caches: bool) -> Tuple[PipelineDescriptionSequence, List[WorkspaceKinds]]:
        pipelines = PipelineDescriptionSequence(config, dryrun=dryrun, debug_caches=debug_caches)
        register_LIA_pipelines(pipelines, produce_angles=config.produce_lia_map)
        required_workspaces = [WorkspaceKinds.LIA]
        return pipelines, required_workspaces

    return do_process_with_pipeline(
            config_opt, builder,
            dl_wait=dl_wait, dl_timeout=dl_timeout,
            searched_items_per_page=searched_items_per_page,
            nb_max_search_retries=nb_max_search_retries,
            dryrun=dryrun,
            debug_caches=debug_caches,
            debug_otb=debug_otb,
            watch_ram=watch_ram,
            debug_tasks=debug_tasks,
    )
