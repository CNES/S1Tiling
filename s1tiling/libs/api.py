#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
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
# Authors: Thierry KOLECK (CNES)
#          Luc HERMITTE (CS Group)
#          Fabien CONTIVAL (CS Group)
#
# =========================================================================

"""
Submodule that defines all API related functions and classes.
"""

from collections.abc import Callable
import contextlib
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast

from distributed.scheduler import KilledWorker
from dask.distributed import Client
from eodag.api.core import EODataAccessGateway

from s1tiling.libs.otbwrappers.gamma_area import NaNifyNoData, ProjectGeoidToDEM

from .S1DateAcquisition import S1DateAcquisition
from .S1FileManager import (
    S1FileManager,
    EODAG_DEFAULT_DOWNLOAD_WAIT,
    EODAG_DEFAULT_DOWNLOAD_TIMEOUT,
    EODAG_DEFAULT_SEARCH_MAX_RETRIES,
    EODAG_DEFAULT_SEARCH_ITEMS_PER_PAGE,
)
from . import exits
from . import exceptions
from . import Utils
from .configuration import (
    Configuration,
    LIAConfiguration,
    dname_fmt_filtered,
    dname_fmt_gamma_area_product,
    dname_fmt_lia_product,
    dname_fmt_tiled,
    fname_fmt_concatenation,
    fname_fmt_filtered,
    fname_fmt_gamma_area_corrected,
    fname_fmt_gamma_area_product,
    fname_fmt_lia_corrected,
)
from .otbpipeline import (
    FirstStep,
    PipelineDescription,
    PipelineDescriptionSequence,
    StepFactory,
    AbstractStep,
)
from .otbwrappers import (
    # Main S1 -> S2 Step Factories
    ExtractSentinel1Metadata,
    AnalyseBorders,
    Calibrate,
    CorrectDenoising,
    CutBorders,
    OrthoRectify,
    Concatenate,
    BuildBorderMask,
    SmoothBorderMask,
    # LIA related Step Factories
    AgglomerateDEMOnS2,
    ProjectDEMToS2Tile,
    ProjectGeoidToS2Tile,
    SumAllHeights,
    ComputeGroundAndSatPositionsOnDEM,
    ComputeGroundAndSatPositionsOnDEMFromEOF,
    ComputeLIAOnS2,
    filter_LIA,
    ComputeNormalsOnS2,
    ComputeEllipsoidNormalsOnS2,
    ComputeIAOnS2,
    ComputeGroundAndSatPositionsOnEllipsoid,
    ApplyLIACalibration,
    # Deprecated LIA related Step Factories
    AgglomerateDEMOnS1,
    SARDEMProjection,
    SARCartesianMeanEstimation,
    ComputeNormalsOnS1,
    OrthoRectifyLIA,
    ComputeLIAOnS1,
    ConcatenateLIA,
    SelectBestCoverage,
    # Gamma Area related Step Factories
    ResampleDEM,
    SARDEMProjectionImageEstimation,
    SARGammaAreaImageEstimation,
    OrthoRectifyGAMMA_AREA,
    ConcatenateGAMMA_AREA,
    SelectGammaNaughtAreaBestCoverage,
    ApplyGammaNaughtRTCCalibration,
    # Filter Step Factories
    SpatialDespeckle,
)
from .outcome     import Outcome
from .orbit       import EOFFileManager
from .utils.dask  import DaskContext
from .utils       import eodag
from .utils.layer import filter_existing_tiles
from .utils.timer import timethis

from .vis import SimpleComputationGraph  # Graphs
from .workspace import DEMWorkspace, WorkspaceKinds, ensure_tiled_workspaces_exist


IntersectingS1FilesOutcome = Outcome[List[Dict]]


logger = logging.getLogger('s1tiling.api')


def main_output_name_formats(configuration: Configuration) -> List[Tuple[str,str]]:
    """
    Helper function that generates the list of output name formats (dirname + filename) in the
    main case scenarios: :ref:`scenario.S1Processor`, :ref:`scenario.S1ProcessorLIA` and
    :ref:`scenario.S1ProcessorRTC`.
    """
    res = []
    if configuration.calibration_type == 'normlim':
        res.append((dname_fmt_tiled(configuration), fname_fmt_lia_corrected(configuration)))
    elif configuration.calibration_type == 'gamma_naught_rtc':
        res.append((dname_fmt_tiled(configuration), fname_fmt_gamma_area_corrected(configuration)))
        res.append((dname_fmt_gamma_area_product(configuration), fname_fmt_gamma_area_product(configuration)))
    else:
        res.append((dname_fmt_tiled(configuration), fname_fmt_concatenation(configuration)))
    if configuration.filter:
        res.append((dname_fmt_filtered(configuration), fname_fmt_filtered(configuration)))
    return res


@timethis("Sanitizing requested tiles", log_level=logging.INFO)
def extract_tiles_to_process(cfg: Configuration, s1_file_manager: Optional[S1FileManager]) -> List[str]:
    """
    Deduce from the configuration all the tiles that need to be processed.

    :return: the sorted list of all the tile names to process
    """
    logger.info('Requested tiles: %s', cfg.tile_list)

    tiles_to_process = []
    if cfg.tile_list[0] == "ALL":
        if not s1_file_manager:
            raise exceptions.ConfigurationError("tile_list=ALL mode is not compatible with this scenario", "")
        # Check already done in the configuration object
        assert not (
            cfg.download and "ALL" in cfg.roi_by_tiles
        ), "Can not request to download 'ROI_by_tiles : ALL' if 'Tiles : ALL'. Change either value or deactivate download instead"
        tiles_to_process = s1_file_manager.get_tiles_covered_by_products()
        logger.info("All tiles for which more than %s%% of the surface is covered by products will be produced: %s",
                100 * cfg.tile_to_product_overlap_ratio, tiles_to_process)
    else:
        tiles_to_process = filter_existing_tiles(cfg.output_grid, cfg.tile_list)

    # We can not require both to process all tiles covered by downloaded products
    # and download all tiles

    logger.info('The following tiles will be processed: %s', tiles_to_process)
    return tiles_to_process


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
    *,
    dsk:                   Dict[str, Union[Tuple, "FirstStep"]],
    tile_name:             str,
    tile_idx:              int,
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
                dsk, required_products, errors = pipelines.generate_tasks(do_watch_ram=do_watch_ram)
                # it's unlikely for errors to appear here
                assert not errors, f"No errors regarding task generation shall appear here: {errors}"
            else:
                raise
    return []


def get_s1_files_for_tile(
    s1_file_manager:     S1FileManager,
    tile_name:           str,
    output_name_formats: List[Tuple[str, str]],
    dryrun:              bool,
) -> IntersectingS1FilesOutcome:
    """
    Returns the list of all S1 files intersecting the given S2 MGRS tile name.

    :return: An :class:`Outcome` of list of S1 image information, or the :class:`RuntimeError` that has happened.
    :raise DownloadS1FileError: if a critical error occurs
    """
    s1_file_manager.keep_X_latest_S1_files(1000, tile_name, output_name_formats)

    try:
        s1_file_manager.download_images(tiles=[tile_name], output_name_formats=output_name_formats, dryrun=dryrun)
        # download_images will have updated the list of know products
    except RuntimeError as e:
        logger.warning('Cannot download S1 images associated to %s: %s', tile_name, e)
        # logger.critical(e, exc_info=True)
        return IntersectingS1FilesOutcome(e)

    except BaseException as e:
        logger.debug('Download error intercepted: %s', e)
        raise exceptions.DownloadS1FileError(tile_name) from e

    intersect_raster_list = s1_file_manager.get_s1_intersect_by_tile(tile_name, output_name_formats)
    logger.debug('%s products found to intersect %s: %s', len(intersect_raster_list), tile_name, intersect_raster_list)
    return IntersectingS1FilesOutcome(intersect_raster_list)


@timethis("Processing of tile {tile_name}", log_level=logging.INFO)
def process_one_tile(  # pylint: disable=too-many-arguments
    *,
    tile_name:               str,
    tile_idx:                int,
    tiles_nb:                int,
    cfg:                     Configuration,
    pipelines:               PipelineDescriptionSequence,
    client:                  Optional[Client],
    required_workspaces:     List[WorkspaceKinds],
    debug_otb:               bool = False,
    do_watch_ram:            bool = False,
    debug_tasks:             bool = False
) -> List[Outcome]:
    """
    Process one S2 tile.

    I.E. run the OTB pipeline on all the S1 images that match the S2 tile.
    """
    ensure_tiled_workspaces_exist(cfg, tile_name, required_workspaces)

    logger.info("Processing tile %s (%s/%s)", tile_name, tile_idx + 1, tiles_nb)

    pipelines.register_extra_parameters_for_input_factories(tile_name=tile_name)
    dsk, required_products, errors = pipelines.generate_tasks(do_watch_ram)
    if errors:
        return errors
    logger.debug('######################################################################')
    logger.debug('Summary of %s tasks related to S1 -> S2 transformations of %s', len(dsk), tile_name)
    for product, how in dsk.items():
        logger.debug('- task: %s <-- %s', product, _how2str(how))

    if debug_otb:
        return _execute_tasks_debug(dsk, tile_name)
    else:
        assert client, "Dask client shall exist when not debugging calls to OTB applications"
        return _execute_tasks_with_dask(
            dsk=dsk,
            tile_name=tile_name,
            tile_idx=tile_idx,
            required_products=required_products,
            client=client,
            pipelines=pipelines,
            do_watch_ram=do_watch_ram,
            debug_tasks=debug_tasks,
        )


def read_config(
    config_opt          : Union[str, Configuration],
    extra_config_checks : Sequence[Tuple[Callable[[Configuration], bool], str]] = (),
) -> Configuration:
    """
    The config_opt can be either the configuration filename or an already initialized configuration
    object
    """
    if isinstance(config_opt, str):
        return Configuration(config_opt, extra_config_checks=extra_config_checks)
    else:
        for check, msg in extra_config_checks:
            if not check(config_opt):
                raise exceptions.ConfigurationError(msg, "")
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
    *,
    ctx_managers           : Sequence[Type] = (),
    extra_config_checks    : Sequence[Tuple[Callable[[Configuration], bool], str]] = (),
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
    config: Configuration  = read_config(config_opt, extra_config_checks)
    extra_opts = {
            "dl_wait"                : dl_wait,
            "dl_timeout"             : dl_timeout,
            "searched_items_per_page": searched_items_per_page,
            "nb_max_search_retries"  : nb_max_search_retries,
    }
    _extend_config(config, extra_opts, overwrite=False)

    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(config.OTBThreads)
    os.environ["GDAL_NUM_THREADS"] = str(config.OTBThreads)

    # For the OTB applications that don't receive the path as a parameter (like SARDEMProjection)
    # -> we set $OTB_GEOID_FILE
    if not os.path.exists(config.GeoidFile):
        raise exceptions.MissingGeoidError(config.GeoidFile)
    os.environ["OTB_GEOID_FILE"] = config.GeoidFile

    dag = eodag.create(config)
    s1_file_manager = S1FileManager(config, dag)
    tiles_to_process = extract_tiles_to_process(config, s1_file_manager)
    nb_tiles = len(tiles_to_process)
    logger.info("%s images to process on %s tiles: %s", s1_file_manager.nb_images, nb_tiles, tiles_to_process)

    if nb_tiles == 0:
        raise exceptions.NoS2TileError()

    # Prepare directories where to store temporary files
    # These directories won't be cleaned up automatically
    S1_tmp_dir = os.path.join(config.tmpdir, 'S1')
    os.makedirs(S1_tmp_dir, exist_ok=True)

    with contextlib.ExitStack() as context:
        for cm in ctx_managers:
            context.enter_context(cm(config, tiles_to_process))

        pipelines, required_workspaces = pipeline_builder(config, dryrun=dryrun, debug_caches=debug_caches)

        # Used by eof
        pipelines.register_extra_parameters_for_input_factories(
                dag=dag,
                s1_file_manager=s1_file_manager,
                dryrun=dryrun,
                # tile_name will be done in process_one_tile
        )

        log_level : Callable[[Any], int] = lambda res: logging.INFO if bool(res) else logging.WARNING
        results = []
        with DaskContext(config, debug_otb) as dask_client:
            for idx, tile_it in enumerate(tiles_to_process):
                res = process_one_tile(
                    tile_name=tile_it,
                    tile_idx=idx,
                    tiles_nb=nb_tiles,
                    cfg=config,
                    pipelines=pipelines,
                    client=dask_client.client,
                    required_workspaces=required_workspaces,
                    debug_otb=debug_otb,
                    do_watch_ram=watch_ram,
                    debug_tasks=debug_tasks,
                )
                results.extend(res)

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
    dem = pipelines.register_pipeline(
        [AgglomerateDEMOnS1],
        'AgglomerateDEM',
        inputs={'insar': 'basename'})

    demproj = pipelines.register_pipeline(
        [ExtractSentinel1Metadata, SARDEMProjection],
        'SARDEMProjection',
        is_name_incremental=True,
        inputs={'insar': 'basename', 'indem': dem})
    xyz = pipelines.register_pipeline(
        [SARCartesianMeanEstimation],
        'SARCartesianMeanEstimation',
        inputs={'insar': 'basename', 'indem': dem, 'indemproj': demproj})
    lia = pipelines.register_pipeline(
        [ComputeNormalsOnS1, ComputeLIAOnS1],
        'Normals|LIA',
        is_name_incremental=True,
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

    ortho_sin = pipelines.register_pipeline(
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


def register_LIA_pipelines_v1_1(
    pipelines: PipelineDescriptionSequence,
    produce_angles: bool,
) -> PipelineDescription:
    """
    Internal function that takes care to register all pipelines related to
    LIA map and sin(LIA) map.
    """
    pipelines.register_inputs('tilename', tilename_first_inputs_factory)
    dem_vrt = pipelines.register_pipeline(
        [AgglomerateDEMOnS2],
        'AgglomerateDEM',
        inputs={'tilename': 'tilename'},
    )

    s2_dem = pipelines.register_pipeline(
        [ProjectDEMToS2Tile],
        "ProjectDEMToS2Tile",
        is_name_incremental=True,
        inputs={"indem": dem_vrt}
    )

    s2_height = pipelines.register_pipeline(
        [
            ProjectGeoidToS2Tile,
            SumAllHeights(
                product_key='height_on_s2',
                key_map={'indem': 'in_s2_dem', 'ingeoid': 'in_s2_geoid'},
                fname_fmt_default='DEM+GEOID_projected_on_{tile_name}.tiff',
                dname_fmt_default='S2/{tile_name}',
                image_description='DEM + GEOID height info projected on S2 tile',
            ),
        ],
        "GenerateHeightForS2Tile",
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



def register_GAMMA_AREA_pipelines(
    pipelines: PipelineDescriptionSequence,
    config: Configuration
) -> PipelineDescription:
    """
    Internal function that takes care to register all pipelines related to
    GAMMA AREA map.
    """
    # Build VRT
    dem = pipelines.register_pipeline(
        [AgglomerateDEMOnS1],
        'AgglomerateDEM',
        inputs={'insar': 'basename'},
    )

    # Resample DEM
    resampled_dem = dem
    if config.use_resampled_dem:
        resampled_dem = pipelines.register_pipeline(
            [NaNifyNoData, ResampleDEM],
            'RigidTransformResample',
            inputs={'indem': dem},
        )

    # Combine dem + geoid in order to optimize SARDEMProjection execution
    # 66% of its execution time would be lost in mutexes for accessing Geoid elevation on each point
    # otherwise.
    heights = pipelines.register_pipeline(
        [
            ProjectGeoidToDEM,
            SumAllHeights(
                product_key='height_4rtc',
                key_map={'indem': 'indem', 'ingeoid': 'ingeoid'},
                fname_fmt_default='DEM+GEOID_{polarless_basename}',
                dname_fmt_default='S1',
                image_description='DEM + GEOID',
            ),
        ],
        'Combine DEM+height',
        is_name_incremental=True,
        inputs={'indem': resampled_dem},
    )

    # Project DEM
    demproj = pipelines.register_pipeline(
        [ExtractSentinel1Metadata, SARDEMProjectionImageEstimation],
        'SARDEMProjection',
        is_name_incremental=True,
        # inputs={'insar': 'basename', 'indem': resampled_dem},
        inputs={'insar': 'basename', 'indem': heights},
    )

    # gamma area
    gamma_area = pipelines.register_pipeline(
        [SARGammaAreaImageEstimation],
        'SARGammaAreaImageEstimation',
        # TODO: indem parameter doesn't make sens in the application code...
        inputs={'insar': 'basename', 'indem': heights, 'indemproj': demproj},
    )

    # ortho gamma area
    ortho_gamma_area = pipelines.register_pipeline(
        [OrthoRectifyGAMMA_AREA],
        'OrthoGAMMA_AREA',
        inputs={'in': gamma_area},
        is_name_incremental=True,
    )
    concat_ortho_gamma_area = pipelines.register_pipeline(
        [ConcatenateGAMMA_AREA],
        'ConcatGAMMA_AREA',
        inputs={'in': ortho_gamma_area},
    )
    best_concat_ortho_gamma_area = pipelines.register_pipeline(
        [SelectGammaNaughtAreaBestCoverage],
        'SelectGAMMA_AREA',
        inputs={'in': concat_ortho_gamma_area},
        product_required=True,
    )

    return best_concat_ortho_gamma_area


def s1_raster_first_inputs_factory(
    *,
    tile_name          : str,
    s1_file_manager    : S1FileManager,
    output_name_formats: List[Tuple[str, str]],
    dryrun             : bool,
    **kwargs,  # pylint: disable=unused-argument
) -> List[Outcome[FirstStep]]:
    """
    :class:`FirstStepFactory` hook dedicated to S1 images.
    """
    matching_rasters = get_s1_files_for_tile(s1_file_manager, tile_name, output_name_formats, dryrun)
    if not matching_rasters:
        return [cast(Outcome[FirstStep], matching_rasters)]
    intersect_raster_list = matching_rasters.value()

    if len(intersect_raster_list) == 0:
        logger.info("No intersection with tile %s", tile_name)
        return []
    return s1_raster_first_inputs_factory_from_rasters(tile_name, intersect_raster_list)


def s1_raster_first_inputs_factory_from_rasters(
    tile_name      : str,
    raster_list : List[Dict],
    **kwargs,  # pylint: disable=unused-argument
) -> List[Outcome[FirstStep]]:
    """
    :class:`FirstStepFactory` hook dedicated to S1 images: converts S1 raster list into
    :class:`FirstStep` instance list.
    """
    assert raster_list
    first_inputs = []
    for raster_info in raster_list:
        raster: S1DateAcquisition = raster_info['raster']

        manifest = raster.get_manifest()
        for image in raster.get_images_list():
            start = FirstStep(tile_name=tile_name,
                              tile_origin=raster_info['tile_origin'],
                              tile_coverage=raster_info['tile_coverage'],
                              manifest=manifest,
                              basename=image)
            first_inputs.append(Outcome(start))

    # Log commented and kept for filling in unit tests
    # logger.debug('Generate first steps from: %s', intersect_raster_list)
    return first_inputs


def tilename_first_inputs_factory(
    *,
    tile_name    : str,
    configuration: Configuration,
    **kwargs,  # pylint: disable=unused-argument
) -> List[Outcome[FirstStep]]:
    """
    :class:`FirstStepFactory` hook dedicated to S2 MGRS tile information: name and footprint origin.
    """
    # TODO: avoid to search this information multiple times
    tiles_db  = configuration.output_grid
    layer     = Utils.Layer(tiles_db)
    tile_info = layer.find_tile_named(tile_name)
    if not tile_info:
        raise RuntimeError(f"Tile {tile_name} cannot be found in {tiles_db!r}")
    tile_footprint = tile_info.GetGeometryRef()
    area_polygon   = tile_footprint.GetGeometryRef(0)
    points         = area_polygon.GetPoints()
    tile_origin    = [(point[0], point[1]) for point in points[:-1]]
    return [
        Outcome(FirstStep(
            tile_name=tile_name,
            tile_origin=tile_origin,  # S2 tile footprint
            basename=f"S2info_{tile_name}",
            out_filename=tiles_db,  # Trick existing file detection
            does_product_exist=lambda: True,
        )),
    ]


def eof_first_inputs_factory(
    tile_name    : str,
    configuration: Configuration,
    dag          : EODataAccessGateway,
    **kwargs,  # pylint: disable=unused-argument
) -> List[Outcome[FirstStep]]:
    """
    :class:`FirstStepFactory` hook dedicated to precise orbit inputs.

    It takes takes of returning or downloading the EOF files on-the-fly according to the single
    relative_orbit number requested in the configuration.

    :precondition: one and only one relative orbit number must have been requested in the configuration.
    :precondition: one and only one mission must have been requested in the configuration.
    """
    assert len(configuration.relative_orbit_list) >= 1
    relative_orbits = configuration.relative_orbit_list
    logger.debug("Configure EOF inputs for tile %s, orbit %s", tile_name, relative_orbits)
    eof_manager = EOFFileManager(configuration, dag)
    eof_founds = eof_manager.search_for(relative_orbits)
    assert len(eof_founds) > 0
    if not eof_founds[0]:
        error = eof_founds[0].error()
        raise exceptions.DownloadEOFFileError(str(error)) from error
    logger.info("Orbit %s OSVs will be taken from %s", relative_orbits, ",".join([f"{eof_file.value()}" for eof_file in eof_founds]))
    # Duplicate the first step for all tile_name (as this is what will be used to attach dropped inputs)
    # TODO: see how to support the case where all inputs are dropped...
    # TODO: keep only one eof_file per series of consecutive files related to a same orbit
    #       => associate orbit+mission to a single EOF file
    steps = []
    for eof_entry in eof_founds:
        if not eof_entry:
            error = eof_entry.error()
            raise exceptions.DownloadEOFFileError(str(error)) from error
        for relorb, product in eof_entry.value().items():
            logger.debug("#  orb=%03d, product=%s", relorb, product.filename)
            assert product, f"Here, we should have a non null instance for {product=}"
            step = FirstStep(
                orbit=f"{relorb:0>3d}",
                basename=str(product.filename),
                flying_unit_code=product.mission.lower(),
                tile_name=tile_name,
            )
            steps.append(step)
    for step in steps:
        logger.debug("- EOF FirstStep = %s", step)
    return [Outcome(step) for step in steps]


def register_LIA_pipelines(
    pipelines: PipelineDescriptionSequence,
    produce_angles: bool,
) -> PipelineDescription:
    """
    Internal function that takes care to register all pipelines related to
    LIA map and sin(LIA) map.
    """
    pipelines.register_inputs('tilename', tilename_first_inputs_factory)
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
        [
            ProjectGeoidToS2Tile,
            SumAllHeights(
                product_key='height_on_s2',
                key_map={'indem': 'in_s2_dem', 'ingeoid': 'in_s2_geoid'},
                fname_fmt_default='DEM+GEOID_projected_on_{tile_name}.tiff',
                dname_fmt_default='S2/{tile_name}',
                image_description='DEM + GEOID height info projected on S2 tile',
            ),
        ],
        "GenerateHeightForS2Tile",
        is_name_incremental=True,
        inputs={"in_s2_dem": s2_dem},
    )

    pipelines.register_inputs('eof', eof_first_inputs_factory)
    xyz = pipelines.register_pipeline(
        [ComputeGroundAndSatPositionsOnDEMFromEOF],
        "ComputeGroundAndSatPositionsOnDEM",
        inputs={'ineof': 'eof', 'inheight': s2_height},
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


def register_IA_pipelines(
        pipelines: PipelineDescriptionSequence,
        # produce_angles: bool,
) -> PipelineDescription:
    """
    Internal function that takes care to register all pipelines related to
    IA map and sin(IA) map.
    """
    pipelines.register_inputs('tilename', tilename_first_inputs_factory)

    pipelines.register_inputs('eof', eof_first_inputs_factory)
    xyz = pipelines.register_pipeline(
        [ComputeGroundAndSatPositionsOnEllipsoid],
        "ComputeGroundAndSatPositionsOnEllipsoid",
        inputs={'tilename': 'tilename', 'ineof': 'eof'},
    )

    # And then this time, normals are computed from S2 tile
    # Always generate sin(IA). If IA° is requested, then it's also a
    # final/requested product.
    # produce_angles is ignored as there is no extra select_IA step
    lia = pipelines.register_pipeline(
        [ComputeEllipsoidNormalsOnS2, ComputeIAOnS2],
        'ComputeIAOnS2',
        is_name_incremental=True,
        inputs={'tilename': 'tilename', 'xyz': xyz},
        product_required=True,
    )
    return lia


def s1_process(  # pylint: disable=too-many-arguments, too-many-locals
    config_opt              : Union[str, Configuration],
    *,
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
    gamma_area_process             = None,
) -> exits.Situation:
    """
    Entry point to :ref:`S1Tiling classic scenario <scenario.S1Processor>` and :ref:`S1Tiling
    NORMLIM scenario <scenario.S1ProcessorLIA>` of on demand Ortho-rectification of Sentinel-1 data
    on Sentinel-2 grid for all calibration kinds.

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
        Permits to override EODAG default wait time in minutes between two download tries.
    :param dl_timeout:
        Permits to override EODAG default maximum time in mins before stop retrying to download
        (default=20)
    :param searched_items_per_page:
        Tells how many items are to be returned by EODAG when searching for S1 images.
    :param dryrun:
        Used for debugging: external (OTB/GDAL) application aren't executed.
    :param debug_otb:
        Used for debugging: Don't execute processing tasks in DASK workers but directly in order to
        be able to analyse OTB/external application through a debugger.
    :param debug_caches:
        Used for debugging: Don't delete the intermediary files but leave them behind.
    :param watch_ram:
        Used for debugging: Monitoring Python/Dask RAM consumption.
    :param debug_tasks:
        Generate SVG images showing task graphs of the processing flows
    :param cache_before_ortho:
        Cutting, calibration and orthorectification are chained in memory unless this option is
        true. In that case, :ref:`Cut and calibrated (aka "OrthoReady") files <orthoready-files>`
        are stored in :ref:`%(tmp) <paths.tmp>`:samp:`/S1/` directory.
        Do not forget to regularly clean up this space.

    :return:
        A *nominal* exit code depending of whether everything could have been downloaded and
        produced.
    :rtype: :class:`s1tiling.libs.exits.Situation`

    :exception Error: A variety of exceptions. See below (follow the link).
    """
    def builder(
        config: Configuration,
        dryrun: bool,
        debug_caches: bool,
    ) -> Tuple[PipelineDescriptionSequence, List[WorkspaceKinds]]:
        assert (not config.filter) or (config.keep_non_filtered_products or not config.mask_cond), \
                'Cannot purge non filtered products when mask are also produced!'

        output_name_formats = main_output_name_formats(config)

        chain_LIA_and_despeckle_inmemory        = config.filter and not config.keep_non_filtered_products
        chain_GAMMA_AREA_and_despeckle_inmemory = config.filter and not config.keep_non_filtered_products
        chain_concat_and_despeckle_inmemory     = False  # See issue #118

        pipelines = PipelineDescriptionSequence(config, dryrun=dryrun, debug_caches=debug_caches)
        pipelines.register_inputs('basename', s1_raster_first_inputs_factory)
        pipelines.register_extra_parameters_for_input_factories(output_name_formats=output_name_formats)

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
            apply_LIA = pipelines.register_pipeline(
                apply_LIA_seq, product_required=True,
                inputs={'sin_LIA': sin_LIA, 'concat_S2': concat_S2},
                is_name_incremental=True,
            )
            last_product_S2 = apply_LIA
            required_workspaces.append(WorkspaceKinds.LIA)

        # GAMMA AREA Calibration (...+ Despeckle)
        elif config.calibration_type == 'gamma_naught_rtc':
            apply_GAMMA_AREA_seq: List[Type[StepFactory]] = [ApplyGammaNaughtRTCCalibration]
            if chain_GAMMA_AREA_and_despeckle_inmemory:
                apply_GAMMA_AREA_seq.append(SpatialDespeckle)
                need_to_keep_non_filtered_products = False
            else:
                need_to_keep_non_filtered_products = True

            GammaNaughtArea_registration = gamma_area_process or register_GAMMA_AREA_pipelines
            gammanaughtareas = GammaNaughtArea_registration(pipelines, config)

            apply_GAMMA_AREA = pipelines.register_pipeline(
                apply_GAMMA_AREA_seq, product_required=True,
                inputs={'gamma_area': gammanaughtareas, 'concat_S2': concat_S2},
                is_name_incremental=True,
            )
            last_product_S2 = apply_GAMMA_AREA
            required_workspaces.append(WorkspaceKinds.GAMMA_AREA)

        # Masking
        if config.mask_cond:
            pipelines.register_pipeline(
                [BuildBorderMask, SmoothBorderMask], 'GenerateMask',
                product_required=True, inputs={'in': last_product_S2})
            required_workspaces.append(WorkspaceKinds.MASK)

        # Despeckle in non-inmemory case
        if config.filter:
            # Use SpatialDespeckle, only if filter ∈ [lee, gammamap, frost, kuan]
            required_workspaces.append(WorkspaceKinds.FILTER)
            if need_to_keep_non_filtered_products:  # config.keep_non_filtered_products:
                # Define another pipeline if chaining cannot be done in memory
                pipelines.register_pipeline(
                    [SpatialDespeckle], product_required=True,
                    inputs={'in': last_product_S2})

        return pipelines, required_workspaces

    def _check_requested_number_of_orbits(config: LIAConfiguration) -> bool:
        # The check is positive: failing to comply => there is an issue in the options
        return config.calibration_type != 'normlim' or len(config.relative_orbit_list) > 0

    return do_process_with_pipeline(
        config_opt, builder,
        extra_config_checks=[(
            _check_requested_number_of_orbits,
            "At least one relative orbit is required for LIA map generation"
        )],
        ctx_managers=[DEMWorkspace],
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
    *,
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
    Entry point to :ref:`LIA Map production scenario <scenario.S1LIAMap>` that generates Local
    Incidence Angle Maps on S2 geometry.

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
        Permits to override EODAG default wait time in minutes between two download tries.
    :param dl_timeout:
        Permits to override EODAG default maximum time in mins before stop retrying to download
        (default=20)
    :param searched_items_per_page:
        Tells how many items are to be returned by EODAG when searching for S1 images.
    :param dryrun:
        Used for debugging: external (OTB/GDAL) application aren't executed.
    :param debug_otb:
        Used for debugging: Don't execute processing tasks in DASK workers but directly in order to
        be able to analyse OTB/external application through a debugger.
    :param debug_caches:
        Used for debugging: Don't delete the intermediary files but leave them behind.
    :param watch_ram:
        Used for debugging: Monitoring Python/Dask RAM consumption.
    :param debug_tasks:
        Generate SVG images showing task graphs of the processing flows

    :return:
        A *nominal* exit code depending of whether everything could have been downloaded and
        produced.
    :rtype: :class:`s1tiling.libs.exits.Situation`

    :exception Error: A variety of exceptions. See below (follow the link).
    """
    def builder(config: Configuration, dryrun: bool, debug_caches: bool) -> Tuple[PipelineDescriptionSequence, List[WorkspaceKinds]]:
        pipelines = PipelineDescriptionSequence(config, dryrun=dryrun, debug_caches=debug_caches)
        pipelines.register_inputs('basename', s1_raster_first_inputs_factory)
        output_name_formats = [
            (dname_fmt_lia_product(config), 'sin_LIA_{flying_unit_code}_{tile_name}_{orbit_direction}_{orbit}.tif'),
        ]
        if config.produce_lia_map:
            output_name_formats.append(
                (dname_fmt_lia_product(config), 'LIA_{flying_unit_code}_{tile_name}_{orbit_direction}_{orbit}.tif')
            )
        pipelines.register_extra_parameters_for_input_factories(output_name_formats=output_name_formats)
        register_LIA_pipelines_v0(pipelines, produce_angles=config.produce_lia_map)
        required_workspaces = [WorkspaceKinds.LIA]
        return pipelines, required_workspaces

    return do_process_with_pipeline(
        config_opt, builder,
        ctx_managers=[DEMWorkspace],
        dl_wait=dl_wait, dl_timeout=dl_timeout,
        searched_items_per_page=searched_items_per_page,
        nb_max_search_retries=nb_max_search_retries,
        dryrun=dryrun,
        debug_caches=debug_caches,
        debug_otb=debug_otb,
        watch_ram=watch_ram,
        debug_tasks=debug_tasks,
    )


def s1_process_lia_v1_1(  # pylint: disable=too-many-arguments
    config_opt             : Union[str, Configuration],
    *,
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
        pipelines.register_inputs('basename', s1_raster_first_inputs_factory)
        output_name_formats = [
            (dname_fmt_lia_product(config), 'sin_LIA_{flying_unit_code}_{tile_name}_{orbit_direction}_{orbit}.tif'),
        ]
        if config.produce_lia_map:
            output_name_formats.append(
                (dname_fmt_lia_product(config), 'LIA_{flying_unit_code}_{tile_name}_{orbit_direction}_{orbit}.tif')
            )
        pipelines.register_extra_parameters_for_input_factories(output_name_formats=output_name_formats)
        register_LIA_pipelines_v1_1(pipelines, produce_angles=config.produce_lia_map)
        required_workspaces = [WorkspaceKinds.LIA]
        return pipelines, required_workspaces

    return do_process_with_pipeline(
        config_opt, builder,
        ctx_managers=[DEMWorkspace],
        dl_wait=dl_wait, dl_timeout=dl_timeout,
        searched_items_per_page=searched_items_per_page,
        nb_max_search_retries=nb_max_search_retries,
        dryrun=dryrun,
        debug_caches=debug_caches,
        debug_otb=debug_otb,
        watch_ram=watch_ram,
        debug_tasks=debug_tasks,
    )


def s1_process_lia_v1_2(  # pylint: disable=too-many-arguments
    config_opt             : Union[str, Configuration],
    *,
    dryrun                 : bool = False,
    debug_otb              : bool = False,
    debug_caches           : bool = False,
    watch_ram              : bool = False,
    debug_tasks            : bool = False,
) -> exits.Situation:
    """
    Entry point to :ref:`LIA Map production scenario <scenario.S1LIAMap>` that generates :ref:`Local
    Incidence Angle Maps on S2 geometry <lia-files>`.

    It performs the following steps:

    1. Register the downloading of missing EOF matching the requested (relative) orbit number
    2. Generate the LIA maps

    :param config_opt:
        Either a :ref:`request configuration file <request-config-file>` or a
        :class:`s1tiling.libs.configuration.Configuration` instance.
    :param dryrun:
        Used for debugging: external (OTB/GDAL) application aren't executed.
    :param debug_otb:
        Used for debugging: Don't execute processing tasks in DASK workers but directly in order to
        be able to analyse OTB/external application through a debugger.
    :param debug_caches:
        Used for debugging: Don't delete the intermediary files but leave them behind.
    :param watch_ram:
        Used for debugging: Monitoring Python/Dask RAM consumption.
    :param debug_tasks:
        Generate SVG images showing task graphs of the processing flows

    :return:
        A *nominal* exit code depending of whether everything could have been downloaded and
        produced.
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
        extra_config_checks=[(
            lambda config: len(config.relative_orbit_list) > 0,
            "At least one relative orbit is required for LIA map generation"
        )],
        ctx_managers=[DEMWorkspace],
        dryrun=dryrun,
        debug_caches=debug_caches,
        debug_otb=debug_otb,
        watch_ram=watch_ram,
        debug_tasks=debug_tasks,
    )


s1_process_lia = s1_process_lia_v1_2


def s1_process_ia(  # pylint: disable=too-many-arguments
    config_opt             : Union[str, Configuration],
    *,
    dryrun                 : bool = False,
    debug_otb              : bool = False,
    debug_caches           : bool = False,
    watch_ram              : bool = False,
    debug_tasks            : bool = False,
) -> exits.Situation:
    """
    Entry point to :ref:`IA Map production scenario <scenario.S1IAMap>` that
    generates :ref:`Incidence Angle Maps on S2 geometry <ia-files>`.

    It performs the following steps:

    1. Register the downloading of missing EOF matching the requested (relative) orbit number
    2. Generate the IA maps

    :param config_opt:
        Either a :ref:`request configuration file <request-config-file>` or a
        :class:`s1tiling.libs.configuration.Configuration` instance.
    :param dryrun:
        Used for debugging: external (OTB/GDAL) application aren't executed.
    :param debug_otb:
        Used for debugging: Don't execute processing tasks in DASK workers but directly in order to
        be able to analyse OTB/external application through a debugger.
    :param debug_caches:
        Used for debugging: Don't delete the intermediary files but leave them behind.
    :param watch_ram:
        Used for debugging: Monitoring Python/Dask RAM consumption.
    :param debug_tasks:
        Generate SVG images showing task graphs of the processing flows

    :return:
        A *nominal* exit code depending of whether everything could have been downloaded and
        produced.
    :rtype: :class:`s1tiling.libs.exits.Situation`

    :exception Error: A variety of exceptions. See below (follow the link).
    """
    def builder(config: Configuration, dryrun: bool, debug_caches: bool) -> Tuple[PipelineDescriptionSequence, List[WorkspaceKinds]]:
        pipelines = PipelineDescriptionSequence(config, dryrun=dryrun, debug_caches=debug_caches)
        register_IA_pipelines(pipelines)
        required_workspaces = [WorkspaceKinds.IA]
        return pipelines, required_workspaces

    return do_process_with_pipeline(
        config_opt, builder,
        extra_config_checks=[(
            lambda config: len(config.relative_orbit_list) > 0,
            "At least one relative orbit is required for Ellipsoid IA map generation"
        )],
        dryrun=dryrun,
        debug_caches=debug_caches,
        debug_otb=debug_otb,
        watch_ram=watch_ram,
        debug_tasks=debug_tasks,
    )


def s1_process_gamma_area(  # pylint: disable=too-many-arguments
    config_opt             : Union[str, Configuration],
    *,
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
    Entry point to :ref:`GAMMA_AREA Map production scenario <scenario.S1GammaAreaMap>` that
    generates Gamma Area Maps on S2 geometry.

    It performs the following steps:

    1. Determine the S1 products to process
        Given a list of S2 tiles, we first determine the day that'll the best coverage of each S2
        tile in terms of S1 products.

        In case there is no single day that gives the best coverage for all S2 tiles, we try to
        determine the best solution that minimizes the number of S1 products to download and
        process.
    2. Process these S1 products

    :param config_opt:
        Either a :ref:`request configuration file <request-config-file>` or a
        :class:`s1tiling.libs.configuration.Configuration` instance.
    :param dl_wait:
        Permits to override EODAG default wait time in minutes between two download tries.
    :param dl_timeout:
        Permits to override EODAG default maximum time in mins before stop retrying to download
        (default=20)
    :param searched_items_per_page:
        Tells how many items are to be returned by EODAG when searching for S1 images.
    :param dryrun:
        Used for debugging: external (OTB/GDAL) application aren't executed.
    :param debug_otb:
        Used for debugging: Don't execute processing tasks in DASK workers but directly in order to
        be able to analyse OTB/external application through a debugger.
    :param debug_caches:
        Used for debugging: Don't delete the intermediary files but leave them behind.
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
        pipelines.register_inputs('basename', s1_raster_first_inputs_factory)
        output_name_formats = [(dname_fmt_gamma_area_product(config), fname_fmt_gamma_area_product(config))]
        pipelines.register_extra_parameters_for_input_factories(output_name_formats=output_name_formats)

        register_GAMMA_AREA_pipelines(pipelines, config=config)
        required_workspaces = [WorkspaceKinds.GAMMA_AREA]
        return pipelines, required_workspaces

    return do_process_with_pipeline(
        config_opt, builder,
        ctx_managers=[DEMWorkspace],
        dl_wait=dl_wait, dl_timeout=dl_timeout,
        searched_items_per_page=searched_items_per_page,
        nb_max_search_retries=nb_max_search_retries,
        dryrun=dryrun,
        debug_caches=debug_caches,
        debug_otb=debug_otb,
        watch_ram=watch_ram,
        debug_tasks=debug_tasks,
    )
