#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2025 (c) CNES.
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
# Authors:
# - Thierry KOLECK (CNES)
# - Luc HERMITTE (CSGROUP)
#
# =========================================================================

"""Centralize workspaces preparation"""

from enum import Enum
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Dict, List, Optional, Protocol, Tuple, Union


from . import exceptions
from .configuration import (
    Configuration, dname_fmt_filtered, dname_fmt_gamma_area_product, dname_fmt_ia_product, dname_fmt_lia_product, dname_fmt_mask, dname_fmt_tiled
)
from .Utils     import fetch_nodata_value, set_nodata_value
from .utils.dem import check_dem_coverage

logger = logging.getLogger('s1tiling.workspace')


class DEMWorkspaceConfiguration(Protocol):
    """
    Specialized protocol for configuration information related to class:`DEMWorkspace` configuration data.

    Can be seen an a ISP compliant concept for Configuration object regarding workspaces.
    """
    cache_dem_by        : str
    tmpdir              : str
    dem                 : str
    output_grid         : str
    dem_db_filepath     : str
    dem_field_ids       : List[str]
    dem_main_field_id   : str
    dem_filename_format : str
    tmp_dem_dir         : str
    GeoidFile           : str
    def register_dems_related_to_S2_tiles(self, dem: Dict[str, Dict]) -> None:
        """
        Workaround that helps caching DEM related information for later use.
        """
        pass


def search_dems_covering_tiles(
        tiles_to_process: List[str],
        cfg             : DEMWorkspaceConfiguration
) -> Tuple[Dict, Dict[str, Dict]]:
    """
    Search the DEM tiles required to process the tiles to process.
    """
    needed_dem_tiles = {}

    # Analyse DEM coverage for MGRS tiles to be processed
    dem_tiles_check = check_dem_coverage(
            cfg.output_grid,
            cfg.dem_db_filepath,
            tiles_to_process,
            cfg.dem_field_ids,
            cfg.dem_main_field_id,
    )

    # For each MGRS tile to process
    for tile in tiles_to_process:
        # logger.debug("Check DEM coverage for %s", tile)
        # Get DEM tiles coverage statistics
        dem_tiles = dem_tiles_check[tile]
        current_coverage = 0
        # Compute global coverage
        for _, dem_info in dem_tiles.items():
            current_coverage += dem_info['_coverage']
        needed_dem_tiles.update(dem_tiles)
        # If DEM coverage of MGRS tile is enough, process it
        # Round coverage at 3 digits as tile footprint has a very limited precision
        current_coverage = round(current_coverage, 3)
        if current_coverage < 1.:
            logger.warning("Tile %s has insufficient DEM coverage (%s%% - %s DEMs)", tile, 100 * current_coverage, len(dem_tiles))
        else:
            logger.info("Tile %s has a DEM coverage of %s%% => OK (%s DEMs)", tile, 100 * current_coverage, len(dem_tiles))

    # Remove duplicates
    return needed_dem_tiles, dem_tiles_check


def check_dem_tiles(cfg: DEMWorkspaceConfiguration, dem_tile_infos: Dict) -> bool:
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


class DEMWorkspace:
    """
    Context manager dedicated to DEM.

    1. It takes care of analysing the DEM files required for the S2 tiles
    2. It initializes a workspace for the session, and clears temporary files on exit.
    3. DEM related information are stored back on the configuration object
    """

    def __init__(self, cfg: DEMWorkspaceConfiguration, tiles_to_process: List[str]) -> None:
        """
        constructor
        """
        # Check tiles
        assert tiles_to_process, "DEM selection needs S2 tiles"

        # Check DEM
        needed_dem_tiles, dems_by_s2_tiles = search_dems_covering_tiles(tiles_to_process, cfg)
        logger.info("Required DEM tiles: %s", list(needed_dem_tiles.keys()))

        if not check_dem_tiles(cfg, needed_dem_tiles):
            raise exceptions.MissingDEMError()

        self.__tmpdemdir      : Optional[tempfile.TemporaryDirectory] = None
        self.__cfg_tmpdir     = cfg.tmpdir
        self.__cfg_dem        = cfg.dem
        self.__caching_option = cfg.cache_dem_by
        assert self.__caching_option in ['copy', 'symlink']
        cfg.tmp_dem_dir = self.tmpdemdir(
                needed_dem_tiles, cfg.dem_filename_format, cfg.GeoidFile)
        cfg.register_dems_related_to_S2_tiles(dems_by_s2_tiles)

    def __enter__(self) -> "DEMWorkspace":
        """
        Turn the class into a context manager, context acquisition function
        """
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """
        Turn the class into a context manager, cleanup function
        """
        if self.__tmpdemdir:
            logger.debug('Cleaning temporary DEM diretory (%s)', self.__tmpdemdir)
            self.__tmpdemdir.cleanup()
            self.__tmpdemdir = None
        return False

    def tmpdemdir(self, dem_tile_infos: Dict, dem_filename_format: str, geoid_file: str) -> str:
        """
        Generate the temporary directory for DEM tiles on the fly,
        and either populate it with symbolic links to the actual DEM
        tiles, or copies of the actual DEM tiles.
        """
        assert self.__caching_option in ['copy', 'symlink']
        if not self.__tmpdemdir:
            # copy all needed DEM & geoid files in a temp directory for orthorectification processing
            self.__tmpdemdir = tempfile.TemporaryDirectory(dir=self.__cfg_tmpdir)
            logger.debug('Create temporary DEM directory (%s) for needed tiles %s', self.__tmpdemdir.name, list(dem_tile_infos.keys()))
            assert Path(self.__tmpdemdir.name).is_dir()
            def do_symlink(src: Union[Path, str], dst: Path):
                logger.debug('- ln -s %s <-- %s', src, dst)
                dst.symlink_to(src)
            def do_copy(src: Union[Path, str], dst: Path):
                logger.debug('- cp %s --> %s', src, dst)
                shutil.copy2(src, dst)
            do_localize = do_symlink if self.__caching_option == 'symlink' else do_copy

            for _, dem_tile_info in dem_tile_infos.items():
                dem_file          = dem_filename_format.format_map(dem_tile_info)
                dem_tile_filepath = Path(self.__cfg_dem, dem_file)
                dem_tile_filelink = Path(self.__tmpdemdir.name, os.path.basename(dem_file))  # for copernicus dem
                dem_tile_filelink.parent.mkdir(parents=True, exist_ok=True)
                do_localize(dem_tile_filepath, dem_tile_filelink)
            # + copy/link geoid
            geoid_filelink = Path(self.__cfg_tmpdir, 'geoid', os.path.basename(geoid_file))
            if not geoid_filelink.exists():
                geoid_filelink.parent.mkdir(parents=True, exist_ok=True)
                do_localize(geoid_file, geoid_filelink)
                # in case there is an associated file like `egm96.grd.hdr` or `egm96.gtx.aux.xml`, copy/symlink it as well
                for suffix in ('hdr', 'aux.xml'):
                    if os.path.isfile(with_extra := f"{geoid_file}.{suffix}"):
                        do_localize(with_extra, geoid_filelink.with_suffix(f'{geoid_filelink.suffix}.{suffix}'))
            # Make sure Geoid nodata value is not something like -88.88
            geoid_nodata = float(fetch_nodata_value(geoid_filelink, is_running_dry=False, default_value=0))
            logger.debug("'%s' nodata is %s", geoid_filelink, geoid_nodata)
            if -200 < geoid_nodata < 200 :
                set_nodata_value(geoid_filelink, is_running_dry=False, value=-32768)

        return self.__tmpdemdir.name


class WorkspaceKinds(Enum):
    """
    Enum used to list the kinds of "workspaces" needed.
    A workspace is a directory where products will be stored.

    :todo: Use a more flexible and OCP (Open-Close Principle) compliant solution.
        Indeed At this moment, only two kinds of workspaces are supported.
    """
    TILE       = 1
    LIA        = 2
    FILTER     = 3
    MASK       = 4
    GAMMA_AREA = 5
    IA         = 6


def ensure_tiled_workspaces_exist(
    cfg: Configuration,
    tile_name: str,
    required_workspaces: List[WorkspaceKinds]
) -> None:
    """
    Makes sure the directories used for :
    - output data/{tile},
    - temporary data/S2/{tile}
    - LIA data (if required)
    - IA data (if required)
    - γ° RTC data (if required)
    all exist
    """
    directories = {
        'out_dir'        : cfg.output_preprocess,
        'tmp_dir'        : cfg.tmpdir,
        'lia_dir'        : cfg.extra_directories['lia_dir'],
        'ia_dir'         : cfg.extra_directories['ia_dir'],
        'gamma_area_dir' : cfg.extra_directories['gamma_area_dir'],
    }

    working_directory = os.path.join(cfg.tmpdir, 'S2', tile_name)
    os.makedirs(working_directory, exist_ok=True)

    if WorkspaceKinds.TILE in required_workspaces:
        wdir = dname_fmt_tiled(cfg).format(**directories, tile_name=tile_name)
        os.makedirs(wdir, exist_ok=True)

    if WorkspaceKinds.MASK in required_workspaces:
        wdir = dname_fmt_mask(cfg).format(**directories, tile_name=tile_name)
        os.makedirs(wdir, exist_ok=True)

    if WorkspaceKinds.FILTER in required_workspaces:
        wdir = dname_fmt_filtered(cfg).format(**directories, tile_name=tile_name)
        os.makedirs(wdir, exist_ok=True)

    # if cfg.calibration_type == 'normlim':
    if WorkspaceKinds.LIA in required_workspaces:
        wdir = dname_fmt_lia_product(cfg).format(**directories, tile_name=tile_name)
        os.makedirs(wdir, exist_ok=True)

    if WorkspaceKinds.IA in required_workspaces:
        wdir = dname_fmt_ia_product(cfg).format(**directories, tile_name=tile_name)
        os.makedirs(wdir, exist_ok=True)

    if WorkspaceKinds.GAMMA_AREA in required_workspaces:
        wdir = dname_fmt_gamma_area_product(cfg).format(**directories, tile_name=tile_name)
        os.makedirs(wdir, exist_ok=True)
