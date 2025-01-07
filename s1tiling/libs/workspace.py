#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2024 (c) CNES.
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
from typing import Dict, List, Optional, Protocol, Union

from .configuration import Configuration, dname_fmt_filtered, dname_fmt_ia_product, dname_fmt_lia_product, dname_fmt_mask, dname_fmt_tiled


logger = logging.getLogger('s1tiling.workspace')

class DEMWorkspaceConfiguration(Protocol):
    """
    Specialized protocol for configuration information related to class:`DEMWorkspace` configuration data.

    Can be seen an a ISP compliant concept for Configuration object regarding workspaces.
    """
    cache_dem_by : str
    tmpdir       : str
    dem          : str


class DEMWorkspace:
    """
    Context manager that initialize a workspace for the session, and clears temporary files on exit.
    """

    def __init__(self, cfg: DEMWorkspaceConfiguration) -> None:
        """
        constructor
        """
        self.__tmpdemdir      : Optional[tempfile.TemporaryDirectory] = None
        self.__cfg_tmpdir     = cfg.tmpdir
        self.__cfg_dem        = cfg.dem
        self.__caching_option = cfg.cache_dem_by
        assert self.__caching_option in ['copy', 'symlink']

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
                # in case there is an associated file like (egm96.grd.hdr), copy/symlink it as well
                if os.path.isfile(with_hdr := f"{geoid_file}.hdr"):
                    do_localize(with_hdr, geoid_filelink.with_suffix(geoid_filelink.suffix+'.hdr'))

        return self.__tmpdemdir.name


class WorkspaceKinds(Enum):
    """
    Enum used to list the kinds of "workspaces" needed.
    A workspace is a directory where products will be stored.

    :todo: Use a more flexible and OCP (Open-Close Principle) compliant solution.
        Indeed At this moment, only two kinds of workspaces are supported.
    """
    TILE   = 1
    LIA    = 2
    FILTER = 3
    MASK   = 4
    IA     = 6


def ensure_tiled_workspaces_exist(
        cfg: Configuration,
        tile_name: str,
        required_workspaces: List[WorkspaceKinds]
) -> None:
    """
    Makes sure the directories used for :
    - output data/{tile},
    - temporary data/S2/{tile}
    - and LIA data (if required)
    all exist
    """
    directories = {
            'out_dir': cfg.output_preprocess,
            'tmp_dir': cfg.tmpdir,
            'lia_dir': cfg.lia_directory,
            'ia_dir' : cfg.ia_directory,
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
