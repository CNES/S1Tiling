#!/usr/bin/env python3
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

""" This sub-module defines the EOFFileManager """

from collections.abc import Sequence
from datetime import timedelta
from dateutil.parser import parse
from enum import Enum
import logging
import os
from typing import Dict, List, Optional, Protocol

from eodag.api.core import EODataAccessGateway
from eof.client import Filename

from ._providers import ASFProvider, DataspaceProvider, Provider
from ._file import SentinelOrbitFile, filter_intersecting_eof_files, glob_eof_files
from ..outcome import DownloadOutcome


EOFOutcome = DownloadOutcome[Filename, Optional[SentinelOrbitFile]]


logger = logging.getLogger('s1tiling.orbit')

class EOFConfiguration(Protocol):
    """
    Specialized protocol for configuration information related to EOF configuration data.

    Can be seen an a ISP compliant concept for Configuration object regarding EOF data.
    """
    first_date    : str
    last_date     : str
    eof_directory : Filename
    platform_list : Sequence[str]
    download      : bool


class ProviderKind(Enum):
    COP_DATASPACE = 1
    EARTHDATA     = 2


class EOFFileManager:
    # TODO: Don't depend on Configuration
    def __init__(self, cfg: EOFConfiguration, dag: EODataAccessGateway):
        """
        constructor
        """
        assert(dag)
        self.__cfg           = cfg
        self.__dag           = dag
        self.__first_date    = parse(cfg.first_date)
        self.__last_date     = parse(cfg.last_date) + timedelta(days=1) - timedelta(seconds=1)
        self.__dest_dir      = cfg.eof_directory
        self.__missions      = cfg.platform_list
        self.__build_options : Dict[ProviderKind, Dict] = {
                ProviderKind.COP_DATASPACE : {
                    'class':   DataspaceProvider,
                    'options': {'dag': dag},
                },
                ProviderKind.EARTHDATA     : {
                    'class': ASFProvider,
                    'options': {},
                },
        }

    def add_extra_build_option(self, provider: ProviderKind, **kwargs):
        """
        Permits to tune construction parameters passed to the :class:`Provider` instances.

        Typically, it can be used to set `cache_dir` when building :class:`ASFProvider`
        """
        self.__build_options[provider]['options'].update(**kwargs)

    def _instanciate_provider(self, provider: ProviderKind) -> Provider:
        """
        Internal method that do instantiate an EOF provider.
        """
        provider_data = self.__build_options[provider]
        return provider_data['class'](**provider_data['options'])

    def _ensure_workspaces_exist(self) -> None:
        """
        Makes sure the directories used for :
        - eof files
        all exist
        """
        for path in [self.__dest_dir]:
            if not os.path.isdir(path):
                os.makedirs(path, exist_ok=True)

    def download_eof(
            self,
            missions  : Sequence[str] = (),
            dryrun    : bool          = False,
    ) -> List[EOFOutcome]:
        """
        Main entry point to search and download the EOF precise orbit files.

        The orbits files are searched in the specified time range (construction
        parameters), for the chosen missions (default is set during construction but can
        be overridden when calling :meth:`download_eof`.
        """
        if not self.__cfg.download:
            logger.info("Using EOF files already downloaded, as per configuration request")
            # TODO: Should do a glob/ls
            return []

        self._ensure_workspaces_exist()

        request = f"between {self.__first_date} and {self.__last_date}"
        errors : List[EOFOutcome] = []

        provider_kinds = [p for p in ProviderKind if self.__build_options[p]['class'].is_configured(self.__dag)]
        if len(provider_kinds) == 0:
            logger.warning("No data provider has been configured for EOF files")
            return [EOFOutcome(RuntimeError("No data provider has been configured for EOF files {request}"), None)]
        logger.debug(
                "EOF files will be searched on %s between %s and %s",
                " and ".join((str(p) for p in provider_kinds)),
                self.__first_date,
                self.__last_date,
        )
        missions = missions or self.__missions
        for provider_kind in provider_kinds:
            try:
                provider = self._instanciate_provider(provider_kind)
                eofs = provider.search(self.__first_date, self.__last_date, missions)
                files = provider.download(eofs, self.__dest_dir)
                return [EOFOutcome(f, SentinelOrbitFile(f)) for f in files]
            except BaseException as e:
                logger.warning(e, exc_info=False)
                logger.debug(e, exc_info=True)
                errors.append(EOFOutcome(e, None))
        else:
            if len(errors) == 0:
                errors = [EOFOutcome(RuntimeError("No data provider has been configured for EOF files {request}"), None)]
            return errors

    def search_for(
            self,
            relative_orbit : int,
            missions       : Sequence[str] = (),
            dryrun         : bool          = False,
    ) -> List[EOFOutcome]:
        # TODO: handle cache...
        # 1. scan dest_dir for EOF having relative_orbit
        #    priority to the files in the time range
        eof_files = glob_eof_files(self.__dest_dir)

        eof_files_matching = self._filter_files(eof_files, relative_orbit, missions)
        if eof_files_matching:
            # Several results possible as we can request several missions...
            # But should we be precise with the target mission as we are with the target relative orbit?
            return [DownloadOutcome(f.filename, f) for f in eof_files_matching]

        # 2. if not, download files in the time range
        #    analyse the new files
        # downloaded_products = self.download_eof(missions, dryrun)
        # @post: for each EOF file detected, build a dict of min-max abs- and/or rel- orbit numbers

        return []

    def _filter_files(
            self,
            eof_files      : List[SentinelOrbitFile],
            relative_orbit : int,
            missions       : Sequence[str] = (),
    ) -> List[SentinelOrbitFile]:
        eof_files_in_range = filter_intersecting_eof_files(
                eof_files,
                self.__first_date,
                self.__last_date,
                missions
        )
        return [ f for f in eof_files_in_range if f.has_relative_orbit(relative_orbit, -1)]


# ===============[ "Internal" functions used to implement the public service
# This organisation eases the writing of unit tests
