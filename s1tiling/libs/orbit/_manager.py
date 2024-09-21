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

from datetime import timedelta
from enum import Enum
import logging
import os
from typing import Dict, List, Sequence
from dateutil.parser import parse

from eodag.api.core import EODataAccessGateway

from ._providers import ASFProvider, DataspaceProvider, Provider
from ._file import filter_intersecting_eof_files, glob_eof_files
from ..configuration import Configuration
from ..outcome import DownloadOutcome


logger = logging.getLogger('s1tiling.orbit')


class ProviderKind(Enum):
    COP_DATASPACE = 1
    EARTHDATA     = 2


class EOFFileManager:
    # TODO: Don't depend on Configuration
    def __init__(self, cfg: Configuration, dag: EODataAccessGateway):
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
    ) -> List[DownloadOutcome]:
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

        request = f"EOF files between {self.__first_date} and {self.__last_date}"
        errors : List[DownloadOutcome] = []

        provider_kinds = [p for p in ProviderKind if self.__build_options[p]['class'].is_configured(self.__dag)]
        if len(provider_kinds) == 0:
            logger.warning("No data provider has been configured for EOF files")
            return [DownloadOutcome(RuntimeError("No data provider has been configured for EOF files"), request)]
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
                return [DownloadOutcome(f, f) for f in files]
            except BaseException as e:
                logger.warning(e, exc_info=False)
                logger.debug(e, exc_info=True)
                errors.append(DownloadOutcome(e, request))
        else:
            if len(errors) == 0:
                errors = [DownloadOutcome(RuntimeError("No data provider has been configured for EOF files"), request)]
            return errors

    def search_for(
            self,
            relative_orbit : int,
            missions       : Sequence[str] = (),
            dryrun         : bool          = False,
    ) -> List[DownloadOutcome]:
        # TODO: handle cache...
        # 1. scan dest_dir for EOF having relative_orbit
        #    priority to the files in the time range
        eof_files = glob_eof_files(self.__dest_dir)

        eof_files_in_range = filter_intersecting_eof_files(
                eof_files,
                self.__first_date,
                self.__last_date,
                missions
        )

        eof_files_matching = [ f for f in eof_files_in_range if f.has_relative_orbit(relative_orbit)]
        if eof_files_matching:
            return [DownloadOutcome(f.filename, f) for f in eof_files_matching]

        # 2. if not, download files in the time range
        #    analyse the new files
        #
        # @post: for each EOF file detected, build a dict of min-max abs- and/or rel- orbit numbers

        return []


