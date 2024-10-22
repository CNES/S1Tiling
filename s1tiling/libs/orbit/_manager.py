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

"""This sub-module defines the EOFFileManager"""

from collections.abc import Sequence
from datetime import timedelta
from enum import Enum
import logging
import os
from typing import Dict, List, Optional, Protocol

from dateutil.parser import parse

from eodag.api.core import EODataAccessGateway
from eof.client import Filename
from portion import closed as interval
from portion import empty as empty_interval

from ._providers import ASFProvider, DataspaceProvider, Provider
from ._file import SentinelOrbitFile, filter_intersecting_eof_file_dict, filter_intersecting_eof_file_list, glob_eof_files, filter_eof_files_according_to_orbit_and_mission, filter_eof_files_containing_orbit
from ..outcome import DownloadOutcome
from ..utils import partition


EOFDownloadOutcome = DownloadOutcome[SentinelOrbitFile]
EOFOutcome         = DownloadOutcome[Dict[int, SentinelOrbitFile]]


logger = logging.getLogger("s1tiling.orbit")


class EOFConfiguration(Protocol):
    """
    Specialized protocol for configuration information related to EOF configuration data.

    Can be seen an a ISP compliant concept for Configuration object regarding EOF data.
    """

    first_date    : str
    last_date     : str
    eof_directory : Filename
    platform_list : List[str]
    download      : bool


class ProviderKind(Enum):
    """
    List of EOF file providers
    """

    COP_DATASPACE = 1
    EARTHDATA     = 2


class EOFFileManager:
    """
    EOF files manager.

    The main service is :meth:`search_for` which returns the name of the EOF file that contains the
    requested orbit information within the required time range.

    Files are firts searched in the cache directory. And if not, they'll be downloaded on-the-fly
    on the EOF providers for which credential information have been set.
    """

    # TODO: Don't depend on Configuration
    def __init__(self, cfg: EOFConfiguration, dag: Optional[EODataAccessGateway]):
        """
        constructor
        """
        self.__cfg           = cfg
        self.__dag           = dag
        self.__first_date    = parse(cfg.first_date)
        self.__last_date     = parse(cfg.last_date) + timedelta(days=1) - timedelta(seconds=1)
        self.__dest_dir      = cfg.eof_directory
        self.__missions      = cfg.platform_list
        self.__build_options : Dict[ProviderKind, Dict] = {
                ProviderKind.COP_DATASPACE : {
                    "class":   DataspaceProvider,
                    "options": {"dag": dag},
                },
                ProviderKind.EARTHDATA     : {
                    "class": ASFProvider,
                    "options": {},
                },
        }

    def add_extra_build_option(self, provider: ProviderKind, **kwargs):
        """
        Permits to tune construction parameters passed to the :class:`Provider` instances.

        Typically, it can be used to set `cache_dir` when building :class:`ASFProvider`
        """
        self.__build_options[provider]["options"].update(**kwargs)

    def _instanciate_provider(self, provider: ProviderKind) -> Provider:
        """
        Internal method that do instantiate an EOF provider.
        """
        provider_data = self.__build_options[provider]
        return provider_data["class"](**provider_data["options"])

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
    ) -> List[EOFDownloadOutcome]:
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
        errors : List[EOFDownloadOutcome] = []

        provider_kinds = [
            p
            for p in ProviderKind
            if self.__dag and self.__build_options[p]["class"].is_configured(self.__dag)
        ]
        if len(provider_kinds) == 0:
            logger.warning("No data provider has been configured for EOF files")
            return [EOFDownloadOutcome(RuntimeError(f"No data provider has been configured for EOF files {request}"))]
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
                return [EOFDownloadOutcome(SentinelOrbitFile(f)) for f in files]
            except BaseException as e:  # pylint: disable=broad-except
                logger.warning(e, exc_info=False)
                # logger.debug(e, exc_info=True)
                errors.append(EOFDownloadOutcome(e))
        if len(errors) == 0:
            errors = [EOFDownloadOutcome(RuntimeError(f"No data provider has been configured for EOF files {request}"))]
        return errors

    def search_for(
            self,
            relative_orbits: List[int],
            missions       : Sequence[str] = (),
            dryrun         : bool          = False,
    ) -> List[EOFOutcome]:
        """
        Search for the precise orbit files within the time range contain the requested orbit.

        :param relative_orbits: List of relative orbit numbers designating the searched orbits
        :param missions:        List of missions searched. By defaut search in all!
        :param dryrun:          Set to True to inhibit actual downloading
        """
        # Several results possible for a pair <mission, orbit> as and sometimes 3 orbits may
        # overlap instead of just 2. e.g.:
        #   - [30584 .. 30600] + [30598 .. 30614]  <-- 3 overlapping
        #   - [30598 .. 30614] + [30613 .. 30629]  <-- 2 overlapping
        #
        # Still, a question:
        # ~> should we be precise (in the configuration file) with the target mission as we
        #    are with the target relative orbit?
        results: List[EOFOutcome]

        # Cached for logs:
        relative_orbits_4logs = ", ".join(f"{relative_orbits}")

        # 1. scan dest_dir for EOF having relative_orbit
        # TODO: handle cache...
        eof_files = glob_eof_files(self.__dest_dir)

        eof_files_matching_orbits = filter_eof_files_according_to_orbit_and_mission(
            eof_files, relative_orbits, margin=-1, missions=missions)

        if len(eof_files_matching_orbits) > 0:
            eof_files_in_time_range = filter_intersecting_eof_file_dict(eof_files_matching_orbits, self.__first_date, self.__last_date)
            if len(eof_files_in_time_range) == 0:
                logger.warning(
                    "Precise orbit files matching orbits %s for %s missions have been found, but not in the requested time range [%s .. %s]",
                    relative_orbits_4logs,
                    "and ".join(missions) or "all",
                    self.__first_date,
                    self.__last_date,
                )
            results = [EOFOutcome(f) for f in eof_files_matching_orbits]
            return results

        # 2. if not, download files in the time range
        #    analyse the new files
        logger.debug(
                "No precise orbit files found in cache that contains OSV for the orbits %s within the time range [%s .. %s]",
                relative_orbits_4logs,
                self.__first_date, self.__last_date,
        )

        # 2.1. First: There may be no files matching the requested relative orbit number, yet the
        #      time range may be fully covered by the files in cache. In that case, no need to download
        #      Indeed, trying to download new EOF files within the time range will get no new EOF
        #      file, and thus the requested relative orbits won't be covered either.
        the_period_is_fully_covered_in_cache = self._has_the_period_fully_covered_in_cache(eof_files)
        if the_period_is_fully_covered_in_cache:
            msg = (f"No occurrence of the requested relative orbits {relative_orbits_4logs} found in the request time range."
                   f" [{self.__first_date} .. {self.__last_date}]")
            logger.warning("%s", msg)
            return [EOFOutcome(RuntimeError(msg))]

        # 2.2. Download if requested or if it would make a difference
        results = []
        if self.__cfg.download:
            downloaded_products = self.download_eof(missions, dryrun)
            eof_products, eof_errors = partition(bool, downloaded_products)
            if eof_products:
                eof_files = [prod.value() for prod in eof_products]
                # First. Let's check all files are in the time range, and match the requested missions
                # if not, there is a download error
                eof_files_in_range = filter_intersecting_eof_file_list(
                    eof_files,
                    self.__first_date,
                    self.__last_date,
                    missions
                )
                if not(eof_files_in_range):
                    # NB: We could also tests whether the list are identical
                    raise RuntimeError(
                        f"EOF files downloaded don't match the requested missions {missions} and "
                        f"time range [{self.__first_date}..{self.__last_date}]: {eof_files}")
                # Then: try to see if matching products have been downloaded
                eof_files_matching = filter_eof_files_according_to_orbit_and_mission(
                    eof_files, relative_orbits, -1, missions)
                results = [EOFOutcome(f) for f in eof_files_matching]
            results.extend((EOFOutcome(e.error()) for e in eof_errors))

        # @post: for each EOF file detected, build a dict of min-max abs- and/or rel- orbit numbers

        if len(results) == 0:
            msg = (f"No precise orbit files found containing OSVs for orbits {relative_orbits_4logs} in the time range"
                   f" [{self.__first_date} .. {self.__last_date}]")
            logger.warning("%s", msg)

            results.append(EOFOutcome(RuntimeError(msg)))
        return results

    # def _filter_files(
    #         self,
    #         eof_files      : List[SentinelOrbitFile],
    #         relative_orbit : int,
    #         missions       : Sequence[str] = (),
    # ) -> List[SentinelOrbitFile]:
    #     """
    #     Filter EOF files that intersect the requested time range, and that contain the requested relative orbit
    #     with a margin of 1 (in order to handle cases around ascending crossing node.
    #     """
    #     eof_files_in_range = filter_intersecting_eof_files(
    #             eof_files,
    #             self.__first_date,
    #             self.__last_date,
    #             missions
    #     )
    #     return filter_eof_files_containing_orbit(eof_files_in_range, relative_orbit, -1)

    def _has_the_period_fully_covered_in_cache(
        self, eof_files: List[SentinelOrbitFile]
    ) -> bool:
        """
        Returns whether the request time range is fully contained by the union of the
        time span of all the EOF files.
        """
        tgt_interval = interval(self.__first_date, self.__last_date)
        cumulated_interval = empty_interval()
        for eof_file in eof_files:
            cumulated_interval |= to_interval(eof_file)
        the_period_is_fully_covered_in_cache = tgt_interval in cumulated_interval
        logger.debug(f"{the_period_is_fully_covered_in_cache=} <== {tgt_interval=} ⊂ {cumulated_interval=}")
        return the_period_is_fully_covered_in_cache


# ===============[ "Internal" functions used to implement the public service
# This organisation eases the writing of unit tests
def to_interval(eof_file: SentinelOrbitFile) -> interval:
    """
    Helper function that returns the time interval associated to a EOF file.
    """
    return interval(
            eof_file.start_time,
            eof_file.stop_time,
    )
