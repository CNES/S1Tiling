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

"""This sub-module defines the EOFFileManager"""

from collections.abc import Collection, Iterable
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Optional, Protocol, Tuple

from dateutil.parser import parse

from eodag.api.core import EODataAccessGateway
from portion import Interval, closed as closed_interval
from portion import empty as empty_interval

from ..utils.eodag        import (
    EODAG_DEFAULT_DOWNLOAD_TIMEOUT, EODAG_DEFAULT_DOWNLOAD_WAIT,
)
from ._file      import (
    ALL_MISSIONS,
    SentinelOrbitFile,
    analyse_obt2eof_map_quality_according_to_request,
    filter_intersecting_eof_file_list,
    filter_uniq_eofs,
    glob_eof_files,
)
from ._providers import EodagProvider, Provider
from ..outcome   import DownloadOutcome
from ..utils     import AnyPath, partition


#: Outcome result of download operation: SentinelOrbitFile
EOFDownloadOutcome = DownloadOutcome[SentinelOrbitFile]

#: Outcome result for searched EOF files: Dict of {relorb -> File}
EOFOutcome         = DownloadOutcome[Dict[int, SentinelOrbitFile]]


logger = logging.getLogger("s1tiling.orbit")


class EOFConfiguration(Protocol):
    """
    Specialized protocol for configuration information related to EOF configuration data.

    Can be seen an a ISP compliant concept for Configuration object regarding EOF data.
    """

    first_date        : str
    last_date         : str
    extra_directories : Dict[str, AnyPath]
    platform_list     : List[str]
    download          : bool


class EOFFileManager:
    """
    EOF files manager.

    The main service is :meth:`search_for` which returns the name of the EOF file that contains the
    requested orbit information within the required time range.

    Files are firts searched in the cache directory. And if not, they'll be downloaded on-the-fly
    on the EOF providers for which credential information have been set.
    """

    # TODO: Don't depend on Configuration
    def __init__(self, cfg: EOFConfiguration, dag: EODataAccessGateway):
        """
        constructor
        """
        self.__cfg           = cfg
        self.__dag           = dag
        self.__first_date    = parse(cfg.first_date)
        self.__last_date     = parse(cfg.last_date) + timedelta(days=1) - timedelta(seconds=1)
        self.__dest_dir      = cfg.extra_directories['eof_dir']
        self.__missions      = cfg.platform_list
        self.__dl_wait       = getattr(cfg, 'dl_wait',    EODAG_DEFAULT_DOWNLOAD_WAIT)
        self.__dl_timeout    = getattr(cfg, 'dl_timeout', EODAG_DEFAULT_DOWNLOAD_TIMEOUT)

    def _instanciate_provider(self) -> Provider:
        """
        Internal method that do instantiate an EOF provider.
        """
        return EodagProvider(dag=self.__dag, dl_wait=self.__dl_wait, dl_timeout=self.__dl_timeout)

    def _ensure_workspaces_exist(self) -> None:
        """
        Makes sure the directories used for:
        - eof files
        all exist
        """
        for path in [self.__dest_dir]:
            if not os.path.isdir(path):
                os.makedirs(path, exist_ok=True)

    def do_download_eof_files(
        self,
        missions : Collection[str] = (),
        dryrun   : bool            = False,
    ) -> List[EOFDownloadOutcome]:
        """
        Raw function to search and download remote EOF precise orbit files, independently of the
        requested orbit numbers.

        The orbits files are searched in the specified time range (construction
        parameters), for the chosen missions (default is set during construction but can
        be overridden when calling :meth:`do_download_eof_files`.

        :return: A list of :class:`Outcome` of the :class:`SentinelOrbitFile` products downloaded or
                 failed to being downloaded.
        :return: An erroneous :class:`Outcome` if no EOF providers have been configured.
        """
        if not self.__cfg.download:
            logger.info("Using EOF files already downloaded, as per configuration request")
            # TODO: Should do a glob/ls
            return []

        self._ensure_workspaces_exist()

        request = f"between {self.__first_date} and {self.__last_date}"
        errors : List[EOFDownloadOutcome] = []

        if not EodagProvider.is_configured(self.__dag):
            logger.warning("No data provider has been configured for EOF files")
            return [EOFDownloadOutcome(RuntimeError(f"No data provider has been configured for EOF files {request}"))]
        logger.debug(
            "EOF files will be searched between %s and %s",
            self.__first_date,
            self.__last_date,
        )
        missions = missions or self.__missions
        try:
            provider = self._instanciate_provider()
            eofs = provider.search(self.__first_date, self.__last_date, missions)
            files = provider.download(list(eofs), self.__dest_dir)
            return [EOFDownloadOutcome(SentinelOrbitFile(f.value())) for f in files]
        except BaseException as e:  # pylint: disable=broad-except
            logger.warning(e, exc_info=False)
            # logger.debug(e, exc_info=True)
            errors.append(EOFDownloadOutcome(e))
        assert len(errors) > 0, "This situation shouldn't happen: either we return a result, or an exception has been caught and converted..."
        return errors

    def _search_on_disk(
        self,
        relative_orbits: List[int],
        missions       : Collection[str],
        first_date     : datetime,
        last_date      : datetime,
    ) -> Tuple[Dict[int, SentinelOrbitFile], Iterable[int], Optional[bool]]:
        """
        Takes care of analysing the EOF found on disk and filter them according to the requested
        orbits.

        When several EOF files cover an orbit, we keep in priority:
        1. The newest EOF file that intersects the requested time range, and that matches the
           requested platform(/mission)
        2. The oldest EOF file otherwise

        :return: A single dictionary that maps a :class:`SentinelOrbitFile` to a relative orbit
                 number.
        :return: A list of missing orbits
        :return: Whether the eof files fully cover the requested time period, when orbits are
                 missing
        """
        # Several results possible for a pair <mission, orbit> as and sometimes 3 orbits may
        # overlap instead of just 2. e.g.:
        #   - [30584 .. 30600] + [30598 .. 30614]  <-- 3 overlapping
        #   - [30598 .. 30614] + [30613 .. 30629]  <-- 2 overlapping

        # Scan dest_dir for EOF having relative_orbit
        # TODO: handle cache as we loop over several tiles
        eof_files = glob_eof_files(self.__dest_dir)
        logger.debug("%s EOF files found in %s", len(eof_files), self.__dest_dir)

        # Keep only one EOF per orbit
        obt2eof_map, missing_orbits = filter_uniq_eofs(
            eof_files,
            first_date, last_date,
            relative_orbits,
            missions or ALL_MISSIONS,
        )
        period_is_covered : Optional[bool] = self._is_the_period_fully_covered_in_cache(eof_files, first_date, last_date) if missing_orbits else None
        return obt2eof_map, missing_orbits, period_is_covered

    def _fetch_eof_files(  # pylint: disable=too-many-arguments
        self,
        relative_orbits   : List[int],
        missions          : Collection[str],
        known_obt2eof_map : Dict[int, SentinelOrbitFile],
        first_date        : datetime,
        last_date         : datetime,
        dryrun            : bool,
    ) -> Tuple[Dict[int, SentinelOrbitFile], Iterable[int], List[EOFOutcome]]:
        if not self.__cfg.download:
            return known_obt2eof_map, set(relative_orbits) - known_obt2eof_map.keys(), []
        obt2eof_map    : Dict[int, SentinelOrbitFile] = {}
        missing_orbits : Iterable[int]                = []
        # 1. Download everything in the specified time range and mission list
        downloaded_products = self.do_download_eof_files(missions, dryrun)
        # 2. Analyse whether errors occured in order:
        #    - to report "no file downloaded" with an exception
        #    - to return whether there is no files in the requested time range
        eof_products, eof_errors = partition(bool, downloaded_products)
        if eof_products:
            eof_files = [prod.value() for prod in eof_products]
            # First. Let's check all files are in the time range, and match the requested missions
            # if not, there is an unexpected download error
            eof_files_in_range = filter_intersecting_eof_file_list(
                eof_files,
                first_date,
                last_date,
                # missions,  # missions will be analysed in filter_uniq_eofs
            )
            if not eof_files_in_range:
                # NB: We could also tests whether the lists are identical
                raise RuntimeError(
                    f"EOF files downloaded don't match the requested missions {missions} and "
                    f"time range [{first_date}..{last_date}]: {eof_files}")
            # Then: try to see if matching products have been downloaded
            obt2eof_map, missing_orbits = filter_uniq_eofs(
                eof_files,
                first_date,
                last_date,
                relative_orbits,
                missions or ALL_MISSIONS,
                known_obt2eof_map,
            )

        # Convert errors from EOFDownloadOutcome0 to EOFOutcome
        errors : List[EOFOutcome] = [EOFOutcome(e.error()) for e in eof_errors]

        # # @post: for each EOF file detected, build a dict of min-max abs- and/or rel- orbit numbers
        # if len(obt2eof_map) == 0:
        #     relative_orbits_4logs = ", ".join((f"{ro}" for ro in relative_orbits))
        #     msg = (f"No precise orbit files found containing OSVs for orbits {relative_orbits_4logs} in the time range"
        #            f" [{first_date} .. {last_date}]")
        #     logger.warning("%s", msg)

        #     errors.append(EOFOutcome(RuntimeError(msg)))
        return obt2eof_map, missing_orbits, errors

    def search_for(  # pylint: disable=too-many-arguments
        self,
        relative_orbits: List[int],
        *,
        missions       : Collection[str]    = (),
        first_date     : Optional[datetime] = None,
        last_date      : Optional[datetime] = None,
        dryrun         : bool               = False,
    ) -> List[EOFOutcome]:
        """
        Searchs for the precise orbit files within the time range that contain the requested orbits.

        :param relative_orbits: List of relative orbit numbers designating the searched orbits
        :param missions:        List of missions searched. By defaut search in all!
        :param first_date:      Optional date to override time from configuration
        :param last_date:       Optional date to override time from configuration
        :param dryrun:          Set to True to inhibit actual downloading
        :return:                The list of all the matching precise orbit files available on disk
                                -- they could have been there or downloaded on-the-fly.
        """
        first_date = first_date or self.__first_date
        last_date  = last_date  or self.__last_date
        # Several results possible for a pair <mission, orbit> as and sometimes 3 orbits may
        # overlap instead of just 2. e.g.:
        #   - [30584 .. 30600] + [30598 .. 30614]  <-- 3 overlapping
        #   - [30598 .. 30614] + [30613 .. 30629]  <-- 2 overlapping
        res : List[EOFOutcome] = []

        # 1. scan dest_dir for EOF having relative_orbit
        obt2eof_map, missing_orbits, period_is_fully_covered = self._search_on_disk(relative_orbits, missions, first_date, last_date)

        # 2. if eof files appear to be missing, download files in the time range for each mission
        # unless the time period is fully covered
        extra_log = ''
        if missing_orbits:
            if period_is_fully_covered:
                logger.info(
                    "Time period [%s..%s] is fully covered by cached EOF files on disk. "
                    "No download attempt is made for the missing orbits %s",
                    first_date, last_date, missing_orbits
                )
            elif self.__cfg.download:
                obt2eof_map, missing_orbits, eof_errors = self._fetch_eof_files(relative_orbits, missions, obt2eof_map, first_date, last_date, dryrun)
                res = eof_errors
            else:
                logger.warning(
                    "No EOF files found for the requested time period and orbits, "
                    "but no download attempt will be made as it has been disabled per configuration.")
                extra_log = ", nor download,"

        # 3. Analyse EOF product quality
        analyse_obt2eof_map_quality_according_to_request(obt2eof_map, first_date, last_date, missions or ALL_MISSIONS,)

        # 4. Convert EOF results and errors into EOFOutcome instances
        res.extend([
            EOFOutcome({relorb: prod})
            for relorb, prod in obt2eof_map.items()
        ])
        res.extend([
            EOFOutcome(RuntimeError(f"Cannot find{extra_log} precise orbit file for orbit {ro:>03d} between {first_date} and {last_date}"))
            for ro in missing_orbits
        ])
        return res

    @staticmethod
    def _is_the_period_fully_covered_in_cache(
        eof_files  : List[SentinelOrbitFile],
        first_date : datetime,
        last_date  : datetime,
    ) -> bool:
        """
        Returns whether the request time range is fully contained by the union of the time spans of
        all the EOF files.
        """
        tgt_interval = closed_interval(first_date, last_date)
        cumulated_interval = empty_interval()
        for eof_file in eof_files:
            cumulated_interval |= to_interval(eof_file)
        the_period_is_fully_covered_in_cache = tgt_interval in cumulated_interval
        logger.debug(f"{the_period_is_fully_covered_in_cache=} <== {tgt_interval=} ⊂ {cumulated_interval=}")
        return the_period_is_fully_covered_in_cache


# ===============[ "Internal" functions used to implement the public service
# This organisation eases the writing of unit tests
def to_interval(eof_file: SentinelOrbitFile) -> Interval:
    """
    Helper function that returns the time interval associated to a EOF file.
    """
    return closed_interval(
        eof_file.start_time,
        eof_file.stop_time,
    )
