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

"""This sub-module defines the Sentinel1 Orbit file class"""

from collections.abc import Sequence
from datetime import datetime
import glob
import logging
import os
from typing import Dict, Iterable, List, Tuple

from eof.client import Filename
from eof.download import SentinelOrbit

from ._conversions import ORBIT_CONVERTERS
from ..utils import lxml as xml


logger = logging.getLogger("s1tiling.orbit")


class SentinelOrbitFile(SentinelOrbit):
    """
    Extends :class:`eof.SentinelOrbit` with min-max absolute orbit info
    """

    def __init__(self, filename: Filename, **kwargs) -> None:
        """
        constructor
        """
        super().__init__(filename, **kwargs)
        assert (
            self.mission in ORBIT_CONVERTERS
        ), f"Unexpected mission ID {self.mission!r}. Only {ORBIT_CONVERTERS.keys()} are supported."

        self.first_abs_orbit, self.last_abs_orbit = extract_min_max_abs_orbit_numbers(filename)

        self.__orbit_converter = ORBIT_CONVERTERS[self.mission]
        self.first_rel_orbit = self.__orbit_converter.to_relative(self.first_abs_orbit)
        self.last_rel_orbit  = self.__orbit_converter.to_relative(self.last_abs_orbit)

    @property
    def nb_orbits_in_mission(self):
        """
        Returns the number of different relative orbit numbers known for the current misions.
        It's likely to always be 175...
        """
        return self.__orbit_converter.modulo

    def does_intersect(self, start: datetime, stop: datetime) -> bool:
        """
        Tells whether an EOF file intersect the given time range
        """
        return self.start_time < stop and self.stop_time > start

    def has_relative_orbit(self, relative_orbit: int, margin: int = 0) -> bool:
        """
        Tells whether a relative orbit is stored in a EOF file.

        A positive margin can be used to relax the constraint.
        A negative margin can be used to restrict the constraint.

        :param relative_orbit: Target relative orbit searched.
        :param margin:         Offset margin to tune the search
        :return: ``self.first_rel_orbit - margin <= relative_orbit <= self.last_rel_orbit + margin``
        """
        min_obt = self.first_rel_orbit - margin
        max_obt = self.last_rel_orbit + margin
        if min_obt < max_obt:
            does_contain = min_obt <= relative_orbit <= max_obt
        else:
            # min_obt is close to 175, and max_obt is close to 0
            does_contain = (min_obt <= relative_orbit <= self.nb_orbits_in_mission) or (1 <= relative_orbit <= max_obt)
        # logger.debug("¿ %s == %s ∈ [%s, %s] ('%s')", does_contain, relative_orbit, min_obt, max_obt, self.filename)
        return does_contain


# ===============[ "Internal" functions used to implement the public service
# This organisation eases the writing of unit tests
def extract_min_max_abs_orbit_numbers(filename: Filename) -> Tuple[int, int]:
    """
    Extract the first and the last absolute orbit numbers found in the EOF file.

    :param filename: Name of an XML .EOF file.
    """
    # ~ 80ms with lxml, 2.7s with xml
    root = xml.parse(filename)
    if not root:
        raise RuntimeError(f"Cannot open EOF file: {filename!r}")
    osv_list = xml.find(
            root,
            "Data_Block/List_of_OSVs",
            str(filename) + "<Earth_Explorer_File/>",
    )

    min_obt = xml.find_as(
            int,
            osv_list,
            "OSV/Absolute_Orbit",
            filename,
    )
    max_obt = xml.find_as(
            int,
            osv_list,
            "OSV[last()]/Absolute_Orbit",
            filename,
    )
    return int(min_obt), int(max_obt)


def glob_eof_files(dirname: Filename) -> List[SentinelOrbitFile]:
    """
    Glob precise orbit files in ``dirname``
    """
    eof_files = sorted([
        SentinelOrbitFile(f)
        for f in glob.glob(os.path.join(dirname, "S1*OPER_AUX_POEORB*.EOF"))
    ])
    return eof_files


def keep_one_eof_per_orbit(
        eof_files_per_orbit : Iterable[Dict[int, SentinelOrbitFile]],
        first_date          : datetime,
        last_date           : datetime,
) -> Dict[int, SentinelOrbitFile]:
    """
    Filters list of {orbit: eof_file} to keep only one product per orbit number.
    If there are several EOF file for a given orbit, we keep in priority the latest eof file that is
    within the time range.

    :return: A single dictionary of one EOF file per relative orbit
    """
    all_eof_per_obt : Dict[int, SentinelOrbitFile] = {}
    for eof_file in eof_files_per_orbit:
        assert len(eof_file) == 1
        obt, product = list(eof_file.items())[0]
        if obt in all_eof_per_obt:
            if not product.does_intersect(first_date, last_date):
                continue
        all_eof_per_obt[obt] = product
    return all_eof_per_obt


def filter_intersecting_eof_file_dict(
        eof_files_per_orbit : Iterable[Dict[int, SentinelOrbitFile]],
        first_date          : datetime,
        last_date           : datetime,
        missions            : Sequence[str] = (),
) -> List[Dict[int, SentinelOrbitFile]]:
    """
    Filter orbit files to keep those intersecting the time range.

    If ``mission`` is set, it's also used as a filtering parameter.
    """
    if missions:
        return [
                f
                for f in eof_files_per_orbit
                for relorb in f
                if f[relorb].does_intersect(first_date, last_date) and f[relorb].mission in missions
        ]
    else:
        return [
                f for f in eof_files_per_orbit
                for relorb in f
                if f[relorb].does_intersect(first_date, last_date)
        ]


def filter_intersecting_eof_file_list(
        eof_files  : Iterable[SentinelOrbitFile],
        first_date : datetime,
        last_date  : datetime,
        missions   : Sequence[str] = (),
) -> List[SentinelOrbitFile]:
    """
    Filter orbit files to keep those intersecting the time range.

    If ``mission`` is set, it's also used as a filtering parameter.
    """
    if missions:
        return [
                f
                for f in eof_files
                if f.does_intersect(first_date, last_date) and f.mission in missions
        ]
    else:
        return [
                f for f in eof_files
                if f.does_intersect(first_date, last_date)
        ]


def filter_eof_files_according_to_orbit_and_mission(
        eof_files      : Iterable[SentinelOrbitFile],
        relative_orbits: Sequence[int],
        margin         : int = 0,
        missions       : Sequence[str] = (),
) -> List[Dict[int, SentinelOrbitFile]]:
    """
    Filter orbit files to keep those containing the requested relative orbit numbers and missions.
    """
    if missions:  # First filter missions
        eof_files = (f for f in eof_files if f.mission in missions)

    # Then filter according to relative orbit
    return [
        {ro: f}
        for f in eof_files
        for ro in relative_orbits
        if f.has_relative_orbit(ro, margin)
    ]


def filter_eof_files_containing_orbit(
        eof_files     : Iterable[SentinelOrbitFile],
        relative_orbit: int,
        margin        : int = 0,
) -> List[SentinelOrbitFile]:
    """
    Filter orbit files to keep those containing the requested relative orbit number.
    """
    return [f for f in eof_files if f.has_relative_orbit(relative_orbit, margin)]


def orbit_range(eof_file: SentinelOrbitFile):
    """
    Generates all possible relativate orbit number between first and last relative numbers in orbit file.
    """
    return orbit_range_internal(
            eof_file.first_rel_orbit,
            eof_file.last_rel_orbit,
            eof_file.nb_orbits_in_mission,
    )


def orbit_range_internal(first: int, last: int, nb_orbits: int):
    """
    Generates all possible relativate orbit number between ``first`` and ``last``.
    >>> list(orbit_range_internal(1, 9, 175))
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> list(orbit_range_internal(3, 11, 175))
    [3, 4, 5, 6, 7, 8, 9, 10, 11]
    >>> list(orbit_range_internal(170, 174, 175))
    [170, 171, 172, 173, 174]
    >>> list(orbit_range_internal(170, 175, 175))
    [170, 171, 172, 173, 174, 175]
    >>> list(orbit_range_internal(170, 176, 175))  # actually a precondition violation
    Traceback (most recent call last):
        ...
    AssertionError
    >>> list(orbit_range_internal(174, 3, 175))
    [174, 175, 1, 2, 3]
    """
    assert 1 <= first <= nb_orbits
    assert 1 <= last <= nb_orbits
    if last < first:
        last += nb_orbits
    while first <= last:
        yield (first - 1) % nb_orbits + 1
        first += 1
