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

""" This sub-module defines the Sentinel1 Orbit file class """

from collections.abc import Sequence
from datetime import datetime
import glob
import logging
import os
from pathlib import Path
from typing import List, Tuple, Union

from eof.download import SentinelOrbit

from ._conversions import ORBIT_CONVERTERS
from ..utils import lxml as xml


logger = logging.getLogger('s1tiling.orbit')

class SentinelOrbitFile(SentinelOrbit):
    """
    Extends :class:`eof.SentinelOrbit` with min-max absolute orbit info
    """
    def __init__(self, filename: Union[str, Path], **kwargs) -> None:
        """
        constructor
        """
        super().__init__(filename, **kwargs)
        assert self.mission in ORBIT_CONVERTERS, (
                f"Unexpected mission ID {self.mission!r}. Only {ORBIT_CONVERTERS.keys()} are supported."
        )

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
        Tells whether a relative orbit is stored in a EOF file
        """
        min = self.first_rel_orbit - margin
        max = self.last_rel_orbit + margin
        if min < max:
            return min <= relative_orbit <= max
        else:
            # min is close to 175, and max is close to 0
            return (min <= relative_orbit <= self.nb_orbits_in_mission) or (1 <= relative_orbit <= max)


def extract_min_max_abs_orbit_numbers(filename: Union[str, Path]) -> Tuple[int, int]:
    # ~ 80ms with lxml, 2.7s with xml
    root = xml.parse(filename)
    if not root:
        raise RuntimeError(f"Cannot open EOF file: {filename!r}")
    osv_list = xml.find(
            root,
            "Data_Block/List_of_OSVs",
            str(filename) + "<Earth_Explorer_File/>",
    );

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


def glob_eof_files(dirname: Union[str, Path]) -> List[SentinelOrbitFile]:
    """
    Glob precise orbit files in ``dirname``
    """
    eof_files = sorted([
        SentinelOrbitFile(f)
        for f in glob.glob(os.path.join(dirname, "S1*OPER_AUX_POEORB*.EOF"))
    ])
    return eof_files


def filter_intersecting_eof_files(
        eof_files : List[SentinelOrbitFile],
        first_date: datetime,
        last_date : datetime,
        missions  : Sequence[str] = (),
) -> List[SentinelOrbitFile]:
    """
    Filter orbit files to keep those intersecting the time range.

    If ``mission`` is set, it's also used as a filtering parameter.
    """
    if missions:
        return [
                f for f in eof_files
                if f.does_intersect(first_date, last_date) and f.mission in missions
        ]
    else:
        return [
                f for f in eof_files
                if f.does_intersect(first_date, last_date)
        ]


def filter_eof_files_containing_orbit(
        eof_files     : List[SentinelOrbitFile],
        relative_orbit: int,
) -> List[SentinelOrbitFile]:
    """
    Filter orbit files to keep those containing the requested relative orbit number.
    """
    return [ f for f in eof_files if f.has_relative_orbit(relative_orbit)]


def orbit_range(eof_file: SentinelOrbitFile):
    """
    Generates all possible relativate orbit number between first and last relative numbers in orbit file.
    """
    return orbit_range_internal(
            eof_file.first_rel_orbit,
            eof_file.last_rel_orbit,
            eof_file.nb_orbits_in_mission,
    )


def orbit_range_internal(first: int , last: int, nb_orbits: int):
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
        yield (first-1) % nb_orbits + 1
        first += 1
