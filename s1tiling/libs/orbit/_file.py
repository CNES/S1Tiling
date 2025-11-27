#!/usr/bin/env python3
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

"""This sub-module defines the Sentinel1 Orbit file class"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses     import asdict, dataclass
from datetime        import datetime
from enum import Enum
import glob
import logging
import os
import re
from typing import Dict, Iterable, List, Literal, Optional, Tuple, TypedDict, cast


from ._conversions import ORBIT_CONVERTERS
from ..utils import lxml as xml
from ..utils.path import AnyPath


#: List of all possible missions. Even the yet to be launched S1D is listed
ALL_MISSIONS = ("S1A", "S1B", "S1C", "S1D")

logger = logging.getLogger("s1tiling.orbit")


_k_POEORB  : Literal['POEORB']  = "POEORB"
_k_RESORB  : Literal['RESORB']  = "RESORB"


class OrbitType(Enum):
    """
    Strong (enum) type to modelize orbit types
    """
    POEORB = _k_POEORB
    RESORB = _k_RESORB  # Ignored for now

    def __str__(self) -> str:
        return self.name


def to_datetime(s: str) -> datetime:
    """
    Build a :class:`datatime` from a timestamp in ISO format
    """
    return datetime.strptime(s, '%Y%m%dT%H%M%S')


@dataclass(frozen=True, eq=False)
class SentinelOrbit:
    """
    Aggregates data related to a Sentinel-1 Orbit File.
    """
    mission:       str
    start_time:    datetime
    stop_time:     datetime
    orbit_type:    OrbitType
    creation_time: datetime

    @classmethod
    def create(  # pylint: disable=too-many-arguments
        cls,
        *,
        mission:       str,
        start_time:    str,
        stop_time:     str,
        orbit_type:    str,
        creation_time: str,
    ) -> SentinelOrbit:
        """
        Factory method

        >>> f='_EOF/S1A_OPER_AUX_POEORB_OPOD_20250302T070634_V20250209T225942_20250211T005942.EOF'
        >>> SentinelOrbit.create(**decode_filename(f))
        SentinelOrbit(mission='S1A', start_time=datetime.datetime(2025, 2, 9, 22, 59, 42), stop_time=datetime.datetime(2025, 2, 11, 0, 59, 42), orbit_type=<OrbitType.POEORB: 'POEORB'>, creation_time=datetime.datetime(2025, 3, 2, 7, 6, 34))
        """
        return SentinelOrbit(
            mission=mission,
            start_time=to_datetime(start_time),
            stop_time=to_datetime(stop_time),
            creation_time=to_datetime(creation_time),
            orbit_type=OrbitType(orbit_type),
        )

    def __eq__(self, rhs) -> bool:
        # compare as_tuple, but ignoring the creation_time
        # TODO: remove creation time from the field list
        return (
            self.mission, self.start_time, self.stop_time, self.orbit_type,
        ) == (
            rhs.mission, rhs.start_time, rhs.stop_time, rhs.orbit_type,
        )

    def __contains__(self, dt: datetime) -> bool:
        return self.start_time <= dt <= self.stop_time

    def __lt__(self, rhs):
        return (self.start_time, self.stop_time) < (rhs.start_time, rhs.stop_time)



RE_EOF = re.compile(
    r"(?P<mission>S1A|S1B|S1C|S1D)_OPER_AUX_"
    r"(?P<orbit_type>[\w_]{6})_OPOD_"
    r"(?P<creation_time>[T\d]{15})_"
    r"V(?P<start_time>[T\d]{15})_"
    r"(?P<stop_time>[T\d]{15})"
)


EOFFields = TypedDict(
    'EOFFields',
    {
        'mission':str, 'orbit_type': str, 'creation_time': str, 'start_time': str, 'stop_time': str
    }
)

# Unfortunately, it seems we cannot generate one from the other. But as least, we can check they are
# consistent
assert EOFFields.__annotations__.keys() == SentinelOrbit.__annotations__.keys()


def decode_filename(filename: AnyPath) -> EOFFields:
    """
    Extract components from Sentinel-1 orbit filenames.
    """
    match = RE_EOF.match(os.path.basename(filename))
    if not match:
        raise ValueError(f"Invalid EOF filename: {filename!r}")
    return cast(EOFFields, match.groupdict())


class SentinelOrbitFile(SentinelOrbit):
    """
    Extends :class:`eof.SentinelOrbit` with min-max absolute orbit info
    """

    def __init__(self, filename: AnyPath, **kwargs) -> None:
        """
        constructor
        """
        super().__init__(**asdict(SentinelOrbit.create(**decode_filename(filename))))
        assert (
            self.mission in ORBIT_CONVERTERS
        ), f"Unexpected mission ID {self.mission!r}. Only {ORBIT_CONVERTERS.keys()} are supported."

        self.first_abs_orbit, self.last_abs_orbit = extract_min_max_abs_orbit_numbers(filename)

        self.__filename : AnyPath = filename
        self.__orbit_converter = ORBIT_CONVERTERS[self.mission]
        self.first_rel_orbit = self.__orbit_converter.to_relative(self.first_abs_orbit)
        self.last_rel_orbit  = self.__orbit_converter.to_relative(self.last_abs_orbit)
        # logger.debug("ABS[%s, %s] -> REL[%s, %s] for %s", self.first_abs_orbit, self.last_abs_orbit, self.first_rel_orbit, self.last_rel_orbit, filename)
        assert 1 <= self.first_rel_orbit <= 175
        assert 1 <= self.last_rel_orbit <= 175

    @property
    def filename(self) -> AnyPath:
        """ Filename accessor """
        return self.__filename

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

    def __str__(self) -> str:
        return (
            f"{self.orbit_type} {self.__class__.__name__} from {self.start_time} to {self.stop_time} "
            f"[{self.first_rel_orbit:>03d} .. {self.last_rel_orbit:0>03d}]"
        )


# ===============[ "Internal" functions used to implement the public service
# This organisation eases the writing of unit tests
def extract_min_max_abs_orbit_numbers(filename: AnyPath) -> Tuple[int, int]:
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


def glob_eof_files(dirname: AnyPath) -> List[SentinelOrbitFile]:
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
    missions            : Iterable[str],
    obt2eof_map         : Optional[Dict[int, SentinelOrbitFile]] = None,
) -> Dict[int, SentinelOrbitFile]:
    """
    Filters list of {orbit: eof_file} to keep only one product per orbit number.

    :param eof_files_per_orbit: List of {orbit: eof_file} maps to filter
    :param first_date:          Start of the requested time range
    :param last_date:           End of the requested time range
    :param missions:            Requested missions -- cannot be empty
    :param obt2eof_map:         Previously map of unique {orbit: eof_file} to update.

    If there are several EOF files for a given orbit, we keep in priority:
    1. the latest eof file that is within the time range.
    2. the eof file associated to the mission/platform requested.

    :return: A single dictionary of one EOF file per relative orbit
    """
    assert missions, "At least one mission is expected"
    assert all(m in ALL_MISSIONS for m in missions), f"Invalid mission names: {missions=}"

    all_eof_per_obt : Dict[int, SentinelOrbitFile] = obt2eof_map or {}
    for eof_file in eof_files_per_orbit:
        assert len(eof_file) == 1
        obt, product = list(eof_file.items())[0]
        if obt in all_eof_per_obt:
            if not product.does_intersect(first_date, last_date):
                continue
            if product.mission not in missions:
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
        missions   : Iterable[str] = (),
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
    missions       : Iterable[str] = (),
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


def filter_uniq_eofs(  # pylint: disable=too-many-arguments
    eof_files         : List[SentinelOrbitFile],
    first_date        : datetime,
    last_date         : datetime,
    relative_orbits   : List[int],
    missions          : Iterable[str],
    known_obt2eof_map : Optional[Dict[int, SentinelOrbitFile]] = None,
) -> Tuple[Dict[int, SentinelOrbitFile], Iterable[int]]:
    """
    Main function used to filter a list of EOF files according to requested orbits, date and
    missions.

    :return: A single dictionary that maps a :class:`SentinelOrbitFile` to a relative orbit number.
    :return: A list of missing orbits
    """
    assert missions, "At least one mission is expected"
    assert all(m in ALL_MISSIONS for m in missions), f"Invalid mission names: {missions=}"

    eof_files_matching_orbits = filter_eof_files_according_to_orbit_and_mission(
        eof_files, relative_orbits, margin=-1, missions=()
    )  # Keep products from any missions

    expected_orbits = set(relative_orbits)
    if len(eof_files_matching_orbits) == 0:
        # No results
        return {}, expected_orbits

    # Good => we have results -- may be not enough, but it will be analysed later on
    obt2eof_map = keep_one_eof_per_orbit(eof_files_matching_orbits, first_date, last_date, missions, known_obt2eof_map)
    missing_orbits = expected_orbits - obt2eof_map.keys()

    return obt2eof_map, missing_orbits


def analyse_obt2eof_map_quality_according_to_request(
    obt2eof_map    : Dict[int, SentinelOrbitFile],
    first_date     : datetime,
    last_date      : datetime,
    missions       : Iterable[str],
) -> None:
    """
    Analyses the final map of {orbit: eof_file} for precise EOF files from an unrequested mission or
    outside the requested time range.

    Nothing is returned. Only warnings are issued.
    """
    assert missions, "At least one mission is expected"
    assert all(m in ALL_MISSIONS for m in missions), f"Invalid mission names: {missions=}"

    for obt, eof in obt2eof_map.items():
        if eof.mission not in missions:
            logger.info(
                "Note: Precise orbit file %s matching orbit %s doesn't match the requested missions %s",
                eof, obt, missions)
        if eof.does_intersect(first_date, last_date):
            logger.info(
                "Note: Precise orbit file %s matching orbit %s is not in the requested time range [%s .. %s]",
                eof, obt, first_date, last_date,
            )


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
    Generates all possible relate orbit number between ``first`` and ``last``.
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
