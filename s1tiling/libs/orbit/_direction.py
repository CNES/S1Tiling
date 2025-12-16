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

"""Defines the strong type :class:`Direction`"""

from __future__ import annotations
from enum import Enum
from typing import Dict, Literal

_k_ascending  : Literal['ascending']  = "ascending"
_k_descending : Literal['descending'] = "descending"


class Direction(Enum):
    """
    Strong (enum) type to modelize orbit directions
    """
    ASC = _k_ascending
    DES = _k_descending

    @staticmethod
    def __from_string(s: str) -> Direction:
        __map : Dict[str, Direction] = {
            _k_ascending      : Direction.ASC,
            _k_descending     : Direction.DES,
            Direction.ASC.name: Direction.ASC,
            Direction.DES.name: Direction.DES,
        }
        if s not in __map:
            raise ValueError(f"{s} is not a valid Direction")
        return __map[s]

    @property
    def short(self) -> Literal['ASC', 'DES']:
        """
        :return str: the short 3 letters version of the Direction

        >>> Direction.ASC.short
        'ASC'
        >>> Direction.DES.short
        'DES'
        """
        return self.name  # type: ignore[return-value]

    @property
    def long(self) -> Literal['ascending']|Literal['descending']:
        """
        :return str: the long version of the Direction

        >>> Direction.ASC.long
        'ascending'
        >>> Direction.DES.long
        'descending'
        """
        return self.value

    @classmethod
    def create(
        cls,
        value: Direction|str,
    ) -> Direction:
        """
        Factory method

        >>> Direction.create(Direction.ASC)
        <Direction.ASC: 'ascending'>
        >>> Direction.create(Direction.DES)
        <Direction.DES: 'descending'>
        >>> Direction.create("ASC")
        <Direction.ASC: 'ascending'>
        >>> Direction.create("DES")
        <Direction.DES: 'descending'>
        >>> Direction.create("ascending")
        <Direction.ASC: 'ascending'>
        >>> Direction.create("descending")
        <Direction.DES: 'descending'>
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return Direction.__from_string(value)
        raise ValueError(f"{value} is not a valid Direction")

    def __str__(self) -> str:
        return self.name
