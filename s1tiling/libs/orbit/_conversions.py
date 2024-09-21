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

""" This sub-module defines how to compute relative orbit number from the absolute orbit number """


class OrbitConverter:
    """
    helper class used to convert an absolute orbit number into relative orbit number.
    """
    def __init__(self, offset, modulo):
        """
        Constructor
        """
        self.offset = offset
        self.modulo = modulo

    def to_relative(self, absolute_orbit_number: int) -> int:
        """
        Applies the offset and modulo to operate the conversion.
        """
        return (absolute_orbit_number - self.offset) % self.modulo + 1;

    def closest_absolute(self, first_absolute: int, tgt_relative: int) -> int:
        """
        Finds the closest absolute orbit number to ``first_absolute`` that would
        match to the ``target_relative`` number.

        As multiple absolute orbit numbers match a same relative orbit number, a choice must be
        made when operating the inverse conversion: we return the closest absolute orbit number
        to the requested target relative number.
        """
        # TODO: check the cases around rel_orb == 0 // 175
        first_relative = self.to_relative(first_absolute);
        res = first_absolute + (tgt_relative - first_relative);
        assert self.to_relative(res) == tgt_relative;
        return res


#: Modulo and offset tables for Sentinel-1A and Sentinel-1B
#: Eventually, this should be patched to support Sentinel-1C...
ORBIT_CONVERTERS = {
        "S1A" : OrbitConverter(73, 175),
        "S1B" : OrbitConverter(27, 175),
}
