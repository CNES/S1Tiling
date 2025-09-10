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

"""
Centralizes domain aspects related to Local Incidence Angle and (Ellipsoid) Incidence Angle aspects
"""


from enum   import Enum
from typing import List, Optional

from .utils         import partial_format
from .configuration import CreationOptionConfiguration, _extended_filename, pixel_type

# logger = logging.getLogger('s1tiling.incidence_angle')

class IA_map(Enum):
    """
    Enum used to distinguish all the possible incidence angle maps
    """
    tsk = 0
    cos = 1
    sin = 2
    tan = 3
    deg = 4


# =========================================================================
# Wrap IA related configuration options


# ----------------------------------------[ fname
__eia_fname_fmt_prefixes = {
    IA_map.cos: 'cos_IA',
    IA_map.sin: 'sin_IA',
    IA_map.tan: 'tan_IA',
    IA_map.deg: 'IA',
    IA_map.tsk: 'TaskIA',
}


def eia_map_fname_fmt(fname_fmt: str, ia_map: IA_map):
    """
    Returns the filename for a given (Ellipsoid) Incidence Angle map.

    :param fname_fmt: filename format string for this kind of product.
    :param ia_map:    Type of Incidence Angle map to produce
    :return: an updated filename format string

    This will replace the key `{IA_kind}` from the format string with the prefix for the actual
    (Ellipsoid) Incidence Angle map.
    """
    return partial_format(fname_fmt, IA_kind=__eia_fname_fmt_prefixes[ia_map])


# ----------------------------------------[ extended_filename
__extended_filenames = {
    IA_map.cos : ('filtered', ['COMPRESS=DEFLATE', 'PREDICTOR=3']),
    IA_map.sin : ('filtered', ['COMPRESS=DEFLATE', 'PREDICTOR=3']),
    IA_map.tan : ('filtered', ['COMPRESS=DEFLATE', 'PREDICTOR=3']),
    IA_map.deg : ('filtered', ['COMPRESS=DEFLATE']),
}


def extended_filename_ia(
    cfg              : CreationOptionConfiguration,
    ia_map           : IA_map,
    disable_streaming: bool,
) -> str:
    """
    Returns the (OTB) extended filename complement for the given Incidence Angle map.

    :param cfg:    Configuration object
    :param ia_map: Type of Incidence Angle map to produce
    :return: The extended filename complement for the given Incidence Angle map.
    """
    product: str
    default: List[str]
    product, default = __extended_filenames[ia_map]

    if disable_streaming:
        extra_ef = ['streaming:type=stripped', 'streaming:sizemode=nbsplits', 'streaming:sizevalue=1']
    else:
        extra_ef = []

    return _extended_filename(cfg, product, default, extra_ef)


# ----------------------------------------[ pixel_type
__pixel_type_fmt_keys = {
    IA_map.cos: '{IA}_cos',
    IA_map.sin: '{IA}_sin',
    IA_map.tan: '{IA}_tan',
    IA_map.deg: '{IA}_deg',
}

__pixel_type_defaults = {
    IA_map.deg: 'uint16',
}


def pixel_type_ia(cfg: CreationOptionConfiguration, ia_map: IA_map, incidence_angle_kind: str) -> str:
    """
    Returns the pixel type for the given Incidence Angle map.

    :param cfg:    Configuration object
    :param ia_map: Type of Incidence Angle map to produce
    :return: The pixel type for the given Incidence Angle map.
    """
    pixel_type_key    : str           = __pixel_type_fmt_keys[ia_map].format(IA=incidence_angle_kind.lower())
    pixel_type_default: Optional[str] = __pixel_type_defaults.get(ia_map, None)
    return pixel_type(cfg, pixel_type_key, pixel_type_default)
