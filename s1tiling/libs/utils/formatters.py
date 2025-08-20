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

"""Collection of Format Helpers"""


from typing import Optional


class _PartialFormatHelper(dict):
    """
    Helper class that returns missing ``{key}`` as themselves
    """
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def partial_format(format_str: str, **kwargs) -> str:
    """
    Permits to apply partial formatting to format string.

    Example:
    --------
    >>> s = "{ab}_bla_{cd}"
    >>> partial_format(s, ab="tot")
    'tot_bla_{cd}'
    """
    return format_str.format_map(_PartialFormatHelper(**kwargs))


class _FormatOrGlobHelper(dict):
    """
    Helper class that returns missing ``{key}`` as '*'
    """
    def __missing__(self, _: str) -> str:
        return "*"


def glob_format(format_str: str, **kwargs) -> str:
    """
    Permits to apply partial formatting to format string, and set unknown keys to the glob-anything
    pattern: '*'

    Example:
    --------
    >>> s = "{ab}_bla_{cd}"
    >>> glob_format(s, ab="tot")
    'tot_bla_*'
    """
    return format_str.format_map(_FormatOrGlobHelper(**kwargs))


class ResilientFormater:
    """
    Very similar to :class:`_PartialFormatHelper` or :class:`_FormatOrGlobHelper`, except we can
    choose the replacement text.

    >>> s = "{ab}_bla_{cd}"
    >>> ResilientFormater().format(s, ab="tot")
    'tot_bla_{cd}'

    >>> ResilientFormater("*").format(s, ab="tot")
    'tot_bla_*'

    >>> ResilientFormater(".*").format(s, ab="tot")
    'tot_bla_.*'
    """
    def __init__(self, default: Optional[str] = None):
        """
        constructor
        """
        self.__default = default

    @property
    def default(self):
        """
        Getter to default replacement
        """
        return self.__default

    def format(self, format_str: str, **kwargs) -> str:
        """
        Overrides the format() function
        """
        outer = self
        class _Formatter(dict):
            def __missing__(self, key: str) -> str:
                if outer.default is None:
                    return "{" + key + "}"
                return outer.default

        return format_str.format_map(_Formatter(**kwargs))


