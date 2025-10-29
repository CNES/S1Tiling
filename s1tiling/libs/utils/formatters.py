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

from __future__ import annotations

# import logging
from string import Formatter
from typing import Optional
from typing_extensions import deprecated


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


@deprecated("Since v1.2")
class ResilientFormatter0:
    """
    Very similar to :class:`_PartialFormatHelper` or :class:`_FormatOrGlobHelper`, except we can
    choose the replacement text.

    .. deprecated:: 1.2

    >>> s = "{ab}_bla_{cd}"
    >>> ResilientFormatter().format(s, ab="tot")
    'tot_bla_{cd}'

    >>> ResilientFormatter("*").format(s, ab="tot")
    'tot_bla_*'

    >>> ResilientFormatter(".*").format(s, ab="tot")
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


class ExtendedFormatter(Formatter):
    """
    Extends :class:`string.Formatter` capabilities to interpret new conversion fields.

    - ``!l`` -> change into lowercase
    - ``!u`` -> change into uppercase
    - ``!c`` -> change the first letter to uppercase, and the other letters to lowercase

    .. todo::
        We may want to support on-the-fly regex substitution as in bash ${var/pat/repl}
        It could be done with: `!s/pattern/replacement`

    # Change a key to uppercase, another to lowercase
    >>> ExtendedFormatter().format("{ab!u}_bla_{cd!l}_bli", ab="tOt", cd='BaR')
    'TOT_bla_bar_bli'

    # Missing key raises a KeyError exception
    >>> ExtendedFormatter().format("{ab!u}_bla_{cd!l}_bli_{ef}", ab="tOt", cd='BaR')
    Traceback (most recent call last):
        ...
    KeyError: 'ef'

    # Use format specifiers to reduce an expansion, and to pad with selected characters, right or left
    >>> ExtendedFormatter().format("{ab:_<5}_bla_{cd:.2}_bli", ab="tOt", cd='BaR')
    'tOt___bla_Ba_bli'
    >>> ExtendedFormatter().format("{ab:X>5}_bla_{cd:.2}_bli", ab="tOt", cd='BaR')
    'XXtOt_bla_Ba_bli'

    # Mix in format specifier and case changing through conversion field
    >>> ExtendedFormatter().format("{ab!u:_>5}_bla_{cd!l:.2}_bli", ab="tOt", cd='BaR')
    '__TOT_bla_ba_bli'

    """
    def convert_field(self, value, conversion):
        """
        Override :meth:`string.Formatter.convert_field` to support ``!l``,``!u``, ``!c``.
        """
        # logging.debug(f"convert {value=} with: {conversion=}")
        if conversion == 'u':
            return value.upper()
        elif conversion == 'l':
            return value.lower()
        elif conversion == 'c':
            return value.capitalize()
        return super().convert_field(value, conversion)


class ResilientFormatter2(ExtendedFormatter):
    """
    Very similar to :class:`_PartialFormatHelper` or :class:`_FormatOrGlobHelper`, except we can
    choose the replacement text. Unlike other text formatters, it never throws :class:`KeyError`
    exceptions.

    It also support the new conversion fields.

    # Basic test: replace one key, leave the other with "{keyname}"
    >>> s = "{ab}_bla_{cd}"
    >>> ResilientFormatter2().format(s, ab="tot")
    'tot_bla_{cd}'

    # Basic test: replace one key, leave the other with the default glob pattern "*"
    >>> ResilientFormatter2("*").format(s, ab="tot")
    'tot_bla_*'

    # Basic test: replace one key, leave the other with the default regex pattern ".*"
    >>> ResilientFormatter2(".*").format(s, ab="tot")
    'tot_bla_.*'

    # Change a key to uppercase, another to lowercase, with default {unknownkey}
    >>> ResilientFormatter2().format("{ab!u}_bla_{cd!l}_bli_{ef}", ab="tOt", cd='BaR')
    'TOT_bla_bar_bli_{ef}'

    # Change a key to uppercase, another to lowercase, with default regex pattern ".*"
    >>> ResilientFormatter2(".*").format("{ab!u}_bla_{cd!l}_bli_{ef}", ab="tOt", cd='BaR')
    'TOT_bla_bar_bli_.*'

    # Use format specifiers to reduce an expansion, and to pad with selected characters, right or left
    >>> ResilientFormatter2(".*").format("{ab:_<5}_bla_{cd:.2}_bli_{ef}", ab="tOt", cd='BaR')
    'tOt___bla_Ba_bli_.*'
    >>> ResilientFormatter2(".*").format("{ab:X>5}_bla_{cd:.2}_bli_{ef}", ab="tOt", cd='BaR')
    'XXtOt_bla_Ba_bli_.*'

    # Mix in format specifier and case changing through conversion field
    >>> ResilientFormatter2(".*").format("{ab!u:_>5}_bla_{cd!l:.2}_bli_{ef}", ab="tOt", cd='BaR')
    '__TOT_bla_ba_bli_.*'

    # This time there is no value associated to the key, and the default/{key}  text shall not be altered
    >>> ResilientFormatter2().format("{ab!u}_bla_{cd!l:.2}_bli_{longkey!l:.3}", ab="tOt", cd='BaR')
    'TOT_bla_ba_bli_{longkey}'
    """

    class Missing:
        """
        Helper class to trick the formatter into passing around "values" that don't get reformatted
        through :meth:`string.Formatter.format_field` or :meth:`string.Formatter.convert_field`.
        """

        def __init__(self, repl):
            self._repl = repl

        def __format__(self, format_spec: str, /) -> str:
            return self._repl

        def lower(self) -> ResilientFormatter2.Missing:
            """
            Disguises :class:`Missing` instances into :class:`str` instances that support
            :meth:`str.lower` method.

            In out case, the implementation is a no-op.
            """
            return self

        def upper(self) -> ResilientFormatter2.Missing:
            """
            Disguises :class:`Missing` instances into :class:`str` instances that support
            :meth:`str.upper` method.

            In out case, the implementation is a no-op.
            """
            return self

        def capitalize(self) -> ResilientFormatter2.Missing:
            """
            Disguises :class:`Missing` instances into :class:`str` instances that support
            :meth:`str.capitalize` method.

            In out case, the implementation is a no-op.
            """
            return self

    def __init__(self, default: Optional[str] = None):
        """
        constructor
        """
        self.__default = default

    def default(self, key):
        """
        Getter to default replacement or "{key}" string if `default` is unset.
        """
        return self.__default if self.__default is not None else f"{{{key}}}"

    def get_value(self, key, args, kwargs):
        """
        Replace :meth:`string.Formatter.get_value` to return an instance of :class:`Missing` when a
        `key` is unknown.

        In that case the :class:`Missing` instance will have a (default) string value that stays
        unaffected by `conversion fields` and by `format specifiers`.
        """
        # logging.debug(f"convert {key=} with: {args=} -- {kwargs=}")
        if isinstance(key, int):
            return args[key]
        else:
            return kwargs.get(key, self.Missing(self.default(key)))

ResilientFormatter = ResilientFormatter2
