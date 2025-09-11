#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2025 (c) CNES.
#   Copyright 2022-2024 (c) CS GROUP France.
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

"""Defines CLI related helpers"""


import logging
from typing import NoReturn
import sys
import click

from s1tiling.libs.exits import translate_exception_into_exit_code

logger = logging.getLogger('s1tiling.cli')


def cli_execute(processing, *args, **kwargs) -> int:
    """
    Factorize code common to all S1Tiling CLI entry points (exception translation into exit
    codes...)
    """
    trace_errors = kwargs.pop('trace_errors', False)
    try:
        situation = processing(*args, **kwargs)
        # logger.debug('nominal exit: %s', situation.code)
        return situation.code
    except BaseException as e:  # pylint: disable=broad-except
        # Logger object won't always exist at this time (like in configuration
        # errors) hence we may use click report mechanism instead.
        if logger:
            logger.critical(e, exc_info=trace_errors)
            # logger.exception(e) # <=> exc_info=True
        else:
            click.echo(f"Error: {e}", err=True)
        return translate_exception_into_exit_code(e)


def cli_main(processing, *args, **kwargs) -> NoReturn:
    """
    Factorize code common to all S1Tiling CLI entry points (exception translation into exit
    codes...) plus program exit
    """
    sys.exit(
        cli_execute(processing, *args, **kwargs)
    )
