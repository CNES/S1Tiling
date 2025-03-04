#!/usr/bin/env python
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

"""Helpers to time code execution"""

from functools import wraps
import inspect
import logging
from timeit import default_timer as timer
from typing import Literal, Optional


logger = logging.getLogger("s1tiling.utils.timer")

class ExecutionTimer:
    """Context manager to help measure execution times

    Example:
    with ExecutionTimer("the code", True) as t:
        Code_to_measure()
    """
    def __init__(self, text, do_measure, log_level: Optional[int]=None) -> None:
        self._text       = text
        self._do_measure = do_measure
        self._log_level  = log_level or logging.INFO

    def __enter__(self) -> "ExecutionTimer":
        self._start = timer()  # pylint: disable=attribute-defined-outside-init
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> Literal[False]:
        if self._do_measure:
            end = timer()
            logger.log(self._log_level, "%s took %fsec", self._text, end - self._start)
        return False


def timethis(fmt: str = "", log_level: Optional[int] = logging.DEBUG, do_measure: bool = True):
    """
    Decorator for timing function calls

    :param fmt:        Format string handled with :func:`string.format` function where possible
                       parameters are the names of the function formal parameters.
                       IOW, beware, this isn't as flexible as an f-string.
                       If empty (default case), copy the decorated line: ``function name(args, kwargs)``
    :param log_level:  To determine the log level used
    :param do_measure: Set to ``False`` to disable time measurement
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not fmt:
                m_args = ', '.join([f"{a!r}" for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()])
                message = f"{func.__name__}({m_args})"
            else:
                # inpect...arguments returns a dict with all parameter names and their values
                kw = inspect.signature(func).bind(*args, **kwargs).arguments
                message = fmt.format(**kw)

            with ExecutionTimer(message, log_level=log_level, do_measure=do_measure):
                return func(*args, **kwargs)
        return wrapper
    return decorator
