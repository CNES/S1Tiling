#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2024 (c) CNES.
#   Copyright 2022-2024 (c) CS GROUP France.
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
# - Luc HERMITTE (CS Group)
#
# =========================================================================

""" Collection of generic algorithms """


from collections.abc import Callable, Iterable
from typing import List, Optional, Tuple, TypeVar

T = TypeVar('T')


def partition(
        predicate: Optional[Callable[[T], bool]],
        inputs:    Iterable[T],
) -> Tuple[List[T], List[T]]:
    """
    Partition a list according to a predicate.

    :param predicate: Boolean predicate used to sort out the elements. If ``None`` then ``bool`` is assumed.
    :param inputs:    Input list to partition
    :return: A tuple of the list of the True elements and the False elements

    >>> partition(None, [True, False, False, True, True])
    ([True, True, True], [False, False])

    >>> partition( lambda i : i % 2 == 0 , [1224, 42, 13, 31, 1426, 5])
    ([1224, 42, 1426], [13, 31, 5])
    """
    yes : List[T] = []
    no  : List[T] = []
    predicate = predicate or bool
    for e in inputs:
        (no, yes)[predicate(e)].append(e)
    return yes, no
