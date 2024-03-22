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
# Authors: Thierry KOLECK (CNES)
#          Luc HERMITTE (CS Group)
# =========================================================================

"""
Submodule that provides :class:`node_queue`
"""

import logging
from typing import Iterable, Iterator, List, Optional, Set, TypeVar

T = TypeVar("T")

logger = logging.getLogger('s1tiling.node_queue')


class node_queue(Iterable[T]):
    """
    Defines a mutable list where elements can be appended while iterating on the list.

    Elements removal is not supported. Nor are indexed access.

    This class is mainly meant for tracking node exploration in breadth-first search algorithms.

    Typical scenarios:

    # Simple iteration
    >>> nq = node_queue((1, 2, 3, 5))
    >>> for n in nq:
    ...     print(n)
    1
    2
    3
    5

    >>> len(nq)
    4

    >>> 2 in nq
    True
    >>> 4 in nq
    False

    # Iterating and mutating
    >>> for n in nq:
    ...     print(n)
    ...     if n < 10:
    ...         nq.add_if_new(n*2)
    1
    False
    2
    True
    3
    True
    5
    True
    4
    True
    6
    True
    10
    8
    True
    12
    16

    >>> len(nq)
    10
    """
    def __init__(self, elements: Optional[Iterable[T]] = None) -> None:
        """
        Initializes the queue from a set of elements.
        """
        self.__list : List[T] = list(elements or [])
        self.__set  : Set[T]  = set(self.__list)

    def __iter__(self) -> Iterator[T]:
        """ The queue is iterable """
        return iter(self.__list)

    def __contains__(self, e : T) -> bool:
        """ We can request whether an element is already present """
        return e in self.__set

    def add_if_new(self, e : T) -> bool:
        """
        Add an element only is it's not already present.

        :param e: Element to insert
        :return: Whether the element has bee added
        """
        if e not in self.__set:
            self.__list.append(e)
            self.__set.add(e)
            return True
        return False

    def __repr__(self) -> str:
        return f"node_queue({self.__list})"

    def __bool__(self) -> bool:
        return bool(self.__list)

    def __len__(self) -> int:
        return len(self.__list)
