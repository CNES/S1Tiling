#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2025 (c) CNES. All rights reserved.
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
# Authors: Thierry KOLECK (CNES)
#          Luc HERMITTE (CS Group)
#
# =========================================================================

import pytest
from s1tiling.libs.Utils import TopologicalSorter

# Direct acyclic graph
dag = {
        3  : [8,10],
        5  : [11],
        7  : [11,8],
        8  : [9],
        11 : [2,9,10],
        }

dag_a_la_dask = {
        3  : ('whatever',  [8,10]),
        5  : ('dont care', [11]),
        7  : ('ignore',    [11,8]),
        8  : ('dummy',     [9]),
        11 : ('noise',     [2,9,10]),
        }

# Direct cyclic graph
dcg = dag.copy()
dcg[9] = [11]

def test_tsort_dag():
    ts = TopologicalSorter(dag)
    elements = list(ts.depth([3,5,7]))
    assert elements.index(7)  < elements.index(11)
    assert elements.index(7)  < elements.index(8)
    assert elements.index(5)  < elements.index(11)
    assert elements.index(3)  < elements.index(8)
    assert elements.index(3)  < elements.index(10)
    assert elements.index(11) < elements.index(2)
    assert elements.index(11) < elements.index(9)
    assert elements.index(11) < elements.index(10)
    assert elements.index(8)  < elements.index(9)

def test_tsort_all():
    ts = TopologicalSorter(dag)
    elements = list(ts.depth(dag.keys()))
    assert elements.index(7)  < elements.index(11)
    assert elements.index(7)  < elements.index(8)
    assert elements.index(5)  < elements.index(11)
    assert elements.index(3)  < elements.index(8)
    assert elements.index(3)  < elements.index(10)
    assert elements.index(11) < elements.index(2)
    assert elements.index(11) < elements.index(9)
    assert elements.index(11) < elements.index(10)
    assert elements.index(8)  < elements.index(9)

def test_tsort_dag_a_la_dask():
    ts = TopologicalSorter(dag_a_la_dask, lambda v : v[1])
    elements = list(ts.depth([3,5,7]))
    assert elements.index(7)  < elements.index(11)
    assert elements.index(7)  < elements.index(8)
    assert elements.index(5)  < elements.index(11)
    assert elements.index(3)  < elements.index(8)
    assert elements.index(3)  < elements.index(10)
    assert elements.index(11) < elements.index(2)
    assert elements.index(11) < elements.index(9)
    assert elements.index(11) < elements.index(10)
    assert elements.index(8)  < elements.index(9)

def test_tsort_dcg():
    with pytest.raises(ValueError):
        ts = TopologicalSorter(dcg)
        ts.depth([3,5,7])
