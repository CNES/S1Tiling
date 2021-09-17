#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

def test_tsort_dag_a_la_dsak():
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
        elements = list(ts.depth([3,5,7]))
