#!/usr/bin/env python
# -*- coding: utf-8 -*-
# https://gist.github.com/jcrist/dc5b7cedfddff123f2177e5238e566e5
# Authors: Jim Crist-Harif, and Silvio Rodrigues, 2018
# Licence: BSD-3 clause

"""
Simple visualization of dask graph pipelines.
"""

import os
import graphviz

from dask.utils import key_split
from dask.dot import _get_display_cls
from dask.core import get_dependencies

__author__ = "Jim Crist-Harif, and Silvio Rodrigues"
__copyright__ = "Copyright, 2018"
__credits__ = ["Jim Crist-Harif", "Silvio Rodrigues"]
__license__ = "BSD-3 clause"
__version__ = "2"
__status__ = "gist"
__contact__ = "https://gist.github.com/jcrist/dc5b7cedfddff123f2177e5238e566e5"


class SimpleComputationGraph:
    def __init__(self):
        return

    @staticmethod
    def _node_key(s):
        if isinstance(s, tuple):
            return s[0]
        return str(s)

    def simple_graph(self,
                     x,
                     filename='simple_computation_graph',
                     format=None):

        if hasattr(x, 'dask'):
            dsk = x.__dask_optimize__(x.dask, x.__dask_keys__())
        else:
            dsk = x

        deps = {k: get_dependencies(dsk, k) for k in dsk}

        g = graphviz.Digraph(graph_attr={'rankdir': 'LR'})

        nodes = set()
        edges = set()
        for k in dsk:
            key = self._node_key(k)
            if key not in nodes:
                g.node(key, label=key_split(k), shape='rectangle')
                nodes.add(key)
            for dep in deps[k]:
                dep_key = self._node_key(dep)
                if dep_key not in nodes:
                    g.node(dep_key, label=key_split(dep), shape='rectangle')
                    nodes.add(dep_key)
                # Avoid circular references
                if dep_key != key and (dep_key, key) not in edges:
                    g.edge(dep_key, key)
                    edges.add((dep_key, key))

        fmts = ['.png', '.pdf', '.dot', '.svg', '.jpeg', '.jpg']
        if format is None and any(filename.lower().endswith(fmt) for fmt in fmts):
            filename, format = os.path.splitext(filename)
            format = format[1:].lower()

        if format is None:
            format = 'png'

        data = g.pipe(format=format)
        if not data:
            raise RuntimeError("Graphviz failed to properly produce an image. "
                               "This probably means your installation of graphviz "
                               "is missing png support. See: "
                               "https://github.com/ContinuumIO/anaconda-issues/"
                               "issues/485 for more information.")

        display_cls = _get_display_cls(format)

        if not filename:
            return display_cls(data=data)

        full_filename = '.'.join([filename, format])
        with open(full_filename, 'wb') as f:
            f.write(data)

        return display_cls(filename=full_filename)
