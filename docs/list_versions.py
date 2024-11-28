#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2022 (c) CNES. All rights reserved.
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
#
# =========================================================================

import fnmatch
import os
import sys
from natsort import natsorted
from jinja2 import Environment, FileSystemLoader, select_autoescape

SCRIPTDIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

def list_dirs(directory, pattern=None):
    """
    Efficient listing of sub-directories in requested directory.

    This version shall be faster than glob to isolate directories only as it keeps in
    "memory" the kind of the entry without needing to stat() the entry again.

    Requires Python 3.5
    """
    if pattern:
        filt = lambda path: path.is_dir() and fnmatch.fnmatch(path.name, pattern)
    else:
        filt = lambda path: path.is_dir()

    with os.scandir(directory) as nodes:
        res = [entry for entry in nodes if filt(entry)]
    return res



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <root-directory>")
        sys.exit(1)
    root = sys.argv[1]
    dirs = list_dirs(root)
    print('Loading templates from %s', (SCRIPTDIR+"/_static/html",))
    env = Environment(
            loader=FileSystemLoader(SCRIPTDIR+"/_static/html"),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True, lstrip_blocks=True # don't add new line, keep indent
            )
    template = env.get_template('versions.html')
    versions=natsorted([d.name for d in dirs])
    # print(template.render(versions=versions))
    destfilename = root+'/versions.html'
    for d in dirs:
        destfilename = d.path + '/_static/html/versions.html'
        with open(destfilename, "w") as destfile:
            destfile.write(template.render(versions=versions))
        print("Version HTML menu saved into %s" % (destfilename,))
