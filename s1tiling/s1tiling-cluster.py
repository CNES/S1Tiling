#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =========================================================================
#   Copyright 2017-2021 (c) CESBIO. All rights reserved.
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

import sys
import os
import ConfigParser

if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " config.cfg")
    sys.exit(1)

CFG = sys.argv[1]
config = ConfigParser.SafeConfigParser()
config.read(CFG)


raw_directory = config.get('Paths', 'S1Images')

output_preprocess = config.get('Paths', 'Output')

tiles_list = [s.strip() for s in config.get('Processing', 'Tiles').split(",")]
try:
    os.remove(os.path.join("./jobs", "*.cfg"))
except OSError:
    pass

for itile, tile in enumerate(tiles_list):
    cfgFilename = os.path.join("./jobs", "job-" + str(itile + 1) + ".cfg")
    if not os.path.exists("./jobs"):
        os.mkdir("./jobs")
    config.set("Processing", "Tiles", tile)
    config.set("PEPS", "ROI_by_tiles", "ALL")
    with open(cfgFilename, 'wb') as configfile:
        config.write(configfile)
    print(itile, " ", tile, "->", cfgFilename)
with open("s1tiling.jobarray.template") as f:
    newtext = f.read().replace("#PBS -J", "#PBS -J 1:" + str(len(tiles_list)) + ":1")
with open("s1tiling.jobarray", "w") as f:
    f.write(newtext)
