# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2024 (c) CNES. All rights reserved.
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

@dem @intersection
Feature: Intersecting DEM grids with polygons
    Checks DEM database can be searched whatever their format.

    @wgs84
    Scenario: Search S1 footprint in WGS84 DEM GPKG database
        Given The shipped WGS84 DEM database
        And   All S1 VV files are known
        When  S1 files footprints are searched in DEM database
        Then  The expected DEM files are found

    @lambert93
    Scenario: Search S1 footprint in Lambert93 DEM GPKG database
        Given The DEM database converted to Lambert93
        And   All S1 VV files are known
        When  S1 files footprints are searched in DEM database
        Then  The expected DEM files are found
