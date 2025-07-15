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

Feature: Test creation options
    Test if default creation options can be properly overridden

    # --[ No default -----------------------------------
    Scenario: No default, override nothing
        Given Default pixel type is unset
        And   Default extended filename is unset
        And   Creation option is unset
        When  Analysing creation option
        Then  Pixel type is not specified
        And   Filename is not extended

    Scenario: No default, override pixel_type
        Given Default pixel type is unset
        And   Default extended filename is unset
        And   Creation option is float32
        When  Analysing creation option
        Then  Pixel type is float32
        And   Filename is not extended

    Scenario: No default, override creation options
        Given Default pixel type is unset
        And   Default extended filename is unset
        And   Creation option is COMPRESS=DEFLATE PREDICTOR=3
        When  Analysing creation option
        Then  Pixel type is not specified
        And   Filename is ?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=3

    Scenario: No default, override everything
        Given Default pixel type is unset
        And   Default extended filename is unset
        And   Creation option is float64 COMPRESS=DEFLATE, PREDICTOR=3
        When  Analysing creation option
        Then  Pixel type is double
        And   Filename is ?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=3

    # --[ pixel_type default ---------------------------
    Scenario: Default on pixel_type, override nothing
        Given Default pixel type is cfloat
        And   Default extended filename is unset
        And   Creation option is unset
        When  Analysing creation option
        Then  Pixel type is cfloat
        And   Filename is not extended

    Scenario: Default on pixel_type, override pixel_type
        Given Default pixel type is cfloat
        And   Default extended filename is unset
        And   Creation option is float32
        When  Analysing creation option
        Then  Pixel type is float32
        And   Filename is not extended

    Scenario: Default on pixel_type, override creation options
        Given Default pixel type is cfloat
        And   Default extended filename is unset
        And   Creation option is COMPRESS=DEFLATE PREDICTOR=3
        When  Analysing creation option
        Then  Pixel type is cfloat
        And   Filename is ?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=3

    Scenario: Default on pixel_type, override everything
        Given Default pixel type is cfloat
        And   Default extended filename is unset
        And   Creation option is float64 COMPRESS=DEFLATE, PREDICTOR=3
        When  Analysing creation option
        Then  Pixel type is double
        And   Filename is ?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=3

    # --[ Default on extended_filename -----------------------------------
    Scenario: Default on extended_filename, override nothing
        Given Default pixel type is unset
        And   Default extended filename is TILED=YES
        And   Creation option is unset
        When  Analysing creation option
        Then  Pixel type is not specified
        And   Filename is ?&gdal:co:TILED=YES

    Scenario: Default on extended_filename, override pixel_type
        Given Default pixel type is unset
        And   Default extended filename is TILED=YES
        And   Creation option is float32
        When  Analysing creation option
        Then  Pixel type is float32
        # Setting the pixel_type reset the other creation options
        # => TODO: extend the syntax to support extend/override
        # And   Filename is ?&gdal:co:TILED=YES
        And   Filename is not extended

    Scenario: Default on extended_filename, override creation options
        Given Default pixel type is unset
        And   Default extended filename is TILED=YES
        And   Creation option is COMPRESS=DEFLATE PREDICTOR=3
        When  Analysing creation option
        Then  Pixel type is not specified
        And   Filename is ?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=3

    Scenario: Default on extended_filename, override everything
        Given Default pixel type is unset
        And   Default extended filename is TILED=YES
        And   Creation option is float64 COMPRESS=DEFLATE, PREDICTOR=3
        When  Analysing creation option
        Then  Pixel type is double
        And   Filename is ?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=3

    # --[ Default on pixel_type and on extended_filename -----------------
    Scenario: Default on pixel_type and on extended_filename, override nothing
        Given Default pixel type is cfloat
        And   Default extended filename is TILED=YES
        And   Creation option is unset
        When  Analysing creation option
        Then  Pixel type is cfloat
        And   Filename is ?&gdal:co:TILED=YES

    Scenario: Default on pixel_type and on extended_filename, override pixel_type
        Given Default pixel type is cfloat
        And   Default extended filename is TILED=YES
        And   Creation option is float32
        When  Analysing creation option
        Then  Pixel type is float32
        # Setting the pixel_type reset the other creation options
        # => TODO: extend the syntax to support extend/override
        # And   Filename is ?&gdal:co:TILED=YES
        And   Filename is not extended

    Scenario: Default on pixel_type and on extended_filename, override creation options
        Given Default pixel type is cfloat
        And   Default extended filename is TILED=YES
        And   Creation option is COMPRESS=DEFLATE PREDICTOR=3
        When  Analysing creation option
        Then  Pixel type is cfloat
        And   Filename is ?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=3

    Scenario: Default on pixel_type and on extended_filename, override everything
        Given Default pixel type is cfloat
        And   Default extended filename is TILED=YES
        And   Creation option is float64 COMPRESS=DEFLATE, PREDICTOR=3
        When  Analysing creation option
        Then  Pixel type is double
        And   Filename is ?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=3

