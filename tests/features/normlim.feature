# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2023 (c) CNES. All rights reserved.
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

Feature: Norlim
    Existing S1 images shall be analysed to deduce normlim related tasks to
    execute.

    Scenario: Generate LIA tasks for a single S1 image
        Given A pipeline that computes LIA
        And   a single S1 image

        When  dependencies are analysed
        And   tasks are generated

        Then  a single LIA image is required
        And   LIA images depend on XYZ images
        And   XYZ images depend on DEM, DEMPROJ and BASE images
        And   DEMPROJ images depend on DEM and BASE images
        And   DEM images depend on BASE images

        And   LIA task(s) is(/are) registered
        And   XYZ task(s) is(/are) registered
        And   DEMPROJ task(s) is(/are) registered
        And   DEM task(s) is(/are) registered

    Scenario: Generate LIA tasks for a pair of VV+VH S1 images
        # Check a reduction of type 'any()': any one between vh or vv is good:
        # just keep one
        Given A pipeline that computes LIA
        And   a pair of VV + VH S1 images

        When  dependencies are analysed
        And   tasks are generated

        Then  a single LIA image is required
        And   LIA images depend on XYZ images
        And   XYZ images depend on DEM, DEMPROJ and BASE images
        And   DEMPROJ images depend on DEM and BASE images
        And   DEM images depend on BASE images

        And   LIA task(s) is(/are) registered
        And   XYZ task(s) is(/are) registered
        And   DEMPROJ task(s) is(/are) registered
        And   DEM task(s) is(/are) registered

    Scenario: Generate LIA tasks for a series of S1 VV images
        # Check a single LIA task will be registered even w/ multiple input
        # images of different acquisition date. => Keep only one LIA
        Given A pipeline that fully computes in LIA S2 geometry
        And   a series of S1 VV images

        When  dependencies are analysed
        And   tasks are generated

        Then  a single S2 LIA image is required
        # TODO fix the dependencies
        And   final LIA image has been selected from one concat LIA
        And   concat LIA depends on 2 ortho LIA images
        And   2 ortho LIA images depend on two LIA images
        And   LIA images depend on XYZ images
        And   XYZ images depend on DEM, DEMPROJ and BASE images
        And   DEMPROJ images depend on DEM and BASE images
        And   DEM images depend on BASE images

        And   a select LIA task is registered
        And   a concat LIA task is registered
        And   ortho LIA task(s) is(/are) registered
        And   LIA task(s) is(/are) registered
        And   XYZ task(s) is(/are) registered
        And   DEMPROJ task(s) is(/are) registered
        And   DEM task(s) is(/are) registered

    Scenario: Full production of orthorectified of normlim calibrated S2 images
        Given A pipeline that normlim calibrates and orthorectifies
        And   that concatenates
        And   A pipeline that fully computes in LIA S2 geometry
        And   that applies LIA

        And   two S1 images

        When  dependencies are analysed
        And   tasks are generated

        # We have everything we usually have + the final bandmath
        # Then  a txxxxxx S2 file is required, and no mask is required
        Then  a txxxxxx S2 file is expected but not required
        And   it depends on 2 ortho files (and two S1 inputs), and no mask on a concatenated product
        And   a concatenation task is registered and produces txxxxxxx S2 file and no mask
        And   two orthorectification tasks are registered

        Then  no S2 LIA image is required
        And   final LIA image has been selected from one concat LIA
        And   concat LIA depends on 2 ortho LIA images
        And   2 ortho LIA images depend on two LIA images
        And   LIA images depend on XYZ images
        And   XYZ images depend on DEM, DEMPROJ and BASE images
        And   DEMPROJ images depend on DEM and BASE images
        And   DEM images depend on BASE images

        Then  a txxxxxx normlim S2 file is required

        And   a select LIA task is registered
        And   a concat LIA task is registered
        And   ortho LIA task(s) is(/are) registered
        And   LIA task(s) is(/are) registered
        And   XYZ task(s) is(/are) registered
        And   DEMPROJ task(s) is(/are) registered
        And   DEM task(s) is(/are) registered

