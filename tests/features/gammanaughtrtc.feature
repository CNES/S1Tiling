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
#          Fabien CONTIVAL (CS Group)
#
# =========================================================================

Feature: gamma_naught_rtc
    Existing S1 images shall be analysed to deduce gamma_naught_rtc related tasks to
    execute.

    # v1.0 Workflows
    Scenario: Generate GAMMA_AREA tasks for a single S1 image w/ v1.0 workflow
        Given A pipeline that computes GAMMA_AREA in S1
        And   a single S1 image

        When  dependencies are analysed
        And   tasks are generated

        Then  a single GAMMA_AREA image is required in S1
        And   GAMMA_AREA images depend on DEM, RESAMPLE_DEM and DEMPROJ and BASE images (S1)
        And   DEMPROJ images depend on RESAMPLED_DEM and BASE images
        And   RESAMPLED_DEM images depend on DEM images
        And   DEM images depend on BASE images

        And   GAMMA_AREA task(s) is(/are) registered (S1)
        And   RESAMPLEDDEMPROJ task(s) is(/are) registered
        And   RESAMPLED_DEM task(s) is(/are) registered
        And   DEM task(s) is(/are) registered

    Scenario: Generate GAMMA_AREA tasks for a pair of VV+VH S1 images
        # Check a reduction of type 'any()': any one between vh or vv is good:
        # just keep one
        Given A pipeline that computes GAMMA_AREA in S1
        And   a pair of VV + VH S1 images

        When  dependencies are analysed
        And   tasks are generated

        Then  a single GAMMA_AREA image is required in S1
        And   GAMMA_AREA images depend on DEM, RESAMPLED_DEM, DEMPROJ and BASE images (S1)
        And   DEMPROJ images depend on RESAMPLED_DEM and BASE images
        And   RESAMPLED_DEM images depend on DEM images
        And   DEM images depend on BASE images

        And   GAMMA_AREA task(s) is(/are) registered (S1)
        And   RESAMPLEDDEMPROJ task(s) is(/are) registered
        And   RESAMPLED_DEM task(s) is(/are) registered
        And   DEM task(s) is(/are) registered

    Scenario: Generate GAMMA_AREA tasks for a series of S1 VV images
        # Check a single GAMMA_AREA task will be registered even w/ multiple input
        # images of different acquisition date. => Keep only one GAMMA_AREA
        Given A pipeline that fully computes in GAMMA_AREA S2 geometry
        And   a series of S1 VV images

        When  dependencies are analysed
        And   tasks are generated

        Then  a single S2 GAMMA_AREA image is required
        # TODO fix the dependencies
        And   final GAMMA_AREA image has been selected from one concat GAMMA_AREA
        And   concat GAMMA_AREA depends on 2 ortho GAMMA_AREA images
        And   2 ortho GAMMA_AREA images depend on two GAMMA_AREA images
        And   GAMMA_AREA images depend on DEM, RESAMPLED_DEM, DEMPROJ and BASE images (S1)
        And   DEMPROJ images depend on RESAMPLED_DEM and BASE images
        And   RESAMPLED_DEM images depend on DEM images
        And   DEM images depend on BASE images

        And   a select GAMMA_AREA task is registered
        And   a concat GAMMA_AREA task is registered
        And   ortho GAMMA_AREA task(s) is(/are) registered
        And   GAMMA_AREA task(s) is(/are) registered (S1)
        And   RESAMPLEDDEMPROJ task(s) is(/are) registered
        And   RESAMPLED_DEM task(s) is(/are) registered
        And   DEM task(s) is(/are) registered

    Scenario: Full production of orthorectified of gamma_naught_rtc calibrated S2 images
        Given A pipeline that gamma_naught_rtc calibrates and orthorectifies
        And   that concatenates
        And   A pipeline that fully computes in GAMMA_AREA S2 geometry
        And   that applies GAMMA_AREA

        And   two S1 images

        When  dependencies are analysed
        And   tasks are generated

        # We have everything we usually have + the final bandmath
        # Then  a txxxxxx S2 file is required, and no mask is required
        Then  a txxxxxx S2 file is expected but not required
        And   it depends on 2 ortho files (and two S1 inputs), and no mask on a concatenated product
        And   a concatenation task is registered and produces txxxxxxx S2 file and no mask
        And   two orthorectification tasks are registered

        Then  no S2 GAMMA_AREA image is required
        And   final GAMMA_AREA image has been selected from one concat GAMMA_AREA
        And   concat GAMMA_AREA depends on 2 ortho GAMMA_AREA images
        And   2 ortho GAMMA_AREA images depend on two GAMMA_AREA images
        And   GAMMA_AREA images depend on DEM, RESAMPLED_DEM, DEMPROJ and BASE images (S1)
        And   RESAMPLEDDEMPROJ images depend on RESAMPLED_DEM and BASE images
        And   RESAMPLED_DEM images depend on DEM images
        And   DEM images depend on BASE images

        Then  a txxxxxx gamma_naught_rtc S2 file is required

        And   a select GAMMA_AREA task is registered
        And   a concat GAMMA_AREA task is registered
        And   ortho GAMMA_AREA task(s) is(/are) registered
        And   GAMMA_AREA task(s) is(/are) registered (S1)
        And   RESAMPLEDDEMPROJ task(s) is(/are) registered
        And   RESAMPLED_DEM task(s) is(/are) registered
        And   DEM task(s) is(/are) registered

