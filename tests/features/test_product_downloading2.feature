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

@complex
Feature: Test download request v2
    Test download requests given requirements and detected files

    Background:
        Given the S1 products:
            | id        | product                                                             |
            | d1t1      | S1A_IW_GRDH_1SDV_20200108T044150_20200108T044215_030704_038506_C7F5 |
            | d1t2      | S1A_IW_GRDH_1SDV_20200108T044215_20200108T044240_030704_038506_D953 |
            | d2T044149 | S1A_IW_GRDH_1SDV_20200120T044149_20200120T044214_030879_038B2D_5671 |
            | d3T044149 | S1A_IW_GRDH_1SDV_20200201T044149_20200201T044214_031054_039149_ED12 |
            | d3T044214 | S1A_IW_GRDH_1SDV_20200201T044214_20200201T044239_031054_039149_CC58 |

        Given the S2 products:
            | id         | products                                            |
            | d1tx_sigma | s1a_33NWB_vh_DES_007_20200108txxxxxx_sigma          , s1a_33NWB_vv_DES_007_20200108txxxxxx_sigma |
            | d1tx_gamma | s1a_33NWB_vh_DES_007_20200108txxxxxx_GammaNaughtRTC |
            | d1t1_sigma | s1a_33NWB_vh_DES_007_20200108t044150_sigma          , s1a_33NWB_vv_DES_007_20200108t044150_sigma |
            | d1t1_gamma | s1a_33NWB_vh_DES_007_20200108t044150_GammaNaughtRTC |
            | d1t2_sigma | s1a_33NWB_vh_DES_007_20200108t044215_sigma          , s1a_33NWB_vv_DES_007_20200108t044215_sigma |
            | d1t2_gamma | s1a_33NWB_vh_DES_007_20200108t044215_GammaNaughtRTC |

        Given the gamma areas:
            | id                   | product                     |
            | 20200108txxxxxx_area | GAMMA_AREA_33NWB_vh_DES_007 |
            | 20200108t044150_area | GAMMA_AREA_33NWB_vh_DES_007 |
            | 20200108t044215_area | GAMMA_AREA_33NWB_vh_DES_007 |

    Scenario Outline: gamma rtc calibration
        Given The following S1 products are available for download: <remote_s1>
        And   The following S1 products are on disk: <local_s1>
        And   The following S2 products are on disk: <local_s2>
        And   We <scenario>
        When  Searching which S1 files to download II
        Then  The following S1 products will be downloaded: <dl_s1>

        ### Usual cases: 2 inputs => expect txxxxxx
        #   Cases where tile is intersected by TWO S1 products
        @complex_sigma_two_s1
        Examples:
            | remote_s1  | local_s1   | local_s2   | dl_s1      | scenario        |
            ## Calibration is σ°
            # Target is here => we don't care
            | d1t1, d1t2 |            | d1tx_sigma |            | sigma calibrate |
            | d1t1, d1t2 | d1t1, d1t2 | d1tx_sigma |            | sigma calibrate |
            | d1t1, d1t2 | d1t1       | d1tx_sigma |            | sigma calibrate |
            | d1t1, d1t2 |       d1t2 | d1tx_sigma |            | sigma calibrate |
            # Half target is here => We need to DL & build... what is not on disk
            # Half target is ignored b/c 2 inputs => expect txxxxxx
            | d1t1, d1t2 | d1t1       | d1t1_sigma |       d1t2 | sigma calibrate |
            | d1t1, d1t2 |       d1t2 | d1t2_sigma | d1t1       | sigma calibrate |
            | d1t1, d1t2 |            | d1t1_sigma | d1t1, d1t2 | sigma calibrate |
            | d1t1, d1t2 |            | d1t2_sigma | d1t1, d1t2 | sigma calibrate |
            # Target is not here => DL & build... what is not on disk
            | d1t1, d1t2 | d1t1, d1t2 |            |            | sigma calibrate |
            | d1t1, d1t2 | d1t1       |            |       d1t2 | sigma calibrate |
            | d1t1, d1t2 |       d1t2 |            | d1t1       | sigma calibrate |
            | d1t1, d1t2 |            |            | d1t1, d1t2 | sigma calibrate |
            | d1t1, d1t2 |            |            | d1t1, d1t2 | sigma calibrate |

        @complex_gamma
        Examples:
            | remote_s1  | local_s1   | local_s2   | dl_s1      | scenario        |
            ## Calibration is γ°RTC
            ## Request γ area files only
            ##### TODO: doc faire requête sur γ avec dates restrintes (car
            ##### sinon, DL bcp trop de choses)

        ### Cases w/ only one input => expect tdddddd
        @complex_sigma_one_s1
        Examples:
            | remote_s1  | local_s1   | local_s2   | dl_s1      | scenario        |
            # Situation "IMPOSSIBLE", and not tested: Target is not here
            # | d1t1       |            | d1tx_sigma | d1t1       | sigma calibrate |
            # | d1t1       | d1t1       | d1tx_sigma |            | sigma calibrate |
            # Target is here => we don't care
            | d1t1       |            | d1t1_sigma |           | sigma calibrate |
            | d1t1       | d1t1       | d1t1_sigma |           | sigma calibrate |
            | d1t1       |            | d1t2_sigma |           | sigma calibrate |
            | d1t1       | d1t1       | d1t2_sigma |           | sigma calibrate |


