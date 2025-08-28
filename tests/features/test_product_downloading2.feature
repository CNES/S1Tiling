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
            | id   | product                                                             |
            | d1t1 | S1A_IW_GRDH_1SDV_20200108T044150_20200108T044215_030704_038506_C7F5 |
            | d1t2 | S1A_IW_GRDH_1SDV_20200108T044215_20200108T044240_030704_038506_D953 |

            | d2t1 | S1A_IW_GRDH_1SDV_20200120T044149_20200120T044214_030879_038B2D_5671 |
            | d2t2 | S1A_IW_GRDH_1SDV_20200120T044214_20200120T044239_030879_038B2D_FDB0 |

            | d3t1 | S1A_IW_GRDH_1SDV_20200201T044149_20200201T044214_031054_039149_ED12 |
            | d3t2 | S1A_IW_GRDH_1SDV_20200201T044214_20200201T044239_031054_039149_CC58 |

        Given the S2 products:
            | id         | products                                            |
            | d1tx_sigma | s1a_33NWB_vh_DES_007_20200108txxxxxx_sigma         , s1a_33NWB_vv_DES_007_20200108txxxxxx_sigma |
            | d1t1_sigma | s1a_33NWB_vh_DES_007_20200108t044150_sigma         , s1a_33NWB_vv_DES_007_20200108t044150_sigma |
            | d1t2_sigma | s1a_33NWB_vh_DES_007_20200108t044215_sigma         , s1a_33NWB_vv_DES_007_20200108t044215_sigma |
            | d1tx_gamma | s1a_33NWB_vh_DES_007_20200108txxxxxx_GammaNaughtRTC, s1a_33NWB_vv_DES_007_20200108txxxxxx_GammaNaughtRTC |
            | d1t1_gamma | s1a_33NWB_vh_DES_007_20200108t044150_GammaNaughtRTC, s1a_33NWB_vv_DES_007_20200108t044150_GammaNaughtRTC |
            | d1t2_gamma | s1a_33NWB_vh_DES_007_20200108t044215_GammaNaughtRTC, s1a_33NWB_vv_DES_007_20200108t044215_GammaNaughtRTC |

            | d2tx_sigma | s1a_33NWB_vh_DES_007_20200120txxxxxx_sigma         , s1a_33NWB_vv_DES_007_20200120txxxxxx_sigma |
            | d2t1_sigma | s1a_33NWB_vh_DES_007_20200120t044149_sigma         , s1a_33NWB_vv_DES_007_20200120t044149_sigma |
            | d2t2_sigma | s1a_33NWB_vh_DES_007_20200120t044214_sigma         , s1a_33NWB_vv_DES_007_20200120t044214_sigma |
            # TODO: use d1t1_gamma and d1t2_gamma

        Given the gamma areas:
            | id         | product                      |
            | gamma_area | GAMMA_AREA_s1a_33NWB_DES_007 |

    Scenario Outline: gamma rtc calibration
        Given The following S1 products are available for download: <remote_s1>
        And   The following S1 products are on disk: <local_s1>
        And   The following S2 products are on disk: <local_s2>
        And   Requested time range is deduced from known remote S1 products
        And   We <scenario>
        When  Searching which S1 files to download II
        Then  The following S1 products will be downloaded: <dl_s1>
        # TODO: Then the following are used in FirstSteps?
        # TODO: Propagate as KO-FirstStep, products that can not be downloaded

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

        ## Request γ area files only
        #  Note: there is no way to know whether a γ area map has been made
        #  from one or two input S1
        @complex_gamma_area_two_s1
        Examples:
            # Target is here => DL nothing
            | remote_s1  | local_s1   | local_s2    | dl_s1      | scenario           |
            | d1t1, d1t2 |            | gamma_area  |            | compute gamma area |
            | d1t1, d1t2 | d1t1, d1t2 | gamma_area  |            | compute gamma area |
            | d1t1, d1t2 | d1t1       | gamma_area  |            | compute gamma area |
            | d1t1, d1t2 |       d1t2 | gamma_area  |            | compute gamma area |
            # Target is not here => DL what is missing
            | d1t1, d1t2 |            |            | d1t1, d1t2 | compute gamma area |
            | d1t1, d1t2 | d1t1, d1t2 |            |            | compute gamma area |
            | d1t1, d1t2 | d1t1       |            |       d1t2 | compute gamma area |
            | d1t1, d1t2 |       d1t2 |            | d1t1       | compute gamma area |

        ## Calibration is γ°RTC
        @complex_gamma_calibrated_two_s1
        Examples:
            | remote_s1  | local_s1   | local_s2               | dl_s1      | scenario                  |
            ## Calibration is γ° RTC
            #  Note: γ-area maps are always "required-product"
            # Both targets are here => we don't care
            | d1t1, d1t2 |            | gamma_area, d1tx_gamma |            | gamma_naught_rtc calibrate |
            | d1t1, d1t2 | d1t1, d1t2 | gamma_area, d1tx_gamma |            | gamma_naught_rtc calibrate |
            | d1t1, d1t2 | d1t1       | gamma_area, d1tx_gamma |            | gamma_naught_rtc calibrate |
            | d1t1, d1t2 |       d1t2 | gamma_area, d1tx_gamma |            | gamma_naught_rtc calibrate |

            # Only γ°RTC calibrated target is here => need to produce γ-area maps
            | d1t1, d1t2 |            |            d1tx_gamma  | d1t1, d1t2 | gamma_naught_rtc calibrate |
            | d1t1, d1t2 | d1t1, d1t2 |            d1tx_gamma  |            | gamma_naught_rtc calibrate |
            | d1t1, d1t2 | d1t1       |            d1tx_gamma  |       d1t2 | gamma_naught_rtc calibrate |
            | d1t1, d1t2 |       d1t2 |            d1tx_gamma  | d1t1       | gamma_naught_rtc calibrate |

            # Only γ-area maps target is here => need to produce γ°RTC calibrated
            | d1t1, d1t2 |            | gamma_area             | d1t1, d1t2 | gamma_naught_rtc calibrate |
            | d1t1, d1t2 | d1t1, d1t2 | gamma_area             |            | gamma_naught_rtc calibrate |
            | d1t1, d1t2 | d1t1       | gamma_area             |       d1t2 | gamma_naught_rtc calibrate |
            | d1t1, d1t2 |       d1t2 | gamma_area             | d1t1       | gamma_naught_rtc calibrate |

            # No target is here => need to produce both
            | d1t1, d1t2 |            |                        | d1t1, d1t2 | gamma_naught_rtc calibrate |
            | d1t1, d1t2 | d1t1, d1t2 |                        |            | gamma_naught_rtc calibrate |
            | d1t1, d1t2 | d1t1       |                        |       d1t2 | gamma_naught_rtc calibrate |
            | d1t1, d1t2 |       d1t2 |                        | d1t1       | gamma_naught_rtc calibrate |

        ### Cases w/ only one input => expect tdddddd
        #   Cases where tile is intersected by only ONE S1 product
        @complex_sigma_one_s1
        Examples:
            | remote_s1  | local_s1   | local_s2   | dl_s1      | scenario        |
            # Situations "IMPOSSIBLE", and not tested:
            # -> The correct half-target is not here, but others with similar
            #    names are.
            # => At this point, these situations have undefined behaviours
            # | d1t1       |            | d1tx_sigma | d1t1       | sigma calibrate |
            # | d1t1       | d1t1       | d1tx_sigma |            | sigma calibrate |
            # | d1t1       |            | d1t2_sigma | d1t1       | sigma calibrate |
            # | d1t1       | d1t1       | d1t2_sigma |            | sigma calibrate |
            # Target is here => we don't care
            | d1t1       |            | d1t1_sigma |            | sigma calibrate |
            | d1t1       | d1t1       | d1t1_sigma |            | sigma calibrate |

        ## Request γ area files only
        #  Note: there is no way to know whether a γ area map has been made
        @complex_gamma_area_one_s1
        Examples:
            | remote_s1  | local_s1   | local_s2    | dl_s1      | scenario        |
            # Target is here => DL nothing
            | d1t1       |            | gamma_area  |            | compute gamma area |
            | d1t1       | d1t1       | gamma_area  |            | compute gamma area |
            # Target is not here => DL what is missing
            | d1t1       |            |             | d1t1       | compute gamma area |
            | d1t1       | d1t1       |             |            | compute gamma area |


        ### Cases of mismatching with other dates
        ## σ° calibration
        @complex_mismatch_dates_sigma
        Examples:
            | remote_s1              | local_s1   | local_s2               | dl_s1      | scenario        |
            # d1 in local
            | d1t1, d1t2, d2t1, d2t2 |            | d1tx_sigma             | d2t1, d2t2 | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d1t1, d1t2 | d1tx_sigma             | d2t1, d2t2 | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d1t1       | d1tx_sigma             | d2t1, d2t2 | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 |       d1t2 | d1tx_sigma             | d2t1, d2t2 | sigma calibrate |

            | d1t1, d1t2, d2t1, d2t2 |            | d2tx_sigma             | d1t1, d1t2 | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d1t1, d1t2 | d2tx_sigma             |            | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d1t1       | d2tx_sigma             |       d1t2 | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 |       d1t2 | d2tx_sigma             | d1t1       | sigma calibrate |

            # d2 in local
            | d1t1, d1t2, d2t1, d2t2 |            | d1tx_sigma             | d2t1, d2t2 | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d2t1, d2t2 | d1tx_sigma             |            | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d2t1       | d1tx_sigma             |       d2t2 | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 |       d2t2 | d1tx_sigma             | d2t1,      | sigma calibrate |

            | d1t1, d1t2, d2t1, d2t2 |            | d2tx_sigma             | d1t1, d1t2 | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d2t1, d2t2 | d2tx_sigma             | d1t1, d1t2 | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d2t1       | d2tx_sigma             | d1t1, d1t2 | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 |       d2t2 | d2tx_sigma             | d1t1, d1t2 | sigma calibrate |

            # Some improbable mix
            | d1t1, d1t2, d2t1, d2t2 |            | d1tx_sigma, d2tx_sigma |            | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d2t1, d2t2 | d1tx_sigma, d2tx_sigma |            | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d2t1       | d1tx_sigma, d2tx_sigma |            | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 |       d2t2 | d1tx_sigma, d2tx_sigma |            | sigma calibrate |

            | d1t1, d1t2, d2t1, d2t2 |            | d1tx_sigma, d2tx_sigma |            | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d2t1, d2t2 | d1tx_sigma, d2tx_sigma |            | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d2t1       | d1tx_sigma, d2tx_sigma |            | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 |       d2t2 | d1tx_sigma, d2tx_sigma |            | sigma calibrate |

            | d1t1, d1t2, d2t1, d2t2 | d1t1, d2t2 |                        | d1t2, d2t1 | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d1t2, d2t2 |                        | d1t1, d2t1 | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d1t1, d2t1 |                        | d1t2, d2t2 | sigma calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d1t2, d2t1 |                        | d1t1, d2t2 | sigma calibrate |

        ## γ-area map production
        @complex_mismatch_dates_gamma_area
        Examples:
            | remote_s1              | local_s1   | local_s2   | dl_s1                  | scenario           |
            # γ-area map exist => never request a download
            | d1t1, d1t2, d2t1, d2t2 |            | gamma_area |                        | compute gamma area |
            | d1t1, d1t2, d2t1, d2t2 | d1t1, d1t2 | gamma_area |                        | compute gamma area |
            | d1t1, d1t2, d2t1, d2t2 | d1t1       | gamma_area |                        | compute gamma area |
            | d1t1, d1t2, d2t1, d2t2 |       d1t2 | gamma_area |                        | compute gamma area |

            | d1t1, d1t2, d2t1, d2t2 |            | gamma_area |                        | compute gamma area |
            | d1t1, d1t2, d2t1, d2t2 | d1t1, d1t2 | gamma_area |                        | compute gamma area |
            | d1t1, d1t2, d2t1, d2t2 | d1t1       | gamma_area |                        | compute gamma area |
            | d1t1, d1t2, d2t1, d2t2 |       d1t2 | gamma_area |                        | compute gamma area |

            # γ-area map exist => make sure everything is downloaded,
            # ... even if we only need one pair, we download all pairs :(
            | d1t1, d1t2, d2t1, d2t2 |            |            | d1t1, d1t2, d2t1, d2t2 | compute gamma area |
            | d1t1, d1t2, d2t1, d2t2 | d1t1, d1t2 |            |             d2t1, d2t2 | compute gamma area |
            | d1t1, d1t2, d2t1, d2t2 | d1t1       |            |       d1t2, d2t1, d2t2 | compute gamma area |
            | d1t1, d1t2, d2t1, d2t2 |       d1t2 |            | d1t1,       d2t1, d2t2 | compute gamma area |

            | d1t1, d1t2, d2t1, d2t2 | d2t1, d2t2 |            | d1t1, d1t2             | compute gamma area |
            | d1t1, d1t2, d2t1, d2t2 | d2t1       |            | d1t1, d1t2,       d2t2 | compute gamma area |
            | d1t1, d1t2, d2t1, d2t2 |       d2t2 |            | d1t1, d1t2, d2t1       | compute gamma area |

            | d1t1, d1t2, d2t1, d2t2 | d1t1, d2t1 |            |       d1t2,       d2t2 | compute gamma area |
            | d1t1, d1t2, d2t1, d2t2 | d1t2, d2t1 |            | d1t1,             d2t2 | compute gamma area |
            | d1t1, d1t2, d2t1, d2t2 | d1t1, d2t2 |            |       d1t2, d2t1       | compute gamma area |
            | d1t1, d1t2, d2t1, d2t2 | d1t2, d2t2 |            | d1t1,       d2t1       | compute gamma area |

        ## γ-area map production
        @complex_mismatch_dates_gamma_calibration
        Examples:
            | remote_s1              | local_s1   | local_s2   | dl_s1                  | scenario                   |
            # γ-area map exist, but not the γ°RTC calibrated products
            # => always request to download missing inputs
            | d1t1, d1t2, d2t1, d2t2 |            | gamma_area | d1t1, d1t2, d2t1, d2t2 | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d1t1, d1t2 | gamma_area |             d2t1, d2t2 | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d1t1       | gamma_area |       d1t2, d2t1, d2t2 | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 |       d1t2 | gamma_area | d1t1,       d2t1, d2t2 | gamma_naught_rtc calibrate |

            | d1t1, d1t2, d2t1, d2t2 | d2t1, d2t2 | gamma_area | d1t1, d1t2,            | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d2t1       | gamma_area | d1t1, d1t2,       d2t2 | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 |       d2t2 | gamma_area | d1t1, d1t2, d2t1,      | gamma_naught_rtc calibrate |

        Examples:
            | remote_s1              | local_s1   | local_s2               | dl_s1                  | scenario                   |
            # γ°RTC calibrated products exist, but not the γ-area maps
            # => always request to download missing inputs as producing γ-area
            #    isn't smart and require every possible input pair
            | d1t1, d1t2, d2t1, d2t2 |            | d1tx_gamma, d2tx_gamma             | d1t1, d1t2, d2t1, d2t2 | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d1t1, d1t2 | d1tx_gamma, d2tx_gamma             |             d2t1, d2t2 | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d1t1       | d1tx_gamma, d2tx_gamma             |       d1t2, d2t1, d2t2 | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 |       d1t2 | d1tx_gamma, d2tx_gamma             | d1t1,       d2t1, d2t2 | gamma_naught_rtc calibrate |

            | d1t1, d1t2, d2t1, d2t2 | d2t1, d2t2 | d1tx_gamma, d2tx_gamma             | d1t1, d1t2,            | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d2t1       | d1tx_gamma, d2tx_gamma             | d1t1, d1t2,       d2t2 | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 |       d2t2 | d1tx_gamma, d2tx_gamma             | d1t1, d1t2, d2t1,      | gamma_naught_rtc calibrate |

            # All three outputs are found => no download required
            | d1t1, d1t2, d2t1, d2t2 |            | d1tx_gamma, d2tx_gamma, gamma_area |                        | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d1t1, d1t2 | d1tx_gamma, d2tx_gamma, gamma_area |                        | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d1t1       | d1tx_gamma, d2tx_gamma, gamma_area |                        | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 |       d1t2 | d1tx_gamma, d2tx_gamma, gamma_area |                        | gamma_naught_rtc calibrate |

            | d1t1, d1t2, d2t1, d2t2 | d2t1, d2t2 | d1tx_gamma, d2tx_gamma, gamma_area |                        | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d2t1       | d1tx_gamma, d2tx_gamma, gamma_area |                        | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 |       d2t2 | d1tx_gamma, d2tx_gamma, gamma_area |                        | gamma_naught_rtc calibrate |

            # One of the γ°RTC calibrated output is missing
            # => download was is related to that missing output, if need be
            | d1t1, d1t2, d2t1, d2t2 |            |             d2tx_gamma, gamma_area | d1t1, d1t2             | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d1t1, d1t2 |             d2tx_gamma, gamma_area |                        | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d1t1       |             d2tx_gamma, gamma_area |       d1t2             | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 |       d1t2 |             d2tx_gamma, gamma_area | d1t1                   | gamma_naught_rtc calibrate |

            | d1t1, d1t2, d2t1, d2t2 | d2t1, d2t2 |             d2tx_gamma, gamma_area | d1t1, d1t2             | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 | d2t1       |             d2tx_gamma, gamma_area | d1t1, d1t2             | gamma_naught_rtc calibrate |
            | d1t1, d1t2, d2t1, d2t2 |       d2t2 |             d2tx_gamma, gamma_area | d1t1, d1t2             | gamma_naught_rtc calibrate |
