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

Feature: Test download requests
    Test download requests given requirements and detected files

        Examples:
            | dates     |
            | 8th jan   |
            | all dates |

    Scenario Outline: Everything was downloaded and generated
        Given Request on <dates>
        And   All S1 files are known
        And   All S2 files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  None are requested for download

    Scenario Outline: Everything was downloaded and nothing was generated
        Given Request on <dates>
        And   All S1 files are known
        And   No S2 files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  None are requested for download

    Scenario Outline: Nothing was downloaded and everything was generated
        Given Request on <dates>
        And   No S1 files are known
        And   All S2 files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  None are requested for download

    Scenario Outline: Nothing was downloaded and nothing was generated
        Given Request on <dates>
        And   No S1 files are known
        And   No S2 files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  All are requested for download

    # + scenarios with VV / VH mismatchs
    Scenario Outline: Everything was downloaded and all VV were generated and requested
        Given Request on <dates>
        And   Request on VV
        And   All S1 files are known
        And   All S2 VV files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  None are requested for download

    Scenario Outline: Everything was downloaded and all VV were generated but VH requested
        Given Request on <dates>
        And   Request on VH
        And   All S1 files are known
        And   All S2 VV files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  None are requested for download

    Scenario Outline: Nothing was downloaded and all VV was generated and requested
        Given Request on <dates>
        And   Request on VV
        And   No S1 files are known
        And   All S2 VV files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  None are requested for download

    Scenario Outline: Nothing was downloaded and all VV was generated but VH requested
        Given Request on <dates>
        And   Request on VH
        And   No S1 files are known
        And   All S2 VV files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  All are requested for download


    # + scenarios with fname_fmt mismatch
    Scenario Outline: Nothing was downloaded and everything was generated but for another calibration
        Given Request on <dates>
        And   Request for _beta
        And   No S1 files are known
        And   All S2 files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  All are requested for download

    Scenario Outline: Nothing was downloaded and everything was generated but for another fname_fmt
        Given Request on <dates>
        And   Request with default fname_fmt_concatenation
        And   No S1 files are known
        And   All S2 files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  All are requested for download


    # + scenarios with existing filtered products
    # > with standard fname_fmt
    # > with different fname_fmt
    # > with different dname_fmt
