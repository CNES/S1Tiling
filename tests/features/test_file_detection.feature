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

#@tasks
Feature: Analysis of existing files
    Analyse existing files according to requested parameters like polarisation

    Scenario: Search VV-VH among VV-VH
        Given All S1 files are known
        When  VV-VH files are searched
        Then  VV files are found
        And   VH files are found
        And   No (other) files are found

    Scenario: Search VV among VV-VH
        Given All S1 files are known
        When  VV files are searched
        Then  VV files are found
        And   No (other) files are found

    Scenario: Search VH among VV-VH
        Given All S1 files are known
        When  VH files are searched
        Then  VH files are found
        And   No (other) files are found


    Scenario: Search VV-VH among VV
        Given All S1 VV files are known
        When  VV-VH files are searched
        Then  VV files are found
        And   No (other) files are found

    Scenario: Search VV among VV
        Given All S1 VV files are known
        When  VV files are searched
        Then  VV files are found
        And   No (other) files are found

    Scenario: Search VH among VV
        Given All S1 VV files are known
        When  VH files are searched
        Then  No (other) files are found


    Scenario: Search VV-VH among VH
        Given All S1 VH files are known
        When  VV-VH files are searched
        Then  VH files are found
        And   No (other) files are found

    Scenario: Search VH among VH
        Given All S1 VH files are known
        When  VH files are searched
        Then  VH files are found
        And   No (other) files are found

    Scenario: Search VV among VH
        Given All S1 VH files are known
        When  VV files are searched
        Then  No (other) files are found

