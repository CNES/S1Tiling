#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2026 (c) CNES.
#
#   This file is part of S1Tiling project
#       https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# =========================================================================
#
# Authors:
# - Thierry KOLECK (CNES)
# - Luc HERMITTE (CSGROUP)
#
# =========================================================================

"""«module-docstring»"""

from typing import Generic, TypeVar

from eodag.utils.exceptions  import NotAvailableError, TimeOutError

from ..outcome import DownloadOutcome


# Value     = TypeVar("Value")
DLOutcome = TypeVar("DLOutcome", bound=DownloadOutcome)


class FileManager(Generic[DLOutcome]):
    """
    Root (implementation class) for all file managers.
    """
    def __init__(self) -> None:
        """
        constructor
        """
        # Failures related to download (e.g. missing products)
        self.__download_failures : list[DLOutcome] = []
        self.__search_failures                     = 0

    def get_search_failures(self) -> int:
        """Returns the number of times querying matching products failed"""
        return self.__search_failures

    def get_download_failures(self) -> list[DLOutcome]:
        """
        Returns the list of download failures as a list of :class:S1DownloadOutcome`
        """
        return self.__download_failures

    def get_download_timeouts(self) -> list[DLOutcome]:
        """
        Returns the list of download timeours as a list of :class:S1DownloadOutcome`
        """
        return list(filter(lambda f: isinstance(f.error(), (NotAvailableError, TimeOutError)), self.__download_failures))

    def _inc_search_failures(self) -> None:
        self.__search_failures += 1

    def _register_download_failures(self, failures: list[DLOutcome]) -> None:
        self.__download_failures.extend(failures)
