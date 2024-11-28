#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2024 (c) CNES.
#   Copyright 2022-2024 (c) CS GROUP France.
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

"""
This module defines S1Tiling specific exception classes.
"""

from typing import List, Optional, Set, Union
from pathlib import Path


class Error(Exception):
    """
    Base class for all S1Tiling specific exceptions.
    """
    pass


class ConfigurationError(Error):
    """
    Generic error for configuration file errors.
    """
    def __init__(self, message: str, configFile: Union[str, Path], *args, **kwargs) -> None:
        """
        Constructor
        """
        super().__init__(f"{message}\nPlease fix the configuration file {str(configFile)!r}.",
                         *args, **kwargs)


class CorruptedDataSAFEError(Error):
    """
    An empty data safe has been found and needs to be removed so it can be fetched again.
    """
    def __init__(self, product: str, details: Optional[str], *args, **kwargs) -> None:
        """
        Constructor
        """
        extra = details or "no manifest, or image files"
        super().__init__(
                f"Product {product!r} appears to be corrupted ({extra}).\nPlease remove the raw data for {product!r} SAFE file.",
                *args, **kwargs)
        self.product = product
        self.details = details

    def __reduce__(self):
        # __reduce__ is required as this error will be pickled from subprocess
        # when transported in the :class:`Outcome` object.
        return (CorruptedDataSAFEError, (self.product, self.details, ))


class DownloadS1FileError(Error):
    """
    Error that signals problems to download images.
    """
    def __init__(self, tile_name: str, *args, **kwargs) -> None:
        """
        Constructor
        """
        super().__init__(f"Cannot download S1 images associated to {tile_name}.",
                         *args, **kwargs)
        self.tile_name = tile_name

    def __reduce__(self):
        # __reduce__ is required as this error will be pickled from subprocess
        # when transported in the :class:`Outcome` object.
        return (DownloadS1FileError, (self.tile_name, ))


class DownloadEOFFileError(Error):
    """
    Error that signals problems to download EOF orbit files.
    """
    def __init__(self, msg, *args, **kwargs) -> None:
        """
        Constructor
        """
        super().__init__(f"Cannot download EOF orbit files: {msg}",
                         *args, **kwargs)
        self.msg    = msg

    def __reduce__(self):
        # __reduce__ is required as this error will be pickled from subprocess
        # when transported in the :class:`Outcome` object.
        return (DownloadEOFFileError, (self.msg, ))


class NoS2TileError(Error):
    """
    Error that signals incorrect Sentinel-2 tile names.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Constructor
        """
        super().__init__("No existing tiles found, exiting...",
                         *args, **kwargs)


class NoS1ImageError(Error):
    """
    No Sentinel-1 product has been found that intersects the :ref:`requested
    Sentinel-2 tiles <DataSource.roi_by_tiles>` within the :ref:`requested
    time range <DataSource.first_date>`
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Constructor
        """
        super().__init__("No S1 tiles to process, exiting...",
                         *args, **kwargs)


class MissingDEMError(Error):
    """
    Cannot find all the :ref:`DEM products <paths.srtm>` that cover the
    :ref:`requested Sentinel-2 tiles <DataSource.roi_by_tiles>`
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Constructor
        """
        super().__init__("Some DEM tiles are missing, exiting...",
                         *args, **kwargs)


class MissingGeoidError(Error):
    """
    The :ref:`geoid file <paths.geoid_file>` is missing or the specified path is incorrect.
    """
    def __init__(self, geoid_file: Union[str, Path], *args, **kwargs) -> None:
        """
        Constructor
        """
        super().__init__(f"Geoid file does not exists ({geoid_file}), exiting...",
                         *args, **kwargs)
        self.geoid_file = geoid_file


class InvalidOTBVersionError(Error):
    """
    Error that signals OTB version incompatible with S1Tiling needs.
    """
    def __init__(self, reason: str, *args, **kwargs) -> None:
        """
        Constructor
        """
        super().__init__(f"{reason}",
                         *args, **kwargs)


class MissingApplication(Error):
    """
    Some processing cannot be done because external applications cannot
    be executed. Likelly OTB and/or NORMLIM related applications aren't
    correctly installed.
    """
    def __init__(self, missing_apps, contexts: Union[Set[str], List[str]], *args, **kwargs) -> None:
        """
        Constructor
        """
        self.missing_apps = missing_apps
        self.contexts     = contexts
        message = ['Cannot execute S1Tiling because of the following reason(s):']
        for req, task_keys in missing_apps.items():
            message.append(f"- {req} for {task_keys}")
        for ctx in contexts:
            message.append(f" --> {ctx}")
        super().__init__("\n".join(message), *args, **kwargs)


class NotCompatibleInput(Error):
    """
    Exception used to report input of incompatible type.
    For instance DEM+geoid on S2 tile is not compatible with ExtractSentinel1Metadata.
    """
    pass
