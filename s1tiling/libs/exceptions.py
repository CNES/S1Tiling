#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

"""
This module defines S1Tiling specific exception classes.
"""

from s1tiling.libs import exits


class Error(Exception):
    """
    Base class for all S1Tiling specific exceptions.
    """
    def __init__(self, *args, **kwargs):
        """
        Constructor
        """
        super().__init__(*args)
        self.code = kwargs.get('code', exits.UNKNOWN_REASON)


class ConfigurationError(Error):
    """
    Generic error for configuration file errors.
    """
    def __init__(self, message, configFile="", *args, **kwargs):
        """
        Constructor
        """
        super().__init__(f"{message}\nPlease fix the configuration file '{configFile}'.",
                         code=exits.CONFIG_ERROR, *args, **kwargs)


class CorruptedDataSAFEError(Error):
    """
    Error that signals invalid data in the data SAFE.
    """
    def __init__(self, manifest, *args, **kwargs):
        """
        Constructor
        """
        super().__init__(f"Problem with {manifest}.\nPlease remove the raw data for {manifest} SAFE file.",
                         code=exits.CORRUPTED_DATA_SAFE, *args, **kwargs)
        self.manifest = manifest

    def __reduce__(self):
        # __reduce__ is required as this error will be pickled from subprocess
        # when transported in the :class:`Outcome` object.
        return (CorruptedDataSAFEError, (self.manifest, ))


class DownloadS1FileError(Error):
    """
    Error that signals problems to download images.
    """
    def __init__(self, tile_name, *args, **kwargs):
        """
        Constructor
        """
        super().__init__(f"Cannot download S1 images associated to {tile_name}.",
                         code=exits.DOWNLOAD_ERROR, *args, **kwargs)
        self.tile_name = tile_name


class NoS2TileError(Error):
    """
    Error that signals incorrect Sentinel-2 tile names.
    """
    def __init__(self, *args, **kwargs):
        """
        Constructor
        """
        super().__init__("No existing tiles found, exiting...",
                         code=exits.NO_S2_TILE, *args, **kwargs)


class NoS1ImageError(Error):
    """
    Error that signals missing Sentinel-1 images.
    """
    def __init__(self, *args, **kwargs):
        """
        Constructor
        """
        super().__init__("No S1 tiles to process, exiting...",
                         code=exits.NO_S1_IMAGE, *args, **kwargs)


class MissingDEMError(Error):
    """
    Error that signals missing DEM file(s).
    """
    def __init__(self, *args, **kwargs):
        """
        Constructor
        """
        super().__init__("Some DEM tiles are missing, exiting...",
                         code=exits.MISSING_SRTM, *args, **kwargs)


class MissingGeoidError(Error):
    """
    Error that signals missing Geoid file.
    """
    def __init__(self, geoid_file, *args, **kwargs):
        """
        Constructor
        """
        super().__init__(f"Geoid file does not exists ({geoid_file}), exiting...",
                         code=exits.MISSING_GEOID, *args, **kwargs)
        self.geoid_file = geoid_file


class InvalidOTBVersionError(Error):
    """
    Error that signals OTB version incompatible with S1Tiling needs.
    """
    def __init__(self, reason, *args, **kwargs):
        """
        Constructor
        """
        super().__init__(f"{reason}",
                         code=exits.CONFIG_ERROR, *args, **kwargs)


