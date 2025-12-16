#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2025 (c) CNES.
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


"""Path related utilities"""

import fnmatch
import logging
import os
from pathlib import Path
import re
from typing import List, Sequence, Union



AnyPath = os.PathLike | str


logger = logging.getLogger("s1tiling.utils.path")


def as_path(path: AnyPath) -> Path:
    """
    Returns a :class:`pathlib:Path` instance of the path.

    If parameter is of the right type, it's directly returned. Otherwise a new object will be built
    and returned.
    """
    if isinstance(path, Path):
        return path
    if isinstance(path, bytes):
        return Path(str(path))
    return Path(path)


def list_files(directory: str, pattern: Union[None,str,re.Pattern] = None) -> List[os.DirEntry]:
    """
    Efficient listing of files in requested directory.

    This version shall be faster than glob to isolate files only as it keeps in "memory"
    the kind of the entry without needing to stat() the entry again.

    Requires Python 3.5
    """
    assert not isinstance(directory, re.Pattern)
    if not pattern:
        filt = lambda path: path.is_file()
    elif isinstance(pattern, re.Pattern):
        filt = lambda path: path.is_file() and re.match(pattern, path.name)
    else:
        filt = lambda path: path.is_file() and fnmatch.fnmatch(path.name, pattern)

    with os.scandir(directory) as nodes:
        res = list(filter(filt, nodes))
    return res


def list_dirs(directory: str, pattern: Union[None,str,re.Pattern] = None) -> List[os.DirEntry]:
    """
    Efficient listing of sub-directories in requested directory.

    This version shall be faster than glob to isolate directories only as it keeps in
    "memory" the kind of the entry without needing to stat() the entry again.

    Requires Python 3.5
    """
    if not pattern:
        filt = lambda path: path.is_dir()
    elif isinstance(pattern, re.Pattern):
        filt = lambda path: path.is_dir() and re.match(pattern, path.name)
    else:
        filt = lambda path: path.is_dir() and fnmatch.fnmatch(path.name, pattern)

    with os.scandir(directory) as nodes:
        res = list(filter(filt, nodes))
        logger.debug("RES(%r)= %s", directory, res)
    return res


def files_exist(files: Union[AnyPath, Sequence[AnyPath]]) -> bool:
    """
    Checks whether a single file, or all files from a list, exist.
    """
    if isinstance(files, (str, os.PathLike)):
        return os.path.isfile(files)
    else:
        for file in files:
            if not os.path.isfile(file):
                return False
        return True
