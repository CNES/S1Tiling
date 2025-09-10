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

"""Centralizes Dask related utilities"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Protocol, Union

from dask.distributed import Client, LocalCluster

from .. import Utils


logger = logging.getLogger('s1tiling.utils.dask')


class DaskConfiguration(Protocol):
    """
    Specialized protocol for configuration information related to class:`DaskContext` configuration data.

    Can be seen an a ISP compliant concept for Configuration object regarding dask context.
    """
    nb_procs: int

    @property
    def log_config(self) -> Dict:
        """Accessor to log property"""
        ...


the_config : DaskConfiguration


class DaskContext:
    """
    Custom context manager for :class:`dask.distributed.Client` +
    :class:`dask.distributed.LocalCluster` classes.
    """
    def __init__(self, config: DaskConfiguration, debug_otb: bool) -> None:
        self.__client    : Optional[Client]       = None
        self.__cluster   : Optional[LocalCluster] = None
        self.__config    : DaskConfiguration      = config
        self.__debug_otb : bool                   = debug_otb

    def __enter__(self) -> "DaskContext":
        if not self.__debug_otb:
            clean_logs(self.__config.log_config, self.__config.nb_procs)
            self.__cluster = LocalCluster(
                    threads_per_worker=1, processes=True, n_workers=self.__config.nb_procs,
                    silence_logs=False)
            self.__client = Client(self.__cluster)
            # Work around: Cannot pickle local object in lambda...
            global the_config
            the_config = self.__config
            self.__client.register_worker_callbacks(
                    lambda dask_worker: setup_worker_logs(the_config.log_config, dask_worker))
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> Literal[False]:
        if self.__client:
            self.__client.close()
            assert self.__cluster, "client existence implies cluster existence"
            self.__cluster.close()
        return False

    @property
    def client(self) -> Optional[Client]:
        """
        Return a :class:`dask.distributed.Client`
        """
        return self.__client


def remove_files(files: List[Union[str, Path]], what: str) -> None:
    """
    Removes the files from the disk
    """
    logger.debug("Remove %s: %s", what, files)
    for file_it in files:
        if os.path.exists(file_it):
            os.remove(file_it)


def clean_logs(config: Dict, nb_workers: int) -> None:
    """
    Clean all the log files.
    Meant to be called once, at startup
    """
    filenames = []
    for _, cfg in config['handlers'].items():
        if 'filename' in cfg and '{kind}' in cfg['filename']:
            filenames += [cfg['filename'].format(kind=f"worker-{w}") for w in range(nb_workers)]
    remove_files(filenames, "logs")


def setup_worker_logs(config: Dict, dask_worker) -> None:
    """
    Set-up the logger on Dask Worker.
    """
    d_logger = logging.getLogger('distributed.worker')
    r_logger = logging.getLogger()
    old_handlers = d_logger.handlers[:]

    for _, cfg in config['handlers'].items():
        if 'filename' in cfg and '{kind}' in cfg['filename']:
            cfg['mode']     = 'a'  # Make sure to not reset worker log file
            cfg['filename'] = cfg['filename'].format(kind=f"worker-{dask_worker.name}")

    logging.config.dictConfig(config)
    # Restore old dask.distributed handlers, and inject them in root handler as well
    for hdlr in old_handlers:
        d_logger.addHandler(hdlr)
        r_logger.addHandler(hdlr)  # <-- this way we send s1tiling messages to dask channel

    # From now on, redirect stdout/stderr messages to s1tiling
    Utils.RedirectStdToLogger(logging.getLogger('s1tiling'))
