#!/usr/bin/env python3
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
# Authors: Thierry KOLECK (CNES)
#          Luc HERMITTE (CS Group)
# =========================================================================

"""
This module defines steps meta data related helper functions
"""

from collections.abc import Iterable
import logging
import os
from typing import Dict, Union

logger = logging.getLogger('s1tiling.metadata')

Meta          = Dict
TaskName      = str


def append_to(meta: Meta, key: str, value) -> Dict:
    """
    Helper function to append to a list that may be empty
    """
    meta[key] = meta.get(key, []) + [value]
    return meta


def in_filename(meta: Meta) -> str:
    """
    Helper accessor to access the input filename of a `Step`.
    """
    assert 'in_filename' in meta
    return meta['in_filename']


def out_filename(meta: Meta) -> str:
    """
    Helper accessor to access the ouput filename(s) of a `Step`.
    """
    assert 'out_filename' in meta
    return meta['out_filename']


def tmp_filename(meta: Meta) -> str:
    """
    Helper accessor to access the temporary ouput filename of a `Step`.
    """
    assert 'out_tmp_filename' in meta
    return meta['out_tmp_filename']


def output_parameter(meta: Meta) -> str:
    """
    Helper accessor to the exact outpout parameter passed to (OTB) applications.
    Most of the time, the correct output parameter is :func:`tmp_filename`, but in some instances,
    the application receives an output parameter that is more like _file pattern_. In that cases we
    need to be able to override the exact output parameter.

    .. warning:: The result still needs to contain the ``.tmp`` pattern.
    """
    if 'output_parameter' in meta:
        res = meta['output_parameter']
        assert ".tmp" in res, f"meta['output_parameter']={res!r} needs to contain '.tmp'\n        In {meta=}"
        return res
    else:
        return tmp_filename(meta)


def out_extended_filename_complement(meta: Meta) -> str:
    """
    Helper accessor to the extended filename to use to produce the image.
    """
    return meta.get('out_extended_filename_complement', '')


def get_task_name(meta: Meta) -> TaskName:
    """
    Helper accessor to the task name related to a `Step`.

    By default, the task name is stored in `out_filename` key.
    In the case of reducing :class:`MergeStep`, a dedicated name shall be
    provided. See :class:`Concatenate`

    Important, task names shall be unique and attributed to a single step.
    """
    if 'task_name' in meta:
        return meta['task_name']
    else:
        return out_filename(meta)


def check_one_product(filename: Union[str, os.PathLike], step_factory_name: str) -> bool:
    """
    Helper function that tells whether a filename-like string corresponds to an existing filename.
    """
    assert isinstance(filename, (str, os.PathLike)), f"[{step_factory_name}] product name {filename=!r} not a string/pathlike, but a {type(filename)}"
    exist_file_name = os.path.isfile(filename)
    logger.debug('     Checking %s product: %s => %s', step_factory_name, filename, '∃' if exist_file_name else '∅')
    return exist_file_name


def check_several_products(filenames: Iterable[Union[str, os.PathLike]], step_factory_name: str) -> bool:
    """
    Helper function that tells whether a series of filename-like strings corresponds to existing
    filenames.
    """
    return all(check_one_product(f, step_factory_name) for f in filenames)


def product_exists(meta: Meta) -> bool:
    """
    Helper accessor that tells whether the product described by the metadata
    already exists.
    """
    if 'does_product_exist' in meta:
        return meta['does_product_exist']()
    output = out_filename(meta)
    if isinstance(output, list):
        return check_several_products(output, meta.get('current_step', '??'))
    else:
        return check_one_product(output, meta.get('current_step', '??'))


def accept_as_compatible_input(output_meta: Meta, input_meta: Meta) -> bool:
    """
    Tells whether ``input_meta`` is a valid and compatible input for ``output_meta``

    Uses the optional meta information ``accept_as_compatible_input`` from ``output_meta``
    to tell whether they are compatible.

    This will be uses for situations where an input file will be used as input
    for several different new files. Typical example: all final normlimed
    outputs on a S2 tile will rely on the same map of sin(LIA). As such,
    the usual __input -> expected output__ approach cannot work.
    """
    if 'accept_as_compatible_input' in output_meta:
        return output_meta['accept_as_compatible_input'](input_meta)
    else:
        return False


def is_running_dry(execution_parameters: Dict) -> bool:
    """
    Helper function to test whether execution parameters have ``dryrun`` property set to True.
    """
    return execution_parameters.get('dryrun', False)


def is_debugging_caches(execution_parameters: Dict) -> bool:
    """
    Helper function to test whether execution parameters have ``debug_caches`` property set to True.
    """
    return execution_parameters.get('debug_caches', False)
