#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2025 (c) CNES.
#   Copyright 2022-2024 (c) CS GROUP France.
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
# - Luc HERMITTE (CS Group)
#
# =========================================================================

""" This module contains various utility functions related to xml library"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional, TypeVar, Union
import xml.etree.ElementTree as ET

from .path import AnyPath

T = TypeVar('T')


def parse(filename: AnyPath) -> ET.ElementTree[ET.Element[str]]:
    """
    Returns root of XML document.
    """
    return ET.parse(filename)


def find(
    element: Union[ET.Element[str], ET.ElementTree[ET.Element[str]]],
    key    : str,
    context: AnyPath,
    keytext: Optional[str] = None,
    **kwargs
) -> ET.Element[str]:
    """
    Helper function that finds an XML tag within a node.

    :param element: node/tree where the search is done
    :param key:     key that identifies the tag name to search
    :param context: extra information used to report where search failures happen
    :param keytext: text to use instead of ``key`` to report a missing key
    :param kwargs:  extra parameters forwarded to :meth:`ET.find`.
    :raise RuntimeError: If the requested ``key`` isn't found.
    :return: The non null node.
    """
    node = element.find(key, **kwargs)
    if node is None:
        kt = keytext or f"{key} node"
        raise RuntimeError(f"Cannot find {kt!r} in {context!s}")
    return node


def find_text(
    element: Union[ET.Element[str], ET.ElementTree[ET.Element[str]]],
    key    : str,
    context: AnyPath,
    keytext: Optional[str] = None,
    **kwargs
) -> str:
    """
    Helper function that finds and returns the text contained in an XML tag
    within a node.

    :param element: node/tree where the search is done
    :param key:     key that identifies the tag name to search
    :param context: extra information used to report where search failures happen
    :param keytext: text to use instead of ``key`` to report a missing key
    :param kwargs:  extra parameters forwarded to :meth:`ET.find`.
    :raise RuntimeError: If the requested ``key`` isn't found.
    :raise RuntimeError: If the node has non value
    :return: The non empty text.
    """
    node = find(element, key, context, keytext, **kwargs)
    if not node.text:
        kt = keytext or f"{key} node"
        raise RuntimeError(f"Empty {kt!r} in {str(context)}")
    return node.text


def find_as(
    to     : Callable[[str], T],
    element: Union[ET.Element[str], ET.ElementTree[ET.Element[str]]],
    key    : str,
    context: AnyPath,
    keytext: Optional[str] = None,
    **kwargs,
) -> T:
    """
    Helper function that finds and returns the text contained in an XML tag
    within a node and converts it to the requested type.

    :param to:      type to which the text shall be converted to
    :param element: node/tree where the search is done
    :param key:     key that identifies the tag name to search
    :param context: extra information used to report where search failures happen
    :param keytext: text to use instead of ``key`` to report a missing key
    :param kwargs:  extra parameters forwarded to :meth:`ET.find`
    :raise RuntimeError: If the requested ``key`` isn't found.
    :raise RuntimeError: If the node has non value.
    :raise RuntimeError: If the node text value cannot be converted to a ``to`` instance.
    :return: The value stored in the node
    """
    text = find_text(element, key, context, keytext, **kwargs)
    try:
        return to(text)
    except ValueError as e:
        kt = keytext or f"{key} node"
        raise RuntimeError(f"Value for {kt!r} is not a valid {to.__name__}: {text=!r}") from e
