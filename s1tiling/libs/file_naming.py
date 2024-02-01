#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2024 (c) CNES.
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
# =========================================================================

"""
This module provide filename generator classes
"""

import os
from abc import ABC
from typing import Dict, List, Union


class OutputFilenameGenerator(ABC):
    """
    Abstract class for generating filenames.
    Several policies are supported as of now:
    - return the input string (default implementation)
    - replace a text with another one
    - {template} strings
    - list of any of the other two
    """
    def generate(self, basename: str, keys: Dict) -> Union[str, List[str]]:  # pylint: disable=unused-argument
        """
        Default implementation does nothing.
        """
        return basename


class ReplaceOutputFilenameGenerator(OutputFilenameGenerator):
    """
    Given a pair ``[text_to_search, text_to_replace_with]``,
    replace the exact matching text with new text in ``basename`` metadata.
    """
    def __init__(self, before_afters: List) -> None:
        assert isinstance(before_afters, list)
        self.__before_afters = before_afters

    def generate(self, basename, keys: Dict) -> str:
        filename = basename.replace(*self.__before_afters)
        return filename


class CannotGenerateFilename(KeyError):
    """
    Exception used to filter out cases where a meta cannot serve as a direct
    input of a :class:`StepFactory`.
    """
    pass


class TemplateOutputFilenameGenerator(OutputFilenameGenerator):
    """
    Given a template: ``"text{key1}_{another_key}_.."``,
    inject the metadata instead of the template keys.

    Most filename format templates can be fine tuned to end-user ideal filenames.
    While the filenames used for intermediary products may be changed, it's not
    recommended for data flow stability.
    See :ref:`[Processing].fname_fmt.* <Processing.fname_fmt>` for the short list
    of filenames meants to be adapted.
    """
    def __init__(self, template) -> None:
        assert isinstance(template, str)
        self.__template = template

    def generate(self, basename, keys: Dict) -> str:
        try:
            rootname = os.path.splitext(basename)[0]
            filename = self.__template.format(**keys, rootname=rootname)
            return filename
        except KeyError as e:
            raise CannotGenerateFilename(f'Impossible to generate a filename matching {self.__template} from {keys}') from e


class OutputFilenameGeneratorList(OutputFilenameGenerator):
    """
    Some steps produce several products.
    This specialization permits to generate several output filenames.

    It's constructed from other filename generators.
    """
    def __init__(self, generators) -> None:
        assert isinstance(generators, list)
        self.__generators = generators

    def generate(self, basename, keys) -> List[str]:
        filenames = [generator.generate(basename, keys) for generator in self.__generators]
        return filenames
