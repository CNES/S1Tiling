#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2024 (c) CNES. All rights reserved.
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
Module relate to :class:`Outcome` monad.
"""

from collections.abc import Callable
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union

Value   = TypeVar("Value")
File    = TypeVar('File')
Product = TypeVar('Product')
T       = TypeVar("T")


class Outcome(Generic[Value]):
    """
    Kind of monad à la C++ ``std::expected<>``, ``boost::Outcome``.

    It stores tasks results which could be:
    - either the filename of task product,
    - or the error message that leads to the task failure.
    """
    def __init__(self, value_or_error : Union[Value, BaseException]) -> None:
        """
        constructor
        """
        self.__value_or_error    = value_or_error
        self.__is_error          = issubclass(type(value_or_error), BaseException)

    def has_value(self) -> bool:
        """
        Tells whether there is an outcome: i.e. a valid value and not an error.
        """
        return not self.__is_error

    def __bool__(self) -> bool:
        """
        Tells whether there is an outcome: i.e. a valid value and not an error.
        """
        return self.has_value()

    def value(self) -> Value:
        """
        Returns the outcome value.

        Requires ``has_value()`` to be ``True``
        """
        assert self.has_value()
        assert not isinstance(self.__value_or_error, BaseException)
        return self.__value_or_error

    def error(self) -> BaseException:
        """
        Returns the error that happened.

        Requires ``has_value()`` to be ``False``
        """
        assert not self.has_value()
        assert isinstance(self.__value_or_error, BaseException)
        return self.__value_or_error

    def __repr__(self) -> str:
        if self.has_value():
            return f'Success: {self.__value_or_error}'
        else:
            return f'Error: {self.error()}'

    def transform(self, f : Callable[[Value], T]) -> "Outcome[T]":
        """
        Transforms the value, if any. Leave the error unchanged.
        """
        if self.has_value():
            return Outcome(f(self.value()))
        else:
            return Outcome(self.error())


class PipelineOutcome(Outcome[Value], Generic[Value, File]):
    """
    Kind of monad à la C++ ``std::expected<>``, ``boost::Outcome`` that is specialized for
    generated products for better error messages.

    It stores tasks results which could be:
    - either the path to the downloaded product,
    - or the error message that leads to the task failure.

    Plus information about the related input files.
    """
    def __init__(self, value_or_error : Union[Value, BaseException]) -> None:
        """
        constructor
        """
        super().__init__(value_or_error)
        self.__related_filenames : List[File] = []
        self.__pipeline_name     : Optional[str] = None

    def related_filenames(self) -> List[File]:
        """
        Returns the list of filenames related to the error or the result.
        """
        return self.__related_filenames

    def add_related_filename(self, filename: File) -> "PipelineOutcome":
        """
        Register a filename(s) related to the result.
        """
        if isinstance(filename, list):
            for f in filename:
                # Some OTB applications expect list passed with ``-il`` e.g.
                self.__related_filenames.append(f)
        else:
            # While other OTB application expect only one file, passed with ``-in`` e.g.
            self.__related_filenames.append(filename)
        return self

    def set_pipeline_name(self, pipeline_name: str) -> "PipelineOutcome":
        """
        Record the name of the pipeline in error
        """
        self.__pipeline_name = pipeline_name
        return self

    def __repr__(self) -> str:
        if self.has_value():
            return f'Success: {self.value()}'
        else:
            msg = f'Failed to produce {self.__related_filenames[-1]}'
            if self.__pipeline_name:
                msg += f' because {self.__pipeline_name} failed.'
            if len(self.__related_filenames) > 1:
                errored_files = ', '.join(map(str, self.__related_filenames[:-1]))
                # errored_files = str(self.__related_filenames)
                msg += f' {errored_files} could not be produced: '
            else:
                msg += ': '
            msg += f'{self.error()}'
            return msg


class DownloadOutcome(Outcome[Value], Generic[Value, Product]):
    """
    Kind of monad à la C++ ``std::expected<>``, ``boost::Outcome`` that is specialized for
    downloaded products for better error messages.

    It stores tasks results which could be:
    - either the path to the downloaded product,
    - or the error message that leads to the task failure.

    Plus information about the related eodag product.
    """
    def __init__(
            self,
            value_or_error : Union[Value, BaseException],
            product: Product
    ) -> None:
        """
        constructor
        """
        super().__init__(value_or_error)
        self.__related_product = product

    def related_product(self) -> Product:
        """
        Property related_product
        """
        return self.__related_product

    def __repr__(self) -> str:
        if self.has_value():
            return f'{self.value()} has been successfully downloaded'
        else:
            return f'Failed to download {self.__related_product}: {self.error()}'


# Let's workaround mypy/Pyright...
def filter_outcome_list(
        outcomes: List[Outcome[T]]
) -> Tuple[List[T], List[Outcome[T]]]:
    """
    Internal helper to filter list of :class:`Outcome`
    """
    values : List = []
    errors : List[Outcome] = []
    for o in outcomes:
        if o:
            values.append(o.value())
        else:
            errors.append(o)
    return values, errors

def filter_outcome_dict(
        outcomes: Dict[str, List[Outcome[T]]]
) -> Tuple[Dict[str, List[T]], List[Outcome[T]]]:
    """
    Internal helper to filter dictionary of lists of :class:`Outcome`
    """
    values : Dict = {}
    errors : List[Outcome] = []
    for k in outcomes:
        values[k], e = filter_outcome_list(outcomes[k])
        errors.extend(e)
    return values, errors


def filter_outcomes(
        outcomes: Union[List[Outcome[T]], Dict[str, List[Outcome[T]]]]
) -> Tuple[Union[List[T], Dict[str, List[T]]], List[Outcome[T]]]:
    """
    Helper function that filters a collection of :class:`Outcome` to return a collection of the
    values, and a list of the outcome errors.
    """
    if isinstance(outcomes, list):
        return filter_outcome_list(outcomes)
    elif isinstance(outcomes, dict):
        return filter_outcome_dict(outcomes)
    assert False, f"Invalid sequence of outcomes: {type(outcomes)=}"
