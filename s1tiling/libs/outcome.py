#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2025 (c) CNES. All rights reserved.
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
Module relate to :class:`Outcome` monad.
"""

from __future__ import annotations
from collections.abc import Callable
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union, cast

try:
    # python 3.11+
    from typing import Self
except (ModuleNotFoundError, ImportError):
    from typing_extensions import Self


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
        return cast(Value, self.__value_or_error)

    def value_or(self, default: Value) -> Value:
        """
        Returns the current value, or ``default`` if the instance holds an error.
        """
        return self.value() if self.has_value() else default

    def error(self) -> BaseException:
        """
        Returns the error that happened.

        Requires ``has_value()`` to be ``False``
        """
        assert not self.has_value()
        # assert isinstance(self.__value_or_error, BaseException)
        return cast(BaseException, self.__value_or_error)

    def __repr__(self) -> str:
        if self.has_value():
            return f'Success: {self.__value_or_error}'
        else:
            return f'Error: {self.error()}'

    def transform(self, f: Callable[[Value], T]) -> Outcome[T]:
        """
        Transforms the value, if any. Leave the error unchanged.

        .. warning::
            This method is not polymorphic. The result type will be Outcome[T] and not Self[T]
            We would need Higher Kinded Types with
            https://returns.readthedocs.io/en/latest/pages/hkt.html for instance (which requires
            Python 3.10)

            In the mean time, use :meth:`Outcome.inplace_transform`
        """
        if self.has_value():
            try:
                return Outcome(f(self.value()))
            except BaseException as e:  # pylint: disable=broad-exception-caught
                return Outcome(e)
        else:
            return Outcome(self.error())

    def inplace_transform(self, f: Callable[[Value], Value]) -> None:
        """
        Transforms the value, if any, inplace. Leave the error unchanged.
        """
        if self.has_value():
            try:
                self.__value_or_error = f(self.value())
            except BaseException as e:  # pylint: disable=broad-exception-caught
                self.__value_or_error = e

    def change_error(self, error: BaseException) -> Self:
        """
        Change the actual error
        """
        self.__value_or_error = error
        return self


class PipelineOutcome(Outcome[Value], Generic[Value, File]):
    """
    Kind of monad à la C++ ``std::expected<>``, ``boost::Outcome`` that is specialized for
    generated products for better error messages.

    It stores tasks results which could be:
    - either the path to the downloaded product,
    - or the error message that leads to the task failure.

    Plus information about the related input files.
    """

    def __init__(self, value_or_error: Union[Value, BaseException]) -> None:
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

    def add_related_filename(self, filename: Union[File, List[File]]) -> Self:
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

    def set_pipeline_name(self, pipeline_name: str) -> Self:
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


class DownloadOutcome(Outcome[Value]):
    """
    Kind of monad à la C++ ``std::expected<>``, ``boost::Outcome`` that is specialized for
    downloaded products for better error messages.

    It stores tasks results which could be:
    - high-level information about the product downloaded,
    - or the error message that leads to the task failure.
    """

    pass


class ProductDownloadOutcome(DownloadOutcome[Value], Generic[Value, Product]):
    """
    Kind of monad à la C++ ``std::expected<>``, ``boost::Outcome`` that is specialized for
    Sentinel-1 downloaded products for better error messages.

    It stores tasks results which could be:
    - either the path to the downloaded product,
    - or the error message that leads to the task failure.

    Plus information about the related (eodag) product.
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


S1DownloadOutcome = ProductDownloadOutcome[Value, Product]


# Let's workaround mypy/Pyright...
def filter_outcome_list(
    outcomes: List[Outcome[T]]
) -> Tuple[List[T], List[Outcome[T]]]:
    """
    Internal helper to filter list of :class:`Outcome`
    """
    values : List[T] = []
    errors : List[Outcome[T]] = []
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
    values : Dict[str, List[T]] = {}
    errors : List[Outcome[T]] = []
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
