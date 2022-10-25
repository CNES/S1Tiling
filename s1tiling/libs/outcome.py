#!/usr/bin/env python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2022 (c) CNES. All rights reserved.
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


class Outcome:
    """
    Kind of monad à la C++ ``std::expected<>``, ``boost::Outcome``.

    It stores tasks results which could be:
    - either the filename of task product,
    - or the error message that leads to the task failure.
    """
    def __init__(self, value_or_error):
        """
        constructor
        """
        self.__value_or_error    = value_or_error
        self.__is_error          = issubclass(type(value_or_error), BaseException)
        self.__related_filenames = []
        self.__pipeline_name     = None

    def has_value(self):
        return not self.__is_error

    def __bool__(self):
        return self.has_value()

    def value(self):
        assert not self.__is_error
        return self.__value_or_error

    def error(self):
        assert self.__is_error
        return self.__value_or_error

    def related_filenames(self):
        return self.__related_filenames

    def add_related_filename(self, filename):
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

    def set_pipeline_name(self, pipeline_name):
        """
        Record the name of the pipeline in error
        """
        self.__pipeline_name = pipeline_name

    def __repr__(self):
        if self.__is_error:
            msg = f'Failed to produce {self.__related_filenames[-1]}'
            if self.__pipeline_name:
                msg += f' because {self.__pipeline_name} failed.'
            if len(self.__related_filenames) > 1:
                errored_files = ', '.join(self.__related_filenames[:-1])
                # errored_files = str(self.__related_filenames)
                msg += f' {errored_files} could not be produced: '
            else:
                msg += ': '
            msg +=  f'{self.__value_or_error}'
            return msg
        else:
            return f'Success: {self.__value_or_error}'


