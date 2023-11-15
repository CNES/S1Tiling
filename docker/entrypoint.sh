#!/bin/bash
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

# export LANG=en_US.utf8
# RuntimeError: Click will abort further execution because Python was
# configured to use ASCII as encoding for the environment. Consult
# https://click.palletsprojects.com/unicode-support/ for mitigation
# steps.
# This system supports the C.UTF-8 locale which is recommended. You
# might be able to resolve your issue by exporting the following
# environment variables:
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
. "${OTB_INSTALL_DIRNAME}/otbenv.profile"
. "${S1TILING_VENV}/bin/activate"
if [ "$1" = "--lia" ] ; then
    shift
    S1LIAMap "$@"
else
    S1Processor "$@"
fi
