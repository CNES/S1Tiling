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
#          Fabien CONTIVAL (CS Group)
# =========================================================================

"""
This modules defines the specialized Python wrappers for binding all the OTB
Applications, and other external executables, used in the pipelines for
S1Tiling needs.
"""

from .s1_to_s2 import (
    ExtractSentinel1Metadata,
    AnalyseBorders,
    Calibrate,
    CorrectDenoising,
    CutBorders,
    _OrthoRectifierFactory,
    OrthoRectify,
    _ConcatenatorFactory,
    Concatenate,
    BuildBorderMask,
    SmoothBorderMask,
    SpatialDespeckle,
)

from .lia import (
    filter_LIA,
    AgglomerateDEMOnS2,
    ProjectDEMToS2Tile,
    ProjectGeoidToS2Tile,
    SumAllHeights,
    ComputeGroundAndSatPositionsOnDEM,
    ComputeGroundAndSatPositionsOnDEMFromEOF,
    ComputeNormalsOnS2,
    ComputeLIAOnS2,
    ApplyLIACalibration,

    SARDEMProjection,
    SARCartesianMeanEstimation,
    OrthoRectifyLIA,
    ComputeNormalsOnS1,
    ComputeLIAOnS1,
    ConcatenateLIA,
    SelectBestCoverage,
)

from .gamma_area import (
    ApplyGammaNaughtRTCCalibration,

    AgglomerateDEMOnS1,
    NaNifyNoData,
    ResampleDEM,
    ProjectGeoidToDEM,
    SARDEMProjectionImageEstimation,
    SARGammaAreaImageEstimation,
    OrthoRectifyGAMMA_AREA,
    ConcatenateGAMMA_AREA,
    SelectGammaNaughtAreaBestCoverage,
)
from .ia import (
    ComputeEllipsoidNormalsOnS2,
    ComputeIAOnS2,
    ComputeGroundAndSatPositionsOnEllipsoid,
)

__all__ = [
    "ExtractSentinel1Metadata",
    "AnalyseBorders",
    "Calibrate",
    "CorrectDenoising",
    "CutBorders",
    "_OrthoRectifierFactory",
    "OrthoRectify",
    "_ConcatenatorFactory",
    "Concatenate",
    "BuildBorderMask",
    "SmoothBorderMask",
    "SpatialDespeckle",
    "filter_LIA",
    "AgglomerateDEMOnS2",
    "ProjectDEMToS2Tile",
    "ProjectGeoidToS2Tile",
    "SumAllHeights",
    "ComputeGroundAndSatPositionsOnDEM",
    "ComputeGroundAndSatPositionsOnDEMFromEOF",
    "ComputeNormalsOnS2",
    "ComputeLIAOnS2",
    "ApplyLIACalibration",

    "SARDEMProjection",
    "SARCartesianMeanEstimation",
    "OrthoRectifyLIA",
    "ComputeNormalsOnS1",
    "ComputeLIAOnS1",
    "ConcatenateLIA",
    "SelectBestCoverage",

    "ComputeGroundAndSatPositionsOnEllipsoid",
    "ComputeEllipsoidNormalsOnS2",
    "ComputeIAOnS2",

    "ApplyGammaNaughtRTCCalibration",

    "AgglomerateDEMOnS1",
    "NaNifyNoData",
    "ResampleDEM",
    "ProjectGeoidToDEM",
    "SARDEMProjectionImageEstimation",
    "SARGammaAreaImageEstimation",
    "OrthoRectifyGAMMA_AREA",
    "ConcatenateGAMMA_AREA",
    "SelectGammaNaughtAreaBestCoverage",
]
