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
# Authors:
# - Thierry KOLECK (CNES)
# - Luc HERMITTE (CSGROUP)
#
# =========================================================================

"""Mocking related helper functions"""


import logging
import fnmatch
import os
import shutil
from typing import List

from .mock_data import FileDB
from .mock_otb  import OTBApplicationsMockContext, isfile, isdir, list_dirs, glob, dirname, makedirs

from s1tiling.libs.meta        import Meta, out_filename
from s1tiling.libs.steps       import _ProducerStep


def remove_dirs(dir_list) -> None:
    for dir in dir_list:
        if os.path.isdir(dir):
            logging.info("rm -r '%s'", dir)
            shutil.rmtree(dir)


def declare_know_files(
        mocker,
        known_files,
        known_dirs,
        tile              : str,
        patterns          : List,
        file_db           : FileDB,
        application_mocker: OTBApplicationsMockContext
) -> None:
    # logging.debug('declare_know_files(%s)', patterns)
    all_files = file_db.all_files() + file_db.all_annotations()
    # logging.debug('- all_files: %s', all_files)
    files = []
    for pattern in patterns:
        files += [fn for fn in all_files if fnmatch.fnmatch(fn, '*'+pattern+'*')]
    known_files.extend(files)
    demtmpdir = f"{file_db.tmpdir}/TMP_DEM"
    known_files.extend(
            map(lambda dem: f"{demtmpdir}/{dem}.hgt", file_db.TILE_DATA[tile]['dems'])
    )
    known_dirs.update([dirname(fn, 3) for fn in known_files])
    known_dirs.update([dirname(fn, 2) for fn in known_files])
    # known_dirs.update([dirname(fn, 1) for fn in known_files])
    known_files.extend(file_db.all_manifests())
    logging.debug('Mocking w/ %s --> %s', patterns, files)
    mocker.patch('s1tiling.libs.workspace.DEMWorkspace.tmpdemdir', lambda slf, dem_tile_info, dem_filename, geoid_file: demtmpdir)
    # Utils.list_dirs has been imported in S1FileManager. This is the one that needs patching!
    mocker.patch('s1tiling.libs.S1FileManager.list_dirs', lambda dir, pat : list_dirs(dir, pat, known_dirs, file_db.inputdir))
    mocker.patch('glob.glob',        lambda pat  : glob(pat, known_files))
    mocker.patch('os.path.isfile',   lambda file : isfile(file, known_files))
    mocker.patch('os.path.isdir',    lambda dir  : isdir(dir, known_dirs))
    mocker.patch('os.makedirs',      lambda dir, **kw  : makedirs(dir, known_dirs))
    def mock_rename(fr, to):
        logging.debug('Renaming: %s --> %s', fr, to)
        known_files.append(to)
        known_files.remove(fr)
    mocker.patch('os.rename',        lambda fr, to: mock_rename(fr, to))
    mocker.patch('os.path.getctime', lambda file : 0)
    # TODO: Test written meta data as well
    # mocker.patch('s1tiling.libs.otbwrappers.OrthoRectify.add_ortho_metadata',    lambda slf, mt, app : True)
    # mocker.patch('s1tiling.libs.otbwrappers.OrthoRectifyLIA.add_ortho_metadata', lambda slf, mt, app : True)
    # mocker.patch('s1tiling.libs.otbwrappers.OrthoRectifyGAMMA_AREA.add_ortho_metadata', lambda slf, mt, app : True)
    def mock_write_image_metadata(slf: _ProducerStep, dryrun: bool):
        img_meta = slf.meta.get('image_metadata', {})
        fullpath = out_filename(slf.meta)
        application_mocker.register_any_unexpected_image_metadata(img_meta, slf.pipeline_name, fullpath)

        logging.debug('Set metadata in %s', fullpath)
        for (kw, val) in img_meta.items():
            assert isinstance(val, (str, list)), f'GDAL metadata shall be strings or lists of strings. "{kw}" is a {val.__class__.__name__} (="{val}")'
            logging.debug(' - %s -> %s', kw, val)
    mocker.patch('s1tiling.libs.steps._ProducerStep._write_image_metadata',  mock_write_image_metadata)
    mocker.patch('s1tiling.libs.steps.commit_execution',    lambda tmp, out : True)
    mocker.patch('s1tiling.libs.Utils.get_origin',          lambda manifest : file_db.get_origin(manifest))
    mocker.patch('s1tiling.libs.Utils.get_orbit_direction', lambda manifest : file_db.get_orbit_direction(manifest))
    mocker.patch('s1tiling.libs.Utils.get_relative_orbit',  lambda manifest : file_db.get_relative_orbit(manifest))
    mocker.patch('s1tiling.libs.Utils.get_orbit_information',  lambda manifest : file_db.get_orbit_information(manifest))
    # Utils.get_orbit_direction has been imported in S1FileManager. This is the one that needs patching!
    mocker.patch('s1tiling.libs.S1FileManager.get_orbit_direction', lambda manifest : file_db.get_orbit_direction(manifest))
    mocker.patch('s1tiling.libs.S1FileManager.get_relative_orbit',  lambda manifest : file_db.get_relative_orbit(manifest))

    def mock_commit_execution_for_SelectLIA(inp, out):
        logging.debug('mock.mv %s %s', inp, out)
        assert os.path.isfile(inp)
        known_files.append(out)
        known_files.remove(inp)
    mocker.patch('s1tiling.libs.otbwrappers.lia.commit_execution', mock_commit_execution_for_SelectLIA)

    def mock_commit_execution_for_SelectGAMMA_AREA(inp, out):
        logging.debug('mock.mv %s %s', inp, out)
        assert os.path.isfile(inp)
        known_files.append(out)
        known_files.remove(inp)
    mocker.patch('s1tiling.libs.otbwrappers.gamma_area.commit_execution', mock_commit_execution_for_SelectGAMMA_AREA)

    def mock_add_image_metadata(slf, mt, *args, **kwargs):
        # TODO: Problem: how can we pass around meta from different pipelines???
        fullpath = mt.get('out_filename')
        logging.debug('Mock Set metadata in %s', fullpath)
        assert 'inputs' in mt, f'Looking for "inputs" in {mt.keys()}'
        inputs = mt['inputs']
        # indem = fetch_input_data('indem', inputs)
        assert 'dems' in mt, f"Metadata don't contain 'dems', only: {mt.keys()}"
        return mt
    mocker.patch('s1tiling.libs.otbwrappers.SARDEMProjection.add_image_metadata', mock_add_image_metadata)
    mocker.patch('s1tiling.libs.otbwrappers.SARDEMProjectionImageEstimation.add_image_metadata', mock_add_image_metadata)

    def mock_direction_to_scan(slf, meta: Meta) -> Meta:
        logging.debug('Mocking direction to scan')
        meta['directiontoscandeml'] = 12
        meta['directiontoscandemc'] = 24
        meta['gain']                = 42
        return meta
    mocker.patch('s1tiling.libs.otbwrappers.SARCartesianMeanEstimation.fetch_direction', lambda slf, ip, mt : mock_direction_to_scan(slf, mt))
    mocker.patch('s1tiling.libs.otbwrappers.SARGammaAreaImageEstimation.fetch_direction', lambda slf, ip, mt: mock_direction_to_scan(slf, mt))
    mocker.patch('s1tiling.libs.otbwrappers.lia._SumAllHeights.fetch_upstream_dem_resampling_method', lambda slf, ip, mt : 'cubic')

    def mock_fetch_nodata_value(inputpath, is_running_dry, default_value, band_nr:int = 1) -> float:
        return default_value
    # mocker.patch('s1tiling.libs.otbwrappers.lia.fetch_nodata_value', mock_fetch_nodata_value)
    mocker.patch('s1tiling.libs.Utils.fetch_nodata_value', mock_fetch_nodata_value)
