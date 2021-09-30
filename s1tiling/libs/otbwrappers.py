#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2021 (c) CESBIO. All rights reserved.
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
This modules defines the specialized Python wrappers for the OTB Applications used in
the pipeline for S1Tiling needs.
"""

import logging
import os
import shutil
import re
import datetime

import numpy as np
from osgeo import gdal
import otbApplication as otb

from .otbpipeline import StepFactory, in_filename, out_filename, Step, AbstractStep, otb_version
from . import Utils
from ..__meta__ import __version__

logger = logging.getLogger('s1tiling')

re_tiff = re.compile(r'\.tiff?$')


def has_too_many_NoData(image, threshold, nodata):
    """
    Analyses whether an image contains NO DATA.

        :param image:     np.array image to analyse
        :param threshold: number of NoData searched
        :param nodata:    no data value
        :return:          whether the number of no-data pixel > threshold
    """
    nbNoData = len(np.argwhere(image == nodata))
    return nbNoData > threshold


class AnalyseBorders(StepFactory):
    """
    StepFactory that analyses whether image borders need to be cut as
    described in :ref:`Margins cutting` documentation.

    The step produced by this actual factory doesn't register any OTB
    application nor execute one.
    However, it loads two lines from the input image to determine whether it
    contains too many NoData.

    Found information will be stored into the `meta` dictionary for later use
    by :class:`CutBorders` step factory.
    """
    def __init__(self, cfg):
        """
        Constructor
        """
        super().__init__('', 'AnalyseBorders')
        self.__override_azimuth_cut_threshold_to = cfg.override_azimuth_cut_threshold_to

    def parameters(self, meta):
        """
        As there is no OTB application associated to :class:`AnalyseBorders`,
        no parameters are returned.
        """
        return None

    def output_directory(self, meta):
        """
        As there is no OTB application associated to :class:`AnalyseBorders`,
        there is no output directory.
        """
        raise TypeError("An AnalyseBorders step don't produce anything!")

    def build_step_output_filename(self, meta):
        """
        Forward the output filename.
        """
        return meta['out_filename']

    def build_step_output_tmp_filename(self, meta):
        """
        As there is no OTB application associated to :class:`AnalyseBorders`,
        there is no temporary filename.
        """
        return self.build_step_output_filename(meta)

    def complete_meta(self, meta):
        """
        Complete meta information with Cutting thresholds.
        """
        meta = super().complete_meta(meta)

        cut_overlap_range   = 1000  # Number of columns to cut on the sides. Here 500pixels = 5km
        cut_overlap_azimuth = 1600  # Number of lines to cut at the top or the bottom
        thr_nan_for_cropping = cut_overlap_range * 2  # When testing we having cut the NaN yet on the border hence this threshold.

        # With proper rasterio execution contexts, it would have been as clean as the following.
        # Alas RasterIO requires GDAL 2.x while OTB requires GDAL 3.x... => We cannot use rasterio.
        # with rasterio.open(meta['out_filename']) as ds_reader:
        #     xsize = ds_reader.width
        #     ysize = ds_reader.height
        #     north = ds_reader.read(1, window=Window(0, 100, xsize + 1, 1))
        #     south = ds_reader.read(1, window=Window(0, ysize - 100, xsize + 1, 1))


        if self.__override_azimuth_cut_threshold_to is None:
            ds_reader = gdal.Open(meta['out_filename'])
            xsize = ds_reader.RasterXSize
            ysize = ds_reader.RasterYSize
            north = ds_reader.ReadAsArray(0, 100, xsize, 1)
            south = ds_reader.ReadAsArray(0, ysize - 100, xsize, 1)
            crop1 = has_too_many_NoData(north, thr_nan_for_cropping, 0)
            crop2 = has_too_many_NoData(south, thr_nan_for_cropping, 0)
            del south
            del north
            del ds_reader
            south = None
            north = None
            ds_reader = None
        else:
            crop1 = self.__override_azimuth_cut_threshold_to
            crop2 = self.__override_azimuth_cut_threshold_to

        logger.debug("   => need to crop north: %s", crop1)
        logger.debug("   => need to crop south: %s", crop2)
        meta['cut'] = {
                'threshold.x'      : cut_overlap_range,
                'threshold.y.start': cut_overlap_azimuth if crop1 else 0,
                'threshold.y.end'  : cut_overlap_azimuth if crop2 else 0,
                }
        return meta


class Calibrate(StepFactory):
    """
    Factory that prepares steps that run
    :std:doc:`Applications/app_SARCalibration` as described in :ref:`SAR
    Calibration` documentation.

    Requires the following information from the configuration object:

    - `ram_per_process`
    - `calibration_type`
    - `removethermalnoise`

    Requires the following information from the metadata dictionary:

    - base name -- to generate typical output filename
    - input filename
    - output filename
    """
    def __init__(self, cfg):
        """
        Constructor
        """
        super().__init__('SARCalibration', 'Calibration')
        # Warning: config object cannot be stored and passed to workers!
        # => We extract what we need
        self.__ram_per_process    = cfg.ram_per_process
        self.__calibration_type   = cfg.calibration_type
        self.__removethermalnoise = cfg.removethermalnoise
        self.__tmpdir             = cfg.tmpdir

    def complete_meta(self, meta):
        """
        Complete GDAL metadata with the kind of calibration done.
        """
        meta = super().complete_meta(meta)
        meta['calibration_type'] = self.__calibration_type
        return meta

    def output_directory(self, meta):
        """
        :std:doc:`Applications/app_SARCalibration` result files would be
        stored in `{tmp}/S1` in case their production is required (i.e. not
        in-memory processing)
        """
        # tile_name = meta['tile_name'] # manifest maybe?
        return os.path.join(self.__tmpdir, 'S1')

    def build_step_output_filename(self, meta):
        """
        Returns the names of typical SARCalibration result files in case their
        production is required (i.e. not in-memory processing)
        """
        filename = meta['basename'].replace(".tiff", "_calOk.tiff")
        return os.path.join(self.output_directory(meta), filename)

    def build_step_output_tmp_filename(self, meta):
        """
        Returns the names of typical SARCalibration temporary result files in case their
        production is required (i.e. not in-memory processing)
        """
        filename = meta['basename'].replace(".tiff", "_calOk.tmp.tiff")
        return os.path.join(self.output_directory(meta), filename)

    def parameters(self, meta):
        """
        Returns the parameters to use with :std:doc:`SARCalibration OTB
        application <Applications/app_SARCalibration>`.
        """
        params = {
                'ram'           : str(self.__ram_per_process),
                # 'progress'    : 'false',
                self.param_in   : in_filename(meta),
                # self.param_out  : out_filename(meta),
                'lut'           : self.__calibration_type,
                }
        if otb_version() >= '7.4.0':
            params['removenoise'] = self.__removethermalnoise
        else:
            # Don't try to do anything, let's keep the noise
            params['noise']       = True
        return params


class CutBorders(StepFactory):
    """
    Factory that prepares steps that run
    :std:doc:`Applications/app_ResetMargin` as described in :ref:`Margins Cutting` documentation.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - base name -- to generate typical output filename
    - input filename
    - output filename
    - `cut`->`threshold.x`       -- from :class:`AnalyseBorders`
    - `cut`->`threshold.y.start` -- from :class:`AnalyseBorders`
    - `cut`->`threshold.y.end`   -- from :class:`AnalyseBorders`
    """
    def __init__(self, cfg):
        """
        Constructor.
        """
        super().__init__('ResetMargin', 'BorderCutting')
        self.__ram_per_process = cfg.ram_per_process
        self.__tmpdir          = cfg.tmpdir

    def output_directory(self, meta):
        """
        Border cutting result files would be stored in :file:`{tmp}/S1`.
        """
        # tile_name = meta['tile_name'] # manifest maybe?
        return os.path.join(self.__tmpdir, 'S1')

    def build_step_output_filename(self, meta):
        """
        Returns the names of typical ResetMargin result files:
        ``{basename}_OrthoReady.tiff``.
        """
        filename = meta['basename'].replace(".tiff", "_OrthoReady.tiff")
        return os.path.join(self.output_directory(meta), filename)

    def build_step_output_tmp_filename(self, meta):
        """
        Returns the names of typical ResetMargin temporary result files
        """
        filename = meta['basename'].replace(".tiff", "_OrthoReady.tmp.tiff")
        return os.path.join(self.output_directory(meta), filename)

    def parameters(self, meta):
        """
        Returns the parameters to use with :std:doc:`ResetMargin OTB
        application <Applications/app_ResetMargin>`.
        """
        params = {
                'ram'              : str(self.__ram_per_process),
                # 'progress'       : 'false',
                self.param_in      : in_filename(meta),
                # self.param_out     : out_filename(meta),
                'threshold.x'      : meta['cut']['threshold.x'],
                'threshold.y.start': meta['cut']['threshold.y.start'],
                'threshold.y.end'  : meta['cut']['threshold.y.end']
                }
        if otb_version() != '7.2.0':  # From 7.3.0 onward actually
            params['mode'] = 'threshold'
        return params


class OrthoRectify(StepFactory):
    """
    Factory that prepares steps that run
    :std:doc:`Applications/app_OrthoRectification` as described in
    :ref:`OrthoRectification` documentation.

    Requires the following information from the configuration object:

    - `ram_per_process`
    - `out_spatial_res`
    - `GeoidFile`
    - `grid_spacing`
    - `tmp_srtm_dir`

    Requires the following information from the metadata dictionary

    - base name -- to generate typical output filename
    - input filename
    - output filename
    - `manifest`
    - `tile_name`
    - `tile_origin`
    """
    def __init__(self, cfg):
        """
        Constructor.
        Extract and cache configuration options.
        """
        super().__init__(
                'OrthoRectification', 'OrthoRectification',
                param_in='io.in', param_out='io.out')
        self.__ram_per_process      = cfg.ram_per_process
        self.__out_spatial_res      = cfg.out_spatial_res
        self.__GeoidFile            = cfg.GeoidFile
        self.__grid_spacing         = cfg.grid_spacing
        self.__interpolation_method = cfg.interpolation_method
        self.__tmp_srtm_dir         = cfg.tmp_srtm_dir
        self.__tmpdir               = cfg.tmpdir
        # Some workaround when ortho is not sequenced long with calibration
        self.__calibration_type = cfg.calibration_type

    def output_directory(self, meta):
        """
        The output directory is a temporary directory with the name of the tile.
        """
        tile_name = meta['tile_name']
        return os.path.join(self.__tmpdir, 'S2', tile_name)

    def build_step_output_filename(self, meta):
        """
        The actual output filename will be built in :func:`complete_meta()`.
        """
        return None

    def build_step_output_tmp_filename(self, meta):
        """
        The actual temporary output filename will be built in
        :func:`complete_meta()`.
        """
        return None

    def complete_meta(self, meta):
        """
        Complete meta information such as filenames, GDAL metadata from
        information found in the current S1 image filename.
        """
        meta = super().complete_meta(meta)
        manifest                = meta['manifest']
        image                   = in_filename(meta)   # meta['in_filename']
        # image                   = meta['basename']
        tile_name               = meta['tile_name']
        tile_origin             = meta['tile_origin']
        logger.debug("OrthoRectify.complete_meta(%s) /// image: %s /// tile_name: %s", meta, image, tile_name)
        current_date            = Utils.get_date_from_s1_raster(image)
        current_polar           = Utils.get_polar_from_s1_raster(image)
        current_platform        = Utils.get_platform_from_s1_raster(image)
        # TODO: if the manifest is no longer here, we may need to look into the geom instead
        # It'd actually be better
        current_orbit_direction = Utils.get_orbit_direction(manifest)
        current_relative_orbit  = Utils.get_relative_orbit(manifest)
        out_utm_zone            = tile_name[0:2]
        out_utm_northern        = (tile_name[2] >= 'N')
        in_epsg                 = 4326
        out_epsg                = 32600 + int(out_utm_zone)
        if not out_utm_northern:
            out_epsg = out_epsg + 100

        x_coord, y_coord, _ = Utils.convert_coord([tile_origin[0]], in_epsg, out_epsg)[0]
        lrx, lry, _         = Utils.convert_coord([tile_origin[2]], in_epsg, out_epsg)[0]

        if not out_utm_northern and y_coord < 0:
            y_coord += 10000000.
            lry     += 10000000.

        working_directory = self.output_directory(meta)
        meta['flying_unit_code'] = current_platform
        meta['polarisation']     = current_polar
        meta['orbit_direction']  = current_orbit_direction
        meta['orbit']            = '{:0>3d}'.format(current_relative_orbit)
        meta['acquisition_time'] = current_date
        ortho_image_name_fmt = current_platform\
                + "_" + tile_name\
                + "_" + current_polar\
                + "_" + current_orbit_direction\
                + '_{:0>3d}'.format(current_relative_orbit)\
                + "_" + current_date\
                + ".%s"
        out_filename_fmt = os.path.join(working_directory, ortho_image_name_fmt)
        meta['out_filename']     = out_filename_fmt % ('tif', )
        # ortho product goes to tmp dir, it's perfect for the tmp file as well
        meta['out_tmp_filename'] = out_filename_fmt % ('tmp.tif', )
        spacing = self.__out_spatial_res
        logger.debug("from %s, lrx=%s, x_coord=%s, spacing=%s", tile_name, lrx, x_coord, spacing)
        meta['params.ortho'] = {
                'opt.ram'          : str(self.__ram_per_process),
                # 'progress'       : 'false',
                self.param_in      : in_filename(meta),
                # self.param_out     : out_filename,
                'interpolator'     : self.__interpolation_method,
                'outputs.spacingx' : spacing,
                'outputs.spacingy' : -spacing,
                'outputs.sizex'    : int(round(abs(lrx - x_coord) / spacing)),
                'outputs.sizey'    : int(round(abs(lry - y_coord) / spacing)),
                'opt.gridspacing'  : self.__grid_spacing,
                'map'              : 'utm',
                'map.utm.zone'     : int(out_utm_zone),
                'map.utm.northhem' : out_utm_northern,
                'outputs.ulx'      : x_coord,
                'outputs.uly'      : y_coord,
                'elev.dem'         : self.__tmp_srtm_dir,
                'elev.geoid'       : self.__GeoidFile
                }
        meta['out_extended_filename_complement'] = "?&writegeom=false&gdal:co:COMPRESS=DEFLATE"
        meta['post'] = meta.get('post', []) + [self.add_ortho_metadata]

        # Some workaround when ortho is not sequenced long with calibration
        meta['calibration_type'] = self.__calibration_type

        return meta

    def parameters(self, meta):
        """
        Returns the parameters to use with :std:doc:`OrthoRectification OTB
        application <Applications/app_OrthoRectification>`.

        The parameters have been precomputed in :func:`complete_meta()`.
        """
        return meta['params.ortho']

    def add_ortho_metadata(self, meta):
        """
        Post-application hook used to complete GDAL metadata.
        """
        fullpath = out_filename(meta)
        logger.debug('Set metadata in %s', fullpath)
        dst = gdal.Open(fullpath, gdal.GA_Update)

        dst.SetMetadataItem('S2_TILE_CORRESPONDING_CODE', meta['tile_name'])
        dst.SetMetadataItem('TIFFTAG_DATETIME',           str(datetime.datetime.now().strftime('%Y:%m:%d %H:%M:%S')))
        dst.SetMetadataItem('ORTHORECTIFIED',             'true')
        dst.SetMetadataItem('CALIBRATION',                str(meta['calibration_type']))
        dst.SetMetadataItem('SPATIAL_RESOLUTION',         str(self.__out_spatial_res))
        dst.SetMetadataItem('IMAGE_TYPE',                 'GRD')
        dst.SetMetadataItem('FLYING_UNIT_CODE',           meta['flying_unit_code'])
        dst.SetMetadataItem('POLARIZATION',               meta['polarisation'])
        dst.SetMetadataItem('ORBIT',                      meta['orbit'])
        dst.SetMetadataItem('ORBIT_DIRECTION',            meta['orbit_direction'])
        dst.SetMetadataItem('TIFFTAG_SOFTWARE',           'S1 Tiling v'+__version__)
        dst.SetMetadataItem('TIFFTAG_IMAGEDESCRIPTION',   'Orthorectified Sentinel-'+meta['flying_unit_code'][1:].upper()+' IW GRD on S2 tile')

        acquisition_time = meta['acquisition_time']
        date = acquisition_time[0:4] + ':' + acquisition_time[4:6] + ':' + acquisition_time[6:8]
        if acquisition_time[9] == 'x':
            date += ' 00:00:00'
        else:
            date += ' ' + acquisition_time[9:11] + ':' + acquisition_time[11:13] + ':' + acquisition_time[13:15]
        dst.SetMetadataItem('ACQUISITION_DATETIME', date)
        del dst


class Concatenate(StepFactory):
    """
    Factory that prepares steps that run
    :std:doc:`Applications/app_Synthetize` as described in
    :ref:`Concatenation` documentation.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """
    def __init__(self, cfg):
        super().__init__('Synthetize', 'Concatenation', param_in='il', param_out='out')
        self.__ram_per_process = cfg.ram_per_process
        self.__outdir          = cfg.output_preprocess
        self.__tmpdir          = cfg.tmpdir

    def tmp_directory(self, meta):
        """
        Directory used to store temporary files before they are renamed into
        their final version.
        """
        return os.path.join(self.__tmpdir, 'S2', meta['tile_name'])

    def output_directory(self, meta):
        """
        Concatenation result files will be stored in :file:`{out}/{tile_name}`
        """
        return os.path.join(self.__outdir, meta['tile_name'])

    def build_step_output_filename(self, meta):
        """
        Returns the names of Concatenation result files.
        Basename is precomputed in :func:`complete_meta()`.
        """
        filename = meta['basename']
        return os.path.join(self.output_directory(meta), filename)

    def build_step_output_tmp_filename(self, meta):
        """
        Returns the names of temporary Concatenation result files.
        Basename is precomputed in :func:`complete_meta()`.

        Unlike output, concatenation result goes into tmp.
        """
        filename = meta['basename']
        return os.path.join(self.tmp_directory(meta), re.sub(re_tiff, r'.tmp\g<0>', filename))

    def update_out_filename(self, meta, with_meta):
        # tester input list
        # basename = un truc ou l'autre en fonction nb inputs (old value ou taskname)
        # out_file = basename
        meta['basename']           = meta['task_basename']
        meta['out_filename']       = self.build_step_output_filename(meta)
        meta['out_tmp_filename']   = self.build_step_output_tmp_filename(meta)
        logger.debug("concatenation.out_tmp_filename for %s updated to %s", meta['task_name'], meta['out_filename'])

    def complete_meta(self, meta):
        """
        Precompute output basename from the input file(s).
        Makes sure the :std:doc:`Synthetize OTB application
        <Applications/app_Synthetize>` would compress its result file,
        through extended filename.

        In concatenation case, the task_name needs to be overridden to stay
        unique and common to all inputs.
        """
        meta = meta.copy()
        out_file = out_filename(meta)
        out_dir = self.output_directory(meta)
        if isinstance(out_file, list):
            logger.debug('Register files to remove after concatenation: %s', out_file)
            meta['files_to_remove'] = out_file
            _, out_file = os.path.split(out_file[0])
            task_basename = re.sub(r'(?<=t)\d+(?=\.)', lambda m: 'x' * len(m.group()), out_file)
            meta['basename'] = task_basename
            logger.debug("Concatenation result of %s goes into %s", out_file, meta['basename'])
            # meta['does_product_exist'] = lambda : os.path.isfile(task_name(meta))
        else:
            _, out_file = os.path.split(out_file)
            task_basename = re.sub(r'(?<=t)\d+(?=\.)', lambda m: 'x' * len(m.group()), out_file)
            meta['basename'] = out_file
            logger.debug("Only one file to concatenate, just move it (%s)", out_file)
        meta = super().complete_meta(meta)  # Needs a valid basename
        meta['task_basename'] = task_basename
        meta['task_name'] = os.path.join(out_dir, task_basename)
        meta['out_extended_filename_complement'] = "?&gdal:co:COMPRESS=DEFLATE"
        meta['post'] = meta.get('post', []) + [self.clear_ortho_tmp]
        meta['update_out_filename'] = self.update_out_filename

        # logger.debug("Concatenate.complete_meta(%s) /// task_name: %s /// out_file: %s", meta, meta['task_name'], out_file)
        return meta

    def clear_ortho_tmp(self, meta):
        """
        Takes care of removing the orthorectified subtiles from the temporary
        directory once the concatenation has been done.
        """
        if 'files_to_remove' in meta:
            logger.debug('Cleaning concatenated files: %s', meta['files_to_remove'])
            Utils.remove_files(meta['files_to_remove'])

    def create_step(self, input: Step, in_memory: bool, previous_steps):
        """
        :func:`create_step` is overridden in :class:`Concatenate` case in
        order to by-pass Concatenation in case there is only a single file.
        """
        # logger.debug('CONCAT::create_step(%s) -> %s', input.out_filename, len(input.out_filename))
        if isinstance(input.out_filename, list) and len(input.out_filename) == 1:
            # This situation should not happen any more, we now a single string as input.
            # The code is kept in case s1tiling kernel changes again.
            concat_in_filename = input.out_filename[0]
        elif isinstance(input.out_filename, str):
            concat_in_filename = input.out_filename
        else:
            return super().create_step(input, in_memory, previous_steps)
        # Back to a single file input case
        logger.debug('By-passing concatenation of %s as there is only a single orthorectified tile to concatenate.', concat_in_filename)
        meta = self.complete_meta(input.meta)
        res = AbstractStep(**meta)
        logger.debug('Renaming %s into %s', concat_in_filename, res.out_filename)
        if not meta.get('dryrun', False):
            shutil.move(concat_in_filename, res.out_filename)
        return res

    def parameters(self, meta):
        """
        Returns the parameters to use with :std:doc:`Synthetize OTB
        application <Applications/app_Synthetize>`.
        """
        return {
                'ram'              : str(self.__ram_per_process),
                # 'progress'       : 'false',
                self.param_in      : in_filename(meta),
                # self.param_out     : out_filename(meta),
                }


class BuildBorderMask(StepFactory):
    """
    Factory that prepares the first step that generates border maks as
    described in :ref:`Border mask generation` documentation.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """
    def __init__(self, cfg):
        """
        Constructor.
        """
        super().__init__('BandMath', 'BuildBorderMask', param_in='il', param_out='out')
        self.__ram_per_process = cfg.ram_per_process
        self.__tmpdir          = cfg.tmpdir

    def output_directory(self, meta):
        """
        If dumped to a file, the result shall go in a temporary directory.
        """
        tile_name = meta['tile_name']
        return os.path.join(self.__tmpdir, 'S2', tile_name)

    def build_step_output_filename(self, meta):
        """
        Provide the output filename.
        """
        filename = meta['basename'].replace(".tif", "_BorderMask_TMP.tif")
        return os.path.join(self.output_directory(meta), filename)

    def build_step_output_tmp_filename(self, meta):
        """
        Provide the typical temporary output filename.
        """
        filename = meta['basename'].replace(".tif", "_BorderMask_TMP.tmp.tif")
        return os.path.join(self.output_directory(meta), filename)

    def set_output_pixel_type(self, app, meta):
        """
        Force the output pixel type to ``UINT8``.
        """
        app.SetParameterOutputImagePixelType(self.param_out, otb.ImagePixelType_uint8)

    def parameters(self, meta):
        """
        Returns the parameters to use with :std:doc:`BandMath OTB application
        <Applications/app_BandMath>` for computing border mask.
        """
        params = {
                'ram'              : str(self.__ram_per_process),
                # 'progress'       : 'false',
                self.param_in      : [in_filename(meta)],
                # self.param_out     : out_filename(meta),
                'exp'              : 'im1b1==0?0:1'
                }
        # logger.debug('%s(%s)', self.appname, params)
        return params


class SmoothBorderMask(StepFactory):
    """
    Factory that prepares the first step that smoothes border maks as
    described in :ref:`Border mask generation` documentation.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """
    def __init__(self, cfg):
        super().__init__(
                'BinaryMorphologicalOperation', 'SmoothBorderMask',
                param_in='in', param_out='out')
        self.__ram_per_process = cfg.ram_per_process
        self.__outdir          = cfg.output_preprocess
        self.__tmpdir          = cfg.tmpdir

    def tmp_directory(self, meta):
        """
        Directory used to store temporary files before they are renamed into
        their final version.
        """
        return os.path.join(self.__tmpdir, 'S2', meta['tile_name'])

    def output_directory(self, meta):
        """
        SmoothBorderMask result files will be stored in
        :file:`{out}/{tile_name}`.
        """
        return os.path.join(self.__outdir, meta['tile_name'])

    def build_step_output_filename(self, meta):
        """
        Returns the names of :class:`SmoothBorderMask` result files.
        """
        filename = meta['basename'].replace(".tif", "_BorderMask.tif")
        return os.path.join(self.output_directory(meta), filename)

    def build_step_output_tmp_filename(self, meta):
        """
        Returns the names of :class:`SmoothBorderMask` temporary result files.
        """
        filename = meta['basename'].replace(".tif", "_BorderMask.tmp.tif")
        return os.path.join(self.tmp_directory(meta), filename)

    def set_output_pixel_type(self, app, meta):
        """
        Force the output pixel type to ``UINT8``.
        """
        app.SetParameterOutputImagePixelType(self.param_out, otb.ImagePixelType_uint8)

    def parameters(self, meta):
        """
        Returns the parameters to use with
        :std:doc:`BinaryMorphologicalOperation OTB application
        <Applications/app_BinaryMorphologicalOperation>` to smooth border
        masks.
        """
        return {
                'ram'                   : str(self.__ram_per_process),
                # 'progress'            : 'false',
                self.param_in           : in_filename(meta),
                # self.param_out          : out_filename(meta),
                'structype'             : 'ball',
                # 'structype.ball.xradius': 5,
                # 'structype.ball.yradius': 5 ,
                'xradius'               : 5,
                'yradius'               : 5 ,
                'filter'                : 'opening'
                }
