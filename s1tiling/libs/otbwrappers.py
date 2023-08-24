#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
# =========================================================================

"""
This modules defines the specialized Python wrappers for the OTB Applications used in
the pipeline for S1Tiling needs.
"""

import logging
import os
import shutil
import re
# import datetime
from abc import abstractmethod
from packaging import version

import numpy as np
from osgeo import gdal
import otbApplication as otb

from .otbpipeline import (StepFactory, _FileProducingStepFactory, OTBStepFactory,
        ExecutableStepFactory, in_filename, out_filename, tmp_filename,
        manifest_to_product_name, AbstractStep, otb_version, _check_input_step_type,
        _fetch_input_data, OutputFilenameGenerator,
        OutputFilenameGeneratorList, TemplateOutputFilenameGenerator,
        ReplaceOutputFilenameGenerator, commit_execution, is_running_dry,
        get_task_name, Step)
from . import Utils
from ..__meta__ import __version__

logger = logging.getLogger('s1tiling.wrappers')

def append_to(meta, key, value):
    """
    Helper function to append to a list that may be empty
    """
    meta[key] = meta.get(key, []) + [value]
    return meta


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


def extract_IPF_version(tifftag_software):
    """
    Extracts a comparable IPF version number
    for packaging.version.parse for instance.
    """
    match = re.search(r'Sentinel-1 IPF (\d+\.\d+)$', tifftag_software)
    if not match:
        logger.warning('Cannot extract IPF version from "%s"', tifftag_software)
        return '000.00'
    return match.group(1)


class ExtractSentinel1Metadata(StepFactory):
    """
    Factory that takes care of extracting meta data from S1 input files.

    Note: At this moment it needs to be used on a separate pipeline to make
    sure the meta is updated when calling :func:`PipelineDescription.expected`.
    """
    def __init__(self, cfg):
        super().__init__(cfg, 'ExtractSentinel1Metadata')

    def build_step_output_filename(self, meta):
        """
        Forward the output filename.
        """
        return meta['out_filename']

    def build_step_output_tmp_filename(self, meta):
        """
        As there is no OTB application associated to :class:`ExtractSentinel1Metadata`,
        there is no temporary filename.
        """
        return self.build_step_output_filename(meta)

    def _update_filename_meta_post_hook(self, meta):
        """
        Complete meta information such as filenames, GDAL metadata from
        information found in the current S1 image filename.
        """
        manifest                = meta['manifest']
        # image                   = in_filename(meta)   # meta['in_filename']
        image                   = meta['basename']

        # TODO: if the manifest is no longer here, we may need to look into the geom instead
        # It'd actually be better

        meta['origin_s1_image']  = meta['basename']  # Will be used to remember the reference image
        # meta['rootname']         = os.path.splitext(meta['basename'])[0]
        meta['flying_unit_code'] = Utils.get_platform_from_s1_raster(image)
        meta['polarisation']     = Utils.get_polar_from_s1_raster(image)
        meta['orbit_direction']  = Utils.get_orbit_direction(manifest)
        meta['orbit']            = '{:0>3d}'.format(Utils.get_relative_orbit(manifest))
        meta['acquisition_time'] = Utils.get_date_from_s1_raster(image)
        meta['acquisition_day']  = re.sub(r"(?<=t)\d+$", lambda m: "x" * len(m.group()), meta['acquisition_time'])

        meta['task_name']        = f'ExtractS1Meta_{meta["basename"]}'
        return meta

    def update_image_metadata(self, meta, all_inputs):  # pylint: disable=unused-argument
        """
        Set the common and root information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['IMAGE_TYPE']            = 'GRD'
        imd['FLYING_UNIT_CODE']      = meta['flying_unit_code']
        imd['ORBIT']                 = meta['orbit']
        imd['ORBIT_DIRECTION']       = meta['orbit_direction']
        imd['POLARIZATION']          = meta['polarisation']
        imd['TIFFTAG_SOFTWARE']      = 'S1 Tiling v'+__version__
        imd['INPUT_S1_IMAGES']       = manifest_to_product_name(meta['manifest'])
        # Only one input image at this point, we don't introduce any
        # ACQUISITION_DATETIMES or ACQUISITION_DATETIME_1...

        acquisition_time = meta['acquisition_time']
        date = f'{acquisition_time[0:4]}:{acquisition_time[4:6]}:{acquisition_time[6:8]}'
        if acquisition_time[9] == 'x':
            # This case should not happen, here
            date += ' 00:00:00'
        else:
            date += f' {acquisition_time[9:11]}:{acquisition_time[11:13]}:{acquisition_time[13:15]}'
        imd['ACQUISITION_DATETIME'] = date

    def _get_canonical_input(self, inputs):
        """
        Helper function to retrieve the canonical input associated to a list of inputs.

        :class:`ExtractSentinel1Metadata` can be used either in usual S1Tiling
        orthorectofication scenario, or in LIA Map generation scenarios.
        In the first case only a single and unnamed input is expected. In LIA
        case, several named inputs are expected, and the canonical input is
        named "insar" in :func:s1tiling.s1_process_lia` pipeline builder.
        """
        _check_input_step_type(inputs)
        keys = set().union(*(input.keys() for input in inputs))
        assert 1 <= len(inputs) <= 2, f'Expecting 1 or 2 inputs. {len(inputs)} are found: {keys}'
        if len(inputs) == 1:  # Usual case
            return list(inputs[0].values())[0]
        else:                 # LIA case
            assert 'insar' in keys
            return [input['insar'] for input in inputs if 'insar' in input.keys()][0]

    def complete_meta(self, meta, all_inputs):
        """
        Complete meta information with inputs
        """
        meta = super().complete_meta(meta, all_inputs)
        meta['inputs'] = all_inputs
        return meta


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
        super().__init__('AnalyseBorders')
        self.__override_azimuth_cut_threshold_to = cfg.override_azimuth_cut_threshold_to

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

    def complete_meta(self, meta, all_inputs):
        """
        Complete meta information with Cutting thresholds.
        """
        meta = super().complete_meta(meta, all_inputs)

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

        # Since 2.9 version of IPF S1, range borders are correctly generated
        # see: https://sentinels.copernicus.eu/documents/247904/2142675/Sentinel-1-masking-no-value-pixels-grd-products-note.pdf/32f11e6f-68b1-4f0a-869b-8d09f80e6788?t=1518545526000
        ds_reader = gdal.Open(meta['out_filename'], gdal.GA_ReadOnly)
        tifftag_software = ds_reader.GetMetadataItem('TIFFTAG_SOFTWARE')
        # Ex: Sentinel-1 IPF 003.10

        # TODO: The margin analysis must extract the width of ipf 2.9 margin correction.
        # see Issue #88
        #Temporary correction:
        #     The cut margin (right and left)  is done for any version of IPF

        #ipf_version = extract_IPF_version(tifftag_software)
        # if version.parse(ipf_version) >= version.parse('2.90'):
        #    cut_overlap_range = 0

        if self.__override_azimuth_cut_threshold_to is None:
            xsize = ds_reader.RasterXSize
            ysize = ds_reader.RasterYSize
            north = ds_reader.ReadAsArray(0, 100, xsize, 1)
            south = ds_reader.ReadAsArray(0, ysize - 100, xsize, 1)
            crop1 = has_too_many_NoData(north, thr_nan_for_cropping, 0)
            crop2 = has_too_many_NoData(south, thr_nan_for_cropping, 0)
            del south
            del north
            south = None
            north = None
        else:
            crop1 = self.__override_azimuth_cut_threshold_to
            crop2 = self.__override_azimuth_cut_threshold_to

        del ds_reader
        ds_reader = None

        logger.debug("   => need to crop north: %s", crop1)
        logger.debug("   => need to crop south: %s", crop2)

        thr_x   = cut_overlap_range
        thr_y_s = cut_overlap_azimuth if crop1 else 0
        thr_y_e = cut_overlap_azimuth if crop2 else 0

        meta['cut'] = {
                'threshold.x'      : cut_overlap_range,
                'threshold.y.start': thr_y_s,
                'threshold.y.end'  : thr_y_e,
                'skip'             : thr_x==0 and thr_y_s==0 and thr_y_e==0,
                }
        return meta


k_calib_convert = {'normlim' : 'beta'}

class Calibrate(OTBStepFactory):
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
        self.cfg=cfg
        fname_fmt = '{rootname}_{calibration_type}_calOk.tiff'
        fname_fmt = cfg.fname_fmt.get('calibration') or fname_fmt
        super().__init__(cfg,
                appname='SARCalibration',
                name='Calibration',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
                gen_output_dir=None,  # Use gen_tmp_dir
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description='{calibration_type} calibrated Sentinel-{flying_unit_code_short} IW GRD',
                )
        # Warning: config object cannot be stored and passed to workers!
        # => We extract what we need
        # Locally override calibration type in case of normlim calibration
        self.__calibration_type   = k_calib_convert.get(cfg.calibration_type, cfg.calibration_type)
        self.__removethermalnoise = cfg.removethermalnoise

    def _update_filename_meta_pre_hook(self, meta):
        """
        Injects the ``calibration_type`` in step metadata.
        """
        meta['calibration_type'] = self.__calibration_type

    def update_image_metadata(self, meta, all_inputs):  # pylint: disable=unused-argument
        """
        Set calibration related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['CALIBRATION']   = meta['calibration_type']
        imd['NOISE_REMOVED'] = str(self.__removethermalnoise)

    def parameters(self, meta):
        """
        Returns the parameters to use with :std:doc:`SARCalibration OTB
        application <Applications/app_SARCalibration>`.
        """
        params = {
                'ram'           : str(self.ram_per_process),
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


class CorrectDenoising(OTBStepFactory):
    """
    Factory that prepares steps that run
    :std:doc:`Applications/app_BandMath` as described in :ref:`SAR Calibration`
    documentation.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - base name -- to generate typical output filename
    - input filename
    - output filename
    - lower_signal_value
    """
    def __init__(self, cfg):
        """
        Constructor.
        """
        fname_fmt = '{rootname}_{calibration_type}_NoiseFixed.tiff'
        fname_fmt = cfg.fname_fmt.get('correct_denoising') or fname_fmt
        super().__init__(cfg,
                appname='BandMath', name='DenoisingCorrection', param_in='il', param_out='out',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
                gen_output_dir=None,  # Use gen_tmp_dir
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description='{calibration_type} calibrated Sentinel-{flying_unit_code_short} IW GRD with noise corrected',
                )
        self.__lower_signal_value = cfg.lower_signal_value

    def update_image_metadata(self, meta, all_inputs):  # pylint: disable=unused-argument
        """
        Set noise correction related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['LOWER_SIGNAL_VALUE'] = str(self.__lower_signal_value)

    def parameters(self, meta):
        """
        Returns the parameters to use with :std:doc:`BandMath OTB application
        <Applications/app_BandMath>` for changing 0.0 into lower_signal_value
        """
        params = {
                'ram'              : str(self.ram_per_process),
                self.param_in      : in_filename(meta),
                # self.param_out     : out_filename(meta),
                'exp'              : f'im1b1==0?{self.__lower_signal_value}:im1b1'
                }
        return params


class CutBorders(OTBStepFactory):
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
        fname_fmt = '{rootname}_{calibration_type}_OrthoReady.tiff'
        fname_fmt = cfg.fname_fmt.get('cut_borders') or fname_fmt
        super().__init__(cfg,
                appname='ResetMargin', name='BorderCutting',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
                gen_output_dir=None,  # Use gen_tmp_dir
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                )

    # def create_step(self, in_memory: bool, previous_steps):
    #     """
    #     This overrides checks whether ResetMargin would cut any border.
    #
    #     In the likelly other case, the method returns ``None`` to say **Don't
    #     register any OTB application and skip this step!**.
    #     """
    #     inputs = self._get_inputs(previous_steps)
    #     inp    = self._get_canonical_input(inputs)
    #     if inp.meta['cut'].get('skip', False):
    #         logger.debug('Margins cutting is not required and thus skipped!')
    #         return None
    #     else:
    #         return super().create_step(in_memory, previous_steps)

    def parameters(self, meta):
        """
        Returns the parameters to use with :std:doc:`ResetMargin OTB
        application <Applications/app_ResetMargin>`.
        """
        params = {
                'ram'              : str(self.ram_per_process),
                self.param_in      : in_filename(meta),
                # self.param_out     : out_filename(meta),
                'threshold.x'      : meta['cut']['threshold.x'],
                'threshold.y.start': meta['cut']['threshold.y.start'],
                'threshold.y.end'  : meta['cut']['threshold.y.end']
                }
        if otb_version() != '7.2.0':  # From 7.3.0 onward actually
            params['mode'] = 'threshold'
        return params


class _OrthoRectifierFactory(OTBStepFactory):
    """
    Abstract factory that prepares steps that run
    :std:doc:`Applications/app_OrthoRectification` as described in
    :ref:`OrthoRectification` documentation.

    This factory will be specialized for calibrated S1 images
    (:class:`OrthoRectify`), or LIA and sin-LIA maps (:class:`OrthoRectifyLIA`)

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
    def __init__(self, cfg, fname_fmt, image_description):
        """
        Constructor.
        Extract and cache configuration options.
        """
        super().__init__(cfg,
                appname='OrthoRectification', name='OrthoRectification',
                param_in='io.in', param_out='io.out',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
                gen_output_dir=None,      # Use gen_tmp_dir,
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description=image_description,
                )
        self.__out_spatial_res      = cfg.out_spatial_res
        self.__GeoidFile            = cfg.GeoidFile
        self.__grid_spacing         = cfg.grid_spacing
        self.__interpolation_method = cfg.interpolation_method
        self.__tmp_srtm_dir         = cfg.tmp_srtm_dir
        # self.__tmpdir               = cfg.tmpdir
        # Some workaround when ortho is not sequenced along with calibration
        # (and locally override calibration type in case of normlim calibration)
        self.__calibration_type     = k_calib_convert.get(cfg.calibration_type, cfg.calibration_type)

    def complete_meta(self, meta, all_inputs):
        """
        Complete meta information such as filenames, GDAL metadata from
        information found in the current S1 image filename.
        """
        meta = super().complete_meta(meta, all_inputs)
        meta['out_extended_filename_complement'] = "?&writegeom=false&gdal:co:COMPRESS=DEFLATE"

        # Some workaround when ortho is not sequenced along with calibration
        meta['calibration_type'] = self.__calibration_type
        return meta

    def update_image_metadata(self, meta, all_inputs):  # pylint: disable=unused-argument
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['S2_TILE_CORRESPONDING_CODE'] = meta['tile_name']
        imd['ORTHORECTIFIED']             = 'true'
        imd['SPATIAL_RESOLUTION']         = str(self.__out_spatial_res)

    @abstractmethod
    def _get_input_image(self, meta):
        raise TypeError("_OrthoRectifierFactory does not know how to fetch input image")

    def parameters(self, meta):
        """
        Returns the parameters to use with :std:doc:`OrthoRectification OTB
        application <Applications/app_OrthoRectification>`.
        """
        image                   = self._get_input_image(meta)
        tile_name               = meta['tile_name']
        tile_origin             = meta['tile_origin']
        logger.debug("%s.parameters(%s) /// image: %s /// tile_name: %s",
                self.__class__.__name__, meta, image, tile_name)
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

        spacing = self.__out_spatial_res
        logger.debug("from %s, lrx=%s, x_coord=%s, spacing=%s", tile_name, lrx, x_coord, spacing)
        parameters = {
                'opt.ram'          : str(self.ram_per_process),
                self.param_in      : image,
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
        return parameters


class OrthoRectify(_OrthoRectifierFactory):
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
        fname_fmt = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_time}_{calibration_type}.tif'
        fname_fmt = cfg.fname_fmt.get('orthorectification') or fname_fmt
        super().__init__(cfg, fname_fmt,
                image_description='{calibration_type} calibrated orthorectified Sentinel-{flying_unit_code_short} IW GRD',
                )

    def _get_input_image(self, meta):
        return in_filename(meta)   # meta['in_filename']


class _ConcatenatorFactory(OTBStepFactory):
    """
    Abstract factory that prepares steps that run
    :std:doc:`Applications/app_Synthetize` as described in
    :ref:`Concatenation` documentation.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg,
                appname='Synthetize', name='Concatenation',
                param_in='il', param_out='out',
                *args, **kwargs
                )

    def complete_meta(self, meta, all_inputs):
        """
        Precompute output basename from the input file(s).
        Makes sure the :std:doc:`Synthetize OTB application
        <Applications/app_Synthetize>` would compress its result file,
        through extended filename.

        In concatenation case, the task_name needs to be overridden to stay
        unique and common to all inputs.

        Also, inject files to remove
        """
        meta = super().complete_meta(meta, all_inputs)  # Needs a valid basename
        meta['out_extended_filename_complement'] = "?&gdal:co:COMPRESS=DEFLATE"

        # logger.debug("Concatenate.complete_meta(%s) /// task_name: %s /// out_file: %s", meta, meta['task_name'], out_file)
        in_file = in_filename(meta)
        if isinstance(in_file, list):
            logger.debug('Register files to remove after concatenation: %s', in_file)
            meta['files_to_remove'] = in_file
        else:
            logger.debug('DONT register single file to remove after concatenation: %s', in_file)
        return meta

    def update_image_metadata(self, meta, all_inputs):  # pylint: disable=unused-argument
        """
        Set concatenation related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        inp = self._get_canonical_input(all_inputs)
        if len(inp.input_metas) >= 2:
            product_names = sorted([manifest_to_product_name(m['manifest']) for m in inp.input_metas])
            imd['INPUT_S1_IMAGES']       = ', '.join(product_names)
            imd[f'ACQUISITION_DATETIME'] = '{YYYY}:{MM}:{DD} {hh}:{mm}:{ss}'.format_map(Utils.extract_product_start_time(os.path.basename(product_names[0])))
            for idx, pn in enumerate(product_names, start=1):
                imd[f'ACQUISITION_DATETIME_{idx}'] = '{YYYY}:{MM}:{DD} {hh}:{mm}:{ss}'.format_map(Utils.extract_product_start_time(os.path.basename(pn)))
        else:
            imd['INPUT_S1_IMAGES'] = manifest_to_product_name(meta['manifest'])

    def parameters(self, meta):
        """
        Returns the parameters to use with :std:doc:`Synthetize OTB
        application <Applications/app_Synthetize>`.
        """
        return {
                'ram'              : str(self.ram_per_process),
                self.param_in      : in_filename(meta),
                # self.param_out     : out_filename(meta),
                }

    def create_step(self, in_memory: bool, previous_steps):
        """
        :func:`create_step` is overridden in :class:`Concatenate` case in
        order to by-pass Concatenation in case there is only a single file.
        """
        inputs = self._get_inputs(previous_steps)
        inp    = self._get_canonical_input(inputs)
        # logger.debug('CONCAT::create_step(%s) -> %s', inp.out_filename, len(inp.out_filename))
        if isinstance(inp.out_filename, list) and len(inp.out_filename) == 1:
            # This situation should not happen any more, we now a single string as inp.
            # The code is kept in case s1tiling kernel changes again.
            concat_in_filename = inp.out_filename[0]
        elif isinstance(inp.out_filename, str):
            concat_in_filename = inp.out_filename
        else:
            return super().create_step(in_memory, previous_steps)
        # Back to a single file inp case
        logger.debug('By-passing concatenation of %s as there is only a single orthorectified tile to concatenate.', concat_in_filename)
        meta = self.complete_meta(inp.meta, inputs)
        res = AbstractStep(**meta)
        logger.debug('Renaming %s into %s', concat_in_filename, res.out_filename)
        if not meta.get('dryrun', False):
            shutil.move(concat_in_filename, res.out_filename)
        return res


class Concatenate(_ConcatenatorFactory):
    """
    Abstract factory that prepares steps that run
    :std:doc:`Applications/app_Synthetize` as described in
    :ref:`Concatenation` documentation.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """
    def __init__(self, cfg):
        # TODO: factorise this recurring test!
        calibration_is_done_in_S1 = cfg.calibration_type in ['sigma', 'beta', 'gamma', 'dn']
        if calibration_is_done_in_S1:
            # This is a required product that shall end-up in outputdir
            gen_output_dir=os.path.join(cfg.output_preprocess, '{tile_name}')
        else:
            # This is a temporary product that shall end-up in tmpdir
            gen_output_dir = None # use gen_tmp_dir
        fname_fmt = cfg.fname_fmt_concatenation
        # logger.debug('but ultimatelly fname_fmt is "%s" --> %s', fname_fmt, cfg.fname_fmt)
        self.__tname_fmt = cfg.fname_fmt_concatenation.replace('{acquisition_stamp}', '{acquisition_day}')
        super().__init__(cfg,
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
                gen_output_dir=gen_output_dir,
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description='{calibration_type} calibrated orthorectified Sentinel-{flying_unit_code_short} IW GRD',
                )

    def update_out_filename(self, meta, with_task_info):  # pylint: disable=unused-argument
        """
        This hook will be triggered everytime a new compatible input is added.
        The effect is quite unique to :class:`Concatenate` as the name of the
        output product depends on the number of inputs are their common
        acquisition date.
        """
        # logger.debug('UPDATING %s from %s', meta['task_name'], meta)
        was = meta['out_filename']
        meta['acquisition_stamp']  = meta['acquisition_day']
        meta['out_filename']       = self.build_step_output_filename(meta)
        meta['out_tmp_filename']   = self.build_step_output_tmp_filename(meta)
        meta['basename']           = self._get_nominal_output_basename(meta)
        logger.debug("concatenation.out_tmp_filename for %s updated to %s (previously: %s)", meta['task_name'], meta['out_filename'], was)
        # Remove acquisition_time that no longer makes sense
        meta.pop('acquisition_time', None)

    def _update_filename_meta_pre_hook(self, meta):
        in_file = out_filename(meta)
        if isinstance(in_file, list):
            meta['acquisition_stamp'] = meta['acquisition_day']
            # Remove acquisition_time that no longer makes sense
            meta.pop('acquisition_time', None)
            # logger.debug("Concatenation result of %s goes into %s", in_file, meta['basename'])
        else:
            meta['acquisition_stamp'] = meta['acquisition_time']
            logger.debug("Only one file to concatenate, just move it (%s)", in_file)

    def _update_filename_meta_post_hook(self, meta):
        """
        Make sure the task_name and the basename are updated
        """
        meta['task_name']     = os.path.join(
                self.output_directory(meta),
                TemplateOutputFilenameGenerator(self.__tname_fmt).generate(meta['basename'], meta))
        meta['basename']      = self._get_nominal_output_basename(meta)
        meta['update_out_filename'] = self.update_out_filename
        in_file               = out_filename(meta)
        if not isinstance(in_file, list):
            def check_product(meta):
                task_name       = get_task_name(meta)
                filename        = out_filename(meta)
                exist_task_name = os.path.isfile(task_name)
                exist_file_name = os.path.isfile(filename)
                logger.debug('Checking concatenation product:\n- %s => %s (task)\n- %s => %s (file)',
                        task_name, '∃' if exist_task_name else '∅',
                        filename,  '∃' if exist_file_name else '∅')
                return exist_task_name or exist_file_name
            meta['does_product_exist'] = lambda : check_product(meta)


# ----------------------------------------------------------------------
# Mask related applications

class BuildBorderMask(OTBStepFactory):
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
        super().__init__(cfg,
                appname='BandMath', name='BuildBorderMask', param_in='il', param_out='out',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
                gen_output_dir=None,  # Use gen_tmp_dir
                gen_output_filename=ReplaceOutputFilenameGenerator(['.tif', '_BorderMask_TMP.tif']),
                image_description='Orthorectified Sentinel-{flying_unit_code_short} IW GRD border mask S2 tile',
                )

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
                'ram'              : str(self.ram_per_process),
                self.param_in      : [in_filename(meta)],
                # self.param_out     : out_filename(meta),
                'exp'              : 'im1b1==0?0:1'
                }
        # logger.debug('%s(%s)', self.appname, params)
        return params


class SmoothBorderMask(OTBStepFactory):
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
        super().__init__(cfg,
                appname='BinaryMorphologicalOperation', name='SmoothBorderMask',
                param_in='in', param_out='out',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
                gen_output_dir=os.path.join(cfg.output_preprocess, '{tile_name}'),
                gen_output_filename=ReplaceOutputFilenameGenerator(['.tif', '_BorderMask.tif']),
                image_description='Orthorectified Sentinel-{flying_unit_code_short} IW GRD smoothed border mask S2 tile',
                )

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
                'ram'                   : str(self.ram_per_process),
                self.param_in           : in_filename(meta),
                # self.param_out          : out_filename(meta),
                'structype'             : 'ball',
                'xradius'               : 5,
                'yradius'               : 5 ,
                'filter'                : 'opening'
                }


# ----------------------------------------------------------------------
# Despeckling related applications
class SpatialDespeckle(OTBStepFactory):
    """
    Factory that prepares the first step that smoothes border maks as
    described in :ref:`Spatial despeckle filtering <spatial-despeckle>`
    documentation.

    Requires the following information from the configuration object:

    - `ram_per_process`
    - `fname_fmt_filtered`
    - `filter`:  the name of the filter method
    - `rad`:     the filter windows radius
    - `nblooks`: the number of looks
    - `deramp`:  the deramp factor

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    - the keys used to generate the filename: `flying_unit_code`, `tile_name`,
      `orbit_direction`, `orbit`, `calibration_type`, `acquisition_stamp`,
      `polarisation`...
    """

    # TODO: (#118) This Step doesn't support yet in-memory chaining after Concatenation.
    # Indeed: Concatenation is a non trival step that:
    # - recognizes 2 compatibles inputs and changes the output filename in consequence
    # - rename orthorectification output when there is only one input.
    #
    # To be chained in-memory SpatialDespeckle would need to:
    # - recognize 2 compatibles inputs and change the output filename in consequence
    # - start from the renamed file (instead of expecting to be chained in-memory) when there is only one input.
    def __init__(self, cfg):
        fname_fmt = cfg.fname_fmt_filtered
        super().__init__(cfg,
                appname='Despeckle', name='Despeckle',
                param_in='in', param_out='out',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
                # TODO: Offer an option to choose output directory name scheme
                # TODO: synchronize with S1FileManager.ensure_tile_workspaces_exist()
                gen_output_dir=os.path.join(cfg.output_preprocess, 'filtered', '{tile_name}'),
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description='Orthorectified and despeckled Sentinel-{flying_unit_code_short} IW GRD S2 tile',
                )
        self.__filter  = cfg.filter
        self.__rad     = cfg.filter_options.get('rad', 0)
        self.__nblooks = cfg.filter_options.get('nblooks', 0)
        self.__deramp  = cfg.filter_options.get('deramp', 0)

        assert self.__rad
        # filter in list => nblooks != 0 ~~~> filter not in list OR nblooks != 0
        assert (self.__filter not in ['lee', 'gammamap', 'kuan']) or (self.__nblooks != 0) \
                , f'Unexpected nblooks value ({self.__nblooks} for {self.__filter} despeckle filter'
        # filter in list => deramp != 0 ~~~> filter not in list OR deramp != 0
        assert (self.__filter not in ['frost']) or (self.__deramp != 0.) \
                , f'Unexpected deramp value ({self.__deramp} for {self.__filter} despeckle filter'
        assert (self.__nblooks != 0.) != (self.__deramp != 0.)

    # def set_output_pixel_type(self, app, meta):
    #     """
    #     Force the output pixel type to ``UINT8``.
    #     """
    #     app.SetParameterOutputImagePixelType(self.param_out, otb.ImagePixelType_uint8)

    def _update_filename_meta_post_hook(self, meta):
        """
        Register ``is_compatible`` hook for
        :func:`s1tiling.libs.otbpipeline.is_compatible`.
        It will tell in the case Despeckle is chained in memory after
        ApplyLIACalibration whether a given sin_LIA input is compatible with
        the current S2 tile.
        """
        # TODO find a better way to reuse the hook from the previous step in case it's chained in memory!
        meta['is_compatible'] = lambda input_meta : self._is_compatible(meta, input_meta)

    def _is_compatible(self, output_meta, input_meta):
        """
        Tells in the case Despeckle is chained in memory after
        ApplyLIACalibration whether a given sin_LIA input is compatible with
        the current S2 tile.

        ``flying_unit_code``, ``tile_name``, ``orbit_direction`` and ``orbit``
        have to be identical.
        """
        fields = ['flying_unit_code', 'tile_name', 'orbit_direction', 'orbit']
        return all(input_meta[k] == output_meta[k] for k in fields)

    def complete_meta(self, meta, all_inputs):
        """
        Complete meta information with inputs, and set compression method to
        DEFLATE.
        """
        meta = super().complete_meta(meta, all_inputs)
        meta['out_extended_filename_complement'] = "?&gdal:co:COMPRESS=DEFLATE"
        return meta

    def update_image_metadata(self, meta, all_inputs):  # pylint: disable=unused-argument
        """
        Set despeckling related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['FILTERED'] = 'true'
        imd['FILTERING_METHOD']        = self.__filter
        imd['FILTERING_WINDOW_RADIUS'] = str(self.__rad)
        if self.__deramp:
            imd['FILTERING_DERAMP']    = str(self.__deramp)
        if self.__nblooks:
            imd['FILTERING_NBLOOKS']   = str(self.__nblooks)

    def parameters(self, meta):
        """
        Returns the parameters to use with
        :std:doc:`Despeckle OTB application
        <Applications/app_Despeckle>` to perform speckle noise reduction.
        """
        assert self.__rad
        params = {
                'ram'                         : str(self.ram_per_process),
                self.param_in                 : in_filename(meta),
                # self.param_out              : out_filename(meta),
                'filter'                      : self.__filter,
                f'filter.{self.__filter}.rad' : self.__rad,
                }
        if self.__nblooks:
            params[f'filter.{self.__filter}.nblooks'] = self.__nblooks
        if self.__deramp:
            params[f'filter.{self.__filter}.deramp']  = self.__deramp
        return params


# ======================================================================
# Applications used to produce LIA

def remove_polarization_marks(name):
    """
    Clean filename of any specific polarization mark like ``vv``, ``vh``, or
    the ending in ``-001`` and ``002``.
    """
    # (?=  marks a 0-length match to ignore the dot
    return re.sub(r'[hv][hv]-|[HV][HV]_|-00[12](?=\.)', '', name)


class AgglomerateDEM(ExecutableStepFactory):
    """
    Factory that produces a :class:`Step` that build a VRT from a list of DEM files.

    The choice has been made to name the VRT file after the basename of the
    root S1 product and not the names of the DEM tiles.
    """

    def __init__(self, cfg, *args, **kwargs):
        """
        constructor
        """
        fname_fmt = 'DEM_{polarless_rootname}.vrt'
        fname_fmt = cfg.fname_fmt.get('dem_s1_agglomeration') or fname_fmt
        super().__init__(cfg,
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
                gen_output_dir=None,      # Use gen_tmp_dir,
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                name="AgglomerateDEM", exename='gdalbuildvrt',
                *args, **kwargs)
        self.__srtm_db_filepath = cfg.srtm_db_filepath
        self.__srtm_dir         = cfg.srtm

    def _update_filename_meta_pre_hook(self, meta):
        """
        Injects the :func:`reduce_inputs_insar` hook in step metadata, and
        provide names clear from polar related informations.
        """
        # Ignore polarization in filenames
        assert 'polarless_basename' not in meta
        meta['polarless_basename'] = remove_polarization_marks(meta['basename'])
        rootname = os.path.splitext(meta['polarless_basename'])[0]
        meta['polarless_rootname'] = rootname
        meta['reduce_inputs_insar'] = lambda inputs : [inputs[0]] # TODO!!!

    def complete_meta(self, meta, all_inputs):
        """
        Factory that takes care of extracting meta data from S1 input files.
        """
        meta = super().complete_meta(meta, all_inputs)
        # find DEMs that intersect the input image
        meta['srtms'] = sorted(Utils.find_srtm_intersecting_raster(
            in_filename(meta), self.__srtm_db_filepath))
        logger.debug("SRTM found for %s: %s", in_filename(meta), meta['srtms'])
        return meta

    def parameters(self, meta):
        # While it won't make much a difference here, we are still using
        # tmp_filename.
        return [tmp_filename(meta)] + [os.path.join(self.__srtm_dir, s+'.hgt') for s in meta['srtms']]


class SARDEMProjection(OTBStepFactory):
    """
    Factory that prepares steps that run :std:doc:`Applications/app_SARDEMProjection`
    as described in :ref:`Normals computation` documentation.

    :std:doc:`Applications/app_SARDEMProjection` application puts a DEM file
    into SAR geometry and estimates two additional coordinates.
    For each point of the DEM input four components are calculated:
    C (colunm into SAR image), L (line into SAR image), Z and Y. XYZ cartesian
    components into projection are also computed for our needs.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename

    It also requires :envvar:`$OTB_GEOID_FILE` to be set in order to ignore any
    DEM information already registered in dask worker (through
    :std:doc:`Applications/app_OrthoRectification` for instance) and only use
    the Geoid.
    """
    def __init__(self, cfg):
        fname_fmt = 'S1_on_DEM_{polarless_basename}'
        fname_fmt = cfg.fname_fmt.get('s1_on_dem') or fname_fmt
        super().__init__(cfg,
                appname='SARDEMProjection', name='SARDEMProjection',
                param_in=None, param_out='out',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
                gen_output_dir=None,  # Use gen_tmp_dir
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description="SARDEM projection onto DEM list",
                )
        self.__srtm_db_filepath = cfg.srtm_db_filepath

    def _update_filename_meta_pre_hook(self, meta):
        """
        Injects the :func:`reduce_inputs_insar` hook in step metadata, and
        provide names clear from polar related informations.
        """
        # Ignore polarization in filenames
        if 'polarless_basename' in meta:
            assert meta['polarless_basename'] == remove_polarization_marks(meta['basename'])
        else:
            meta['polarless_basename'] = remove_polarization_marks(meta['basename'])

        meta['reduce_inputs_insar'] = lambda inputs : [inputs[0]] # TODO!!!

    def complete_meta(self, meta, all_inputs):
        """
        Complete meta information with hook for updating image metadata
        w/ directiontoscandemc, directiontoscandeml and gain.
        """
        meta = super().complete_meta(meta, all_inputs)
        append_to(meta, 'post', self.add_image_metadata)
        assert 'inputs' in meta, "Meta data shall have been filled with inputs"
        # meta['inputs'] = all_inputs

        # TODO: The following has been duplicated from AgglomerateDEM.
        # See to factorize this code
        # find DEMs that intersect the input image
        meta['srtms'] = sorted(Utils.find_srtm_intersecting_raster(
            in_filename(meta), self.__srtm_db_filepath))
        logger.debug("SARDEMProjection: SRTM found for %s: %s", in_filename(meta), meta['srtms'])
        _, inbasename = os.path.split(in_filename(meta))
        meta['inbasename'] = inbasename
        return meta

    def update_image_metadata(self, meta, all_inputs):  # pylint: disable=unused-argument
        """
        Set SARDEMProjection related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['POLARIZATION'] = ""  # Clear polarization information (makes no sense here)
        imd['DEM_LIST']     = ', '.join(meta['srtms'])

    def add_image_metadata(self, meta, app):  # pylint: disable=no-self-use
        """
        Post-application hook used to complete GDAL metadata.

        As :func:`update_image_metadata` is not designed to access OTB
        application information (``directiontoscandeml``...), we need this
        extra hook to fetch and propagate the PRJ information.
        """
        fullpath = out_filename(meta)
        logger.debug('Set metadata in %s', fullpath)
        dst = gdal.Open(fullpath, gdal.GA_Update)
        assert dst

        # Pointless here! :(
        assert app
        meta['directiontoscandeml'] = app.GetParameterInt('directiontoscandeml')
        meta['directiontoscandemc'] = app.GetParameterInt('directiontoscandemc')
        meta['gain']                = app.GetParameterFloat('gain')
        dst.SetMetadataItem('PRJ.DIRECTIONTOSCANDEML', str(meta['directiontoscandeml']))
        dst.SetMetadataItem('PRJ.DIRECTIONTOSCANDEMC', str(meta['directiontoscandemc']))
        dst.SetMetadataItem('PRJ.GAIN',                str(meta['gain']))
        dst.FlushCache()  # We really need to be sure it has been flushed now, if not closed
        del dst

    def parameters(self, meta):
        """
        Returns the parameters to use with
        :std:doc:`SARDEMProjection OTB application
        <Applications/app_SARDEMProjection>` to project S1 geometry onto DEM tiles.
        """
        nodata = meta.get('nodata', -32768)
        assert 'inputs' in meta, f'Looking for "inputs" in {meta.keys()}'
        inputs = meta['inputs']
        indem = _fetch_input_data('indem', inputs).out_filename
        return {
                'ram'        : str(self.ram_per_process),
                'insar'      : in_filename(meta),
                'indem'      : indem,
                'withxyz'    : True,
                # 'withh'      : True,  # uncomment to analyse/debug height computed
                'nodata'     : nodata
                }

    def requirement_context(self):
        """
        Return the requirement context that permits to fix missing requirements.
        SARDEMProjection comes from DiapOTB.
        """
        return "Please install DiapOTB."


class SARCartesianMeanEstimation(OTBStepFactory):
    """
    Factory that prepares steps that run
    :std:doc:`Applications/app_SARCartesianMeanEstimation` as described in
    :ref:`Normals computation` documentation.


    :std:doc:`Applications/app_SARCartesianMeanEstimation` estimates a simulated
    cartesian mean image thanks to a DEM file.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename

    Note: It cannot be chained in memory because of the ``directiontoscandem*`` parameters.
    """
    def __init__(self, cfg):
        fname_fmt = 'XYZ_{polarless_basename}'
        fname_fmt = cfg.fname_fmt.get('xyz') or fname_fmt
        super().__init__(cfg,
                appname='SARCartesianMeanEstimation2', name='SARCartesianMeanEstimation',
                param_in=None, param_out='out',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
                gen_output_dir=None,  # Use gen_tmp_dir
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description='Cartesian XYZ coordinates estimation',
                )

    def _update_filename_meta_pre_hook(self, meta):
        """
        Injects the :func:`reduce_inputs_insar` hook in step metadata, and
        provide names clear from polar related informations.
        """
        # Ignore polarization in filenames
        if 'polarless_basename' in meta:
            assert meta['polarless_basename'] == remove_polarization_marks(meta['basename'])
        else:
            meta['polarless_basename'] = remove_polarization_marks(meta['basename'])
        meta['reduce_inputs_insar'] = lambda inputs : [inputs[0]] # TODO!!!

    def _get_canonical_input(self, inputs):
        """
        Helper function to retrieve the canonical input associated to a list of inputs.

        In :class:`SARCartesianMeanEstimation` case, the canonical input comes
        from the "indem" pipeline defined in :func:s1tiling.s1_process_lia`
        pipeline builder.
        """
        _check_input_step_type(inputs)
        keys = set().union(*(input.keys() for input in inputs))
        assert len(inputs) == 3, f'Expecting 3 inputs. {len(inputs)} are found: {keys}'
        assert 'indemproj' in keys
        return [input['indemproj'] for input in inputs if 'indemproj' in input.keys()][0]

    def complete_meta(self, meta, all_inputs):
        """
        Complete meta information with hook for updating image metadata
        w/ directiontoscandemc, directiontoscandeml and gain.
        """
        inputpath = out_filename(meta)  # needs to be done before super.complete_meta!!
        meta = super().complete_meta(meta, all_inputs)
        meta['inputs'] = all_inputs
        if 'directiontoscandeml' not in meta or 'directiontoscandemc' not in meta:
            self.fetch_direction(inputpath, meta)
        indem     = _fetch_input_data('indem',     all_inputs).out_filename
        indemproj = _fetch_input_data('indemproj', all_inputs).out_filename
        meta['files_to_remove'] = [indem, indemproj]
        _, inbasename = os.path.split(in_filename(meta))
        meta['inbasename'] = inbasename
        logger.debug('Register files to remove after XYZ computation: %s', meta['files_to_remove'])
        return meta

    def update_image_metadata(self, meta, all_inputs):  # pylint: disable=unused-argument
        """
        Set SARCartesianMeanEstimation related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        # Clear PRJ.* information: makes no sense anymore
        imd['PRJ.DIRECTIONTOSCANDEML'] = ""
        imd['PRJ.DIRECTIONTOSCANDEMC'] = ""
        imd['PRJ.GAIN']                = ""

    def fetch_direction(self, inputpath, meta):  # pylint: disable=no-self-use
        """
        Extract back direction to scan DEM from SARDEMProjected image metadata.
        """
        logger.debug("Fetch PRJ.DIRECTIONTOSCANDEM* from '%s'", inputpath)
        if not is_running_dry(meta):
            dst = gdal.Open(inputpath, gdal.GA_ReadOnly)
            if not dst:
                raise RuntimeError(f"Cannot open SARDEMProjected file '{inputpath}' to collect scan direction metadata.")
            meta['directiontoscandeml'] = dst.GetMetadataItem('PRJ.DIRECTIONTOSCANDEML')
            meta['directiontoscandemc'] = dst.GetMetadataItem('PRJ.DIRECTIONTOSCANDEMC')
            if meta['directiontoscandeml'] is None or meta['directiontoscandemc'] is None:
                raise RuntimeError(f"Cannot fetch direction to scan from SARDEMProjected file '{inputpath}'")
            del dst
        else:
            meta['directiontoscandeml'] = 42
            meta['directiontoscandemc'] = 42

    def parameters(self, meta):
        """
        Returns the parameters to use with
        :std:doc:`SARCartesianMeanEstimation OTB application
        <Applications/app_SARCartesianMeanEstimation>` to compute cartesian
        coordinates of each point of the origin S1 image.
        """
        assert 'inputs' in meta, f'Looking for "inputs" in {meta.keys()}'
        inputs = meta['inputs']
        insar     = _fetch_input_data('insar', inputs).out_filename
        indem     = _fetch_input_data('indem', inputs).out_filename
        indemproj = _fetch_input_data('indemproj', inputs).out_filename
        return {
                'ram'             : str(self.ram_per_process),
                'insar'           : insar,
                'indem'           : indem,
                'indemproj'       : indemproj,
                'indirectiondemc' : int(meta['directiontoscandemc']),
                'indirectiondeml' : int(meta['directiontoscandeml']),
                'mlran'           : 1,
                'mlazi'           : 1,
                }

    def requirement_context(self):
        """
        Return the requirement context that permits to fix missing requirements.
        SARCartesianMeanEstimation comes from DiapOTB.
        """
        return "Please install DiapOTB."


class ComputeNormals(OTBStepFactory):
    """
    Factory that prepares steps that run
    :std:doc:`ExtractNormalVector <Applications/app_ExtractNormalVector>`
    as described in :ref:`Normals computation` documentation.


    :std:doc:`ExtractNormalVector <Applications/app_ExtractNormalVector>`
    computes surface normals.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """
    def __init__(self, cfg):
        fname_fmt = 'Normals_{polarless_basename}'
        fname_fmt = cfg.fname_fmt.get('normals') or fname_fmt
        super().__init__(cfg,
                appname='ExtractNormalVector', name='ComputeNormals',
                param_in='xyz', param_out='out',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
                gen_output_dir=None,  # Use gen_tmp_dir
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description='Image normals on Sentinel-{flying_unit_code_short} IW GRD',
                )

    def _update_filename_meta_pre_hook(self, meta):
        """
        Injects the :func:`reduce_inputs_insar` hook in step metadata.
        """
        # Ignore polarization in filenames
        assert 'polarless_basename' in meta
        assert meta['polarless_basename'] == remove_polarization_marks(meta['basename'])

    def complete_meta(self, meta, all_inputs):
        """
        Override :func:`complete_meta()` to inject files to remove
        """
        meta = super().complete_meta(meta, all_inputs)
        in_file = in_filename(meta)
        meta['files_to_remove'] = [in_file]
        logger.debug('Register files to remove after normals computation: %s', meta['files_to_remove'])
        return meta

    def parameters(self, meta):
        """
        Returns the parameters to use with
        :std:doc:`ExtractNormalVector OTB application
        <Applications/app_ExtractNormalVector>` to generate surface normals
        for each point of the origin S1 image.
        """
        nodata = meta.get('nodata', -32768)
        xyz = in_filename(meta)
        return {
                'ram'             : str(self.ram_per_process),
                'xyz'             : xyz,
                'nodata'          : float(nodata),
                }

    def requirement_context(self):
        """
        Return the requirement context that permits to fix missing requirements.
        ComputeNormals comes from normlim_sigma0.
        """
        return "Please install https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0."


class ComputeLIA(OTBStepFactory):
    """
    Factory that prepares steps that run
    :std:doc:`SARComputeLocalIncidenceAngle <Applications/app_SARComputeLocalIncidenceAngle>`
    as described in :ref:`Normals computation` documentation.


    :std:doc:`SARComputeLocalIncidenceAngle <Applications/app_SARComputeLocalIncidenceAngle>`
    computes Local Incidende Angle Map.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """
    def __init__(self, cfg):
        fname_fmt_lia = cfg.fname_fmt.get('s1_lia')     or 'LIA_{polarless_basename}'
        fname_fmt_sin = cfg.fname_fmt.get('s1_sin_lia') or 'sin_LIA_{polarless_basename}'
        fname_fmt = [
                TemplateOutputFilenameGenerator(fname_fmt_lia),
                TemplateOutputFilenameGenerator(fname_fmt_sin)]
        super().__init__(cfg,
                appname='SARComputeLocalIncidenceAngle', name='ComputeLIA',
                # In-memory connected to in.normals
                param_in='in.normals', param_out=['out.lia', 'out.sin'],
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
                gen_output_dir=None,  # Use gen_tmp_dir
                gen_output_filename=OutputFilenameGeneratorList(fname_fmt),
                image_description='LIA on Sentinel-{flying_unit_code_short} IW GRD',
                )

    def _update_filename_meta_pre_hook(self, meta):
        """
        Injects the :func:`reduce_inputs_insar` hook in step metadata.
        """
        assert 'polarless_basename' in meta
        assert meta['polarless_basename'] == remove_polarization_marks(meta['basename'])

    def _update_filename_meta_post_hook(self, meta):
        """
        Override "does_product_exist" hook to take into account the multiple
        output files produced by ComputeLIA
        """
        meta['does_product_exist'] = lambda : all(os.path.isfile(of) for of in out_filename(meta))
        return meta

    def complete_meta(self, meta, all_inputs):
        """
        Complete meta information with inputs
        """
        meta = super().complete_meta(meta, all_inputs)
        meta['inputs'] = all_inputs
        return meta

    def _get_inputs(self, previous_steps):
        """
        Extract the last inputs to use at the current level from all previous
        products seens in the pipeline.

        This method will is overridden in order to fetch N-1 "xyz" input.
        It has been specialized for S1Tiling exact pipelines.
        """
        assert len(previous_steps) > 1

        # "normals" is expected at level -1, likelly named '__last'
        normals = _fetch_input_data('__last', previous_steps[-1])
        # "xyz"     is expected at level -2, likelly named 'xyz'
        xyz = _fetch_input_data('xyz', previous_steps[-2])

        inputs = [{'normals': normals, 'xyz': xyz}]
        _check_input_step_type(inputs)
        return inputs

    def parameters(self, meta):
        """
        Returns the parameters to use with
        :std:doc:`SARComputeLocalIncidenceAngle OTB application
        <Applications/app_SARComputeLocalIncidenceAngle>`.
        """
        assert 'inputs' in meta, f'Looking for "inputs" in {meta.keys()}'
        inputs = meta['inputs']
        xyz     = _fetch_input_data('xyz', inputs).out_filename
        normals = _fetch_input_data('normals', inputs).out_filename
        nodata  = meta.get('nodata', -32768)
        return {
                'ram'             : str(self.ram_per_process),
                'in.xyz'          : xyz,
                'in.normals'      : normals,
                'nodata'          : float(nodata),
                }

    def requirement_context(self):
        """
        Return the requirement context that permits to fix missing requirements.
        ComputeLIA comes from normlim_sigma0.
        """
        return "Please install https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0."


class _FilterStepFactory(StepFactory):
    """
    Helper root class for all LIA/sin filtering steps.

    This class will be specialized on the fly by :func:`filter_LIA` which
    will inject the static data ``_LIA_kind``.
    """

    # Useless definition used to trick pylint in believing self._LIA_kind is set.
    # Indeed, it's expected to be set in child classes. But pylint has now way to know that.
    _LIA_kind = None

    def _update_filename_meta_pre_hook(self, meta):
        meta = super()._update_filename_meta_pre_hook(meta)
        assert self._LIA_kind, "LIA kind should have been set in filter_LIA()"
        meta['LIA_kind'] = self._LIA_kind
        return meta

    def _get_input_image(self, meta):
        # Flatten should be useless, but kept for better error messages
        related_inputs = [f for f in Utils.flatten_stringlist(in_filename(meta))
                if re.search(rf'\b{self._LIA_kind}_', f)]
        assert len(related_inputs) == 1, f"Incorrect number ({len(related_inputs)}) of S1 LIA products of type '{self._LIA_kind}' in {in_filename(meta)} found: {related_inputs}"
        return related_inputs[0]

    def build_step_output_filename(self, meta):
        """
        Forward the output filename.
        """
        inp = self._get_input_image(meta)
        logger.debug('%s KEEP %s from %s', self.__class__.__name__, inp, in_filename(meta))
        return inp

    def build_step_output_tmp_filename(self, meta):
        """
        As there is no OTB application associated to :class:`ExtractSentinel1Metadata`,
        there is no temporary filename.
        """
        return self.build_step_output_filename(meta)


def filter_LIA(LIA_kind):
    """
    Generates a new :class:`StepFactory` class that filters which LIA product
    shall be processed: LIA maps or sin LIA maps.
    """
    # We return a new class
    return type("Filter_"+LIA_kind,   # Class name
            (_FilterStepFactory,),    # Parent
            { '_LIA_kind': LIA_kind}
            )


class OrthoRectifyLIA(_OrthoRectifierFactory):
    """
    Factory that prepares steps that run
    :std:doc:`Applications/app_OrthoRectification` on LIA maps.

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
        fname_fmt = '{LIA_kind}_{flying_unit_code}_{tile_name}_{orbit_direction}_{orbit}_{acquisition_time}.tif'
        fname_fmt = cfg.fname_fmt.get('lia_orthorectification') or fname_fmt
        super().__init__(cfg, fname_fmt,
                image_description='Orthorectified {LIA_kind} Sentinel-{flying_unit_code_short} IW GRD',
                )

    def _update_filename_meta_pre_hook(self, meta):
        meta = super()._update_filename_meta_pre_hook(meta)
        assert 'LIA_kind' in meta, "This StepFactory shall be registered after a call to filter_LIA()"
        return meta

    def _get_input_image(self, meta):
        inp = in_filename(meta)
        assert isinstance(inp, str), f"A single string inp was expected, got {inp}"
        return inp   # meta['in_filename']

    def update_image_metadata(self, meta, all_inputs):  # pylint: disable=unused-argument
        """
        Set LIA kind related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        types = {
                'sin_LIA': 'SIN(LIA)',
                'LIA': '100 * degree(LIA)'
                }
        assert 'LIA_kind' in meta, "This StepFactory shall be registered after a call to filter_LIA()"
        kind = meta['LIA_kind']
        assert kind in types, f'The only LIA kind accepted are {types.keys()}'
        imd = meta['image_metadata']
        imd['DATA_TYPE'] = types[kind]

    def set_output_pixel_type(self, app, meta):

        """
        Force LIA output pixel type to ``INT8``.
        """
        if meta.get('LIA_kind', '') == 'LIA':
            app.SetParameterOutputImagePixelType(self.param_out, otb.ImagePixelType_int16)


class ConcatenateLIA(_ConcatenatorFactory):
    """
    Factory that prepares steps that run
    :std:doc:`Applications/app_Synthetize` on LIA images.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """
    def __init__(self, cfg):
        fname_fmt = '{LIA_kind}_{flying_unit_code}_{tile_name}_{orbit_direction}_{orbit}_{acquisition_day}.tif'
        fname_fmt = cfg.fname_fmt.get('lia_concatenation') or fname_fmt
        super().__init__(cfg,
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
                gen_output_dir=None,  # Use gen_tmp_dir
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description='Orthorectified {LIA_kind} Sentinel-{flying_unit_code_short} IW GRD',
                )

    def _update_filename_meta_post_hook(self, meta):
        """
        Override "update_out_filename" hook to help select the input set with
        the best coverage.
        """
        assert 'LIA_kind' in meta
        meta['update_out_filename'] = self.update_out_filename  # <- needs to be done in post_hook!
        # Remove acquisition_time that no longer makes sense
        meta.pop('acquisition_time', None)

    def update_image_metadata(self, meta, all_inputs):  # pylint: disable=unused-argument
        """
        Update concatenated LIA related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        imd = meta['image_metadata']
        imd['DEM_LIST']  = ""  # Clear DEM_LIST information (a merge of 2 lists should be done actually)

    def update_out_filename(self, meta, with_task_info):  # pylint: disable=no-self-use
        """
        Unlike usual :class:`Concatenate`, the output filename will always ends
        in "txxxxxx".

        However we want to update the coverage of the current pair as a new
        input file has been registered.

        TODO: Find a better name for the hook as it handles two different
        services.
        """
        inputs = with_task_info.inputs['in']
        dates = set([re.sub(r'txxxxxx|t\d+', '', inp['acquisition_time']) for inp in inputs])
        assert len(dates) == 1, f"All concatenated files shall have the same date instead of {dates}"
        date = min(dates)
        logger.debug('[ConcatenateLIA] at %s:', date)
        coverage = 0.
        for inp in inputs:
            if re.sub(r'txxxxxx|t\d+', '', inp['acquisition_time']) == date:
                s1_cov = inp['tile_coverage']
                coverage += s1_cov
                logger.debug(' - %s => %s%% coverage', inp['basename'], s1_cov)
        # Round coverage at 3 digits as tile footprint has a very limited precision
        coverage = round(coverage, 3)
        logger.debug('[ConcatenateLIA] => total coverage at %s: %s%%', date, coverage*100)
        meta['tile_coverage'] = coverage

    def set_output_pixel_type(self, app, meta):
        """
        Force LIA output pixel type to ``INT8``.
        """
        if meta.get('LIA_kind', '') == 'LIA':
            app.SetParameterOutputImagePixelType(self.param_out, otb.ImagePixelType_int16)


class SelectBestCoverage(_FileProducingStepFactory):
    """
    StepFactory that helps select only one path after LIA concatenation:
    the one that have the best coverage of the S2 tile target.

    If several concatenated products have the same coverage, the oldest one
    will be selected.

    The coverage is extracted from ``tile_coverage`` step metadata.

    The step produced does nothing: it only only rename the selected product
    into the final expected name. Note: in LIA case two files will actually
    renamed.

    """
    def __init__(self, cfg):
        fname_fmt = '{LIA_kind}_{flying_unit_code}_{tile_name}_{orbit_direction}_{orbit}.tif'
        fname_fmt = cfg.fname_fmt.get('lia_product') or fname_fmt
        super().__init__(cfg, name='SelectBestCoverage',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
                gen_output_dir=cfg.lia_directory,
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                )

    def _update_filename_meta_pre_hook(self, meta):
        """
        Inject the :func:`reduce_LIAs` hook in step metadata.
        """
        def reduce_LIAs(inputs):
            """
            Select the concatenated pair of LIA files that have the best coverage of the considered
            S2 tile.
            """
            # TODO: quid if different dates have best different coverage on a set of tiles?
            # How to avoid computing LIA again and again on a same S1 zone?
            # dates = set([re.sub(r'txxxxxx|t\d+', '', inp['acquisition_time']) for inp in inputs])
            best_covered_input = max(inputs, key=lambda inp: inp['tile_coverage'])
            logger.debug('Best coverage is %s at %s among:', best_covered_input['tile_coverage'], best_covered_input['acquisition_day'])
            for inp in inputs:
                logger.debug(' - %s: %s', inp['acquisition_day'], inp['tile_coverage'])
            return [best_covered_input]

        meta['reduce_inputs_in'] = reduce_LIAs

    def create_step(self, in_memory: bool, previous_steps):
        logger.debug("Directly execute %s step", self.name)
        inputs = self._get_inputs(previous_steps)
        inp = self._get_canonical_input(inputs)
        meta = self.complete_meta(inp.meta, inputs)

        # Let's reuse commit_execution as it does exactly what we need
        if not is_running_dry(meta):
            commit_execution(out_filename(inp.meta), out_filename(meta))

        # Return a dummy Step
        # logger.debug("%s step executed!", self.name)
        res = AbstractStep('move', **meta)
        return res


class ApplyLIACalibration(OTBStepFactory):
    """
    Factory that concludes σ0 with NORMLIM calibration.

    It builds steps that multiply images calibrated with β0 LUT, and
    orthorectified to S2 grid, with the sin(LIA) map for the same S2 tile (and
    orbit number and direction).

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    - flying_unit_code
    - tile_name
    - polarisation
    - orbit_direction
    - orbit
    - acquisition_stamp
    """

    def __init__(self, cfg):
        """
        Constructor.
        """
        fname_fmt = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_NormLim.tif'
        fname_fmt = cfg.fname_fmt.get('s2_lia_corrected') or fname_fmt
        super().__init__(cfg,
                appname='BandMath', name='ApplyLIACalibration', param_in='il', param_out='out',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
                gen_output_dir=os.path.join(cfg.output_preprocess, '{tile_name}'),
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description='Sigma0 Normlim Calibrated Sentinel-{flying_unit_code_short} IW GRD',
                )

    def complete_meta(self, meta, all_inputs):
        """
        Complete meta information with inputs, and set compression method to
        DEFLATE.
        """
        meta = super().complete_meta(meta, all_inputs)
        meta['out_extended_filename_complement'] = "?&gdal:co:COMPRESS=DEFLATE"
        meta['inputs']           = all_inputs
        meta['calibration_type'] = 'Normlim'  # Update meta from now on
        return meta

    def update_image_metadata(self, meta, all_inputs):  # pylint: disable=unused-argument
        """
        Set σ° normlim calibration related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        inputs = meta['inputs']
        in_sin_LIA   = _fetch_input_data('sin_LIA',   inputs).out_filename
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['CALIBRATION'] = meta['calibration_type']
        imd['LIA_FILE']    = os.path.basename(in_sin_LIA)

    def _get_canonical_input(self, inputs):
        """
        Helper function to retrieve the canonical input associated to a list of inputs.

        In current case, the canonical input comes from the "concat_S2"
        pipeline defined in :func:`s1tiling.s1_process` pipeline builder.
        """
        _check_input_step_type(inputs)
        keys = set().union(*(input.keys() for input in inputs))
        assert len(inputs) == 2, f'Expecting 2 inputs. {len(inputs)} are found: {keys}'
        assert 'concat_S2' in keys
        return [input['concat_S2'] for input in inputs if 'concat_S2' in input.keys()][0]

    def _update_filename_meta_post_hook(self, meta):
        """
        Register ``is_compatible`` hook for
        :func:`s1tiling.libs.otbpipeline.is_compatible`.
        It will tell whether a given sin_LIA input is compatible with the
        current S2 tile.
        """
        meta['is_compatible']    = lambda input_meta : self._is_compatible(meta, input_meta)
        meta['basename']         = self._get_nominal_output_basename(meta)
        meta['calibration_type'] = 'Normlim'

    def _is_compatible(self, output_meta, input_meta):
        """
        Tells whether a given sin_LIA input is compatible with the the current
        S2 tile.

        ``flying_unit_code``, ``tile_name``, ``orbit_direction`` and ``orbit``
        have to be identical.
        """
        fields = ['flying_unit_code', 'tile_name', 'orbit_direction', 'orbit']
        return all(input_meta[k] == output_meta[k] for k in fields)

    def parameters(self, meta):
        """
        Returns the parameters to use with :std:doc:`BandMath OTB application
        <Applications/app_BandMath>` for applying sin(LIA) to β0 calibrated
        image orthorectified to S2 tile.
        """
        assert 'inputs' in meta, f'Looking for "inputs" in {meta.keys()}'
        inputs = meta['inputs']
        in_concat_S2 = _fetch_input_data('concat_S2', inputs).out_filename
        in_sin_LIA   = _fetch_input_data('sin_LIA',   inputs).out_filename
        nodata = meta.get('nodata', -32768)
        params = {
                'ram'         : str(self.ram_per_process),
                self.param_in : [in_concat_S2, in_sin_LIA],
                'exp'         : f'im2b1 == {nodata} ? {nodata} : im1b1*im2b1'
                }
        return params
