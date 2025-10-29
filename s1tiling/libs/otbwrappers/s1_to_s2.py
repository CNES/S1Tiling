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
# Authors: Thierry KOLECK (CNES)
#          Luc HERMITTE (CS Group)
# =========================================================================

"""
This modules defines the specialized Python wrappers for the OTB Applications used in
the pipeline for S1Tiling needs.
"""

import logging
import os
import re
from typing import Dict, List, Protocol, Union, cast
from packaging import version

import numpy as np
from osgeo import gdal

from ..file_naming   import (
    ReplaceOutputFilenameGenerator,
    TemplateOutputFilenameGenerator,
)
from ..meta import (
    Meta,
    get_task_name,
    in_filename,
    out_filename,
)
from ..steps import (
    FirstStep,
    InputList,
    MergeStep,
    OTBParameters,
    _check_input_step_type,
    AbstractStep,
    StepFactory,
    OTBStepFactory,
    _OTBStep,
    SkippedStep,
    manifest_to_product_name,
    ram,
)
from ..otbpipeline import (
    TaskInputInfo,
    fetch_input_data,
)
from ..otbtools      import otb_version
from ..              import exceptions
from ..              import Utils
from ..configuration import (
    Configuration,
    dname_fmt_mask,
    dname_fmt_tiled,
    dname_fmt_filtered,
    extended_filename_filtered,
    extended_filename_hidden,
    extended_filename_mask,
    extended_filename_tiled,
    fname_fmt_concatenation,
    fname_fmt_filtered,
)
from ..configuration import pixel_type as cfg_pixel_type  # avoid name hiding
from ._applications import (
    _ConcatenatorFactory,
    _OrthoRectifierFactory,
)
from .helpers        import does_sin_lia_match_s2_tile_for_orbit

logger = logging.getLogger('s1tiling.wrappers')


class InputStep(Protocol):
    """
    Protocol for all input stepts (:class:`FirstStep` and :class:`MergeStep`).

    Their public interface is to expose :meth:`input_metas` that returns a list of :class:`Meta`
    object for each input step.
    """
    @property
    def input_metas(self) -> List[Meta]:
        """
        Return a list of the :class:`Meta` objects from each input step.
        """
        return []


def has_too_many_NoData(image, threshold: int, nodata: Union[float, int]) -> bool:
    """
    Analyses whether an image contains NO DATA.

        :param image:     np.array image to analyse
        :param threshold: number of NoData searched
        :param nodata:    no data value
        :return:          whether the number of no-data pixel > threshold
    """
    nbNoData = len(np.argwhere(image == nodata))
    return nbNoData > threshold


def extract_IPF_version(tifftag_software: str) -> str:
    """
    Extracts a comparable IPF version number
    for packaging.version.parse for instance.
    """
    match = re.search(r'Sentinel-1 IPF (\d+\.\d+)$', tifftag_software)
    if not match:
        logger.warning('Cannot extract IPF version from "%s"', tifftag_software)
        return '000.00'
    ipf_version = match.group(1)
    # logger.debug(f"IPF version is {ipf_version}; {tifftag_software=!r} ")
    return ipf_version


class ExtractSentinel1Metadata(StepFactory):
    """
    Factory that takes care of extracting meta data from S1 input files.

    Note: At this moment it needs to be used on a separate pipeline to make
    sure the meta is updated when calling :func:`PipelineDescription.expected`.
    """

    def __init__(self, cfg: Configuration) -> None:  # pylint: disable=unused-argument
        super().__init__('ExtractSentinel1Metadata')

    def build_step_output_filename(self, meta: Meta) -> str:
        """
        Forward the output filename.
        """
        return meta['out_filename']

    def build_step_output_tmp_filename(self, meta: Meta) -> str:
        """
        As there is no OTB application associated to :class:`ExtractSentinel1Metadata`,
        there is no temporary filename.
        """
        return self.build_step_output_filename(meta)

    def _update_filename_meta_post_hook(self, meta: Meta) -> None:
        """
        Complete meta information such as filenames, GDAL metadata from
        information found in the current S1 image filename.
        """
        if 'manifest' not in meta:
            raise exceptions.NotCompatibleInput(f"{out_filename(meta)} is not an input for ExtractSentinel1Metadata.")
        manifest                 = meta['manifest']
        # image                   = in_filename(meta)   # meta['in_filename']
        image                    = meta['basename']

        # TODO: if the manifest is no longer here, we may need to look into the geom instead
        # It'd actually be better

        orbit_information = Utils.get_orbit_information(manifest)

        meta['origin_s1_image']  = meta['basename']  # Will be used to remember the reference image
        # meta['rootname']         = os.path.splitext(meta['basename'])[0]
        meta['flying_unit_code'] = Utils.get_platform_from_s1_raster(image)
        meta['polarisation']     = Utils.get_polar_from_s1_raster(image)
        meta['orbit_direction']  = orbit_information['orbit_direction']
        meta['orbit']            = f"{orbit_information['relative_orbit']:0>3d}"
        meta['absolute_orbit']   = f"{orbit_information['absolute_orbit']:0>6d}"
        meta['acquisition_time'] = Utils.get_date_from_s1_raster(image)
        meta['acquisition_day']  = re.sub(r"(?<=t)\d+$", lambda m: "x" * len(m.group()), meta['acquisition_time'])

        meta['task_name']        = f'ExtractS1Meta_{meta["basename"]}'

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set the common and root information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['IMAGE_TYPE']            = 'GRD'
        imd['FLYING_UNIT_CODE']      = meta['flying_unit_code']
        imd['RELATIVE_ORBIT_NUMBER'] = meta['orbit']
        imd['ORBIT_NUMBER']          = meta['absolute_orbit']
        imd['ORBIT_DIRECTION']       = meta['orbit_direction']
        imd['POLARIZATION']          = meta['polarisation']
        imd['INPUT_S1_IMAGES']       = manifest_to_product_name(meta['manifest'])
        # Only one input image at this point, we don't introduce any
        # ACQUISITION_DATETIMES or ACQUISITION_DATETIME_1...

        acquisition_time = meta['acquisition_time']
        date = f'{acquisition_time[0:4]}:{acquisition_time[4:6]}:{acquisition_time[6:8]}'
        if acquisition_time[9] == 'x':
            # This case should not happen, here
            date += 'T00:00:00Z'
        else:
            date += f'T{acquisition_time[9:11]}:{acquisition_time[11:13]}:{acquisition_time[13:15]}Z'
        imd['ACQUISITION_DATETIME'] = date

    def _get_canonical_input(self, inputs: InputList) -> AbstractStep:
        """
        Helper function to retrieve the canonical input associated to a list of inputs.

        :class:`ExtractSentinel1Metadata` can be used either in usual S1Tiling
        orthorectification scenario, or in LIA Map generation scenarios.
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

    def __init__(self, cfg: Configuration) -> None:
        """
        Constructor
        """
        super().__init__('AnalyseBorders')
        self.__override_azimuth_cut_threshold_to = cfg.override_azimuth_cut_threshold_to

    def build_step_output_filename(self, meta: Meta) -> str:
        """
        Forward the output filename.
        """
        return meta['out_filename']

    def build_step_output_tmp_filename(self, meta: Meta) -> str:
        """
        As there is no OTB application associated to :class:`AnalyseBorders`,
        there is no temporary filename.
        """
        return self.build_step_output_filename(meta)

    def complete_meta(  # pylint: disable=too-many-locals
        self, meta: Meta, all_inputs: InputList
    ) -> Meta:
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
        tifftag_software = ds_reader.GetMetadataItem('TIFFTAG_SOFTWARE')  # Ex: Sentinel-1 IPF 003.10

        # Starting from IPF 2.90+, no margin correction is done on the sides.
        # With prior versions, the cut margin (right and left) is done.

        ipf_version = extract_IPF_version(tifftag_software)
        if version.parse(ipf_version) >= version.parse('2.90'):
            cut_overlap_range = 0

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

        logger.debug("   => need to crop north: %s", crop1)
        logger.debug("   => need to crop south: %s", crop2)

        thr_x   = cut_overlap_range
        thr_y_s = cut_overlap_azimuth if crop1 else 0
        thr_y_e = cut_overlap_azimuth if crop2 else 0

        meta['cut'] = {
            'threshold.x'      : thr_x,
            'threshold.y.start': thr_y_s,
            'threshold.y.end'  : thr_y_e,
            'skip'             : thr_x == 0 and thr_y_s == 0 and thr_y_e == 0,
        }
        return meta


k_calib_convert = {'normlim': 'beta', 'gamma_naught_rtc': 'sigma'}


class Calibrate(OTBStepFactory):
    """
    Factory that prepares steps that run :external+OTB:doc:`Applications/app_SARCalibration` as
    described in :ref:`SAR Calibration` documentation.

    Requires the following information from the configuration object:

    - `ram_per_process`
    - `calibration_type`
    - `removethermalnoise`

    Requires the following information from the metadata dictionary:

    - base name -- to generate typical output filename
    - input filename
    - output filename
    """

    def __init__(self, cfg: Configuration) -> None:
        """
        Constructor
        """
        fname_fmt = '{rootname}_{calibration_type}_calOk.tiff'
        fname_fmt = cfg.fname_fmt.get('calibration', fname_fmt)
        super().__init__(
            cfg,
            appname='SARCalibration',
            name='Calibration',
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
            gen_output_dir=None,  # Use gen_tmp_dir
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
            extended_filename=extended_filename_hidden(cfg, 'calibration'),
            image_description='{calibration_type} calibrated Sentinel-{flying_unit_code_short} IW GRD',
        )
        # Warning: config object cannot be stored and passed to workers!
        # => We extract what we need
        # Locally override calibration type in case of normlim calibration
        self.__calibration_type   = k_calib_convert.get(cfg.calibration_type, cfg.calibration_type)
        self.__removethermalnoise = cfg.removethermalnoise

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:
        """
        Injects the ``calibration_type`` in step metadata.
        """
        meta['calibration_type'] = self.__calibration_type
        return meta

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set calibration related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['CALIBRATION']   = meta['calibration_type']
        imd['NOISE_REMOVED'] = str(self.__removethermalnoise)

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external+OTB:doc:`SARCalibration OTB application
        <Applications/app_SARCalibration>`.
        """
        params : OTBParameters = {
            'ram'           : ram(self.ram_per_process),
            self.param_in   : in_filename(meta),
            # self.param_out  : out_filename(meta),
            'lut'           : self.__calibration_type,
        }
        if otb_version() >= '7.4.0':
            params['removenoise'] = self.__removethermalnoise
        else:  # pragma: no cover
            # Don't try to do anything, let's keep the noise
            params['noise'] = True
        return params


class CorrectDenoising(OTBStepFactory):
    """
    Factory that prepares steps that run :external+OTB:doc:`Applications/app_BandMath` as described
    in :ref:`SAR Calibration` documentation.

    It requires the following information from the configuration object:

    - `ram_per_process`

    It requires the following information from the metadata dictionary:

    - base name -- to generate typical output filename
    - input filename
    - output filename
    - lower_signal_value
    """

    def __init__(self, cfg: Configuration) -> None:
        """
        Constructor.
        """
        fname_fmt = '{rootname}_{calibration_type}_NoiseFixed.tiff'
        fname_fmt = cfg.fname_fmt.get('correct_denoising', fname_fmt)
        super().__init__(
            cfg,
            appname='BandMath',
            name='DenoisingCorrection',
            param_in='il',
            param_out='out',
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
            gen_output_dir=None,  # Use gen_tmp_dir
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
            extended_filename=extended_filename_hidden(cfg, 'correct_denoising'),
            image_description='{calibration_type} calibrated Sentinel-{flying_unit_code_short} IW GRD with noise corrected',
        )
        self.__lower_signal_value = cfg.lower_signal_value

    def _get_inputs(self, previous_steps: List[InputList]) -> InputList:
        """
        Extract the last inputs to use at the current level from all previous
        products seens in the pipeline.

        This method is overridden in order to fetch N-3 "in_sar" input.
        It has been specialized for S1Tiling exact pipelines.
        """
        assert len(previous_steps) > 1

        # "in_sar" is expected at level -2, likelly named '__last'
        in_sar = fetch_input_data('__last', previous_steps[-2])
        # "in_cal" is expected at level -1, likelly named '__last'
        in_cal = fetch_input_data('__last', previous_steps[-1])

        inputs = [{'in_sar': in_sar, 'in_cal': in_cal}]
        _check_input_step_type(inputs)
        logging.debug("%s inputs: %s", self.__class__.__name__, inputs)
        return inputs

    def _get_canonical_input(self, inputs: InputList) -> AbstractStep:
        """
        Helper function to retrieve the canonical input associated to a list of inputs.

        In current case, the canonical input comes from the "in_cal" step instanciated in
        :func:`s1tiling.s1_process` pipeline builder.
        """
        _check_input_step_type(inputs)
        keys = set().union(*(input.keys() for input in inputs))
        assert len(keys) == 2, f'Expecting 2 inputs. {len(inputs)} is/are found: {keys}'
        assert 'in_cal' in keys
        return [input['in_cal'] for input in inputs if 'in_cal' in input.keys()][0]

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set noise correction related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['LOWER_SIGNAL_VALUE'] = str(self.__lower_signal_value)

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external+OTB:doc:`BandMath OTB application
        <Applications/app_BandMath>` for changing no-non-data 0.0 into lower_signal_value, and force
        nodata to 0.

        The nodata mask comes from the input SAR image.

        Expression used: ``exp = im{sar}b1 == 0 ? 0 : im{cal}b1 == 0 ? 1e-7 : im{cal}b1``
        """
        assert 'inputs' in meta, f'Looking for "inputs" in {meta.keys()}'
        inputs = meta['inputs']
        # "in_sar" needs to be "im2" as _do_create_actual_step() will add extra filenames to inputlist
        # after in_memory pipelines
        in_cal = fetch_input_data('in_cal', inputs).out_filename
        in_sar = fetch_input_data('in_sar', inputs).out_filename
        params : OTBParameters = {
            'ram'              : ram(self.ram_per_process),
            self.param_in      : [in_cal, in_sar],
            # self.param_out     : out_filename(meta),
            # 'exp'              : f'im1b1==0?{self.__lower_signal_value}:im1b1'
            'exp'              : f'im2b1==0?0:im1b1==0?{self.__lower_signal_value}:im1b1'
        }
        return params


class CutBorders(OTBStepFactory):
    """
    Factory that prepares steps that run :external+OTB:doc:`Applications/app_ResetMargin` as
    described in :ref:`Margins Cutting` documentation.

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

    def __init__(self, cfg: Configuration) -> None:
        """
        Constructor.
        """
        fname_fmt = '{rootname}_{calibration_type}_OrthoReady.tiff'
        fname_fmt = cfg.fname_fmt.get('cut_borders', fname_fmt)
        super().__init__(
            cfg,
            appname='ResetMargin',
            name='BorderCutting',
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
            gen_output_dir=None,  # Use gen_tmp_dir
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
            extended_filename=extended_filename_hidden(cfg, 'cut_borders'),
        )

    def create_step(
        self,
        execution_parameters: Dict,
        previous_steps: List[InputList]
    ) -> AbstractStep:
        """
        This overrides checks whether ResetMargin would cut any border.

        In the likelly other case, the method returns a
        :class:`s1tiling.libs.steps.SkippedStep` to say **Don't register any OTB
        application and skip this step!**.
        """
        inputs = self._get_inputs(previous_steps)
        inp    = self._get_canonical_input(inputs)
        if inp.meta['cut'].get('skip', False):
            logger.debug('Margins cutting is not required and thus skipped!')
            meta = self.complete_meta(inp.meta, inputs)
            assert isinstance(inp, _OTBStep)
            return SkippedStep(inp.app, **meta)
        else:
            return super().create_step(execution_parameters, previous_steps)

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external+OTB:doc:`ResetMargin OTB application
        <Applications/app_ResetMargin>`.
        """
        params = {
            'ram'              : ram(self.ram_per_process),
            self.param_in      : in_filename(meta),
            # self.param_out     : out_filename(meta),
            'threshold.x'      : meta['cut']['threshold.x'],
            'threshold.y.start': meta['cut']['threshold.y.start'],
            'threshold.y.end'  : meta['cut']['threshold.y.end']
        }
        if otb_version() != '7.2.0':  # From 7.3.0 onward actually
            params['mode'] = 'threshold'
        return params


class OrthoRectify(_OrthoRectifierFactory):
    """
    Factory that prepares steps that run :external+OTB:doc:`Applications/app_OrthoRectification` as
    described in :ref:`OrthoRectification` documentation.

    Requires the following information from the configuration object:

    - `ram_per_process`
    - `out_spatial_res`
    - `GeoidFile`
    - `grid_spacing`
    - `tmp_dem_dir`

    Requires the following information from the metadata dictionary

    - base name -- to generate typical output filename
    - input filename
    - output filename
    - `manifest`
    - `tile_name`
    - `tile_origin`
    """

    def __init__(self, cfg: Configuration) -> None:
        """
        Constructor.
        Extract and cache configuration options.
        """
        fname_fmt = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_time}_{calibration_type}.tif'
        fname_fmt = cfg.fname_fmt.get('orthorectification', fname_fmt)
        extended_filename = extended_filename_tiled(cfg)
        if otb_version() < '8.0.0':
            extended_filename += '&writegeom=false'
        super().__init__(
            cfg,
            fname_fmt=fname_fmt,
            image_description='{calibration_type} calibrated orthorectified Sentinel-{flying_unit_code_short} IW GRD',
            extended_filename=extended_filename,
            pixel_type=cfg_pixel_type(cfg, 'tiled'),
        )
        # Some workaround when ortho is not sequenced along with calibration
        # (and locally override calibration type in case of normlim calibration)
        self.__calibration_type = k_calib_convert.get(cfg.calibration_type, cfg.calibration_type)

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        Complete meta information such as filenames, GDAL metadata from
        information found in the current S1 image filename.
        """
        meta = super().complete_meta(meta, all_inputs)

        # Some workaround when ortho is not sequenced along with calibration
        meta['calibration_type'] = self.__calibration_type
        return meta

    def _get_input_image(self, meta: Meta) -> str:
        return in_filename(meta)  # meta['in_filename']


class Concatenate(_ConcatenatorFactory):
    """
    Abstract factory that prepares steps that run :external+OTB:doc:`Applications/app_Synthetize` as
    described in :ref:`Concatenation` documentation.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """

    __RE_ACQ_STAMPS = re.compile(r'{acquisition_stamp}|{acquisition_start}')

    def __init__(self, cfg: Configuration) -> None:
        # TODO: factorise this recurring test!
        calibration_is_done_in_S1 = cfg.calibration_type in ['sigma', 'beta', 'gamma', 'dn']
        if calibration_is_done_in_S1:
            # This is a required product that shall end-up in outputdir
            gen_output_dir = dname_fmt_tiled(cfg)
        else:
            # This is a temporary product that shall end-up in tmpdir
            gen_output_dir = None  # use gen_tmp_dir
        fname_fmt = fname_fmt_concatenation(cfg)
        # logger.debug('but ultimatelly fname_fmt is "%s" --> %s', fname_fmt, cfg.fname_fmt)
        self.__tname_fmt = re.sub(self.__RE_ACQ_STAMPS, '{acquisition_day}', fname_fmt)

        super().__init__(
            cfg,
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
            gen_output_dir=gen_output_dir,
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
            image_description='{calibration_type} calibrated orthorectified Sentinel-{flying_unit_code_short} IW GRD',
            extended_filename=extended_filename_tiled(cfg),
            pixel_type=cfg_pixel_type(cfg, 'tiled'),
        )

    def update_out_filename(self, meta: Meta, with_task_info: TaskInputInfo) -> None:
        """
        This hook will be triggered everytime a new compatible input is added.
        The effect is quite unique to :class:`Concatenate` as the name of the output product depends
        on the number of inputs are their common acquisition date.
        """
        # logger.debug('UPDATING %s from %s', meta['task_name'], meta)
        was = meta['out_filename']
        meta['acquisition_stamp']  = meta['acquisition_day']
        meta['acquisition_start']  = min((m['acquisition_time'] for m in with_task_info.input_metas))
        meta['out_filename']       = self.build_step_output_filename(meta)
        meta['out_tmp_filename']   = self.build_step_output_tmp_filename(meta)
        meta['basename']           = self._get_nominal_output_basename(meta)
        logger.debug(
            "concatenation.out_tmp_filename for %s updated to %s (previously: %s) ; acquisition_start: %s",
            meta['task_name'], meta['out_filename'], was, meta['acquisition_start'])
        # Remove acquisition_time that no longer makes sense, use either acquisition_stamp or acquisition_start
        meta.pop('acquisition_time', None)

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:
        in_file = out_filename(meta)
        if isinstance(in_file, list):
            meta['acquisition_stamp'] = meta['acquisition_day']
            # Remove acquisition_time that no longer makes sense, use either acquisition_stamp or acquisition_start
            meta.pop('acquisition_time', None)
            logger.debug("Concatenation result of %s goes into %s", in_file, meta['basename'])
        else:
            meta['acquisition_stamp'] = meta['acquisition_time']
            logger.debug("Only one file to concatenate, just move it (%s)", in_file)

        if 'inputs' in meta:
            inputs : InputList = meta['inputs']
            input_step = self._get_canonical_input(inputs)  # input_metas in FirstStep, MergeStep
            assert isinstance(input_step, (FirstStep, MergeStep))
            # for inp in inputs:
            #     for key, step in inp.items():
            #         logger.debug("input %r is %s -> %s", key, step.__class__.__name__, step)
            #         assert isinstance(step, (FirstStep, MergeStep))
            #         for sub_meta in step.input_metas:
            #             logger.debug('input acq: %s', sub_meta['acquisition_time'])
            meta['acquisition_start'] = min(
                ( sub_meta['acquisition_time']
                 for sub_meta in cast(InputStep, input_step).input_metas)
            )
        else:
            meta['acquisition_start'] = meta['acquisition_time']
        logger.debug('task for %s acquisition_start found at: %s', in_file, meta['acquisition_start'])

        return meta

    def _update_filename_meta_post_hook(self, meta: Meta) -> None:
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
            def check_product(meta: Meta):
                task_name       = get_task_name(meta)
                filename        = out_filename(meta)
                exist_task_name = os.path.isfile(task_name)
                exist_file_name = os.path.isfile(filename)
                logger.debug('Checking concatenation product:\n- %r => %s (task)\n- %r => %s (file)',
                        task_name, '∃' if exist_task_name else '∅',
                        filename,  '∃' if exist_file_name else '∅')
                return exist_task_name or exist_file_name

            meta['does_product_exist'] = lambda: check_product(meta)

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set concatenation related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['IMAGE_TYPE'] = 'BACKSCATTERING'
        # imd['debug_acquisition_start'] = meta['acquisition_start']


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

    def __init__(self, cfg: Configuration) -> None:
        """
        Constructor.
        """
        super().__init__(
            cfg,
            appname='BandMath',
            name='BuildBorderMask',
            param_in='il',
            param_out='out',
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
            gen_output_dir=None,  # Use gen_tmp_dir
            gen_output_filename=ReplaceOutputFilenameGenerator(['.tif', '_BorderMask_TMP.tif']),
            pixel_type=cfg_pixel_type(cfg, 'mask', 'uint8'),
            image_description='Orthorectified Sentinel-{flying_unit_code_short} IW GRD border mask S2 tile',
        )

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set mask related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['IMAGE_TYPE'] = 'MASK'

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external+OTB:doc:`BandMath OTB application
        <Applications/app_BandMath>` for computing border mask.
        """
        params : OTBParameters = {
                'ram'              : ram(self.ram_per_process),
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

    def __init__(self, cfg: Configuration) -> None:
        dname_fmt = dname_fmt_mask(cfg)
        super().__init__(
            cfg,
            appname='BinaryMorphologicalOperation',
            name='SmoothBorderMask',
            param_in='in',
            param_out='out',
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
            gen_output_dir=dname_fmt,
            gen_output_filename=ReplaceOutputFilenameGenerator(['.tif', '_BorderMask.tif']),
            extended_filename=extended_filename_mask(cfg),
            pixel_type=cfg_pixel_type(cfg, 'mask', 'uint8'),
            image_description='Orthorectified Sentinel-{flying_unit_code_short} IW GRD smoothed border mask S2 tile',
        )

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external+OTB:doc:`BinaryMorphologicalOperation OTB
        application <Applications/app_BinaryMorphologicalOperation>` to smooth border masks.
        """
        return {
            'ram'          : ram(self.ram_per_process),
            self.param_in  : in_filename(meta),
            # self.param_out : out_filename(meta),
            'structype'    : 'ball',
            'xradius'      : 5,
            'yradius'      : 5,
            'filter'       : 'opening'
        }


# ----------------------------------------------------------------------
# Despeckling related applications
class SpatialDespeckle(OTBStepFactory):
    """
    Factory that prepares the first step that smoothes border maks as
    described in :ref:`Spatial despeckle filtering <spatial-despeckle>`
    documentation.

    It requires the following information from the configuration object:

    - `ram_per_process`
    - `fname_fmt_filtered`
    - `dname_fmt_filtered`
    - `filter`:  the name of the filter method
    - `rad`:     the filter windows radius
    - `nblooks`: the number of looks
    - `deramp`:  the deramp factor

    It requires the following information from the metadata dictionary

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
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = fname_fmt_filtered(cfg)
        dname_fmt = dname_fmt_filtered(cfg)
        super().__init__(
            cfg,
            appname='Despeckle',
            name='Despeckle',
            param_in='in',
            param_out='out',
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
            gen_output_dir=dname_fmt,
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
            image_description='Orthorectified and despeckled Sentinel-{flying_unit_code_short} IW GRD S2 tile',
            extended_filename=extended_filename_filtered(cfg),
            pixel_type=cfg_pixel_type(cfg, 'filtered'),
        )
        self.__filter  = cfg.filter
        self.__rad     = cfg.filter_options.get('rad', 0)
        self.__nblooks = cfg.filter_options.get('nblooks', 0)
        self.__deramp  = cfg.filter_options.get('deramp', 0)

        assert self.__rad
        # filter in list => nblooks != 0 ~~~> filter not in list OR nblooks != 0
        assert (self.__filter not in ['lee', 'gammamap', 'kuan']) or (self.__nblooks != 0), (
            f'Unexpected nblooks value ({self.__nblooks} for {self.__filter} despeckle filter'
        )
        # filter in list => deramp != 0 ~~~> filter not in list OR deramp != 0
        assert (self.__filter not in ['frost']) or (self.__deramp != 0.0), (
            f'Unexpected deramp value ({self.__deramp} for {self.__filter} despeckle filter'
        )
        assert (self.__nblooks != 0.0) != (self.__deramp != 0.0)

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:
        """
        Injects the ``filter_method`` in step metadata.
        """
        meta['filter_method'] = self.__filter
        return meta

    def _update_filename_meta_post_hook(self, meta: Meta) -> None:
        """
        Register ``accept_as_compatible_input`` hook for
        :func:`s1tiling.libs.meta.accept_as_compatible_input`.
        It will tell in the case Despeckle is chained in memory after
        ApplyLIACalibration whether a given sin_LIA input is compatible with
        the current S2 tile.
        """
        # TODO find a better way to reuse the hook from the previous step in case it's chained in memory!
        meta['accept_as_compatible_input'] = lambda input_meta: does_sin_lia_match_s2_tile_for_orbit(meta, input_meta)

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
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
        imd['IMAGE_TYPE']              = 'BACKSCATTERING'

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external+OTB:doc:`Despeckle OTB application
        <Applications/app_Despeckle>` to perform speckle noise reduction.
        """
        assert self.__rad
        params = {
            'ram'                         : ram(self.ram_per_process),
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
