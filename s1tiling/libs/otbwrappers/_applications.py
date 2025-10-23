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
# Authors:
# - Thierry KOLECK (CNES)
# - Luc HERMITTE (CS Group)
#
# =========================================================================

"""Factorize :class:`StepFactory` definitions common to multiple leaf StepFactories"""

from abc import abstractmethod
import shutil
from typing import Dict, List, Optional
import logging
import os
import re

from osgeo import gdal

from s1tiling.libs.otbpipeline import TaskInputInfo, fetch_input_data

from .. import Utils
from .helpers        import depolarize_4_filename_pre_hook
from ..configuration import Configuration, nodata_DEM, nodata_XYZ
from ..file_naming   import TemplateOutputFilenameGenerator
from ..meta          import Meta, append_to, in_filename, is_running_dry, out_filename, tmp_filename
from ..steps         import (
    AbstractStep,
    AnyProducerStep,
    ExeParameters,
    FirstStep,
    InputList,
    MergeStep,
    OTBParameters,
    OTBStepFactory,
    _FileProducingStepFactory,
    _check_input_step_type,
    commit_execution,
    manifest_to_product_name,
    ram,
)


logger = logging.getLogger('s1tiling.wrappers.applications')


def s2_tile_extent(tile_name: str, tile_origin: Utils.Polygon, in_epsg: int, spacing: float) -> Dict:
    """
    Helper function that computes and returns contant-sized extents of S2 tiles in the
    S2 tile spatial reference.
    """
    out_utm_zone     = int(tile_name[0:2])
    out_utm_northern = tile_name[2] >= 'N'
    out_epsg         = 32600 + out_utm_zone
    if not out_utm_northern:
        out_epsg = out_epsg + 100
    x_coord, y_coord, _ = Utils.convert_coord([tile_origin[0]], in_epsg, out_epsg)[0]
    lrx, lry, _         = Utils.convert_coord([tile_origin[2]], in_epsg, out_epsg)[0]

    if not out_utm_northern and y_coord < 0:
        y_coord += 10000000.
        lry     += 10000000.
    sizex = int(round(abs(lrx - x_coord) / spacing))
    sizey = int(round(abs(lry - y_coord) / spacing))
    logger.debug("from %s, lrx=%s, x_coord=%s, spacing=%s", tile_name, lrx, x_coord, spacing)
    return {
            'xmin'        : x_coord,
            'ymin'        : y_coord - sizey * spacing,
            'xmax'        : x_coord + sizex * spacing,
            'ymax'        : y_coord,
            'xsize'       : sizex,
            'ysize'       : sizey,
            'epsg'        : out_epsg,
            'utm_zone'    : out_utm_zone,
            'utm_northern': out_utm_northern,
    }


class _ConcatenatorFactory(OTBStepFactory):
    """
    Abstract factory that prepares steps that run :external+OTB:doc:`Applications/app_Synthetize` as
    described in :ref:`Concatenation` documentation.

    In the case no concatenation is required, the input file will be move to the expected product.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """
    def __init__(
            self,
            cfg              : Configuration,
            extended_filename: Optional[str],
            pixel_type       : Optional[int],
            *args, **kwargs,
    ) -> None:
        super().__init__(  # type: ignore # mypy issue 4335
            cfg,
            appname='Synthetize',
            name='Concatenation',
            param_in='il',
            param_out='out',
            extended_filename=extended_filename,
            pixel_type=pixel_type,
            *args, **kwargs
        )

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        Precompute output basename from the input file(s).

        In concatenation case, the task_name needs to be overridden to stay unique and common to all
        inputs.

        Also, inject files to remove
        """
        meta = super().complete_meta(meta, all_inputs)  # Needs a valid basename

        # logger.debug("Concatenate.complete_meta(%s) /// task_name: %s /// out_file: %s", meta, meta['task_name'], out_file)
        in_file = in_filename(meta)
        if isinstance(in_file, list):
            logger.debug('Register files to remove after concatenation: %s', in_file)
            meta['files_to_remove'] = in_file
        else:
            logger.debug('DONT register single file to remove after concatenation: %s', in_file)
        return meta

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set concatenation related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        inp = self._get_canonical_input(all_inputs)  # input_metas in FirstStep, MergeStep
        assert isinstance(inp, (FirstStep, MergeStep))

        product_names = sorted([manifest_to_product_name(m['manifest']) for m in inp.input_metas])
        imd['INPUT_S1_IMAGES']       = ', '.join(product_names)
        acq_time = Utils.extract_product_start_time(os.path.basename(product_names[0]))
        imd['ACQUISITION_DATETIME'] = '{YYYY}:{MM}:{DD}T{hh}:{mm}:{ss}Z'.format_map(acq_time) if acq_time else '????'
        for idx, pn in enumerate(product_names, start=1):
            acq_time = Utils.extract_product_start_time(os.path.basename(pn))
            imd[f'ACQUISITION_DATETIME_{idx}'] = '{YYYY}:{MM}:{DD}T{hh}:{mm}:{ss}Z'.format_map(acq_time) if acq_time else '????'

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external+OTB:doc:`Synthetize OTB application
        <Applications/app_Synthetize>`.
        """
        return {
            'ram'              : ram(self.ram_per_process),
            self.param_in      : in_filename(meta),
            # self.param_out     : out_filename(meta),
        }

    def _do_create_actual_step(
        self,
        execution_parameters: Dict,
        input_step: AbstractStep,
        meta: Meta
    ) -> AbstractStep:
        """
        Overrides default implementation of
        :meth:`s1tiling.libs.steps.OTBStepFactory._do_create_actual_step` to return and execute an
        :class:`s1tiling.libs.steps.AnyProducerStep` when there is only one orthorectified input
        product, and thus when no actual concatenation is required.
        """
        # Nominal case: two products will be concatenated
        if not isinstance(input_step.out_filename, str):
            return super()._do_create_actual_step(execution_parameters, input_step, meta)

        # Corner case: there is only product, it will be renamed
        concat_in_filename = input_step.out_filename
        logger.debug('By-passing concatenation of %r as there is only a single orthorectified tile to concatenate.', concat_in_filename)
        res = AnyProducerStep(action=_ConcatenatorFactory.rename, **meta)

        # We do rename to temporary filename: it will be used later on in the update_image_metadata
        # and the commit_execution phases
        # While it may appear unnecessary, the `rename` action renames to the :meth:`tmp_filename`.
        # - It simplifies the integration in the kernel,
        # - It permits update image metadata
        parameters = [concat_in_filename, tmp_filename(meta)]

        # Make sure the renaming is done now
        res.execute_and_write_output(parameters, execution_parameters)
        return res

    @staticmethod
    def rename(parameters: ExeParameters, dryrun: bool) -> None:
        """
        ``action`` meant to be registered in an :class:`s1tiling.libs.steps.AnyProducerStep`.
        It takes care of renaming a product when there is only orthorectified product and no need
        for concatenation.
        """
        src, dst = parameters
        logger.critical('Renaming %r into %r', src, dst)
        if not dryrun:
            shutil.move(src, dst)


class _ConcatenatorFactoryForMaps(_ConcatenatorFactory):
    """
    StepFactory dedicated to concatenate half-maps of orthorectified LIA or γ area products.
    """
    def __init__(
        self,
        cfg: Configuration,
        *,
        fname_fmt:         str,
        image_description: str,
        extended_filename: Optional[str],
    ) -> None:
        super().__init__(
            cfg,
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
            gen_output_dir=None,  # Use gen_tmp_dir
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
            image_description=image_description,
            extended_filename=extended_filename,
            pixel_type=None,         # will be set later...
        )
        self.__dem_info = cfg.dem_info

    def _update_filename_meta_post_hook(self, meta: Meta) -> None:
        """
        Override "update_out_filename" hook to help select the input set with the best coverage.
        """
        meta['update_out_filename'] = self.update_out_filename  # <- needs to be done in post_hook!
        # Remove acquisition_time that no longer makes sense
        meta.pop('acquisition_time', None)

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Update concatenated DEM related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        imd = meta['image_metadata']
        imd['DEM_INFO']  = self.__dem_info
        imd['DEM_LIST']  = ""  # Clear DEM_LIST information (a merge of 2 lists should be done actually)

    def update_out_filename(self, meta: Meta, with_task_info: TaskInputInfo) -> None:
        """
        Unlike usual :class:`Concatenate`, the output filename will always ends in "txxxxxx".

        However we want to update the coverage of the current pair as a new input file has been
        registered.

        TODO: Find a better name for the hook as it handles two different services.
        """
        inputs = with_task_info.inputs['in']
        dates = {re.sub(r'txxxxxx|t\d+', '', inp['acquisition_time']) for inp in inputs}
        assert len(dates) == 1, f"All concatenated files shall have the same date instead of {dates}"
        date = min(dates)
        logger.debug('[%s] at %s:', self.name, date)
        coverage = 0.
        for inp in inputs:
            if re.sub(r'txxxxxx|t\d+', '', inp['acquisition_time']) == date:
                s1_cov = inp['tile_coverage']
                coverage += s1_cov
                logger.debug(' - %s => %s%% coverage', inp['basename'], s1_cov)
        # Round coverage at 3 digits as tile footprint has a very limited precision
        coverage = round(coverage, 3)
        logger.debug('[%s] => total coverage at %s: %s%%', self.name, date, coverage * 100)
        meta['tile_coverage'] = coverage


class _OrthoRectifierFactory(OTBStepFactory):
    """
    Abstract factory that prepares steps that run
    :external+OTB:doc:`Applications/app_OrthoRectification` as described in
    :ref:`OrthoRectification` documentation.

    This factory will be specialized for calibrated S1 images
    (:class:`OrthoRectify`), or LIA and sin-LIA maps (:class:`OrthoRectifyLIA`)

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
    def __init__(  # pylint: disable=too-many-arguments
        self,
        cfg              : Configuration,
        *,
        fname_fmt        : str,
        image_description: str,
        extended_filename: Optional[str] = None,
        pixel_type       : Optional[int] = None,
    ) -> None:
        """
        Constructor.
        Extract and cache configuration options.
        """
        super().__init__(
            cfg,
            appname='OrthoRectification', name='OrthoRectification',
            param_in='io.in', param_out='io.out',
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
            gen_output_dir=None,  # Use gen_tmp_dir,
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
            image_description=image_description,
            extended_filename=extended_filename,
            pixel_type=pixel_type,
        )
        self.__out_spatial_res      = cfg.out_spatial_res
        self.__GeoidFile            = os.path.join(cfg.tmpdir, 'geoid', os.path.basename(cfg.GeoidFile))
        assert os.path.isfile(self.__GeoidFile), f"geoid file {self.__GeoidFile!r} is not accessible"
        self.__grid_spacing         = cfg.grid_spacing
        self.__interpolation_method = cfg.interpolation_method
        self.__tmp_dem_dir          = cfg.tmp_dem_dir
        self.__dem_info             = cfg.dem_info
        # self.__tmpdir               = cfg.tmpdir
        ## # Some workaround when ortho is not sequenced along with calibration
        ## # (and locally override calibration type in case of normlim calibration)
        ## self.__calibration_type     = k_calib_convert.get(cfg.calibration_type, cfg.calibration_type)

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['ORTHORECTIFICATION_INTERPOLATOR'] = self.__interpolation_method
        imd['ORTHORECTIFIED']                  = 'true'
        imd['S2_TILE_CORRESPONDING_CODE']      = meta['tile_name']
        imd['SPATIAL_RESOLUTION']              = str(self.__out_spatial_res)
        imd['DEM_INFO']                        = self.__dem_info
        imd['RELATIVE_ORBIT_NUMBER']           = meta['orbit']
        imd['ORBIT_NUMBER']                    = meta['absolute_orbit']
        imd['ORBIT_DIRECTION']                 = meta['orbit_direction']
        # S1 -> S2 => remove all SAR specific metadata inserted by OTB
        meta_to_remove_in_s2 = (
            'SARCalib*', 'SAR', 'PRF', 'RadarFrequency', 'RedDisplayChannel',
            'GreenDisplayChannel', 'BlueDisplayChannel', 'AbsoluteCalibrationConstant',
            'AcquisitionStartTime', 'AcquisitionStopTime', 'AcquisitionDate',
            'AverageSceneHeight', 'BeamMode', 'BeamSwath', 'Instrument', 'LineSpacing',
            'Mission', 'Mode','OrbitDirection', 'OrbitNumber', 'PixelSpacing', 'SensorID',
            'Swath', 'NumberOfLines', 'NumberOfColumns',
        )
        for kw in meta_to_remove_in_s2:
            imd[kw] = ''

    @abstractmethod
    def _get_input_image(self, meta: Meta):
        raise TypeError("_OrthoRectifierFactory does not know how to fetch input image")

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external+OTB:doc:`OrthoRectification OTB application
        <Applications/app_OrthoRectification>`.
        """
        image       = self._get_input_image(meta)
        tile_name   = meta['tile_name']
        tile_origin = meta['tile_origin']
        spacing     = self.__out_spatial_res

        extent      = s2_tile_extent(tile_name, tile_origin, in_epsg=4326, spacing=spacing)
        logger.debug("%s.parameters(%s) /// image: %s /// tile_name: %s",
                self.__class__.__name__, meta, image, tile_name)

        parameters = {
                'opt.ram'          : ram(self.ram_per_process),
                self.param_in      : image,
                # self.param_out     : out_filename,
                'interpolator'     : self.__interpolation_method,
                'outputs.spacingx' : spacing,
                'outputs.spacingy' : -spacing,
                'outputs.sizex'    : extent['xsize'],
                'outputs.sizey'    : extent['ysize'],
                'opt.gridspacing'  : self.__grid_spacing,
                'map'              : 'utm',
                'map.utm.zone'     : extent['utm_zone'],
                'map.utm.northhem' : extent['utm_northern'],
                'outputs.ulx'      : extent['xmin'],
                'outputs.uly'      : extent['ymax'],  # ymax, not ymin!!!
                'elev.dem'         : self.__tmp_dem_dir,
                'elev.geoid'       : self.__GeoidFile,
        }
        return parameters


class _ProjectGeoidTo(OTBStepFactory):
    def __init__(
        self,
        cfg: Configuration,
        *,
        name: str,
        gen_tmp_dir       : str,
        output_fname_fmt  : str,
        image_description : str,
        extended_filename : Optional[str],
    ) -> None:
        super().__init__(
            cfg,
            param_in="inr",
            param_out="out",
            appname='Superimpose',
            name=name,
            gen_tmp_dir=gen_tmp_dir,
            gen_output_dir=None,  # Use gen_tmp_dir
            gen_output_filename=TemplateOutputFilenameGenerator(output_fname_fmt),
            image_description=image_description,
            extended_filename=extended_filename,
        )
        self.__GeoidFile            = os.path.join(cfg.tmpdir, 'geoid', os.path.basename(cfg.GeoidFile))
        assert os.path.isfile(self.__GeoidFile), f"geoid file {self.__GeoidFile!r} is not accessible"
        self.__interpolation_method = cfg.interpolation_method
        self.__out_spatial_res      = cfg.out_spatial_res  # TODO: should extract this information from reference image
        self.__nodata               = nodata_DEM(cfg)

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set projection information, that'll be carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['GEOID_ORTHORECTIFICATION_INTERPOLATOR'] = self.__interpolation_method
        imd['SPATIAL_RESOLUTION']                    = str(self.__out_spatial_res)

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external+OTB:std:doc:`super impose
        <Applications/app_Superimpose>` to projected the Geoid onto the target geometry
        """
        in_s2_dem = in_filename(meta)
        return {
            'ram'                     : ram(self.ram_per_process),
            'inr'                     : in_s2_dem,  # Reference input is the DEM projected on S2
            'inm'                     : self.__GeoidFile,
            'interpolator'            : self.__interpolation_method,  # TODO: add parameter
            'interpolator.bco.radius' : 2,  # 2 is the default value for bco
            'fv'                      : self.__nodata,  # Make sure meta data are correctly set
        }


class _SARDEMProjectionFamily(OTBStepFactory):
    def __init__(
        self,
        cfg: Configuration,
        *,
        name: str,
        gen_tmp_dir       : str,
        output_fname_fmt  : str,
        image_description : str,
        extended_filename : Optional[str],
    ) -> None:
        """
        constructor
        """
        super().__init__(
            cfg,
            appname='SARDEMProjection2',
            name=name,
            param_in=None,
            param_out='out',
            gen_tmp_dir=gen_tmp_dir,
            gen_output_dir=None,  # Use gen_tmp_dir
            gen_output_filename=TemplateOutputFilenameGenerator(output_fname_fmt),
            image_description=image_description,
            extended_filename=extended_filename,
        )
        self.__dem_db_filepath   = cfg.dem_db_filepath
        self.__dem_field_ids     = cfg.dem_field_ids
        self.__dem_main_field_id = cfg.dem_main_field_id
        self.__dem_info          = cfg.dem_info
        self.__nodata            = nodata_XYZ(cfg)

    @property
    def nodata(self):
        """
        nodata_XYZ() accessor.
        """
        return self.__nodata

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:
        """
        Injects the :func:`reduce_inputs_insar` hook in step metadata, and
        provide names clear from polar related information.
        """
        depolarize_4_filename_pre_hook(meta)
        meta['reduce_inputs_insar'] = lambda inputs : [inputs[0]]  # TODO!!!
        return meta

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        - Complete meta information with hook for updating image metadata w/ directiontoscandemc,
          directiontoscandeml and gain.
        - Computes dem information and add them to the meta structure, to be used later to fill-in
          the image metadata.
        """
        meta = super().complete_meta(meta, all_inputs)
        append_to(meta, 'post', self.add_image_metadata)
        assert 'inputs' in meta, "Meta data shall have been filled with inputs"

        # TODO: The following has been duplicated from AgglomerateDEM.
        # See to factorize this code
        # find DEMs that intersect the input image
        meta['dem_infos'] = Utils.find_dem_intersecting_raster(
            in_filename(meta), self.__dem_db_filepath, self.__dem_field_ids, self.__dem_main_field_id
        )
        meta['dems'] = sorted(meta['dem_infos'].keys())

        logger.debug("%s: DEM found for %s: %s", self.name, in_filename(meta), meta['dems'])
        _, inbasename = os.path.split(in_filename(meta))
        meta['inbasename'] = inbasename
        return meta

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set SARDEMProjection related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['POLARIZATION'] = ""  # Clear polarization information (makes no sense here)
        imd['DEM_INFO']     = self.__dem_info
        imd['DEM_LIST']     = ', '.join(meta['dems'])

    def add_image_metadata(self, meta: Meta, app) -> None:  # pragma: no cover
        """
        Post-application hook used to complete GDAL metadata.

        As :func:`update_image_metadata` is not designed to access OTB application information
        (``directiontoscandeml``...), we need this extra hook to fetch and propagate the PRJ
        information.
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

    def requirement_context(self) -> str:
        """
        Return the requirement context that permits to fix missing requirements.
        SARDEMProjection2 comes from normlim_sigma0.
        """
        return "Please install https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0."


class _PostSARDEMProjectionFamily(OTBStepFactory):

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:
        """
        Injects the :func:`reduce_inputs_insar` hook in step metadata, and provide names clear from
        polar related information.
        """
        depolarize_4_filename_pre_hook(meta)
        meta['reduce_inputs_insar'] = lambda inputs : [inputs[0]]  # TODO!!!
        return meta

    def _get_canonical_input(self, inputs: InputList) -> AbstractStep:
        """
        Helper function to retrieve the canonical input associated to a list of inputs.

        In :class:`SARGammaAreaImageEstimation` case, the canonical input comes from the "indem"
        pipeline defined in :func:s1tiling.s1_process_gamma_area` pipeline builder.

        In :class:`SARCartesianMeanEstimation` case, the canonical input comes from the "indem"
        pipeline defined in :func:`s1tiling.s1_process_lia` pipeline builder.
        """
        _check_input_step_type(inputs)
        keys = set().union(*(input.keys() for input in inputs))
        assert len(inputs) == 3, f'Expecting 3 inputs. {len(inputs)} are found: {keys}'
        assert 'indemproj' in keys
        return [input['indemproj'] for input in inputs if 'indemproj' in input.keys()][0]

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        Complete meta information with hook for updating image metadata w/ directiontoscandemc,
        directiontoscandeml and gain.
        """
        inputpath = out_filename(meta)  # needs to be done before super.complete_meta!!
        meta = super().complete_meta(meta, all_inputs)
        if 'directiontoscandeml' not in meta or 'directiontoscandemc' not in meta:
            self.fetch_direction(inputpath, meta)
        indem     = fetch_input_data('indem',     all_inputs).out_filename
        indemproj = fetch_input_data('indemproj', all_inputs).out_filename
        meta['files_to_remove'] = [indem, indemproj]
        logger.debug('Register files to remove after %s computation: %s', self.name, meta['files_to_remove'])
        _, inbasename = os.path.split(in_filename(meta))
        meta['inbasename'] = inbasename
        return meta

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
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

    def fetch_direction(self, inputpath, meta: Meta) -> None:  # pragma: no cover
        """
        Extract back direction to scan DEM from SARDEMProjected image metadata.
        """
        logger.debug("Fetch PRJ.DIRECTIONTOSCANDEM* from '%s'", inputpath)
        if not is_running_dry(meta):  # FIXME: this info is no longer in meta!
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


class _SelectBestCoverage(_FileProducingStepFactory):
    def __init__(
        self,
        cfg: Configuration,
        *,
        name     :   str,
        fname_fmt:   str,
        dname_fmt:   str,
        gen_tmp_dir: str,
    ) -> None:
        super().__init__(
            cfg,
            name=name,
            gen_tmp_dir=gen_tmp_dir,
            gen_output_dir=dname_fmt,
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt)
        )

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:
        """
        Inject the :func:`reduce_inputs` hook in step metadata.
        """
        def reduce_inputs(inputs):
            """
            Select the concatenated pair of GAMMA AREA files that have the best coverage of the considered
            S2 tile.
            """
            # TODO: quid if different dates have best different coverage on a set of tiles?
            # How to avoid computing γ area/LIA again and again on a same S1 zone?
            # dates = set([re.sub(r'txxxxxx|t\d+', '', inp['acquisition_time']) for inp in inputs])
            best_covered_input = max(inputs, key=lambda inp: inp['tile_coverage'])
            logger.debug('Best coverage is %s at %s among:', best_covered_input['tile_coverage'], best_covered_input['acquisition_day'])
            for inp in inputs:
                logger.debug(' - %s: %s', inp['acquisition_day'], inp['tile_coverage'])
            return [best_covered_input]

        meta['reduce_inputs_in'] = reduce_inputs
        return meta

    def create_step(
        self,
        execution_parameters: Dict,
        previous_steps:       List[InputList]
    ) -> AbstractStep:
        logger.debug("Directly execute %s step", self.name)
        inputs = self._get_inputs(previous_steps)
        inp = self._get_canonical_input(inputs)
        meta = self.complete_meta(inp.meta, inputs)

        # Let's reuse commit_execution as it does exactly what we need
        if not is_running_dry(execution_parameters):
            commit_execution(out_filename(inp.meta), out_filename(meta))

        # Return a dummy Step
        # logger.debug("%s step executed!", self.name)
        res = AbstractStep('move', **meta)
        return res
