#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2024 (c) CNES.
#   Copyright 2022-2024 (c) CS GROUP France.
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
# Authors: Fabien CONTIVAL (CS Group)
#          Luc HERMITTE (CS Group)
# =========================================================================

"""
This modules defines the specialized Python wrappers for the OTB Applications used in
the pipeline for GAMMA AREA production needs.
"""

import logging
import os
import re
from typing import Dict, List
# from packaging import version

from osgeo import gdal

from s1tiling.libs.otbtools import otb_version

from ..file_naming   import TemplateOutputFilenameGenerator
from ..meta import (
        Meta, append_to, in_filename, out_filename, tmp_filename, is_running_dry,
)
from ..steps import (
        InputList, OTBParameters, ExeParameters,
        _check_input_step_type,
        AbstractStep,
        _FileProducingStepFactory, AnyProducerStepFactory, OTBStepFactory,
        commit_execution,
        ram,
)
from ..otbpipeline   import (
    fetch_input_data, TaskInputInfo,
)
from .helpers        import (
        does_gamma_area_match_s2_tile_for_orbit, remove_polarization_marks,
)
from .s1_to_s2       import (
        _ConcatenatorFactory, _OrthoRectifierFactory,
)
from ..              import Utils
from ..configuration import (
        Configuration,
        dname_fmt_gamma_area_product, dname_fmt_tiled,
        extended_filename_gamma_area, extended_filename_tiled,
        pixel_type,
)


logger = logging.getLogger('s1tiling.wrappers.gamma_area')


class ApplyGammaNaughtRTCCalibration(OTBStepFactory):
    """
    Factory that concludes σ0 with GAMMA NAUGHT RTC calibration.

    It builds steps that multiply images calibrated with β0 LUT, and
    orthorectified to S2 grid, with the gamma area map for the same S2 tile (and
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
    - `fname_fmt`  -- optional key: `s2_gamma_area_corrected`
    - `dname_fmt`  -- optional key: `tiled`
    """

    def __init__(self, cfg: Configuration) -> None:
        """
        Constructor.
        """
        fname_fmt = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_GammaNaughtRTC.tif'
        fname_fmt = cfg.fname_fmt.get('s2_gamma_area_corrected', fname_fmt)
        dname_fmt = dname_fmt_tiled(cfg)

        super().__init__(
            cfg,
            appname='SARGammaAreaToGammaNaughtRTCImageEstimation', name='ApplyGammaNaughtRTCCalibration',
            param_in=None, param_out='out',
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
            gen_output_dir=dname_fmt,
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
            image_description='Gamma0 RTC Calibrated Sentinel-{flying_unit_code_short} IW GRD',
        )
        self.mingammaarea = cfg.min_gamma_area
        self.calibfactor  = cfg.calibration_factor
        self.nostreaming  = cfg.gamma_area_to_gamma_naught_rtc_nostreaming
        self.outputnodata = cfg.output_nodata

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        Complete meta information with inputs, and set compression method to
        DEFLATE.
        """
        meta = super().complete_meta(meta, all_inputs)
        meta['inputs']           = all_inputs
        meta['calibration_type'] = 'GammaNaughtRTC'  # Update meta from now on

        # As of v1.1, when S2 product is marked required iff calibration_is_done_in_S1,
        # IOW, it's not required in normlim case, and we can safely remove the calibrated β0 file.
        in_concat_S2 = fetch_input_data('concat_S2', all_inputs).out_filename
        meta['files_to_remove'] = [in_concat_S2]
        return meta

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set σ° normlim calibration related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        inputs = meta['inputs']
        in_GAMMA_AREA = fetch_input_data('GAMMA_AREA', inputs).out_filename
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['CALIBRATION']     = meta['calibration_type']
        imd['GAMMA_AREA_FILE'] = os.path.basename(in_GAMMA_AREA)

    def _get_canonical_input(self, inputs: InputList) -> AbstractStep:
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

    def _update_filename_meta_post_hook(self, meta: Meta) -> None:
        """
        Register ``accept_as_compatible_input`` hook for
        :func:`s1tiling.libs.meta.accept_as_compatible_input`.
        It will tell whether a given gamma area input is compatible with the
        current S2 tile.
        """
        meta['accept_as_compatible_input'] = lambda input_meta : does_gamma_area_match_s2_tile_for_orbit(meta, input_meta)
        meta['basename']                   = self._get_nominal_output_basename(meta)
        meta['calibration_type']           = 'GammaNaughtRTC'

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external:doc:`Gamma To Gamma Naught RTC application
        <Applications/app_SARGammaAreaToGammaNaughtRTCImageEstimation>` for applying GAMMA AREA to β0 calibrated
        image orthorectified to S2 tile.
        """
        assert 'inputs' in meta, f'Looking for "inputs" in {meta.keys()}'
        inputs = meta['inputs']
        in_concat_S2  = fetch_input_data('concat_S2',  inputs).out_filename
        in_GAMMA_AREA   = fetch_input_data('GAMMA_AREA',   inputs).out_filename
        params : OTBParameters = {
                'ram'         : ram(self.ram_per_process),
                'ingammaarea' : in_GAMMA_AREA,
                'inbetanaught': in_concat_S2,
                'mingammaarea': self.mingammaarea,
                'calibfactor' : self.calibfactor,
                'nostreaming' : self.nostreaming,
                'outputnodata': self.outputnodata,
                'nodata'      : 0,
        }
        return params


class AgglomerateDEMOnS1(AnyProducerStepFactory):
    """
    Factory that produces a :class:`Step` that builds a VRT from a list of DEM files.

    The choice has been made to name the VRT file after the basename of the
    root S1 product and not the names of the DEM tiles.
    """

    def __init__(self, cfg: Configuration, *args, **kwargs) -> None:
        """
        constructor
        """
        fname_fmt = 'DEM_{polarless_rootname}.vrt'
        fname_fmt = cfg.fname_fmt.get('dem_s1_agglomeration', fname_fmt)
        super().__init__(  # type: ignore # mypy issue 4335
            cfg,
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
            gen_output_dir=None,      # Use gen_tmp_dir,
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
            name="AgglomerateDEMOnS1",
            action=AgglomerateDEMOnS1.agglomerate,
            *args, **kwargs)
        self.__dem_db_filepath     = cfg.dem_db_filepath
        self.__dem_dir             = cfg.dem
        self.__dem_filename_format = cfg.dem_filename_format
        self.__dem_field_ids       = cfg.dem_field_ids
        self.__dem_main_field_id   = cfg.dem_main_field_id

    @staticmethod
    def agglomerate(parameters: ExeParameters, dryrun: bool) -> None:
        """
        The function that calls :func:`gdal.BuildVRT()`.
        """
        logger.info("gdal.BuildVRT(%s, %s)", parameters[0], parameters[1:])
        assert len(parameters) > 0
        if not dryrun:
            gdal.BuildVRT(parameters[0], parameters[1:])

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:
        """
        Injects the :func:`reduce_inputs_insar` hook in step metadata, and
        provide names clear from polar related information.
        """
        # Ignore polarization in filenames
        assert 'polarless_basename' not in meta
        meta['polarless_basename']  = remove_polarization_marks(meta['basename'])
        rootname = os.path.splitext(meta['polarless_basename'])[0]
        meta['polarless_rootname']  = rootname
        meta['reduce_inputs_insar'] = lambda inputs : [inputs[0]]  # TODO!!!
        return meta

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        Factory that takes care of extracting meta data from S1 input files.
        """
        meta = super().complete_meta(meta, all_inputs)
        # find DEMs that intersect the input image
        meta['dem_infos'] = Utils.find_dem_intersecting_raster(
            in_filename(meta), self.__dem_db_filepath, self.__dem_field_ids, self.__dem_main_field_id)
        meta['dems'] = sorted(meta['dem_infos'].keys())
        logger.debug("DEM found for %s: %s", in_filename(meta), meta['dems'])
        dem_files = map(
                lambda s: os.path.join(self.__dem_dir, self.__dem_filename_format.format_map(meta['dem_infos'][s])),
                meta['dem_infos'])
        missing_dems = list(filter(lambda f: not os.path.isfile(f), dem_files))
        if len(missing_dems) > 0:
            raise RuntimeError(
                    f"Cannot create DEM vrt for {meta['polarless_rootname']}: the following DEM files are missing: {', '.join(missing_dems)}")
        return meta

    def parameters(self, meta: Meta) -> ExeParameters:
        # While it won't make much a difference here, we are still using tmp_filename.
        return [tmp_filename(meta)] \
                + [os.path.join(self.__dem_dir,
                                self.__dem_filename_format.format_map(meta['dem_infos'][s]))
                   for s in meta['dem_infos']]


class ResampleDEM(OTBStepFactory):
    """
    Factory that prepares steps that run :external:doc:`Applications/app_RigidTransformResample`
    as described in :ref:`Gamma area computation` documentation.

    :external:doc:`Applications/app_RigidTransformResample` application resample a DEM by some factor (at least 2).

    Requires the following information from the configuration object:

    - `ram_per_process`
    - `dem_db_filepath`   -- to fill-up image metadata
    - `dem_field_ids`     -- to fill-up image metadata
    - `dem_main_field_id` -- to fill-up image metadata
    - `tmp_dir`           -- useless in the in-memory nomical case
    - `fname_fmt`         -- optional key: `s1_on_geoid_dem`, useless in the in-memory nominal case

    Requires the following information from the metadata dictionary

    - `basename`
    - `input filename`
    - `output filename`
    - `nodata` -- optional
    """

    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = 'RESAMPLED_DEM_{polarless_basename}'
        fname_fmt = cfg.fname_fmt.get('resampled_dem', fname_fmt)
        super().__init__(
            cfg,
            appname='RigidTransformResample', name='ResampleDEM',
            param_in=None, param_out='out',
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
            gen_output_dir=None,  # Use gen_tmp_dir
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
            image_description="DEM resampling",
        )
        self.factor_x            = cfg.resample_dem_factor_x
        self.factor_y            = cfg.resample_dem_factor_y

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:
        """
        Injects the :func:`reduce_inputs_insar` hook in step metadata, and
        provide names clear from polar related information.
        """
        # Ignore polarization in filenames
        if 'polarless_basename' in meta:
            assert meta['polarless_basename'] == remove_polarization_marks(meta['basename'])
        else:
            meta['polarless_basename'] = remove_polarization_marks(meta['basename'])

        meta['reduce_inputs_insar'] = lambda inputs: [inputs[0]]  # TODO!!!
        return meta

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        - Complete meta information with hook for updating image metadata
          w/ directiontoscandemc, directiontoscandeml and gain.
        - Computes dem information and add them to the meta structure, to be used
          later to fill-in the image metadata.
        """
        meta = super().complete_meta(meta, all_inputs)
        meta['inputs'] = all_inputs
        assert 'inputs' in meta, "Meta data shall have been filled with inputs"

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

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with
        :external:doc:`RigidTransformResample OTB application
        <Applications/app_RigidTransformResample>` to resample DEM.
        """
        assert 'inputs' in meta, f'Looking for "inputs" in {meta.keys()}'
        inputs = meta['inputs']
        indem  = fetch_input_data('indem', inputs).out_filename

        params : OTBParameters = {
                "ram"                      : ram(self.ram_per_process),
                "in"                       : indem,
                "transform.type"           : "id",
                "transform.type.id.scalex" : self.factor_x,
                "transform.type.id.scaley" : self.factor_y,
        }

        return params

    def requirement_context(self) -> str:
        """
        Return the requirement context that permits to fix missing requirements.
        RigidTransformResample comes from gamma0-rtc.
        """
        return "Please install https://gitlab.orfeo-toolbox.org/s1-tiling/gamma0-rtc."


class SARDEMProjectionImageEstimation(OTBStepFactory):
    """
    Factory that prepares steps that run :external:doc:`Applications/app_SARDEMProjectionImageEstimation`
    as described in :ref:`Normals computation` documentation.

    :external:doc:`Applications/app_SARDEMProjectionImageEstimation` application puts a DEM file
    into SAR geometry and estimates two additional coordinates.
    For each point of the DEM input four components are calculated:
    C (colunm into SAR image), L (line into SAR image), Z and Y. XYZ cartesian
    components into projection are also computed for our needs.

    Requires the following information from the configuration object:

    - `ram_per_process`
    - `dem_db_filepath`   -- to fill-up image metadata
    - `dem_field_ids`     -- to fill-up image metadata
    - `dem_main_field_id` -- to fill-up image metadata
    - `tmp_dir`           -- useless in the in-memory nomical case
    - `fname_fmt`         -- optional key: `s1_on_dem`, useless in the in-memory nominal case

    Requires the following information from the metadata dictionary

    - `basename`
    - `input filename`
    - `output filename`
    - `nodata` -- optional

    It also requires :envvar:`$OTB_GEOID_FILE` to be set in order to ignore any
    DEM information already registered in dask worker (through
    :external:doc:`Applications/app_OrthoRectification` for instance) and only use
    the Geoid.
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = 'S1_on_DEM_{polarless_basename}'
        fname_fmt = cfg.fname_fmt.get('s1_on_dem', fname_fmt)
        super().__init__(
                cfg,
                appname='SARDEMProjectionImageEstimation', name='SARDEMProjectionImageEstimation',
                param_in=None, param_out='out',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
                gen_output_dir=None,  # Use gen_tmp_dir
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description="SARDEM projection onto DEM list",
        )
        self.__dem_db_filepath   = cfg.dem_db_filepath
        self.__dem_field_ids     = cfg.dem_field_ids
        self.__dem_main_field_id = cfg.dem_main_field_id
        self.__GeoidFile         = os.path.join(cfg.tmpdir, 'geoid', os.path.basename(cfg.GeoidFile))

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:
        """
        Injects the :func:`reduce_inputs_insar` hook in step metadata, and
        provide names clear from polar related information.
        """
        # Ignore polarization in filenames
        if 'polarless_basename' in meta:
            assert meta['polarless_basename'] == remove_polarization_marks(meta['basename'])
        else:
            meta['polarless_basename'] = remove_polarization_marks(meta['basename'])

        meta['reduce_inputs_insar'] = lambda inputs : [inputs[0]]  # TODO!!!
        return meta

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        - Complete meta information with hook for updating image metadata
          w/ directiontoscandemc, directiontoscandeml and gain.
        - Computes dem information and add them to the meta structure, to be used
          later to fill-in the image metadata.
        """
        meta = super().complete_meta(meta, all_inputs)
        append_to(meta, 'post', self.add_image_metadata)
        assert 'inputs' in meta, "Meta data shall have been filled with inputs"

        # TODO: The following has been duplicated from AgglomerateDEM.
        # See to factorize this code
        # find DEMs that intersect the input image
        meta['dem_infos'] = Utils.find_dem_intersecting_raster(
            in_filename(meta), self.__dem_db_filepath, self.__dem_field_ids, self.__dem_main_field_id)
        meta['dems'] = sorted(meta['dem_infos'].keys())

        logger.debug("SARDEMProjectionImageEstimation: DEM found for %s: %s", in_filename(meta), meta['dems'])
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
        imd['DEM_LIST']     = ', '.join(meta['dems'])

    def add_image_metadata(self, meta: Meta, app) -> None:
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

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with
        :external:doc:`SARDEMProjectionImageEstimation OTB application
        <Applications/app_SARDEMProjectionImageEstimation>` to project S1 geometry onto DEM tiles.
        """
        assert 'inputs' in meta, f'Looking for "inputs" in {meta.keys()}'
        inputs = meta['inputs']
        indem = fetch_input_data('indem', inputs).out_filename

        params : OTBParameters = {
                'ram'       : ram(self.ram_per_process),
                'insar'     : in_filename(meta),
                'indem'     : indem,
                'withxyz'   : True,
                'nodata'    : -32768,
                'elev.geoid': self.__GeoidFile,
        }

        return params

    def requirement_context(self) -> str:
        """
        Return the requirement context that permits to fix missing requirements.
        SARDEMProjectionImageEstimation comes from gamma0-rtc.
        """
        return "Please install https://gitlab.orfeo-toolbox.org/s1-tiling/gamma0-rtc."


class SARGammaAreaImageEstimation(OTBStepFactory):
    """
    Factory that prepares steps that run
    :external:doc:`Applications/app_SARGammaAreaImageEstimation` as described in
    :ref:`Normals computation` documentation.


    :external:doc:`Applications/app_SARGammaAreaImageEstimation` estimates a simulated
    cartesian mean image thanks to a DEM file.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename

    Note: It cannot be chained in memory because of the ``directiontoscandem*`` parameters.
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = 'GAMMA_AREA_{polarless_basename}'
        fname_fmt = cfg.fname_fmt.get('gamma_area', fname_fmt)
        super().__init__(
                cfg,
                appname='SARGammaAreaImageEstimation', name='SARGammaAreaImageEstimation',
                param_in=None, param_out='out',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
                gen_output_dir=None,  # Use gen_tmp_dir
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description='Gamma area image estimation',
        )
        self.distributearea         = cfg.distribute_area
        self.nostreaming            = cfg.gamma_area_nostreaming
        self.innermarginratiostatus = cfg.inner_margin_ratio_status
        self.outermarginratiostatus = cfg.outer_margin_ratio_status
        self.innermarginratio       = cfg.inner_margin_ratio
        self.outermarginratio       = cfg.outer_margin_ratio

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:
        """
        Injects the :func:`reduce_inputs_insar` hook in step metadata, and
        provide names clear from polar related information.
        """
        # Ignore polarization in filenames
        if 'polarless_basename' in meta:
            assert meta['polarless_basename'] == remove_polarization_marks(meta['basename'])
        else:
            meta['polarless_basename'] = remove_polarization_marks(meta['basename'])
        meta['reduce_inputs_insar'] = lambda inputs : [inputs[0]]  # TODO!!!
        return meta

    def _get_canonical_input(self, inputs: InputList) -> AbstractStep:
        """
        Helper function to retrieve the canonical input associated to a list of inputs.

        In :class:`SARCartesianMeanEstimation` case, the canonical input comes
        from the "indem" pipeline defined in :func:s1tiling.s1_process_gamma_area`
        pipeline builder.
        """
        _check_input_step_type(inputs)
        keys = set().union(*(input.keys() for input in inputs))
        assert len(inputs) == 3, f'Expecting 3 inputs. {len(inputs)} are found: {keys}'
        assert 'indemproj' in keys
        return [input['indemproj'] for input in inputs if 'indemproj' in input.keys()][0]

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        Complete meta information with hook for updating image metadata
        w/ directiontoscandemc, directiontoscandeml and gain.
        """
        inputpath = out_filename(meta)  # needs to be done before super.complete_meta!!
        meta = super().complete_meta(meta, all_inputs)
        meta['inputs'] = all_inputs
        if 'directiontoscandeml' not in meta or 'directiontoscandemc' not in meta:
            self.fetch_direction(inputpath, meta)
        indem     = fetch_input_data('indem',     all_inputs).out_filename
        indemproj = fetch_input_data('indemproj', all_inputs).out_filename
        meta['files_to_remove'] = [indem, indemproj]
        logger.debug('Register files to remove after GAMMA_AREA computation: %s', meta['files_to_remove'])
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

    def fetch_direction(self, inputpath, meta: Meta) -> None:
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

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with
        :external:doc:`SARCartesianMeanEstimation OTB application
        <Applications/app_SARCartesianMeanEstimation>` to compute cartesian
        coordinates of each point of the origin S1 image.
        """
        assert 'inputs' in meta, f'Looking for "inputs" in {meta.keys()}'
        inputs = meta['inputs']
        insar     = fetch_input_data('insar', inputs).out_filename
        indem     = fetch_input_data('indem', inputs).out_filename
        indemproj = fetch_input_data('indemproj', inputs).out_filename

        params : OTBParameters = {
                'ram'                   : ram(self.ram_per_process),
                'insar'                 : insar,
                'indem'                 : indem,
                'indemproj'             : indemproj,
                'indirectiondemc'       : int(meta['directiontoscandemc']),
                'indirectiondeml'       : int(meta['directiontoscandeml']),
                'mlran'                 : 1,
                'mlazi'                 : 1,
                'distributearea'        : self.distributearea,
                'nostreaming'           : self.nostreaming,
                'nodata'                : -32768,
                'innermarginratiostatus': self.innermarginratiostatus,
                'outermarginratiostatus': self.outermarginratiostatus,
        }
        if self.innermarginratio:
            params["innermarginratio"] = self.innermarginratio
        if self.outermarginratio:
            params["outermarginratio"] = self.outermarginratio

        return params

    def requirement_context(self) -> str:
        """
        Return the requirement context that permits to fix missing requirements.
        SARGammaAreaImageEstimation comes from gamma0-rtc.
        """
        return "Please install https://gitlab.orfeo-toolbox.org/s1-tiling/gamma0-rtc."


class ConcatenateGAMMA_AREA(_ConcatenatorFactory):
    """
    Factory that prepares steps that run
    :external:doc:`Applications/app_Synthetize` on GAMMA_AREA images.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = 'GAMMA_AREA_{flying_unit_code}_{tile_name}_{orbit_direction}_{orbit}_{acquisition_day}.tif'
        fname_fmt = cfg.fname_fmt.get('gamma_area_concatenation', fname_fmt)
        super().__init__(
                cfg,
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
                gen_output_dir=None,  # Use gen_tmp_dir
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description='Orthorectified GAMMA_AREA Sentinel-{flying_unit_code_short} IW GRD',
                extended_filename=extended_filename_gamma_area(cfg),
                pixel_type=None,         # will be set later...
        )

    def _update_filename_meta_post_hook(self, meta: Meta) -> None:
        """
        Override "update_out_filename" hook to help select the input set with
        the best coverage.
        """
        meta['update_out_filename'] = self.update_out_filename  # <- needs to be done in post_hook!
        # Remove acquisition_time that no longer makes sense
        meta.pop('acquisition_time', None)

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Update concatenated GAMMA AREA related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        imd = meta['image_metadata']
        imd['DEM_LIST']  = ""  # Clear DEM_LIST information (a merge of 2 lists should be done actually)
        #imd['POLARIZATION'] = ""  # Clear polarization information (makes no sense here)

    def update_out_filename(self, meta: Meta, with_task_info: TaskInputInfo) -> None:
        """
        Unlike usual :class:`Concatenate`, the output filename will always ends
        in "txxxxxx".

        However we want to update the coverage of the current pair as a new
        input file has been registered.

        TODO: Find a better name for the hook as it handles two different
        services.
        """
        inputs = with_task_info.inputs['in']
        dates = {re.sub(r'txxxxxx|t\d+', '', inp['acquisition_time']) for inp in inputs}
        assert len(dates) == 1, f"All concatenated files shall have the same date instead of {dates}"
        date = min(dates)
        logger.debug('[ConcatenateGAMMA_AREA] at %s:', date)
        coverage = 0.
        for inp in inputs:
            if re.sub(r'txxxxxx|t\d+', '', inp['acquisition_time']) == date:
                s1_cov = inp['tile_coverage']
                coverage += s1_cov
                logger.debug(' - %s => %s%% coverage', inp['basename'], s1_cov)
        # Round coverage at 3 digits as tile footprint has a very limited precision
        coverage = round(coverage, 3)
        logger.debug('[ConcatenateGAMMA_AREA] => total coverage at %s: %s%%', date, coverage * 100)
        meta['tile_coverage'] = coverage

class OrthoRectifyGAMMA_AREA(_OrthoRectifierFactory):
    """
    Factory that prepares steps that run
    :external:doc:`Applications/app_OrthoRectification` on GAMMA AREA maps.

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
        fname_fmt = 'GAMMA_AREA_{flying_unit_code}_{tile_name}_{orbit_direction}_{orbit}_{acquisition_time}.tif'
        fname_fmt = cfg.fname_fmt.get('gamma_area_orthorectification', fname_fmt)
        extended_filename = extended_filename_gamma_area(cfg)
        if otb_version() < '8.0.0':
            extended_filename += '&writegeom=false'
        super().__init__(
                cfg,
                fname_fmt,
                image_description='Orthorectified GAMMA_AREA Sentinel-{flying_unit_code_short} IW GRD',
                extended_filename=extended_filename,
        )

    def _get_input_image(self, meta: Meta) -> str:
        inp = in_filename(meta)
        assert isinstance(inp, str), f"A single string inp was expected, got {inp}"
        return inp   # meta['in_filename']

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set DATA_TYPE metadata, and prevent PixelSpacing and LineSpacing from beeing discarded
        (which is :func:`_OrthoRectifierFactory.update_image_metadata` default behaviour)
        """
        super().update_image_metadata(meta, all_inputs)
        imd = meta['image_metadata']
        imd['DATA_TYPE']    = 'meters^2'
        # Original Line/PixelSpacing should not be discarded => unregister its removal
        assert 'PixelSpacing' in imd,     "PixelSpacing should have been registered for removal. Let's keep it!"
        assert 'LineSpacing' in imd,      "LineSpacing should have been registered for removal. Let's keep it!"
        assert imd['PixelSpacing'] == '', "PixelSpacing should have been registered for removal. Let's keep it!"
        assert imd['LineSpacing'] == '',  "LineSpacing should have been registered for removal. Let's keep it!"
        del imd['LineSpacing']
        del imd['PixelSpacing']

class SelectGammaNaughtAreaBestCoverage(_FileProducingStepFactory):
    """
    StepFactory that helps select only one path after GAMMA AREA concatenation:
    the one that have the best coverage of the S2 tile target.

    If several concatenated products have the same coverage, the oldest one
    will be selected.

    The coverage is extracted from ``tile_coverage`` step metadata.

    The step produced does nothing: it only rename the selected product
    into the final expected name. Note: in GAMMA AREA case two files will actually
    renamed.

    Requires the following information from the metadata dictionary

    - `acquisition_day`
    - `tile_coverage`
    - `flying_unit_code`
    - `tile_name`
    - `orbit_direction`
    - `orbit`
    - `fname_fmt`  -- optional key: `gamma_area_product`
    - `dname_fmt`  -- optional key: `gamma_area_product`
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = 'GAMMA_AREA_{flying_unit_code}_{tile_name}_{orbit_direction}_{orbit}.tif'
        fname_fmt = cfg.fname_fmt.get('gamma_area', fname_fmt)
        dname_fmt = dname_fmt_gamma_area_product(cfg)
        super().__init__(
                cfg,
                name='SelectGammaNaughtAreaBestCoverage',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
                gen_output_dir=dname_fmt,
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt)
        )

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:
        """
        Inject the :func:`reduce_GAMMA_AREAs` hook in step metadata.
        """
        def reduce_GAMMA_AREAs(inputs):
            """
            Select the concatenated pair of GAMMA AREA files that have the best coverage of the considered
            S2 tile.
            """
            # TODO: quid if different dates have best different coverage on a set of tiles?
            # How to avoid computing GAMMA_NAUGHT AREA again and again on a same S1 zone?
            # dates = set([re.sub(r'txxxxxx|t\d+', '', inp['acquisition_time']) for inp in inputs])
            best_covered_input = max(inputs, key=lambda inp: inp['tile_coverage'])
            logger.debug('Best coverage is %s at %s among:', best_covered_input['tile_coverage'], best_covered_input['acquisition_day'])
            for inp in inputs:
                logger.debug(' - %s: %s', inp['acquisition_day'], inp['tile_coverage'])
            return [best_covered_input]

        meta['reduce_inputs_in'] = reduce_GAMMA_AREAs
        return meta

    def create_step(
            self,
            execution_parameters: Dict,
            previous_steps: List[InputList]
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
