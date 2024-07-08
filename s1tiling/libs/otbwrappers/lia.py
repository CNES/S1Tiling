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
# Authors: Thierry KOLECK (CNES)
#          Luc HERMITTE (CS Group)
# =========================================================================

"""
This modules defines the specialized Python wrappers for the OTB Applications used in
the pipeline for LIA production needs.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Type
# from packaging import version

from osgeo import gdal
import otbApplication as otb

from s1tiling.libs.otbtools import otb_version

from ..file_naming   import (
        OutputFilenameGeneratorList, TemplateOutputFilenameGenerator,
)
from ..meta import (
        Meta, append_to, in_filename, out_filename, tmp_filename, is_running_dry,
)
from ..steps import (
        InputList, OTBParameters, ExeParameters,
        _check_input_step_type,
        AbstractStep, StepFactory,
        _FileProducingStepFactory, AnyProducerStepFactory, ExecutableStepFactory, OTBStepFactory,
        commit_execution,
        ram,
)
from ..otbpipeline   import (
    fetch_input_data, fetch_input_data_all_inputs, TaskInputInfo,
)
from .helpers        import (
        does_s2_data_match_s2_tile, does_sin_lia_match_s2_tile_for_orbit, remove_polarization_marks,
)
from .s1_to_s2       import (
        s2_tile_extent, _ConcatenatorFactory, _OrthoRectifierFactory,
)
from ..              import Utils
from ..configuration import (
        Configuration,
        dname_fmt_lia_product, dname_fmt_tiled,
        extended_filename_lia_degree, extended_filename_lia_sin, extended_filename_tiled,
        nodata_DEM, nodata_LIA, nodata_SAR, nodata_XYZ,
        pixel_type,
)

logger = logging.getLogger('s1tiling.wrappers.lia')


class AgglomerateDEMOnS2(AnyProducerStepFactory):
    """
    Factory that produces a :class:`Step` that builds a VRT from a list of DEM files.

    The choice has been made to name the VRT file after the basename of the
    root S1 product and not the names of the DEM tiles.
    """

    def __init__(self, cfg: Configuration, *args, **kwargs) -> None:
        """
        constructor
        """
        fname_fmt = 'DEM_{tile_name}.vrt'
        fname_fmt = cfg.fname_fmt.get('dem_s2_agglomeration', fname_fmt)
        super().__init__(  # type: ignore # mypy issue 4335
                cfg,
                # Because VRT links temporary files, it must not be reused in case of a crash => use tmp_dem_dir
                gen_tmp_dir=os.path.join(cfg.tmpdir, cfg.tmp_dem_dir),
                gen_output_dir=None,      # Use gen_tmp_dir,
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                name="AgglomerateDEMOnS2",
                action=AgglomerateDEMOnS2.agglomerate,
                *args, **kwargs)
        self.__cfg = cfg  # Will be used to access cached DEM intersecting S2 tile
        self.__dem_dir             = cfg.tmp_dem_dir
        self.__dem_filename_format = cfg.dem_filename_format

    @staticmethod
    def agglomerate(parameters: ExeParameters, dryrun: bool) -> None:
        """
        The function that calls :func:`gdal.BuildVRT()`.
        """
        logger.info("gdal.BuildVRT(%s, %s)", parameters[0], parameters[1:])
        assert len(parameters) > 0
        if not dryrun:
            gdal.BuildVRT(parameters[0], parameters[1:])

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        Factory that takes care of extracting meta data from S1 input files.
        """
        meta = super().complete_meta(meta, all_inputs)
        # find DEMs that intersect the input image
        meta['dem_infos'] = self.__cfg.get_dems_covering_s2_tile(meta['tile_name'])
        meta['dems'] = sorted(meta['dem_infos'].keys())
        logger.debug("DEM found for %s: %s", in_filename(meta), meta['dems'])
        dem_files = list(map(
                lambda s: os.path.join(
                    self.__dem_dir,    # Use copies/links from cached DEM directory
                    os.path.basename(  # => Strip any dirname from the input dem_filename_format
                        self.__dem_filename_format.format_map(meta['dem_infos'][s]))),
                meta['dem_infos']))
        meta['dem_files'] = dem_files
        missing_dems = list(filter(lambda f: not os.path.isfile(f), dem_files))
        if len(missing_dems) > 0:
            raise RuntimeError(
                    f"Cannot create DEM vrt for {meta['tile_name']}: the following DEM files are missing: {', '.join(missing_dems)}")
        return meta

    def parameters(self, meta: Meta) -> ExeParameters:
        # While it won't make much a difference here, we are still using tmp_filename.
        return [tmp_filename(meta)] + meta['dem_files']


class ProjectDEMToS2Tile(ExecutableStepFactory):
    """
    Factory that produces a :class:`ExecutableStep` that projects DEM onto target S2 tile
    as described in :ref:`Project DEM to S2 tile <project_dem_to_s2-proc>`.

    It requires the following information from the configuration object:

    - `ram_per_process`
    - `tmp_dir`
    - `fname_fmt`  -- optional key: `dem_on_s2`
    - `out_spatial_res`
    - `interpolation_method` -- OTB key converted to GDAL equivalent
    - `nb_procs`
    - `nodatas.DEM`

    It requires the following information from the metadata dictionary:

    - `tile_name`
    - `tile_origin`
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = 'DEM_projected_on_{tile_name}.tiff'
        fname_fmt = cfg.fname_fmt.get('dem_on_s2', fname_fmt)
        super().__init__(
                cfg,
                exename='gdalwarp', name='ProjectDEMToS2Tile',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
                gen_output_dir=None,      # Use gen_tmp_dir,
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description="Warped DEM to S2 tile",
        )
        self.__out_spatial_res   = cfg.out_spatial_res
        self.__resampling_method = cfg.dem_warp_resampling_method
        self.__nb_threads        = cfg.nb_procs
        self.__nodata            = nodata_DEM(cfg)

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        Register temporary files from previous step for removal.
        """
        meta = super().complete_meta(meta, all_inputs)

        in_file = in_filename(meta)
        meta['files_to_remove'] = [in_file]  # DEM VRT
        logger.debug('Register files to remove after DEM warping on S2 computation: %s', meta['files_to_remove'])
        return meta

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set S2 related information, that should have been carried around...
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['S2_TILE_CORRESPONDING_CODE'] = meta['tile_name']
        imd['SPATIAL_RESOLUTION']         = str(self.__out_spatial_res)
        imd['DEM_RESAMPLING_METHOD']      = self.__resampling_method
        # TODO: shall we set "ORTHORECTIFIED = True" ??
        # TODO: DEM_LIST

    def parameters(self, meta: Meta) -> ExeParameters:
        """
        Returns the parameters to use with :external:std:doc:`gdalwarp
        <programs/gdalwarp>` to projected the DEM onto the S2 geometry.
        """
        image       = in_filename(meta)
        tile_name   = meta['tile_name']
        tile_origin = meta['tile_origin']
        spacing     = self.__out_spatial_res
        logger.debug("%s.parameters(%s) /// image: %s /// tile_name: %s",
                self.__class__.__name__, meta, image, tile_name)

        extent = s2_tile_extent(tile_name, tile_origin, in_epsg=4326, spacing=spacing)

        parameters = [
                "-wm", str(self.ram_per_process*1024*1024),
                "-multi", "-wo", f"{self.__nb_threads}",  # It's already quite fast...
                "-t_srs", f"epsg:{extent['epsg']}",
                "-tr", f"{spacing}", f"-{spacing}",
                "-ot", "Float32",
                # "-crop_to_cutline",
                "-te", f"{extent['xmin']}", f"{extent['ymin']}", f"{extent['xmax']}", f"{extent['ymax']}",
                "-r", self.__resampling_method,
                "-dstnodata", str(self.__nodata),
                image,
                tmp_filename(meta),
        ]
        return parameters


class ProjectGeoidToS2Tile(OTBStepFactory):
    """
    Factory that produces a :class:`Step` that projects any kind of Geoid onto
    target S2 tile as described in :ref:`Project Geoid to S2 tile
    <project_geoid_to_s2-proc>`.

    This particular implementation uses another file in the expected geometry
    and :external:std:doc:`super impose <Applications/app_Superimpose>` the
    Geoid onto it. Unlike :external:std:doc:`gdalwarp <programs/gdalwarp>`,
    OTB application supports non-raster geoid formats.

    It requires the following information from the configuration object:

    - `ram_per_process`
    - `tmp_dir`    -- useless in the in-memory nomical case
    - `fname_fmt`  -- optional key: `geoid_on_s2`, useless in the in-memory nominal case
    - `interpolation_method` -- for use by :external:std:doc:`super impose
      <Applications/app_Superimpose>`
    - `out_spatial_res` -- as a workaround...
    - `nodatas.DEM`

    It requires the following information from the metadata dictionary:

    - `tile_name`
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = 'GEOID_projected_on_{tile_name}.tiff'
        fname_fmt = cfg.fname_fmt.get('geoid_on_s2', fname_fmt)
        super().__init__(
                cfg,
                param_in="inr", param_out="out",
                appname='Superimpose', name='ProjectGeoidToS2Tile',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
                gen_output_dir=None,      # Use gen_tmp_dir,
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description="Geoid superimposed on S2 tile",
        )
        self.__GeoidFile            = os.path.join(cfg.tmpdir, 'geoid', os.path.basename(cfg.GeoidFile))
        self.__interpolation_method = cfg.interpolation_method
        self.__out_spatial_res      = cfg.out_spatial_res  # TODO: should extract this information from reference image
        self.__nodata               = nodata_DEM(cfg)

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set S2 related information, that'll be carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['S2_TILE_CORRESPONDING_CODE'] = meta['tile_name']
        imd['SPATIAL_RESOLUTION']         = str(self.__out_spatial_res)

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external:std:doc:`super impose
        <Applications/app_Superimpose>` to projected the Geoid onto the S2 geometry.
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


class SumAllHeights(OTBStepFactory):
    """
    Factory that produces a :class:`Step` that adds DEM + Geoid that cover a
    same footprint, as described in :ref:`Sum DEM + Geoid
    <sum_dem_geoid_on_s2-proc>`.

    It requires the following information from the configuration object:

    - `ram_per_process`
    - `tmp_dir`    -- useless in the in-memory nomical case
    - `fname_fmt`  -- optional key: `height_on_s2`, useless in the in-memory nominal case
    - `nodata.DEM` -- optional

    It requires the following information from the metadata dictionary:

    """
    def __init__(self, cfg: Configuration) -> None:
        """
        Constructor.
        """
        fname_fmt = 'DEM+GEOID_projected_on_{tile_name}.tiff'
        fname_fmt = cfg.fname_fmt.get('height_on_s2', fname_fmt)
        super().__init__(
                cfg,
                appname='BandMath', name='SumAllHeights', param_in='il', param_out='out',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
                gen_output_dir=None,      # Use gen_tmp_dir,
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description='DEM + GEOID height info projected on S2 tile',
        )
        self.__nodata = nodata_DEM(cfg)

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        Complete meta information with inputs, and
        register temporary files from previous step for removal.
        """
        meta = super().complete_meta(meta, all_inputs)
        meta['inputs'] = all_inputs
        dem_on_s2  = fetch_input_data('in_s2_dem', all_inputs).out_filename
        meta['files_to_remove'] = [dem_on_s2]  # DEM on S2
        logger.debug('Register files to remove after height_on_S2 computation: %s', meta['files_to_remove'])
        # Make sure to set nodata metadata in output image
        meta['out_extended_filename_complement'] = f'?&nodata={self.__nodata}'
        return meta

    def _get_inputs(self, previous_steps: List[InputList]) -> InputList:
        """
        Extract the last inputs to use at the current level from all previous
        products seens in the pipeline.

        This method is overridden in order to fetch N-1 "in_s2_dem" input.
        It has been specialized for S1Tiling exact pipelines.
        """
        assert len(previous_steps) > 1

        # "in_s2_geoid" is expected at level -1, likelly named '__last'
        s2_geoid = fetch_input_data('__last', previous_steps[-1])
        # "in_s2_dem"     is expected at level -2, likelly named 'in_s2_dem'
        s2_dem   = fetch_input_data('in_s2_dem', previous_steps[-2])

        inputs = [{'in_s2_geoid': s2_geoid, 'in_s2_dem': s2_dem}]
        _check_input_step_type(inputs)
        logging.debug("%s inputs: %s", self.__class__.__name__, inputs)
        return inputs

    def _get_canonical_input(self, inputs: InputList) -> AbstractStep:
        """
        Helper function to retrieve the canonical input associated to a list of inputs.

        In current case, the canonical input comes from the "in_s2_geoid"
        step instanciated in :func:`s1tiling.s1_process_lia` pipeline builder.
        """
        _check_input_step_type(inputs)
        keys = set().union(*(input.keys() for input in inputs))
        assert len(keys) == 2, f'Expecting 2 inputs. {len(inputs)} is/are found: {keys}'
        assert 'in_s2_geoid' in keys
        return [input['in_s2_geoid'] for input in inputs if 'in_s2_geoid' in input.keys()][0]

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external:doc:`BandMath OTB
        application <Applications/app_BandMath>` for additionning DEM and Geoid
        data projected on S2.
        """
        assert 'inputs' in meta, f'Looking for "inputs" in {meta.keys()}'
        inputs = meta['inputs']
        in_s2_dem   = fetch_input_data('in_s2_dem',   inputs).out_filename
        in_s2_geoid = fetch_input_data('in_s2_geoid', inputs).out_filename
        dem_nodata = Utils.fetch_nodata_value(in_s2_dem, is_running_dry(meta), self.__nodata)  # usually -32768
        params : OTBParameters = {
                'ram'         : ram(self.ram_per_process),
                self.param_in : [in_s2_geoid, in_s2_dem],
                'exp'         : f'{Utils.test_nodata_for_bandmath(dem_nodata,"im2b1")} ? {self.__nodata} : im1b1+im2b1'
        }
        return params


class ComputeGroundAndSatPositionsOnDEM(OTBStepFactory):
    """
    Factory that prepares steps that run :external:doc:`Applications/app_SARDEMProjection`
    as described in :ref:`Normals computation` documentation to obtain the XYZ
    ECEF coordinates of the ground and of the satellite positions associated
    to the pixel from input the `heigth` file.

    :external:doc:`Applications/app_SARDEMProjection` application fill a
    multi-bands image anchored on the footprint of the input DEM image.
    In each pixel in the DEM/output image, we store the XYZ ECEF coordinate of
    the ground point (associated to the pixel), and the XYZ coordinates of the
    satellite position (associated to the pixel...)

    Requires the following information from the configuration object:

    - `ram_per_process`
    - `dem_db_filepath`   -- to fill-up image metadata
    - `dem_field_ids`     -- to fill-up image metadata
    - `dem_main_field_id` -- to fill-up image metadata
    - `tmp_dir`           -- useless in the in-memory nomical case
    - `fname_fmt`         -- optional key: `ground_and_sat_s2`, useless in the in-memory nominal case
    - `nodata.LIA`        -- optional

    Requires the following information from the metadata dictionary

    - `basename`
    - `input filename`
    - `output filename`

    It also requires :envvar:`$OTB_GEOID_FILE` to be set in order to ignore any
    DEM information already registered in dask worker (through
    :external:doc:`Applications/app_OrthoRectification` for instance) and only use
    the Geoid.
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = 'XYZ_projected_on_{tile_name}_{orbit_direction}_{orbit}.tiff'
        fname_fmt = cfg.fname_fmt.get('ground_and_sat_s2', fname_fmt)
        super().__init__(
                cfg,
                appname='SARDEMProjection2', name='SARDEMProjection',
                param_in=None, param_out='out',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
                gen_output_dir=None,  # Use gen_tmp_dir
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description="XYZ ground and satellite positions on S2 tile",
        )
        self.__cfg = cfg  # Will be used to access cached DEM intersecting S2 tile
        self.__nodata = nodata_XYZ(cfg)

    @staticmethod
    def reduce_inputs(inputs: List[Meta]) -> List:
        """
        Filters which insar input will be kept.

        Given several (usually 2 is the maximum) possible input S1 files
        ("insar" input channels), select and return the one that maximize its
        time coverage with the current S2 destination tile.

        Actually, this function returns the first S1 image that time-covers
        the whole S2 tile. The S1 images are searched in reverse order of
        their footprint-coverage.
        """
        # Sort by coverages (already computed), then return the first that time-covers everything
        # Or the first if none time-covers
        sorted_inputs = sorted(inputs, key=lambda inp: inp['tile_coverage'], reverse=True)
        best_covered_input = sorted_inputs[0]
        logger.debug('Best coverage is %.2f%% at %s among:', best_covered_input['tile_coverage'], best_covered_input['acquisition_time'])
        for inp in sorted_inputs:
            logger.debug(' - %s: %.2f%%', inp['acquisition_time'], inp['tile_coverage'])
        for inp in sorted_inputs:
            product = out_filename(inp).replace("measurement", "annotation").replace(".tiff", ".xml")
            az_start, az_stop, obt_start, obt_stop = Utils.get_s1image_orbit_time_range(product)
            dt = az_stop - az_start
            is_enough = (obt_start <= az_start - dt) and (az_stop + dt < obt_stop)
            logger.debug(" - %s AZ: %s, OBT: %s: 2xAZ ∈ OBT: %s", os.path.dirname(inp['manifest']), [str(az_start), str(az_stop)], [str(obt_start), str(obt_stop)], is_enough)
            if is_enough:
                logger.debug(
                        "Using %s which has orbit data that covers entirelly %s, and with a %.2f%% footprint coverage",
                        out_filename(best_covered_input), best_covered_input['tile_name'], best_covered_input['tile_coverage']
                )
                return [inp]
        logger.warning(
                "None of the orbit state vector sequence from input S1 products seems wide enough to cover entirelly %s tile. Returning %s which has the best footprint coverage: %.2f%%",
                best_covered_input['tile_name'], out_filename(best_covered_input), best_covered_input['tile_coverage']
        )
        return [best_covered_input]

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

        meta['reduce_inputs_insar'] = ComputeGroundAndSatPositionsOnDEM.reduce_inputs
        return meta

    def _update_filename_meta_post_hook(self, meta: Meta) -> None:
        """
        Register ``accept_as_compatible_input`` hook for
        :func:`s1tiling.libs.meta.accept_as_compatible_input`.
        It will tell whether a given heights file on S2 tile input is
        compatible with the current S2 tile.
        """
        meta['accept_as_compatible_input'] = lambda input_meta : does_s2_data_match_s2_tile(meta, input_meta)

    def _get_inputs(self, previous_steps: List[InputList]) -> InputList:
        """
        Extract the last inputs to use at the current level from all previous
        products seens in the pipeline.

        This method is overridden in order to fetch N-2 "insar" and "inheight" inputs.
        It has been specialized for S1Tiling exact pipelines.
        """
        for i, st in enumerate(previous_steps):
            logger.debug("INPUTS: %s previous step[%s] = %s", self.__class__.__name__, i, st)

        inputs = [fetch_input_data_all_inputs({"insar", "inheight"}, previous_steps)]
        _check_input_step_type(inputs)
        logging.debug("%s inputs: %s", self.__class__.__name__, inputs)
        return inputs

    def _get_canonical_input(self, inputs: InputList) -> AbstractStep:
        assert inputs, "No inputs found in ComputeGroundAndSatPositionsOnDEM"
        assert 'insar' in inputs[0], f"'insar' input is missing from ComputeGroundAndSatPositionsOnDEM inputs: {inputs[0].keys()}"
        return inputs[0]['insar']

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        Computes dem information and adds them to the meta structure, to be used
        later to fill-in the image metadata.

        Also register temporary files from previous step for removal.
        """
        # logger.debug("ComputeGroundAndSatPositionsOnDEM inputs are: %s", all_inputs)
        meta = super().complete_meta(meta, all_inputs)
        meta['inputs'] = all_inputs
        assert 'inputs' in meta, "Meta data shall have been filled with inputs"

        # Cannot register height_on_s2 for ulterior removal as the file can be
        # used with different orbits
        # => TODO count how many XYZ files depend on the height_on_s2 file
        # height_on_s2  = fetch_input_data('inheight', all_inputs)
        # meta['files_to_remove'] = [height_on_s2.out_filename]
        # logger.debug('Register files to remove after ground+satpos XYZ computation: %s', meta['files_to_remove'])

        sar = fetch_input_data('insar', all_inputs).meta

        # TODO: Check whether the DEM_LIST is already there and automatically propagated!
        meta['dem_infos'] = self.__cfg.get_dems_covering_s2_tile(meta['tile_name'])
        meta['dems'] = sorted(meta['dem_infos'].keys())

        logger.debug("SARDEMProjection: DEM found for %s: %s", in_filename(sar), meta['dems'])
        _, inbasename = os.path.split(in_filename(sar))
        meta['inbasename'] = inbasename
        return meta

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set SARDEMProjection related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['POLARIZATION']             = ""  # Clear polarization information (makes no sense here)
        imd['DEM_LIST']                 = ', '.join(meta['dems'])
        imd['band.DirectionToScanDEM*'] = ''
        imd['band.Gain']                = ''

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with
        :external:doc:`SARDEMProjection OTB application
        <Applications/app_SARDEMProjection>` to project S1 geometry onto DEM tiles.
        """
        nodata = self.__nodata
        assert 'inputs' in meta, f'Looking for "inputs" in {meta.keys()}'
        inputs = meta['inputs']
        inheight = fetch_input_data('inheight', inputs).out_filename
        insar    = fetch_input_data('insar'   , inputs).out_filename
        # `elev.geoid='@'` tells SARDEMProjection2 that GEOID shall not be used
        # from $OTB_GEOID_FILE, indeed geoid information is already in
        # DEM+Geoid input.
        return {
                'ram'        : ram(self.ram_per_process),
                'insar'      : insar,
                'indem'      : inheight,
                'elev.geoid' : '@',
                'withcryz'   : False,
                'withxyz'    : True,
                'withsatpos' : True,
                # 'withh'      : True,  # uncomment to analyse/debug height computed
                'nodata'     : str(nodata)
        }

    def requirement_context(self) -> str:
        """
        Return the requirement context that permits to fix missing requirements.
        SARDEMProjection2 comes from normlim_sigma0.
        """
        return "Please install https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0."


class _ComputeNormals(OTBStepFactory):
    """
    Abstract factory that prepares steps that run
    :external:doc:`ExtractNormalVector <Applications/app_ExtractNormalVector>`
    as described in :ref:`Normals computation <compute_normals-proc>` documentation.

    :external:doc:`ExtractNormalVector <Applications/app_ExtractNormalVector>`
    computes surface normals.

    Requires the following information from the configuration object:

    - `ram_per_process`
    - `nodata.LIA`      -- optional
    - `fname_fmt`       -- optional key: `normals`, useless in the in-memory nominal case

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """
    def __init__(
            self,
            cfg               : Configuration,
            gen_tmp_dir       : str,
            output_fname_fmt  : str,
            image_description : str,
    ) -> None:
        super().__init__(
                cfg,
                appname='ExtractNormalVector', name='ComputeNormals',
                param_in='xyz', param_out='out',
                gen_tmp_dir=gen_tmp_dir,
                gen_output_dir=None,  # Use gen_tmp_dir
                gen_output_filename=TemplateOutputFilenameGenerator(output_fname_fmt),
                image_description=image_description,
        )
        self.__nodata = nodata_XYZ(cfg)

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:
        """
        Injects ``polarless_basename`` -- Old LIA workflow case
        """
        # Ignore polarization in filenames
        assert 'polarless_basename' in meta
        assert meta['polarless_basename'] == remove_polarization_marks(meta['basename'])
        return meta

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        Override :func:`complete_meta()` to inject files to remove
        """
        meta = super().complete_meta(meta, all_inputs)
        in_file = in_filename(meta)
        meta['files_to_remove'] = [in_file]
        logger.debug('Register files to remove after normals computation: %s', meta['files_to_remove'])
        return meta

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with
        :external:doc:`ExtractNormalVector OTB application
        <Applications/app_ExtractNormalVector>` to generate surface normals
        for each point of the origin S1 image.
        """
        nodata = self.__nodata
        xyz = in_filename(meta)
        logger.debug("nodata(ComputeNormals) == %s", nodata)
        return {
                'ram'             : ram(self.ram_per_process),
                'xyz'             : xyz,
                'nodata'          : str(nodata),
        }

    def requirement_context(self) -> str:
        """
        Return the requirement context that permits to fix missing requirements.
        ComputeNormals comes from normlim_sigma0.
        """
        return "Please install https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0."


class ComputeNormalsOnS2(_ComputeNormals):
    """
    Factory that prepares steps that run
    :external:doc:`ExtractNormalVector <Applications/app_ExtractNormalVector>`
    on images in S2 geometry as described in :ref:`Normals
    computation <compute_normals-proc>` documentation.

    :external:doc:`ExtractNormalVector <Applications/app_ExtractNormalVector>`
    computes surface normals.

    Requires the following information from the configuration object:

    - `ram_per_process`
    - `nodata.LIA`      -- optional
    - `fname_fmt`       -- optional key: `normals_on_s2`, useless in the in-memory nominal case

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = 'Normals_on_{tile_name}'
        fname_fmt = cfg.fname_fmt.get('normals_on_s2', fname_fmt)
        super().__init__(
                cfg,
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2'),
                output_fname_fmt=fname_fmt,
                image_description='Image normals on S2 grid',
        )


class _ComputeLIA(OTBStepFactory):
    """
    Abstract factory that prepares steps that run
    :external:doc:`SARComputeLocalIncidenceAngle <Applications/app_SARComputeLocalIncidenceAngle>`
    as described in :ref:`LIA maps computation <compute_lia-proc>` documentation.

    :external:doc:`SARComputeLocalIncidenceAngle <Applications/app_SARComputeLocalIncidenceAngle>`
    computes Local Incidende Angle Map.

    Requires the following information from the configuration object:

    - `ram_per_process`
    - `nodata.LIA`      -- optional

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """
    def __init__(  # pylint: disable=too-many-arguments
            self,
            cfg               : Configuration,
            fname_fmt_sin     : str,
            fname_fmt_lia     : str,
            gen_tmp_dir       : str,
            gen_output_dir    : Optional[str],
            image_description : str,
    ) -> None:
        fname_fmt          = [ TemplateOutputFilenameGenerator(fname_fmt_sin) ]
        param_out          = ['out.sin']
        extended_filenames = [ extended_filename_lia_sin(cfg) ]
        pixel_types        = [ pixel_type(cfg, 'lia_sin') ]
        if cfg.produce_lia_map:
            # We always produce out.sin, and optionally we produce out.lia.
            # Anyway, their production is always done in output_dir!
            fname_fmt.append(TemplateOutputFilenameGenerator(fname_fmt_lia))
            param_out.append('out.lia')
            extended_filenames.append(extended_filename_lia_degree(cfg))
            pixel_types.append(pixel_type(cfg, 'lia_deg', 'uint16'))
        super().__init__(
                cfg,
                appname='SARComputeLocalIncidenceAngle', name='ComputeLIA',
                param_in='in.normals',  # In-memory connected to in.normals
                param_out=param_out,
                gen_tmp_dir=gen_tmp_dir,
                gen_output_dir=gen_output_dir,
                gen_output_filename=OutputFilenameGeneratorList(fname_fmt),
                image_description=image_description,
                extended_filename=extended_filenames,
                pixel_type=pixel_types,
        )
        self.__nodata = nodata_LIA(cfg)

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:
        """
        Injects ``polarless_basename``.
        """
        assert 'polarless_basename' in meta
        assert meta['polarless_basename'] == remove_polarization_marks(meta['basename'])
        return meta

    def _update_filename_meta_post_hook(self, meta: Meta) -> None:
        """
        Override "does_product_exist" hook to take into account the multiple
        output files produced by ComputeLIA
        """
        meta['does_product_exist'] = lambda : all(os.path.isfile(of) for of in out_filename(meta))

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        Complete meta information with inputs
        """
        meta = super().complete_meta(meta, all_inputs)
        meta['inputs'] = all_inputs
        return meta

    def _get_inputs(self, previous_steps: List[InputList]) -> InputList:
        """
        Extract the last inputs to use at the current level from all previous
        products seens in the pipeline.

        This method is overridden in order to fetch N-1 "xyz" input.
        It has been specialized for S1Tiling exact pipelines.
        """
        assert len(previous_steps) > 1

        # "normals" is expected at level -1, likelly named '__last'
        normals = fetch_input_data('__last', previous_steps[-1])
        # "xyz"     is expected at level -2, likelly named 'xyz'
        xyz = fetch_input_data('xyz', previous_steps[-2])

        inputs = [{'normals': normals, 'xyz': xyz}]
        _check_input_step_type(inputs)
        return inputs

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with
        :external:doc:`SARComputeLocalIncidenceAngle OTB application
        <Applications/app_SARComputeLocalIncidenceAngle>`.
        """
        assert 'inputs' in meta, f'Looking for "inputs" in {meta.keys()}'
        inputs = meta['inputs']
        xyz     = fetch_input_data('xyz', inputs).out_filename
        normals = fetch_input_data('normals', inputs).out_filename
        # TODO: should distinguish deg(LIA) nodata from sin(LIA) nodata
        nodata  = self.__nodata  # Best nodata value here is NaN
        return {
                'ram'             : ram(self.ram_per_process),
                'in.xyz'          : xyz,
                'in.normals'      : normals,
                'nodata'          : str(nodata),
        }

    def requirement_context(self) -> str:
        """
        Return the requirement context that permits to fix missing requirements.
        ComputeLIA comes from normlim_sigma0.
        """
        return "Please install https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0."


class ComputeLIAOnS2(_ComputeLIA):
    """
    Factory that prepares steps that run
    :external:doc:`SARComputeLocalIncidenceAngle <Applications/app_SARComputeLocalIncidenceAngle>`
    on images in S2 geometry as described in :ref:`LIA maps computation <compute_lia-proc>` documentation.

    :external:doc:`SARComputeLocalIncidenceAngle <Applications/app_SARComputeLocalIncidenceAngle>`
    computes Local Incidende Angle Map.

    Requires the following information from the configuration object:

    - `ram_per_process`
    - `fname_fmt`       -- optional key: `lia_product`
    - `dname_fmt`       -- optional key: `lia_product`
    - `nodata.LIA`      -- optional

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt0 = '{LIA_kind}_{flying_unit_code}_{tile_name}_{orbit_direction}_{orbit}.tif'
        fname_fmt0 = cfg.fname_fmt.get('lia_product', fname_fmt0)
        fname_fmt_lia = Utils.partial_format(fname_fmt0, LIA_kind="LIA")
        fname_fmt_sin = Utils.partial_format(fname_fmt0, LIA_kind="sin_LIA")
        dname_fmt = dname_fmt_lia_product(cfg)
        super().__init__(
                cfg,
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2'),
                gen_output_dir=dname_fmt,
                fname_fmt_lia=fname_fmt_lia,
                fname_fmt_sin=fname_fmt_sin,
                image_description='LIA on S2 grid',
        )


class _FilterLIAStepFactory(StepFactory):
    """
    Helper root class for all LIA/sin filtering steps.

    This class will be specialized on the fly by :func:`filter_LIA` which
    will inject the static data ``_LIA_kind``.

    Related step will forward the selected input under a new task-name (that differs from the filename).
    """

    # Useless definition used to trick pylint in believing self._LIA_kind is set.
    # Indeed, it's expected to be set in child classes. But pylint has now way to know that.
    _LIA_kind : Optional[str] = None

    def __init__(self, cfg: Configuration) -> None:
        """
        Constructor.
        Required to ignore the ``cfg`` parameter, and correctly forward the ``name`` parameter.
        """
        super().__init__(self.__class__.__name__)

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:
        meta = super()._update_filename_meta_pre_hook(meta)
        assert self._LIA_kind, "LIA kind should have been set in filter_LIA()"
        meta['LIA_kind'] = self._LIA_kind
        return meta

    def _update_filename_meta_post_hook(self, meta: Meta) -> None:
        """
        Update task name to avoid collision with inputs as file aren't renamed by this filter.
        """
        meta['task_name']        = f'{out_filename(meta)}_FilterLIA'

    def _get_input_image(self, meta: Meta) -> str:
        # Flatten should be useless, but kept for better error messages
        related_inputs = [f for f in Utils.flatten_stringlist(in_filename(meta))
                if re.search(rf'\b{self._LIA_kind}_', f)]
        assert len(related_inputs) == 1, (
            f"Incorrect number ({len(related_inputs)}) of S1 LIA products of type '{self._LIA_kind}' in {in_filename(meta)} found: {related_inputs}"
        )
        return related_inputs[0]

    def build_step_output_filename(self, meta: Meta) -> str:
        """
        Forward the output filename.
        """
        inp = self._get_input_image(meta)
        logger.debug('%s KEEP %s from %s', self.__class__.__name__, inp, in_filename(meta))
        return inp

    def build_step_output_tmp_filename(self, meta: Meta) -> str:
        """
        As there is no producer associated to :class:`_FilterLIAStepFactory`,
        there is no temporary filename.
        """
        return self.build_step_output_filename(meta)


def filter_LIA(LIA_kind: str) -> Type[_FilterLIAStepFactory]:
    """
    Generates a new :class:`StepFactory` class that filters which LIA product
    shall be processed: LIA maps or sin LIA maps.
    """
    # We return a new class
    return type(
            f"Filter_{LIA_kind}",      # Class name
            (_FilterLIAStepFactory,),  # Parent
            { '_LIA_kind': LIA_kind}
    )


class ApplyLIACalibration(OTBStepFactory):
    """
    Factory that concludes σ0 with NORMLIM calibration.

    It builds steps that multiply images calibrated with β0 LUT, and
    orthorectified to S2 grid, with the sin(LIA) map for the same S2 tile (and
    orbit number and direction).

    Requires the following information from the configuration object:

    - `ram_per_process`
    - lower_signal_value
    - `fname_fmt`       -- optional key: `s2_lia_corrected`
    - `dname_fmt`       -- optional key: `tiled`
    - `nodata.SAR`      -- optional
    - `nodata.LIA`      -- optional

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

    def __init__(self, cfg: Configuration) -> None:
        """
        Constructor.
        """
        fname_fmt = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_NormLim.tif'
        fname_fmt = cfg.fname_fmt.get('s2_lia_corrected', fname_fmt)
        dname_fmt = dname_fmt_tiled(cfg)
        super().__init__(
                cfg,
                appname='BandMath', name='ApplyLIACalibration', param_in='il', param_out='out',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
                gen_output_dir=dname_fmt,
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description='Sigma0 Normlim Calibrated Sentinel-{flying_unit_code_short} IW GRD',
                extended_filename=extended_filename_tiled(cfg),
                pixel_type=pixel_type(cfg, 'tiled'),
        )
        self.__lower_signal_value = cfg.lower_signal_value
        self.__nodata_SAR = nodata_SAR(cfg)
        self.__nodata_LIA = nodata_LIA(cfg)

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        Complete meta information with inputs, and set compression method to
        DEFLATE.
        """
        meta = super().complete_meta(meta, all_inputs)
        meta['inputs']           = all_inputs
        meta['calibration_type'] = 'Normlim'  # Update meta from now on

        # As of v1.1, when S2 product is marked required iff calibration_is_done_in_S1,
        # IOW, it's not required in normlim case, and we can safely remove the calibrated β0 file.
        in_concat_S2 = fetch_input_data('concat_S2', all_inputs).out_filename
        meta['files_to_remove'] = [in_concat_S2]
        # Make sure to set nodata metadata in output image
        meta['out_extended_filename_complement'] += f'&nodata={self.__nodata_SAR}'
        return meta

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set σ° normlim calibration related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        inputs = meta['inputs']
        in_sin_LIA   = fetch_input_data('sin_LIA',   inputs).out_filename
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['CALIBRATION'] = meta['calibration_type']
        imd['LIA_FILE']    = os.path.basename(in_sin_LIA)

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
        It will tell whether a given sin_LIA input is compatible with the
        current S2 tile.
        """
        meta['accept_as_compatible_input'] = lambda input_meta : does_sin_lia_match_s2_tile_for_orbit(meta, input_meta)
        meta['basename']                   = self._get_nominal_output_basename(meta)
        meta['calibration_type']           = 'Normlim'

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external:doc:`BandMath OTB application
        <Applications/app_BandMath>` for applying sin(LIA) to β0 calibrated
        image orthorectified to S2 tile.
        """
        assert 'inputs' in meta, f'Looking for "inputs" in {meta.keys()}'
        inputs = meta['inputs']
        in_concat_S2 = fetch_input_data('concat_S2', inputs).out_filename
        in_sin_LIA   = fetch_input_data('sin_LIA',   inputs).out_filename
        # We can expect consistency and blindy use LOWER_SIGNAL_VALUE from previous step
        lower_signal_value = self.__lower_signal_value
        # Read the nodata values from input images
        running_dry = is_running_dry(meta)
        sar_nodata = Utils.fetch_nodata_value(in_concat_S2, running_dry, self.__nodata_SAR)  # usually 0
        lia_nodata = Utils.fetch_nodata_value(in_sin_LIA,   running_dry, self.__nodata_LIA)  # usually what we have chosen, likelly -32768
        # exp is:
        # - if im{LIA} is LIA_nodata => SAR_nodata
        # - if im{SAR} is SAR_nodata = SAR_nodata
        # - else max(lower_signal_value, im{LIA} * im{SAR}
        # Note: if either is NaN, `max(?, nan*nan)` should be = NaN
        # => NaN should be supported, but tests are required
        is_LIA_nodata = Utils.test_nodata_for_bandmath(lia_nodata, "im2b1")
        is_SAR_nodata = Utils.test_nodata_for_bandmath(sar_nodata, "im1b1")
        params : OTBParameters = {
                'ram'         : ram(self.ram_per_process),
                self.param_in : [in_concat_S2, in_sin_LIA],
                'exp'         : f'({is_LIA_nodata} || {is_SAR_nodata}) ? {sar_nodata} : max({lower_signal_value}, im1b1*im2b1)'
        }
        return params


# ======================================================================
# Deprecated wrappers.
# They were used in S1Tiling 1.0 when the worflow was done in S1 SAR geometry
# until the production of the LIA map that was eventuall orthorectified and
# concatenated.

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
        meta['polarless_basename'] = remove_polarization_marks(meta['basename'])
        rootname = os.path.splitext(meta['polarless_basename'])[0]
        meta['polarless_rootname'] = rootname
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


class SARDEMProjection(OTBStepFactory):
    """
    Factory that prepares steps that run :external:doc:`Applications/app_SARDEMProjection`
    as described in :ref:`Normals computation` documentation.

    :external:doc:`Applications/app_SARDEMProjection` application puts a DEM file
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
                appname='SARDEMProjection2', name='SARDEMProjection',
                param_in=None, param_out='out',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
                gen_output_dir=None,  # Use gen_tmp_dir
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description="SARDEM projection onto DEM list",
        )
        self.__dem_db_filepath     = cfg.dem_db_filepath
        self.__dem_field_ids       = cfg.dem_field_ids
        self.__dem_main_field_id   = cfg.dem_main_field_id

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
        # meta['inputs'] = all_inputs

        # TODO: The following has been duplicated from AgglomerateDEM.
        # See to factorize this code
        # find DEMs that intersect the input image
        meta['dem_infos'] = Utils.find_dem_intersecting_raster(
            in_filename(meta), self.__dem_db_filepath, self.__dem_field_ids, self.__dem_main_field_id)
        meta['dems'] = sorted(meta['dem_infos'].keys())

        logger.debug("SARDEMProjection: DEM found for %s: %s", in_filename(meta), meta['dems'])
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
        :external:doc:`SARDEMProjection OTB application
        <Applications/app_SARDEMProjection>` to project S1 geometry onto DEM tiles.
        """
        nodata = meta.get('nodata', -32768)
        assert 'inputs' in meta, f'Looking for "inputs" in {meta.keys()}'
        inputs = meta['inputs']
        indem = fetch_input_data('indem', inputs).out_filename
        return {
                'ram'        : ram(self.ram_per_process),
                'insar'      : in_filename(meta),
                'indem'      : indem,
                'withxyz'    : True,
                # 'withh'      : True,  # uncomment to analyse/debug height computed
                'nodata'     : nodata
                }

    def requirement_context(self) -> str:
        """
        Return the requirement context that permits to fix missing requirements.
        SARDEMProjection2 comes from normlim_sigma0.
        """
        return "Please install https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0."


class SARCartesianMeanEstimation(OTBStepFactory):
    """
    Factory that prepares steps that run
    :external:doc:`Applications/app_SARCartesianMeanEstimation` as described in
    :ref:`Normals computation` documentation.


    :external:doc:`Applications/app_SARCartesianMeanEstimation` estimates a simulated
    cartesian mean image thanks to a DEM file.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename

    Note: It cannot be chained in memory because of the ``directiontoscandem*`` parameters.
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = 'XYZ_{polarless_basename}'
        fname_fmt = cfg.fname_fmt.get('xyz', fname_fmt)
        super().__init__(
                cfg,
                appname='SARCartesianMeanEstimation2', name='SARCartesianMeanEstimation',
                param_in=None, param_out='out',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
                gen_output_dir=None,  # Use gen_tmp_dir
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description='Cartesian XYZ coordinates estimation',
        )

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
        from the "indem" pipeline defined in :func:s1tiling.s1_process_lia`
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
        logger.debug('Register files to remove after XYZ computation: %s', meta['files_to_remove'])
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
        return {
                'ram'             : ram(self.ram_per_process),
                'insar'           : insar,
                'indem'           : indem,
                'indemproj'       : indemproj,
                'indirectiondemc' : int(meta['directiontoscandemc']),
                'indirectiondeml' : int(meta['directiontoscandeml']),
                'mlran'           : 1,
                'mlazi'           : 1,
        }

    def requirement_context(self) -> str:
        """
        Return the requirement context that permits to fix missing requirements.
        SARCartesianMeanEstimation2 comes from normlim_sigma0.
        """
        return "Please install https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0."


class ComputeNormalsOnS1(_ComputeNormals):
    """
    Factory that prepares steps that run
    :external:doc:`ExtractNormalVector <Applications/app_ExtractNormalVector>`
    on images in S1 geometry as described in :ref:`Normals
    computation <compute_normals-proc>` documentation.

    :external:doc:`ExtractNormalVector <Applications/app_ExtractNormalVector>`
    computes surface normals.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    - `fname_fmt`  -- optional key: `normals_on_s1`, useless in the in-memory nominal case
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = 'Normals_{polarless_basename}'
        fname_fmt = cfg.fname_fmt.get('normals_on_s1', fname_fmt)
        super().__init__(
                cfg,
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
                output_fname_fmt=fname_fmt,
                image_description='Image normals on Sentinel-{flying_unit_code_short} IW GRD',
        )


class ComputeLIAOnS1(_ComputeLIA):
    """
    Factory that prepares steps that run
    :external:doc:`SARComputeLocalIncidenceAngle <Applications/app_SARComputeLocalIncidenceAngle>`
    on images in S1 geometry as described in :ref:`LIA maps computation <compute_lia-proc>` documentation.

    :external:doc:`SARComputeLocalIncidenceAngle <Applications/app_SARComputeLocalIncidenceAngle>`
    computes Local Incidende Angle Map.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    - `fname_fmt`  -- optional key: `s1_lia`
    - `fname_fmt`  -- optional key: `s1_sin_lia`
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt_lia = cfg.fname_fmt.get('s1_lia',     'LIA_{polarless_basename}')
        fname_fmt_sin = cfg.fname_fmt.get('s1_sin_lia', 'sin_LIA_{polarless_basename}')
        super().__init__(
                cfg,
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
                gen_output_dir=None,
                fname_fmt_lia=fname_fmt_lia,
                fname_fmt_sin=fname_fmt_sin,
                image_description='LIA on Sentinel-{flying_unit_code_short} IW GRD',
        )


class OrthoRectifyLIA(_OrthoRectifierFactory):
    """
    Factory that prepares steps that run
    :external:doc:`Applications/app_OrthoRectification` on LIA maps.

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
        fname_fmt = '{LIA_kind}_{flying_unit_code}_{tile_name}_{orbit_direction}_{orbit}_{acquisition_time}.tif'
        fname_fmt = cfg.fname_fmt.get('lia_orthorectification', fname_fmt)
        super().__init__(
                cfg,
                fname_fmt,
                image_description='Orthorectified {LIA_kind} Sentinel-{flying_unit_code_short} IW GRD',
        )
        extra_ef = '&writegeom=false' if otb_version() < '8.0.0' else ''
        self._extended_filenames = {
                'LIA'     : extended_filename_lia_degree(cfg) + extra_ef,
                'sin_LIA' : extended_filename_lia_sin(cfg) + extra_ef,
        }

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:
        meta = super()._update_filename_meta_pre_hook(meta)
        assert 'LIA_kind' in meta, "This StepFactory shall be registered after a call to filter_LIA()"
        return meta

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        meta = super().complete_meta(meta, all_inputs)
        assert 'out_extended_filename_complement' not in meta, f'{meta["out_extended_filename_complement"]=!r} nothing was expected'
        kind = meta['LIA_kind']
        meta['out_extended_filename_complement'] = self._extended_filenames[kind]
        return meta

    def _get_input_image(self, meta: Meta) -> str:
        inp = in_filename(meta)
        assert isinstance(inp, str), f"A single string inp was expected, got {inp}"
        return inp   # meta['in_filename']

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
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

    def set_output_pixel_type(self, app, meta: Meta) -> None:
        """
        Force LIA output pixel type to ``INT16``.
        """
        if meta.get('LIA_kind', '') == 'LIA':
            app.SetParameterOutputImagePixelType(self.param_out, otb.ImagePixelType_int16)


class ConcatenateLIA(_ConcatenatorFactory):
    """
    Factory that prepares steps that run
    :external:doc:`Applications/app_Synthetize` on LIA images.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = '{LIA_kind}_{flying_unit_code}_{tile_name}_{orbit_direction}_{orbit}_{acquisition_day}.tif'
        fname_fmt = cfg.fname_fmt.get('lia_concatenation', fname_fmt)
        super().__init__(
                cfg,
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
                gen_output_dir=None,  # Use gen_tmp_dir
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
                image_description='Orthorectified {LIA_kind} Sentinel-{flying_unit_code_short} IW GRD',
                extended_filename=None,  # will be set later...
                pixel_type=None,         # will be set later...
        )
        self._extended_filenames = {
                'LIA'     : extended_filename_lia_degree(cfg),
                'sin_LIA' : extended_filename_lia_sin(cfg),
        }

    def _update_filename_meta_post_hook(self, meta: Meta) -> None:
        """
        Override "update_out_filename" hook to help select the input set with
        the best coverage.
        """
        assert 'LIA_kind' in meta
        meta['update_out_filename'] = self.update_out_filename  # <- needs to be done in post_hook!
        # Remove acquisition_time that no longer makes sense
        meta.pop('acquisition_time', None)

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Update concatenated LIA related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        imd = meta['image_metadata']
        imd['DEM_LIST']  = ""  # Clear DEM_LIST information (a merge of 2 lists should be done actually)

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        meta = super().complete_meta(meta, all_inputs)
        assert 'out_extended_filename_complement' not in meta, f'{meta["out_extended_filename_complement"]=!r} nothing was expected'
        kind = meta['LIA_kind']
        meta['out_extended_filename_complement'] = self._extended_filenames[kind]
        return meta

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
        logger.debug('[ConcatenateLIA] at %s:', date)
        coverage = 0.
        for inp in inputs:
            if re.sub(r'txxxxxx|t\d+', '', inp['acquisition_time']) == date:
                s1_cov = inp['tile_coverage']
                coverage += s1_cov
                logger.debug(' - %s => %s%% coverage', inp['basename'], s1_cov)
        # Round coverage at 3 digits as tile footprint has a very limited precision
        coverage = round(coverage, 3)
        logger.debug('[ConcatenateLIA] => total coverage at %s: %s%%', date, coverage * 100)
        meta['tile_coverage'] = coverage

    def set_output_pixel_type(self, app, meta: Meta) -> None:
        """
        Force LIA output pixel type to ``INT16``.
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

    Requires the following information from the metadata dictionary

    - `acquisition_day`
    - `tile_coverage`
    - `LIA_kind`
    - `flying_unit_code`
    - `tile_name`
    - `orbit_direction`
    - `orbit`
    - `fname_fmt`  -- optional key: `lia_product`
    - `dname_fmt`  -- optional key: `lia_product`
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = '{LIA_kind}_{flying_unit_code}_{tile_name}_{orbit_direction}_{orbit}.tif'
        fname_fmt = cfg.fname_fmt.get('lia_product', fname_fmt)
        dname_fmt = dname_fmt_lia_product(cfg)
        super().__init__(
                cfg,
                name='SelectBestCoverage',
                gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
                gen_output_dir=dname_fmt,
                gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
        )

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:
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
