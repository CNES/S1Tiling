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
#
# =========================================================================

"""
This modules defines the specialized Python wrappers for the OTB Applications used in
the pipeline for LIA production needs.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Type

from osgeo import gdal
import otbApplication as otb


from ..file_naming     import (
    OutputFilenameGenerator,
    OutputFilenameGeneratorList,
    TemplateOutputFilenameGenerator,
)
from ..incidence_angle import IA_map, extended_filename_ia, pixel_type_ia

from ..meta            import (
    Meta,
    in_filename,
    out_filename,
    tmp_filename,
    is_running_dry,
)
from ..otbtools        import otb_version
from ..steps           import (
    AbstractStep,
    AnyProducerStepFactory,
    ExeParameters,
    ExecutableStepFactory,
    InputList,
    OTBParameters,
    OTBStepFactory,
    StepFactory,
    _check_input_step_type,
    ram,
)
from ..otbpipeline     import (
    fetch_input_data,
    fetch_input_data_all_inputs,
)
from ._applications import (
    _ConcatenatorFactoryForMaps,
    _OrthoRectifierFactory,
    _PostSARDEMProjectionFamily,
    _ProjectGeoidTo,
    _SARDEMProjectionFamily,
    _SelectBestCoverage,
    s2_tile_extent,
)
from .helpers          import (
    depolarize_4_filename_pre_hook,
    does_s2_data_match_s2_tile,
    does_sin_lia_match_s2_tile_for_orbit,
)
from ..                 import Utils
from ..utils.formatters import partial_format
from ..configuration    import (
    DEFAULT_FNAME_FMTS,
    Configuration,
    dname_fmt_lia_product,
    dname_fmt_tiled,
    extended_filename_hidden,
    extended_filename_lia_degree,
    extended_filename_lia_sin,
    extended_filename_s1_on_dem,
    extended_filename_tiled,
    fname_fmt_lia_corrected,
    nodata_DEM,
    nodata_LIA,
    nodata_SAR,
    nodata_XYZ,
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
            gen_output_dir=None,  # Use gen_tmp_dir,
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
            name="AgglomerateDEMOnS2",
            action=AgglomerateDEMOnS2.agglomerate,
            *args,
            **kwargs,
        )
        self.__cfg = cfg  # Will be used to access cached DEM intersecting S2 tile
        self.__dem_dir             = cfg.tmp_dem_dir
        self.__dem_filename_format = cfg.dem_filename_format

    @staticmethod
    def agglomerate(parameters: ExeParameters, dryrun: bool) -> None:  # pragma: no cover
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
        # Sort the parameters to have reproductible tests
        meta['dem_files'] = sorted(dem_files)
        missing_dems = list(filter(lambda f: not os.path.isfile(f), dem_files))
        if len(missing_dems) > 0:
            raise RuntimeError(f"Cannot create DEM vrt for {meta['tile_name']}: the following DEM files are missing: {', '.join(missing_dems)}")
        return meta

    def parameters(self, meta: Meta) -> ExeParameters:
        # While it won't make much a difference here, we are still using tmp_filename.
        return [tmp_filename(meta)] + meta['dem_files']


class ProjectDEMToS2Tile(ExecutableStepFactory):
    """
    Factory that produces a :class:`ExecutableStep` that projects DEM onto target S2 tile as
    described in :ref:`Project DEM to S2 tile <project_dem_to_s2-proc>`.

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
            exename='gdalwarp',
            name='ProjectDEMToS2Tile',
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
            gen_output_dir=None,  # Use gen_tmp_dir,
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
            image_description="Warped DEM to S2 tile",
        )
        self.__dem_info          = cfg.dem_info
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
        imd['DEM_INFO']                   = self.__dem_info
        imd['DEM_RESAMPLING_METHOD']      = self.__resampling_method
        imd['ORTHORECTIFIED']             = 'true'
        # TODO: Import DEM_LIST from input VRT image

    def parameters(self, meta: Meta) -> ExeParameters:
        """
        Returns the parameters to use with :external:std:doc:`gdalwarp
        <programs/gdalwarp>` to projected the DEM onto the S2 geometry.
        """
        image       = in_filename(meta)
        tile_name   = meta['tile_name']
        tile_origin = meta['tile_origin']
        spacing     = self.__out_spatial_res
        logger.debug("%s.parameters(%s) /// image: %s /// tile_name: %s", self.__class__.__name__, meta, image, tile_name)

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


class ProjectGeoidToS2Tile(_ProjectGeoidTo):
    """
    Factory that produces a :class:`Step` that projects any kind of Geoid onto target S2 tile as
    described in :ref:`Project Geoid to S2 tile <project_geoid_to_s2-proc>`.

    This particular implementation uses another file in the expected geometry and
    :external+OTB:std:doc:`super impose <Applications/app_Superimpose>` the Geoid onto it. Unlike
    :external:std:doc:`gdalwarp <programs/gdalwarp>`, OTB application supports non-raster geoid
    formats.

    It requires the following information from the configuration object:

    - `ram_per_process`
    - `tmp_dir`    -- useless in the in-memory nomical case
    - `fname_fmt`  -- optional key: `geoid_on_s2`, useless in the in-memory nominal case
    - `interpolation_method` -- for use by :external+OTB:std:doc:`super impose
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
            name='ProjectGeoidToS2Tile',
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
            output_fname_fmt=fname_fmt,
            extended_filename=extended_filename_hidden(cfg, 'geoid_on_s2'),
            image_description="Geoid superimposed on S2 tile",
        )

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set S2 related information, that'll be carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['ORTHORECTIFIED']                        = 'true'
        imd['S2_TILE_CORRESPONDING_CODE']            = meta['tile_name']


class _SumAllHeights(OTBStepFactory):
    """
    Factory that produces a :class:`Step` that adds DEM + Geoid that cover a same footprint, as
    described in :ref:`Sum DEM + Geoid <sum_dem_geoid_on_s2-proc>`, and :ref:`Sum DEM + Geoid
    <sum_dem_geoid_on_s1-proc>`.

    It requires the following information from the configuration object:

    - `ram_per_process`
    - `tmp_dir`    -- useless in the in-memory nomical case
    - `fname_fmt`  -- optional key: `{product_key}`, useless in the in-memory nominal case
    - `nodata.DEM` -- optional

    It requires the following information from the metadata dictionary:
    """
    # Useless definition used to trick pylint in believing self._key_map is set.
    # Indeed, it's expected to be set in child classes. But pylint has now way of knowing that.
    _key_map           : Optional[Dict[str, str]] = None
    _fname_fmt_default : Optional[str]            = None
    _dname_fmt_default : Optional[str]            = None
    _product_key       : Optional[str]            = None
    _image_description : Optional[str]            = None

    def __init__(self, cfg: Configuration) -> None:
        """
        Constructor.
        """
        # Preconditions
        assert self._fname_fmt_default, "This class shall be specialized through SumAllHeights() function"
        assert self._dname_fmt_default, "This class shall be specialized through SumAllHeights() function"
        assert self._key_map,           "This class shall be specialized through SumAllHeights() function"
        assert self._product_key,       "This class shall be specialized through SumAllHeights() function"
        assert self._image_description, "This class shall be specialized through SumAllHeights() function"

        fname_fmt = cfg.fname_fmt.get(self._product_key, self._fname_fmt_default)
        dname_fmt = os.path.join(cfg.tmpdir, self._dname_fmt_default)
        super().__init__(
            cfg,
            appname='BandMath',
            name='SumAllHeights',
            param_in='il',
            param_out='out',
            gen_tmp_dir=dname_fmt,
            gen_output_dir=None,  # Use gen_tmp_dir,
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
            extended_filename=extended_filename_hidden(cfg, self._product_key),
            image_description=self._image_description,
        )
        self.__nodata = nodata_DEM(cfg)
        self.__indem   = self._key_map['indem']
        self.__ingeoid = self._key_map['ingeoid']

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        Complete meta information with inputs, and register temporary files from previous step for
        removal.
        """
        meta = super().complete_meta(meta, all_inputs)
        indem = fetch_input_data(self.__indem, all_inputs).out_filename
        meta['files_to_remove'] = [indem]  # input DEM on S1 or S2 footprint
        logger.debug('Register files to remove after %s computation: %s', self._product_key, meta['files_to_remove'])
        # Make sure to set nodata metadata in output image
        meta['out_extended_filename_complement'] = f'?&nodata={self.__nodata}'
        return meta

    def _get_inputs(self, previous_steps: List[InputList]) -> InputList:
        """
        Extract the last inputs to use at the current level from all previous products seens in the
        pipeline.

        This method is overridden in order to fetch N-1 "indem" input.
        It has been specialized for S1Tiling exact pipelines.
        """
        assert len(previous_steps) > 1

        # "in_geoid" is expected at level -1, likelly named '__last'
        in_geoid = fetch_input_data('__last', previous_steps[-1])
        # "in_dem"     is expected at level -2, likelly named self.__indem
        in_dem   = fetch_input_data(self.__indem, previous_steps[-2])

        inputs = [{self.__ingeoid: in_geoid, self.__indem: in_dem}]
        _check_input_step_type(inputs)
        logging.debug("%s inputs: %s", self.__class__.__name__, inputs)
        return inputs

    def _get_canonical_input(self, inputs: InputList) -> AbstractStep:
        """
        Helper function to retrieve the canonical input associated to a list of inputs.

        In current case, the canonical input comes from the "ingeoid" step instanciated in
        :func:`s1tiling.s1_process_lia` pipeline builder.
        """
        _check_input_step_type(inputs)
        keys = set().union(*(input.keys() for input in inputs))
        assert len(keys) == 2, f'Expecting 2 inputs. {len(inputs)} is/are found: {keys}'
        assert self.__ingeoid in keys
        return [input[self.__ingeoid] for input in inputs if self.__ingeoid in input.keys()][0]

    def fetch_upstream_dem_resampling_method(self, inputpath: str, meta: Meta):  # pragma: no cover
        """
        Extracts DEM_RESAMPLING_METHOD from from input image metadata.
        """
        logger.debug("Fetch DEM_RESAMPLING_METHOD from '%s'", inputpath)
        if not is_running_dry(meta):  # FIXME: this info is no longer in meta!
            dst = gdal.Open(inputpath, gdal.GA_ReadOnly)
            if not dst:
                raise RuntimeError(f"Cannot open DEM/{self._dname_fmt_default} file '{inputpath}' to collect DEM_RESAMPLING_METHOD metadata.")
            res = dst.GetMetadataItem('DEM_RESAMPLING_METHOD')
            del dst
            return res
        return 'No idea in dry run mode'

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Metadata coming from the DEM image are lost => we fetch them in the DEM file.
        """
        super().update_image_metadata(meta, all_inputs)

        in_dem   = fetch_input_data(self.__indem,   all_inputs).out_filename
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['TIFFTAG_GDAL_NODATA'] = str(self.__nodata)
        dem_resampling_method = self.fetch_upstream_dem_resampling_method(in_dem, meta)
        if dem_resampling_method:
            imd['DEM_RESAMPLING_METHOD'] = dem_resampling_method

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external+OTB:doc:`BandMath OTB application
        <Applications/app_BandMath>` for additionning DEM and Geoid data projected on S1/S2
        footprint.
        """
        assert 'inputs' in meta, f'Looking for "inputs" in {meta.keys()}'
        inputs = meta['inputs']
        in_dem    = fetch_input_data(self.__indem,   inputs).out_filename
        in_geoid  = fetch_input_data(self.__ingeoid, inputs).out_filename
        dem_nodata   = Utils.fetch_nodata_value(in_dem,   is_running_dry(meta), self.__nodata)  # usually -32768
        # geoid_nodata = Utils.fetch_nodata_value(in_geoid, is_running_dry(meta), self.__nodata)  # usually -32768/-88.8888
        params : OTBParameters = {
            'ram'         : ram(self.ram_per_process),
            self.param_in : [in_geoid, in_dem],
            # 'exp'         : f'({Utils.test_nodata_for_bandmath(dem_nodata,"im2b1")} || {Utils.test_nodata_for_bandmath(geoid_nodata,"im1b1")}) ? {self.__nodata} : im1b1+im2b1'
            'exp'         : f'{Utils.test_nodata_for_bandmath(dem_nodata,"im2b1")} ? {self.__nodata} : im1b1+im2b1'
        }
        return params


def SumAllHeights(
    product_key      : str,
    key_map          : Dict[str, str],
    fname_fmt_default: str,
    dname_fmt_default: str,
    image_description: str,
) -> type:
    """
    Factory function that returns a :class:`_SumAllHeights` child class created on the fly.

    The new class name will be :samp:`SumAllHeights_{{product_key}}`.

    :param str product_key:       Key used to fetch information from :class:`Configuration` object.
    :param dict key_map:          Maps formal input names (``indem`` and ``ingeoid``) to actual
                                  input names
    :param str fname_fmt_default: Default filename format string.
    :param str dname_fmt_default: Default dirname format string.
    :param str image_description: Image description tag
    """
    # We return a new class
    return type(
        f"SumAllHeights_{product_key}",  # Class name
        (_SumAllHeights,),         # Parent
        {
            '_key_map'          : key_map,
            '_fname_fmt_default': fname_fmt_default,
            '_dname_fmt_default': dname_fmt_default,
            '_product_key'      : product_key,
            '_image_description': image_description,
        }
    )

class ComputeGroundAndSatPositionsOnDEMFromEOF(OTBStepFactory):
    """
    Factory that prepares steps that run
    :external:doc:`Applications/app_SARComputeGroundAndSatPositionsOnDEM` as described in
    :ref:`Compute ECEF ground and satellite positions on S2` documentation to obtain the XYZ ECEF
    coordinates of the ground and of the satellite positions associated to the pixels from the input
    `height` file.

    :external:doc:`Applications/app_SARComputeGroundAndSatPositionsOnDEM` application fills a
    multi-bands image anchored on the footprint of the input DEM image.  In each pixel in the
    DEM/output image, we store the XYZ ECEF coordinate of the ground point (associated to the
    pixel), and the XYZ coordinates of the satellite position (associated to the pixel...)

    Requires the following information from the configuration object:

    - `ram_per_process`
    - `dem_db_filepath`   -- to fill-up image metadata
    - `dem_field_ids`     -- to fill-up image metadata
    - `dem_main_field_id` -- to fill-up image metadata
    - `tmpdir`            -- useless in the in-memory nomical case
    - `fname_fmt`         -- optional key: `ground_and_sat_s2`, useless in the in-memory nominal case
    - `nodata.LIA`        -- optional
    - DEM intersecting S2 tiles

    Requires the following information from the metadata dictionary

    - `basename`
    - `input filename`
    - `output filename`
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = 'XYZ_projected_on_{tile_name}_{orbit}.tiff'
        fname_fmt = cfg.fname_fmt.get('ground_and_sat_s2', fname_fmt)
        super().__init__(
            cfg,
            appname='SARComputeGroundAndSatPositionsOnDEM',
            name='SARComputeGroundAndSatPositionsOnDEM',
            param_in=None,
            param_out='out',
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
            gen_output_dir=None,  # Use gen_tmp_dir
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
            extended_filename=extended_filename_hidden(cfg, 'ground_and_sat_s2'),
            image_description="XYZ ground and satellite positions on S2 tile",
        )
        self.__cfg = cfg  # Will be used to access cached DEM intersecting S2 tile
        self.__nodata = nodata_XYZ(cfg)
        self.__dem_info = cfg.dem_info

    def _update_filename_meta_post_hook(self, meta: Meta) -> None:
        """
        Register ``accept_as_compatible_input`` hook for
        :func:`s1tiling.libs.meta.accept_as_compatible_input`.
        It will tell whether a given height file on S2 tile input is compatible with the current S2
        tile.
        """
        meta['accept_as_compatible_input'] = lambda input_meta: does_s2_data_match_s2_tile(meta, input_meta)

    def _get_inputs(self, previous_steps: List[InputList]) -> InputList:
        """
        Extract the last inputs to use at the current level from all previous products seens in the
        pipeline.

        This method is overridden in order to fetch N-2 "ineof" and "inheight" inputs.
        It has been specialized for S1Tiling exact pipelines.
        """
        for i, st in enumerate(previous_steps):
            logger.debug("INPUTS: %s previous step[%s] = %s", self.__class__.__name__, i, st)

        inputs = [fetch_input_data_all_inputs({"ineof", "inheight"}, previous_steps)]
        _check_input_step_type(inputs)
        logging.debug("%s inputs: %s", self.__class__.__name__, inputs)
        return inputs

    def _get_canonical_input(self, inputs: InputList) -> AbstractStep:
        assert inputs, "No inputs found in ComputeGroundAndSatPositionsOnDEMFromEOF"
        assert 'ineof' in inputs[0], f"'ineof' input is missing from ComputeGroundAndSatPositionsOnDEMFromEOF inputs: {inputs[0].keys()}"
        return inputs[0]['ineof']

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        Computes dem information and adds them to the meta structure, to be used later to fill-in
        the image metadata.

        .. note::
            Don't register previously produced height/S2 files for removal as they could be used for
            a different orbit or a different platform.
        """
        # logger.debug("ComputeGroundAndSatPositionsOnDEMFromEOF inputs are: %s", all_inputs)
        meta = super().complete_meta(meta, all_inputs)
        assert 'inputs' in meta, "Meta data shall have been filled with inputs"

        # Cannot register height_on_s2 for ulterior removal as the file can be
        # used with different orbits
        # => TODO count how many XYZ files depend on the height_on_s2 file
        # height_on_s2  = fetch_input_data('inheight', all_inputs)
        # meta['files_to_remove'] = [height_on_s2.out_filename]
        # logger.debug('Register files to remove after ground+satpos XYZ computation: %s', meta['files_to_remove'])

        eof = fetch_input_data('ineof', all_inputs).meta

        # TODO: Check whether the DEM_LIST is already there and automatically propagated!
        meta['dem_infos'] = self.__cfg.get_dems_covering_s2_tile(meta['tile_name'])
        meta['dems'] = sorted(meta['dem_infos'].keys())

        eof_file = out_filename(eof)
        logger.debug("ComputeGroundAndSatPositionsOnDEMFromEOF: DEM found for %s: %s", eof_file, meta['dems'])
        _, inbasename = os.path.split(eof_file)
        meta['inbasename'] = inbasename
        return meta

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set ComputeGroundAndSatPositionsOnDEMFromEOF related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['POLARIZATION']             = ""  # Clear polarization information (makes no sense here)
        imd['DEM_INFO']                 = self.__dem_info
        imd['DEM_LIST']                 = ', '.join(meta['dems'])
        imd['band.DirectionToScanDEM*'] = ''
        imd['band.Gain']                = ''
        imd['EOF_FILE']                 = meta['inbasename']
        # RELATIVE_ORBIT_NUMBER & ORBIT_DIRECTION are set by the application
        # imd['RELATIVE_ORBIT_NUMBER']    = meta['orbit']
        imd['IMAGE_TYPE']               = 'XYZ'
        imd['ORTHORECTIFIED']           = 'true'

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with
        :external:doc:`SARComputeGroundAndSatPositionsOnDEMFromEOF OTB application
        <Applications/app_SARComputeGroundAndSatPositionsOnDEMFromEOF>` to project S1 geometry onto
        DEM tiles.
        """
        nodata = self.__nodata
        assert 'inputs' in meta, f'Looking for "inputs" in {meta.keys()}'
        assert 'orbit'  in meta, f'Looking for "orbit" in {meta.keys()}'
        inputs = meta['inputs']
        inheight = fetch_input_data('inheight', inputs).out_filename
        ineof    = fetch_input_data('ineof'   , inputs).out_filename
        # `elev.geoid='@'` tells ComputeGroundAndSatPositionsOnDEMFromEOF that GEOID shall not be
        # used from $OTB_GEOID_FILE, indeed geoid information is already in DEM+Geoid input.
        return {
                'ram'        : ram(self.ram_per_process),
                'ineof'      : ineof,
                'indem'      : inheight,
                'inrelorb'   : int(meta['orbit']),
                'elev.geoid' : '@',
                'withxyz'    : True,
                'withsatpos' : True,
                # 'withh'      : True,  # uncomment to analyse/debug height computed
                'nodata'     : str(nodata)
        }

    def requirement_context(self) -> str:
        """
        Return the requirement context that permits to fix missing requirements.
        SARComputeGroundAndSatPositionsOnDEM comes from normlim_sigma0.
        """
        return "Please install https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0."


class ComputeGroundAndSatPositionsOnDEM(OTBStepFactory):
    """
    Factory that prepares steps that run :external:doc:`Applications/app_SARDEMProjection` as
    described in :ref:`Normals computation` documentation to obtain the XYZ ECEF coordinates of the
    ground and of the satellite positions associated to the pixels from the input `height` file.

    :external:doc:`Applications/app_SARDEMProjection` application fills a multi-bands image anchored
    on the footprint of the input DEM image.  In each pixel in the DEM/output image, we store the
    XYZ ECEF coordinate of the ground point (associated to the pixel), and the XYZ coordinates of
    the satellite position (associated to the pixel...)

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

    It also requires :envvar:`$OTB_GEOID_FILE` to be set in order to ignore any DEM information
    already registered in dask worker (through :external+OTB:doc:`Applications/app_OrthoRectification`
    for instance) and only use the Geoid.
    """
    def __init__(self, cfg: Configuration) -> None:
        # fname_fmt = 'XYZ_projected_on_{tile_name}_{orbit_direction}_{orbit}.tiff'
        fname_fmt = 'XYZ_projected_on_{tile_name}_{orbit}.tiff'
        fname_fmt = cfg.fname_fmt.get('ground_and_sat_s2', fname_fmt)
        super().__init__(
            cfg,
            appname='SARDEMProjection2',
            name='SARDEMProjection',
            param_in=None,
            param_out='out',
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
            gen_output_dir=None,  # Use gen_tmp_dir
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
            image_description="XYZ ground and satellite positions on S2 tile",
        )
        self.__cfg = cfg  # Will be used to access cached DEM intersecting S2 tile
        self.__nodata = nodata_XYZ(cfg)
        self.__dem_info = cfg.dem_info

    @staticmethod
    def reduce_inputs(inputs: List[Meta]) -> List:
        """
        TODO: filter EOF inputs
        Filters which insar input will be kept.

        Given several (usually 2 is the maximum) possible input S1 files ("insar" input channels),
        select and return the one that maximize its time coverage with the current S2 destination
        tile.

        Actually, this function returns the first S1 image that time-covers the whole S2 tile. The
        S1 images are searched in reverse order of their footprint-coverage.
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
            logger.debug(" - %s AZ: %s, OBT: %s: 2xAZ ∈ OBT: %s",
                         os.path.dirname(inp['manifest']), [str(az_start), str(az_stop)], [str(obt_start), str(obt_stop)], is_enough)
            if is_enough:
                logger.debug(
                    "Using %s which has orbit data that covers entirelly %s, and with a %.2f%% footprint coverage",
                    out_filename(best_covered_input), best_covered_input['tile_name'], best_covered_input['tile_coverage']
                )
                return [inp]
        logger.warning(
            "None of the orbit state vector sequence from input S1 products seems wide enough to cover entirelly %s tile. "
            "Returning %s which has the best footprint coverage: %.2f%%",
            best_covered_input['tile_name'], out_filename(best_covered_input), best_covered_input['tile_coverage']
        )
        return [best_covered_input]

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:
        """
        Injects the :func:`reduce_inputs_insar` hook in step metadata, and provide names clear from
        polar related information.
        """
        depolarize_4_filename_pre_hook(meta)
        meta['reduce_inputs_insar'] = ComputeGroundAndSatPositionsOnDEM.reduce_inputs
        return meta

    def _update_filename_meta_post_hook(self, meta: Meta) -> None:
        """
        Register ``accept_as_compatible_input`` hook for
        :func:`s1tiling.libs.meta.accept_as_compatible_input`.
        It will tell whether a given heights file on S2 tile input is compatible with the current S2
        tile.
        """
        meta['accept_as_compatible_input'] = lambda input_meta: does_s2_data_match_s2_tile(meta, input_meta)

    def _get_inputs(self, previous_steps: List[InputList]) -> InputList:
        """
        Extract the last inputs to use at the current level from all previous products seens in the
        pipeline.

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
        Computes dem information and adds them to the meta structure, to be used later to fill-in
        the image metadata.

        Also register temporary files from previous step for removal.
        """
        # logger.debug("ComputeGroundAndSatPositionsOnDEM inputs are: %s", all_inputs)
        meta = super().complete_meta(meta, all_inputs)
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
        imd['DEM_INFO']                 = self.__dem_info
        imd['DEM_LIST']                 = ', '.join(meta['dems'])
        imd['band.DirectionToScanDEM*'] = ''
        imd['band.Gain']                = ''
        imd['IMAGE_TYPE']               = 'XYZ'
        imd['ORTHORECTIFIED']           = 'true'

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
        # `elev.geoid='@'` tells SARDEMProjection2 that GEOID shall not be used from
        # $OTB_GEOID_FILE, indeed geoid information is already in DEM+Geoid input.
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
    Abstract factory that prepares steps that run :external:doc:`ExtractNormalVector
    <Applications/app_ExtractNormalVector>` as described in :ref:`Normals computation
    <compute_normals-proc>` documentation.

    :external:doc:`ExtractNormalVector <Applications/app_ExtractNormalVector>` computes surface
    normals.

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
        extended_filename : Optional[str],
    ) -> None:
        super().__init__(
            cfg,
            appname='ExtractNormalVector',
            name='ComputeNormals',
            param_in='xyz',
            param_out='out',
            gen_tmp_dir=gen_tmp_dir,
            gen_output_dir=None,  # Use gen_tmp_dir
            gen_output_filename=TemplateOutputFilenameGenerator(output_fname_fmt),
            extended_filename=extended_filename,
            image_description=image_description,
        )
        self.__nodata = nodata_XYZ(cfg)

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        Override :func:`complete_meta()` to inject files to remove
        """
        meta = super().complete_meta(meta, all_inputs)
        in_file = in_filename(meta)
        meta['files_to_remove'] = [in_file]
        logger.debug('Register files to remove after normals computation: %s', meta['files_to_remove'])
        return meta

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set Normals related information.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['IMAGE_TYPE'] = 'NORMALS'

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external:doc:`ExtractNormalVector OTB application
        <Applications/app_ExtractNormalVector>` to generate surface normals for each point of the
        origin S1 image.
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
    Factory that prepares steps that run :external:doc:`ExtractNormalVector
    <Applications/app_ExtractNormalVector>` on images in S2 geometry as described in :ref:`Normals
    computation <compute_normals-proc>` documentation.

    :external:doc:`ExtractNormalVector <Applications/app_ExtractNormalVector>` computes surface
    normals.

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
            extended_filename=extended_filename_hidden(cfg, 'normals_on_s2'),
            image_description='Image normals on S2 grid',
        )


class _ComputeIncidenceAngle(OTBStepFactory):
    """
    Abstract factory that prepares steps that run :external:doc:`SARComputeIncidenceAngle
    <Applications/app_SARComputeIncidenceAngle>` as described in :ref:`IA map <compute_eia-proc>`
    and :ref:`LIA map <compute_lia-proc>` computations documentation.

    :external:doc:`SARComputeIncidenceAngle <Applications/app_SARComputeIncidenceAngle>` computes
    Local Incidence Angle Map.

    Requires the following information from the configuration object:

    - `ram_per_process`
    - `nodata.LIA`      -- optional

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """

    _data_type_fmts = {
        IA_map.cos: 'cos({IA})',
        IA_map.sin: 'sin({IA})',
        IA_map.tan: 'tan({IA})',
        IA_map.deg: '100 * degrees({IA})',
    }

    def __init__(  # pylint: disable=too-many-arguments
        self,
        cfg                    : Configuration,
        *,
        gen_tmp_dir            : str,
        gen_output_dir         : Optional[str],
        image_description_dict : Dict[IA_map, str],
        incidence_angle_kind   : str,  # "IA" or "LIA"
        tname_fmt              : str,
        fname_fmt_cos          : Optional[str] = None,
        fname_fmt_sin          : Optional[str] = None,
        fname_fmt_tan          : Optional[str] = None,
        fname_fmt_deg          : Optional[str] = None,
    ) -> None:
        params_out         : List[str] = []
        fname_fmts         : List[OutputFilenameGenerator] = []
        extended_filenames : List[str] = []
        pixel_types        : List      = []  # List[PixelType==int]
        self.__data_types  : List[str] = []
        image_description  : List[str] = []
        def register_output(fname_fmt, ia_map: IA_map):
            if fname_fmt:
                default_disable_streaming = otb_version() < '9.1.1'
                params_out        .append(f'out.{ia_map.name}')
                fname_fmts        .append(TemplateOutputFilenameGenerator(fname_fmt))
                extended_filenames.append(
                    extended_filename_ia(
                        cfg,
                        ia_map,
                        cfg.disable_streaming.get('normals_on_s2', default_disable_streaming) and incidence_angle_kind == "LIA"
                    ))
                pixel_types       .append(pixel_type_ia(cfg, ia_map, incidence_angle_kind))
                self.__data_types .append(self._data_type_fmts[ia_map].format(IA=incidence_angle_kind))
                image_description .append(image_description_dict[ia_map])
                logger.debug('Registering %s %s map -> %s', incidence_angle_kind, ia_map.name, fname_fmt)

        register_output(fname_fmt_cos, IA_map.cos)
        register_output(fname_fmt_sin, IA_map.sin)
        register_output(fname_fmt_tan, IA_map.tan)
        register_output(fname_fmt_deg, IA_map.deg)
        assert None not in fname_fmts

        super().__init__(
            cfg,
            appname='SARComputeIncidenceAngle',
            name='ComputeXIA',
            param_in='in.normals',  # In-memory connected to in.normals
            param_out=params_out,
            gen_tmp_dir=gen_tmp_dir,
            gen_output_dir=gen_output_dir,
            gen_output_filename=OutputFilenameGeneratorList(fname_fmts),
            image_description=image_description,
            extended_filename=extended_filenames,
            pixel_type=pixel_types,
        )
        self.__incidence_angle_kind = incidence_angle_kind
        self.__nodata               = nodata_LIA(cfg)
        self.__task_name_fmt        = tname_fmt

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set σ° normlim calibration related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['DATA_TYPE']  = self.__data_types
        imd['IMAGE_TYPE'] = self.__incidence_angle_kind

    def _update_filename_meta_post_hook(self, meta: Meta) -> None:
        """
        The task name is the root basename, with no list
        """
        # logger.debug('%s(%s)._update_filename_meta_post_hook -> out_filename=%s',
                     # self.__class__.__name__, self._burst_index, out_filename(meta))
        generator = TemplateOutputFilenameGenerator(self.__task_name_fmt)
        meta['task_name'] = generator.generate(meta['basename'], meta)
        logger.debug('Setting task_name to %s', meta['task_name'])

    def _get_inputs(self, previous_steps: List[InputList]) -> InputList:
        """
        Extract the last inputs to use at the current level from all previous products seens in the
        pipeline.

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
        logging.debug("%s inputs: %s", self.__class__.__name__, inputs)
        return inputs

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external:doc:`SARComputeIncidenceAngle OTB application
        <Applications/app_SARComputeIncidenceAngle>`.
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


class ComputeLIAOnS2(_ComputeIncidenceAngle):
    """
    Factory that prepares steps that run :external:doc:`SARComputeIncidenceAngle
    <Applications/app_SARComputeIncidenceAngle>` on images in S2 geometry as described in
    :ref:`LIA maps computation <compute_lia-proc>` documentation.

    :external:doc:`SARComputeIncidenceAngle <Applications/app_SARComputeIncidenceAngle>` computes
    Local Incidence Angle Map.

    Requires the following information from the configuration object:

    - `ram_per_process`
    - `fname_fmt`       -- optional key: `lia_product`
    - `dname_fmt`       -- optional key: `lia_product`
    - `nodata.LIA`      -- optional

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """
    _image_descriptions = {
        IA_map.sin: 'sin(LIA) on S2 grid',
        IA_map.deg: '100 * degrees(LIA) on S2 grid',
    }

    def __init__(self, cfg: Configuration) -> None:
        fname_fmt0 = DEFAULT_FNAME_FMTS['lia_product']
        fname_fmt0 = cfg.fname_fmt.get('lia_product', fname_fmt0)
        tname_fmt     = partial_format(fname_fmt0, LIA_kind="TaskLIA")
        fname_fmt_deg = partial_format(fname_fmt0, LIA_kind="LIA")     if cfg.produce_lia_map else None
        fname_fmt_sin = partial_format(fname_fmt0, LIA_kind="sin_LIA")
        dname_fmt = dname_fmt_lia_product(cfg)
        super().__init__(
            cfg,
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2'),
            gen_output_dir=dname_fmt,
            tname_fmt=tname_fmt,
            fname_fmt_deg=fname_fmt_deg,
            fname_fmt_sin=fname_fmt_sin,
            image_description_dict=self._image_descriptions,
            incidence_angle_kind='LIA',
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

    def __init__(self, cfg: Configuration) -> None:  # pylint: disable=unused-argument
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
        related_inputs = [
            f for f in Utils.flatten_stringlist(in_filename(meta))
            if re.search(rf'\b{self._LIA_kind}_', f)
        ]
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
    Generates a new :class:`StepFactory` class that filters which LIA product shall be processed:
    LIA maps or sin LIA maps.
    """
    # We return a new class
    return type(
        f"Filter_{LIA_kind}",      # Class name
        (_FilterLIAStepFactory,),  # Parent
        {'_LIA_kind': LIA_kind}
    )


class ApplyLIACalibration(OTBStepFactory):
    """
    Factory that concludes σ0 with NORMLIM calibration.

    It builds steps that multiply images calibrated with β0 LUT, and orthorectified to S2 grid, with
    the sin(LIA) map for the same S2 tile (and orbit number and direction).

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
        fname_fmt = fname_fmt_lia_corrected(cfg)
        dname_fmt = dname_fmt_tiled(cfg)
        super().__init__(
            cfg,
            appname='BandMath',
            name='ApplyLIACalibration',
            param_in='il',
            param_out='out',
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
        Complete meta information with inputs, and set compression method to DEFLATE.
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
        imd['IMAGE_TYPE']  = 'BACKSCATTERING'

    def _get_canonical_input(self, inputs: InputList) -> AbstractStep:
        """
        Helper function to retrieve the canonical input associated to a list of inputs.

        In current case, the canonical input comes from the "concat_S2" pipeline defined in
        :func:`s1tiling.s1_process` pipeline builder.
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
        It will tell whether a given sin_LIA input is compatible with the current S2 tile.
        """
        meta['accept_as_compatible_input'] = lambda input_meta : does_sin_lia_match_s2_tile_for_orbit(meta, input_meta)
        meta['basename']                   = self._get_nominal_output_basename(meta)
        meta['calibration_type']           = 'Normlim'

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external+OTB:doc:`BandMath OTB application
        <Applications/app_BandMath>` for applying sin(LIA) to β0 calibrated image orthorectified to
        S2 tile.
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


class SARDEMProjection(_SARDEMProjectionFamily):
    """
    Factory that prepares steps that run :external:doc:`Applications/app_SARDEMProjection` as
    described in :ref:`Normals computation` documentation.

    :external:doc:`Applications/app_SARDEMProjection` application puts a DEM file into SAR geometry
    and estimates two additional coordinates.  For each point of the DEM input four components are
    calculated: C (colunm into SAR image), L (line into SAR image), Z and Y. XYZ cartesian
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

    It also requires :envvar:`$OTB_GEOID_FILE` to be set in order to ignore any DEM information
    already registered in dask worker (through
    :external+OTB:doc:`Applications/app_OrthoRectification` for instance) and only use the Geoid.

    .. deprecated:: 1.1
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = 'S1_on_DEM_{polarless_basename}'
        fname_fmt = cfg.fname_fmt.get('s1_on_dem', fname_fmt)
        super().__init__(
            cfg,
            name='SARDEMProjection',
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
            output_fname_fmt=fname_fmt,
            extended_filename=extended_filename_s1_on_dem(cfg),
            image_description="SARDEM projection onto DEM list",
        )

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external:doc:`SARDEMProjection OTB application
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


class SARCartesianMeanEstimation(_PostSARDEMProjectionFamily):
    """
    Factory that prepares steps that run :external:doc:`Applications/app_SARCartesianMeanEstimation`
    as described in :ref:`Normals computation` documentation.

    :external:doc:`Applications/app_SARCartesianMeanEstimation` estimates a simulated cartesian mean
    image thanks to a DEM file.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename

    Note: It cannot be chained in memory because of the ``directiontoscandem*`` parameters.

    .. deprecated:: 1.1
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = 'XYZ_{polarless_basename}'
        fname_fmt = cfg.fname_fmt.get('xyz', fname_fmt)
        super().__init__(
            cfg,
            appname='SARCartesianMeanEstimation2',
            name='SARCartesianMeanEstimation',
            param_in=None,
            param_out='out',
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
            gen_output_dir=None,  # Use gen_tmp_dir
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
            image_description='Cartesian XYZ coordinates estimation',
        )

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external:doc:`SARCartesianMeanEstimation OTB application
        <Applications/app_SARCartesianMeanEstimation>` to compute cartesian coordinates of each
        point of the origin S1 image.
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
    Factory that prepares steps that run :external:doc:`ExtractNormalVector
    <Applications/app_ExtractNormalVector>` on images in S1 geometry as described in :ref:`Normals
    computation <compute_normals-proc>` documentation.

    :external:doc:`ExtractNormalVector <Applications/app_ExtractNormalVector>` computes surface
    normals.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    - `fname_fmt`  -- optional key: `normals_on_s1`, useless in the in-memory nominal case

    .. deprecated:: 1.1
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = 'Normals_{polarless_basename}'
        fname_fmt = cfg.fname_fmt.get('normals_on_s1', fname_fmt)
        super().__init__(
            cfg,
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
            output_fname_fmt=fname_fmt,
            extended_filename=extended_filename_hidden(cfg, 'normals_on_s1'),
            image_description='Image normals on Sentinel-{flying_unit_code_short} IW GRD',
        )


class ComputeLIAOnS1(_ComputeIncidenceAngle):
    """
    Factory that prepares steps that run :external:doc:`SARComputeIncidenceAngle
    <Applications/app_SARComputeIncidenceAngle>` on images in S1 geometry as described in
    :ref:`LIA maps computation <compute_lia-proc>` documentation.

    :external:doc:`SARComputeIncidenceAngle <Applications/app_SARComputeIncidenceAngle>` computes
    Local Incidence Angle Map.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    - `fname_fmt`  -- optional key: `s1_lia`
    - `fname_fmt`  -- optional key: `s1_sin_lia`

    .. deprecated:: 1.1
    """
    _image_descriptions = {
        IA_map.sin: 'sin(LIA) on Sentinel-{flying_unit_code_short} IW GRD',
        IA_map.deg: '100 * degrees(LIA) on Sentinel-{flying_unit_code_short} IW GRD',
    }

    def __init__(self, cfg: Configuration) -> None:
        tname_fmt     = 'TaskLIA_{polarless_basename}'
        fname_fmt_deg = cfg.fname_fmt.get('s1_lia',     'LIA_{polarless_basename}') if cfg.produce_lia_map else None
        fname_fmt_sin = cfg.fname_fmt.get('s1_sin_lia', 'sin_LIA_{polarless_basename}')
        super().__init__(
            cfg,
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S1'),
            gen_output_dir=None,
            tname_fmt=tname_fmt,
            fname_fmt_deg=fname_fmt_deg,
            fname_fmt_sin=fname_fmt_sin,
            image_description_dict=self._image_descriptions,
            incidence_angle_kind='LIA',
        )


class OrthoRectifyLIA(_OrthoRectifierFactory):
    """
    Factory that prepares steps that run :external+OTB:doc:`Applications/app_OrthoRectification` on
    LIA maps.

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

    .. deprecated:: 1.1
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
            fname_fmt=fname_fmt,
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
        data_types = {
            'sin_LIA': 'sin(LIA)',
            'LIA': '100 * degrees(LIA)'
        }
        assert 'LIA_kind' in meta, "This StepFactory shall be registered after a call to filter_LIA()"
        kind = meta['LIA_kind']
        assert kind in data_types, f'The only LIA kind accepted are {data_types.keys()}'
        imd = meta['image_metadata']
        imd['DATA_TYPE'] = data_types[kind]

    def set_output_pixel_type(self, app, meta: Meta) -> None:
        """
        Force LIA output pixel type to ``INT16``.
        """
        if meta.get('LIA_kind', '') == 'LIA':
            app.SetParameterOutputImagePixelType(self.param_out, otb.ImagePixelType_int16)


class ConcatenateLIA(_ConcatenatorFactoryForMaps):
    """
    Factory that prepares steps that run :external+OTB:doc:`Applications/app_Synthetize` on LIA
    images.

    Requires the following information from the configuration object:

    - `ram_per_process`

    Requires the following information from the metadata dictionary

    - input filename
    - output filename

    .. deprecated:: 1.1
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = '{LIA_kind}_{flying_unit_code}_{tile_name}_{orbit_direction}_{orbit}_{acquisition_day}.tif'
        fname_fmt = cfg.fname_fmt.get('lia_concatenation', fname_fmt)
        super().__init__(
            cfg,
            fname_fmt=fname_fmt,
            image_description='Orthorectified {LIA_kind} Sentinel-{flying_unit_code_short} IW GRD',
            extended_filename=None,  # will be set later...
        )
        self._extended_filenames = {
            'LIA'     : extended_filename_lia_degree(cfg),
            'sin_LIA' : extended_filename_lia_sin(cfg),
        }

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        meta = super().complete_meta(meta, all_inputs)
        assert 'out_extended_filename_complement' not in meta, f'{meta["out_extended_filename_complement"]=!r} nothing was expected'
        kind = meta['LIA_kind']
        meta['out_extended_filename_complement'] = self._extended_filenames[kind]
        return meta

    def set_output_pixel_type(self, app, meta: Meta) -> None:
        """
        Force LIA output pixel type to ``INT16``.
        """
        if meta.get('LIA_kind', '') == 'LIA':
            app.SetParameterOutputImagePixelType(self.param_out, otb.ImagePixelType_int16)


class SelectBestCoverage(_SelectBestCoverage):
    """
    StepFactory that helps select only one path after LIA concatenation: the one that have the best
    coverage of the S2 tile target.

    If several concatenated products have the same coverage, the oldest one will be selected.

    The coverage is extracted from ``tile_coverage`` step metadata.

    The step produced does nothing: it only only rename the selected product into the final expected
    name. Note: in LIA case two files will actually renamed.

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

    .. deprecated:: 1.1
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = '{LIA_kind}_{flying_unit_code}_{tile_name}_{orbit_direction}_{orbit}.tif'
        fname_fmt = cfg.fname_fmt.get('lia_product', fname_fmt)
        dname_fmt = dname_fmt_lia_product(cfg)
        super().__init__(
            cfg,
            name='SelectBestCoverage',
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
            dname_fmt=dname_fmt,
            fname_fmt=fname_fmt
        )
