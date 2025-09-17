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
# - Luc HERMITTE (CSGROUP)
#
# =========================================================================

"""
This modules defines the specialized Python wrappers for the OTB Applications used in
the pipeline for LIA production needs.
"""

import logging
import os
from typing import List



from .lia              import _ComputeIncidenceAngle
from ._applications    import s2_tile_extent
from ..configuration   import DEFAULT_FNAME_FMTS, Configuration, dname_fmt_ia_product, extended_filename_hidden, nodata_XYZ
from ..file_naming     import TemplateOutputFilenameGenerator
from ..incidence_angle import IA_map, eia_map_fname_fmt
from ..meta            import Meta, out_filename
from ..otbpipeline     import fetch_input_data, fetch_input_data_all_inputs
from ..steps           import (
    AbstractStep,
    InputList,
    OTBParameters,
    OTBStepFactory,
    ram,
)


logger = logging.getLogger('s1tiling.wrappers.ia')


class ComputeGroundAndSatPositionsOnEllipsoid(OTBStepFactory):
    """
    Factory that prepares steps that run
    :external:doc:`Applications/app_SARComputeGroundAndSatPositionsOnEllipsoid` as described in
    ":ref:`compute_wgs4_xyz_n_sat_s2-proc`" documentation to obtain the XYZ ECEF coordinates of the
    ellipsoid surface and of the satellite positions associated to the requested footprint.

    :external:doc:`Applications/app_SARComputeGroundAndSatPositionsOnEllipsoid` application fills a
    multi-bands image anchored on the requested footprint. In each pixel in the output image, we
    store the XYZ ECEF coordinate of the ellipsoid surface (associated to the pixel), and the XYZ
    coordinates of the satellite position (associated to the pixel...)

    Requires the following information from the configuration object:

    - `ram_per_process`
    - `tmpdir`            -- useless in the in-memory nomical case
    - `fname_fmt`         -- optional key: `ground_and_sat_s2_ellipsoid`, useless in the in-memory nominal case
    - `nodata.XYZ`        -- optional
    - `out_spatial_res`

    Requires the following information from the metadata dictionary

    - `output filename`
    - `tile_name`
    - `tile_origin`
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt = 'XYZ_projected_on_ellipsoid_{tile_name}_{orbit}.tiff'
        fname_fmt = cfg.fname_fmt.get('ground_and_sat_s2_ellipsoid', fname_fmt)
        super().__init__(
            cfg,
            appname='SARComputeGroundAndSatPositionsOnEllipsoid',
            name='SARComputeGroundAndSatPositionsOnEllipsoid',
            param_in=None,
            param_out='out',
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
            gen_output_dir=None,  # Use gen_tmp_dir
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
            image_description="XYZ surface and satellite positions on S2 tile on ellipsoid",
        )
        self.__nodata          = nodata_XYZ(cfg)
        self.__out_spatial_res = cfg.out_spatial_res

    def _update_filename_meta_post_hook(self, meta: Meta) -> None:
        """
        Register ``accept_as_compatible_input`` hook for
        :func:`s1tiling.libs.meta.accept_as_compatible_input`.
        It will tell whether a given tile information input is compatible with the current EOF file
        and S2 tile.
        """
        def ellipsoid_compatible(output_meta, input_meta):
            logger.debug('TEST compat:\nOUT -> %s\nIN  -> %s', output_meta, input_meta)
            return output_meta['tile_name'] == input_meta['tile_name']
        meta['accept_as_compatible_input'] = lambda input_meta: ellipsoid_compatible(meta, input_meta)

    def _get_inputs(self, previous_steps: List[InputList]) -> InputList:
        """
        Extract the last inputs to use at the current level from all previous products seen in the
        pipeline.

        This method is overridden in order to fetch N-2 "ineof" input.
        It has been specialized for S1Tiling exact pipelines.
        """
        for i, st in enumerate(previous_steps):
            logger.debug("INPUTS: %s previous step[%s] = %s", self.__class__.__name__, i, st)

        inputs = [fetch_input_data_all_inputs({"ineof", "tilename"}, previous_steps)]
        # _check_input_step_type(inputs)
        logging.debug("%s inputs: %s", self.__class__.__name__, inputs)
        return inputs

    def _get_canonical_input(self, inputs: InputList) -> AbstractStep:
        assert inputs, "No inputs found in ComputeGroundAndSatPositionsOnEllipsoid"
        assert 'ineof' in inputs[0], f"'ineof' input is missing from ComputeGroundAndSatPositionsOnEllipsoid inputs: {inputs[0].keys()}"
        return inputs[0]['ineof']

    def complete_meta(self, meta: Meta, all_inputs: InputList) -> Meta:
        """
        Extracts S2 tile footprint to be used later to fill-in the application parameters.
        Also extracts the EOF name to store later in the image metadata.
        """
        # logger.debug("ComputeGroundAndSatPositionsOnEllipsoid inputs are: %s", all_inputs)
        meta = super().complete_meta(meta, all_inputs)
        assert 'inputs' in meta, "Meta data shall have been filled with inputs"

        eof = fetch_input_data('ineof', all_inputs).meta
        tile_info = fetch_input_data('tilename', all_inputs).meta

        eof_file = out_filename(eof)
        _, inbasename = os.path.split(eof_file)
        meta['inbasename'] = inbasename
        meta['tile_origin'] = tile_info['tile_origin']
        return meta

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set ComputeGroundAndSatPositionsOnEllipsoid related information that'll get carried around.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['EOF_FILE']                   = meta['inbasename']
        imd['IMAGE_TYPE']                 = 'XYZ'
        imd['ORTHORECTIFIED']             = 'true'
        # RELATIVE_ORBIT_NUMBER & ORBIT_DIRECTION are set by the application
        # imd['RELATIVE_ORBIT_NUMBER']      = meta['orbit']
        imd['S2_TILE_CORRESPONDING_CODE'] = meta['tile_name']
        imd['SPATIAL_RESOLUTION']         = str(self.__out_spatial_res)

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external:doc:`SARComputeGroundAndSatPositionsOnEllipsoid
        OTB application <Applications/app_SARComputeGroundAndSatPositionsOnEllipsoid>` to project
        XYZ coordinates of ellipsoid surface and associated satellite position onto S2 footprint.
        """
        tile_name   = meta['tile_name']
        tile_origin = meta['tile_origin']
        spacing     = self.__out_spatial_res

        extent      = s2_tile_extent(tile_name, tile_origin, in_epsg=4326, spacing=spacing)
        nodata = self.__nodata
        assert 'orbit'  in meta, f'Looking for "orbit" in {meta.keys()}'
        inputs = meta['inputs']
        ineof    = fetch_input_data('ineof'   , inputs).out_filename
        logger.debug("%s.parameters(%s) /// tile_name: %s", self.__class__.__name__, meta, tile_name)
        return {
            'ram'              : ram(self.ram_per_process),
            'outputs.spacingx' : spacing,
            'outputs.spacingy' : -spacing,
            'outputs.sizex'    : extent['xsize'],
            'outputs.sizey'    : extent['ysize'],
            'map'              : 'utm',
            'map.utm.zone'     : extent['utm_zone'],
            'map.utm.northhem' : extent['utm_northern'],
            'outputs.ulx'      : extent['xmin'],
            'outputs.uly'      : extent['ymax'],  # ymax, not ymin!!!
            'ineof'            : ineof,
            'inrelorb'         : int(meta['orbit']),
            'withxyz'          : True,
            'withsatpos'       : True,
            # 'withh'          : True,  # uncomment to analyse/debug height computed
            'nodata'           : str(nodata)
        }

    def requirement_context(self) -> str:
        """
        Return the requirement context that permits to fix missing requirements.
        SARComputeGroundAndSatPositionsOnEllipsoid comes from normlim_sigma0.
        """
        return "Please install https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0."


class ComputeEllipsoidNormalsOnS2(OTBStepFactory):
    """
    Factory that prepares steps that run :external:doc:`ExtractNormalVectorToEllipsoid
    <Applications/app_ExtractNormalVectorToEllipsoid>` as described in :ref:`Normals computation
    <ellipsoid_normals_computation-maths>` documentation.

    :external:doc:`ExtractNormalVectorToEllipsoid <Applications/app_ExtractNormalVectorToEllipsoid>`
    computes ellipsoid surface normals.

    Requires the following information from the configuration object:

    - `ram_per_process`
    - `fname_fmt`      -- optional key: `normals_wgs84_on_s2`, useless in the in-memory nominal case

    Requires the following information from the metadata dictionary

    - `tile_name`
    - `tile_origin`
    - output filename
    """
    def __init__(
        self,
        cfg               : Configuration,
    ) -> None:
        fname_fmt = 'NormalsToEllipsoid_on_{tile_name}'
        fname_fmt = cfg.fname_fmt.get('normals_wgs84_on_s2', fname_fmt)
        super().__init__(
            cfg,
            appname='ExtractNormalVectorToEllipsoid',
            name='ComputeNormalsToEllipsoid',
            param_in=None,
            param_out='out',
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2', '{tile_name}'),
            gen_output_dir=None,  # Use gen_tmp_dir
            gen_output_filename=TemplateOutputFilenameGenerator(fname_fmt),
            extended_filename=extended_filename_hidden(cfg, 'normals_wgs84_on_s2'),
            image_description='Image normals To WGS84 Ellipsoid on S2 grid',
        )
        self.__out_spatial_res      = cfg.out_spatial_res

    def update_image_metadata(self, meta: Meta, all_inputs: InputList) -> None:
        """
        Set Normals related information.
        """
        super().update_image_metadata(meta, all_inputs)
        assert 'image_metadata' in meta
        imd = meta['image_metadata']
        imd['IMAGE_TYPE']                 = 'NORMALS'
        imd['ORTHORECTIFIED']             = 'true'
        imd['S2_TILE_CORRESPONDING_CODE'] = meta['tile_name']
        imd['SPATIAL_RESOLUTION']         = str(self.__out_spatial_res)

    def _get_canonical_input(self, inputs: InputList) -> AbstractStep:
        assert inputs, f"No inputs found in {self.__class__.__name__}"
        return fetch_input_data('tilename',   inputs)

    def parameters(self, meta: Meta) -> OTBParameters:
        """
        Returns the parameters to use with :external:doc:`ExtractNormalVectorToEllipsoid OTB
        application <Applications/app_ExtractNormalVectorToEllipsoid>` to generate surface normals
        for each point of the origin S1 image.
        """
        tile_name   = meta['tile_name']
        tile_origin = meta['tile_origin']
        spacing     = self.__out_spatial_res

        extent      = s2_tile_extent(tile_name, tile_origin, in_epsg=4326, spacing=spacing)
        logger.debug("%s.parameters(%s) /// tile_name: %s",
                self.__class__.__name__, meta, tile_name)

        return {
            'ram'             : ram(self.ram_per_process),
            'outputs.spacingx' : spacing,
            'outputs.spacingy' : -spacing,
            'outputs.sizex'    : extent['xsize'],
            'outputs.sizey'    : extent['ysize'],
            'map'              : 'utm',
            'map.utm.zone'     : extent['utm_zone'],
            'map.utm.northhem' : extent['utm_northern'],
            'outputs.ulx'      : extent['xmin'],
            'outputs.uly'      : extent['ymax'],  # ymax, not ymin!!!
        }

    def requirement_context(self) -> str:
        """
        Return the requirement context that permits to fix missing requirements.
        ComputeNormalsToEllipsoid comes from normlim_sigma0.
        """
        return "Please install https://gitlab.orfeo-toolbox.org/s1-tiling/normlim_sigma0."

    def _update_filename_meta_post_hook(self, meta: Meta) -> None:
        """
        Register ``accept_as_compatible_input`` hook for
        :func:`s1tiling.libs.meta.accept_as_compatible_input`.
        It will tell whether a given sin_IA input is compatible with the current S2 tile.
        """
        def ellipsoid_normal_compatible(output_meta, input_meta):
            logger.debug('TEST compat:\nOUT -> %s\nIN  -> %s', output_meta, input_meta)
            return output_meta['tile_name'] == input_meta['tile_name']
        meta['accept_as_compatible_input'] = ellipsoid_normal_compatible


class ComputeIAOnS2(_ComputeIncidenceAngle):
    """
    Factory that prepares steps that run :external:doc:`SARComputeIncidenceAngle
    <Applications/app_SARComputeIncidenceAngle>` on images in S2 geometry as described in :ref:`IA
    maps computation <compute_eia-proc>` documentation.

    :external:doc:`SARComputeIncidenceAngle <Applications/app_SARComputeIncidenceAngle>` computes
    Incidence Angle Map.

    .. todo::

        Detect when we only need to generate deg among "deg, cos" when cos products are already on
        the disk

    Requires the following information from the configuration object:

    - `ram_per_process`
    - `fname_fmt`       -- optional key: `ia_product`
    - `dname_fmt`       -- optional key: `ia_product`
    - `nodata.IA`       -- optional

    Requires the following information from the metadata dictionary

    - input filename
    - output filename
    """
    def __init__(self, cfg: Configuration) -> None:
        fname_fmt0 = DEFAULT_FNAME_FMTS['ia_product']
        fname_fmt0 = cfg.fname_fmt.get('ia_product', fname_fmt0)
        def fname_fmt(ia_map: IA_map):
            if ia_map.name in cfg.ia_maps_to_produce + [IA_map.tsk.name]:
                fmt = eia_map_fname_fmt(fname_fmt0, ia_map)
                assert isinstance(fmt, str), f"fname_fmt({ia_map}) -> {fmt=!r} is not a string"
                # logger.debug("Registering IA %s map -> %s", ia_map.name, fmt)
                assert fmt
                return fmt
            logger.debug("Not registering IA %s map", ia_map.name)
            return None
        dname_fmt = dname_fmt_ia_product(cfg)
        image_descriptions = {
            IA_map.cos: 'cos(IA) on S2 grid',
            IA_map.sin: 'sin(IA) on S2 grid',
            IA_map.tan: 'tan(IA) on S2 grid',
            IA_map.deg: '100 * degrees(IA) on S2 grid',
        }
        tname_fmt=fname_fmt(IA_map.tsk)
        assert tname_fmt is not None
        super().__init__(
            cfg,
            gen_tmp_dir=os.path.join(cfg.tmpdir, 'S2'),
            gen_output_dir=dname_fmt,
            tname_fmt=tname_fmt,
            fname_fmt_deg=fname_fmt(IA_map.deg),
            fname_fmt_cos=fname_fmt(IA_map.cos),
            fname_fmt_sin=fname_fmt(IA_map.sin),
            fname_fmt_tan=fname_fmt(IA_map.tan),
            image_description_dict=image_descriptions,
            incidence_angle_kind='IA',
        )
        assert self.has_several_outputs()

    def _update_filename_meta_pre_hook(self, meta: Meta) -> Meta:
        # This hooks can be called in two situations
        # 1. when building the expected() result name. In that case we don't have enough information
        #    when called from "xyz" input, and not called at all from "tilename" input.
        # 2. from create_step()->complete_meta(), in which case meta['inputs'] exists, and the exact
        #    information need to be extracted from the "xyz" input.
        meta = super()._update_filename_meta_pre_hook(meta)
        if 'inputs' in meta:
            logger.debug("%s inputs are %s", self.__class__.__name__, meta['inputs'])
            xyz = fetch_input_data('xyz', meta['inputs'])
            meta['flying_unit_code'] = xyz.meta['flying_unit_code']
            meta['orbit']            = xyz.meta['orbit']
        return meta

    def _update_filename_meta_post_hook(self, meta: Meta) -> None:
        """
        Register ``accept_as_compatible_input`` hook for
        :func:`s1tiling.libs.meta.accept_as_compatible_input`.
        It will tell whether a given sin_IA input is compatible with the current S2 tile.
        """
        super()._update_filename_meta_post_hook(meta)
        def ellipsoid_normal_compatible(input_meta):
            logger.debug('TEST2 compat:\nOUT -> %s\nIN  -> %s', meta, input_meta)
            return meta['tile_name'] == input_meta['tile_name']
        meta['accept_as_compatible_input'] = ellipsoid_normal_compatible
