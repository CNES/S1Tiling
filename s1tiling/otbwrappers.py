#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright (c) CESBIO. All rights reserved.
#
#   See LICENSE for details.
#
#   This software is distributed WITHOUT ANY WARRANTY; without even
#   the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#   PURPOSE.  See the above copyright notices for more information.
#
# =========================================================================
#
# Authors: Thierry KOLECK (CNES)
#          Luc HERMITTE (CS Group)
# =========================================================================

import gdal, rasterio
from rasterio.windows import Window
import numpy as np
import tempfile
import logging
import os
from s1tiling.otbpipeline import StepFactory, in_filename, out_filename
from s1tiling import Utils
import otbApplication as otb


def has_too_many_NoData(image, threshold, nodata):
    """
    Analyses whether an image contains NO DATA.

        :param image:     np.array image to analyse
        :param threshold: number of NoData searched
        :param nodata:    no data value
        :return:          whether the number of no-data pixel > threshold
    """
    nbNoData = len(np.argwhere(image==nodata))
    return nbNoData>threshold


class AnalyseBorders(StepFactory):
    """
    StepFactory that analyses whether image borders need to be cut.

    The step produced by this actual factory doesn't register any OTB
    application nor execute one.
    However, it loads two lines from the input image to determine whether it
    contains too many NoData.
    Found information will be stored into the `meta` dictionary for later use
    by `CutBorders` step factory.
    """
    def __init__(self, cfg):
        super().__init__('')
        pass
    def parameters(self, meta):
        return None
    def output_directory(self, meta):
        raise TypeError("An AnalyseBorders step don't produce anything!")
    def build_step_output_filename(self, meta):
        return meta['out_filename']
    def complete_meta(self, meta):
        meta = super().complete_meta(meta)

        cut_overlap_range = 1000 # Number of columns to cut on the sides. Here 500pixels = 5km
        cut_overlap_azimuth = 1600 # Number of lines to cut at top or bottom
        thr_nan_for_cropping = cut_overlap_range*2 #Quand on fait les tests, on a pas encore couper les nan sur le cote, d'ou l'utilisatoin de ce thr
        with rasterio.open(meta['out_filename']) as ds_reader:
            xsize = ds_reader.width
            ysize = ds_reader.height
            north = ds_reader.read(1, window=Window(0, 100, xsize+1, 1))
            south = ds_reader.read(1, window=Window(0, ysize-100, xsize+1, 1))

        crop1 = has_too_many_NoData(north, thr_nan_for_cropping, 0)
        crop2 = has_too_many_NoData(south, thr_nan_for_cropping, 0)
        logging.debug("   => need to crop north: %s", crop1)
        logging.debug("   => need to crop south: %s", crop2)
        meta['cut'] = {
                'threshold.x'      : cut_overlap_range,
                'threshold.y.start': cut_overlap_azimuth if crop1 else 0,
                'threshold.y.end'  : cut_overlap_azimuth if crop2 else 0,
                }
        return meta


class Calibrate(StepFactory):
    """
    Factory that prepares steps that run `SARCalibration`.

    Requires the following information from the configuration object:
    - `ram_per_process`
    - `calibration_type`
    - `removethermalnoise`
    Requires the following information from the metadata dictionary
    - base name -- to generate typical output filename
    - input filename
    - output filename
    """
    def __init__(self, cfg):
        super().__init__('SARCalibration')
        # Warning: config object cannot be stored and passed to workers!
        # => We extract what we need
        self.__ram_per_process    = cfg.ram_per_process
        self.__calibration_type   = cfg.calibration_type
        self.__removethermalnoise = cfg.removethermalnoise
        self.__tmpdir             = cfg.tmpdir
    def output_directory(self, meta):
        tile_name = meta['tile_name']
        return os.path.join(self.__tmpdir, tile_name)
    def build_step_output_filename(self, meta):
        filename = meta['basename'].replace(".tiff", "_calOk.tiff")
        return os.path.join(self.output_directory(meta), filename)
    def parameters(self, meta):
        return {
                'ram'           : str(self.__ram_per_process),
                # 'progress'    : 'false',
                self.param_in   : in_filename(meta),
                # self.param_out  : out_filename(meta),
                'lut'           : self.__calibration_type,
                'noise'         : str(self.__removethermalnoise).lower()
                }


class CutBorders(StepFactory):
    """
    Factory that prepares steps that run `ClampROI`.

    Requires the following information from the configuration object:
    - `ram_per_process`
    Requires the following information from the metadata dictionary
    - base name -- to generate typical output filename
    - input filename
    - output filename
    - `cut`->`threshold.x`       -- from AnalyseBorders
    - `cut`->`threshold.y.start` -- from AnalyseBorders
    - `cut`->`threshold.y.end`   -- from AnalyseBorders
    """
    def __init__(self, cfg):
        super().__init__('ClampROI')
        self.__ram_per_process    = cfg.ram_per_process
        self.__tmpdir             = cfg.tmpdir
    def output_directory(self, meta):
        tile_name = meta['tile_name']
        return os.path.join(self.__tmpdir, tile_name)
    def build_step_output_filename(self, meta):
        filename = meta['basename'].replace(".tiff", "_OrthoReady.tiff")
        return os.path.join(self.output_directory(meta), filename)
    def parameters(self, meta):
        return {
                'ram'              : str(self.__ram_per_process),
                # 'progress'       : 'false',
                self.param_in      : in_filename(meta),
                # self.param_out     : out_filename(meta),
                'threshold.x'      : meta['cut']['threshold.x'],
                'threshold.y.start': meta['cut']['threshold.y.start'],
                'threshold.y.end'  : meta['cut']['threshold.y.end']
                }


class OrthoRectify(StepFactory):
    """
    Factory that prepares steps that run `OrthoRectification`.

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
        super().__init__('OrthoRectification', param_in='io.in', param_out='io.out')
        self.__ram_per_process    = cfg.ram_per_process
        self.__out_spatial_res    = cfg.out_spatial_res
        self.__GeoidFile          = cfg.GeoidFile
        self.__grid_spacing       = cfg.grid_spacing
        self.__tmp_srtm_dir       = cfg.tmp_srtm_dir
        self.__tmpdir             = cfg.tmpdir
    def output_directory(self, meta):
        tile_name = meta['tile_name']
        return os.path.join(self.__tmpdir, tile_name)
    def build_step_output_filename(self, meta):
        # Will be get around in complete_meta
        return None
    def complete_meta(self, meta):
        meta = super().complete_meta(meta)
        # TODO: need manifest!!!
        manifest                = meta['manifest']
        image                   = in_filename(meta)   # meta['in_filename']
        tile_name               = meta['tile_name']
        tile_origin             = meta['tile_origin']
        logging.debug("meta :%s", meta)
        logging.debug("image :%s", image)
        logging.debug("tile_name :%s", tile_name)
        current_date            = Utils.get_date_from_s1_raster(image)
        current_polar           = Utils.get_polar_from_s1_raster(image)
        current_platform        = Utils.get_platform_from_s1_raster(image)
        current_orbit_direction = Utils.get_orbit_direction(manifest)
        current_relative_orbit  = Utils.get_relative_orbit(manifest)
        out_utm_zone            = tile_name[0:2]
        out_utm_northern        = (tile_name[2] >= 'N')
        in_epsg                 = 4326
        out_epsg                = 32600+int(out_utm_zone)
        if not out_utm_northern:
            out_epsg = out_epsg+100

        x_coord, y_coord, _  = Utils.convert_coord([tile_origin[0]], in_epsg, out_epsg)[0]
        lrx, lry, _          = Utils.convert_coord([tile_origin[2]], in_epsg, out_epsg)[0]

        if not out_utm_northern and y_coord < 0:
            y_coord += 10000000.
            lry     += 10000000.

        # TODO: mkdir cannot work in multiproc env...
        working_directory = self.output_directory(meta)
        if os.path.exists(working_directory) == False:
            os.makedirs(working_directory)
        ortho_image_name = current_platform\
                           +"_"+tile_name\
                           +"_"+current_polar\
                           +"_"+current_orbit_direction\
                           +'_{:0>3d}'.format(current_relative_orbit)\
                           +"_"+current_date\
                           +".tif"\
                           +"?&writegeom=false&gdal:co:COMPRESS=DEFLATE"
        out_filename = os.path.join(working_directory, ortho_image_name)
        meta['out_filename'] = out_filename
        spacing = self.__out_spatial_res
        logging.debug("from %s, lrx=%s, x_coord=%s, spacing=%s", tile_name, lrx, x_coord, spacing)
        meta['params.ortho'] = {
                'opt.ram'          : str(self.__ram_per_process),
                # 'progress'       : 'false',
                self.param_in      : in_filename(meta),
                # self.param_out     : out_filename,
                'interpolator'     : 'nn',
                'outputs.spacingx' : spacing,
                'outputs.spacingy' : -self.__out_spatial_res,
                'outputs.sizex'    : int(round(abs(lrx-x_coord)/spacing)),
                'outputs.sizey'    : int(round(abs(lry-y_coord)/spacing)),
                'opt.gridspacing'  : self.__grid_spacing,
                'map'              : 'utm',
                'map.utm.zone'     : int(out_utm_zone),
                'map.utm.northhem' : str(out_utm_northern).lower(),
                'outputs.ulx'      : x_coord,
                'outputs.uly'      : y_coord,
                'elev.dem'         : self.__tmp_srtm_dir,
                'elev.geoid'       : self.__GeoidFile
                }
        # TODO
        # meta['post'] = get(meta, 'post', []) + add_ortho_metadata
        return meta
    def parameters(self, meta):
        return meta['params.ortho']


class Concatenate(StepFactory):
    """
    Factory that prepares steps that run `Synthetize`.

    Requires the following information from the configuration object:
    - `ram_per_process`
    Requires the following information from the metadata dictionary
    - input filename
    - output filename
    """
    def __init__(self, cfg):
        super().__init__('Synthetize', param_in='il', param_out='out')
        self.__ram_per_process    = cfg.ram_per_process
        self.__outdir             = cfg.output_preprocess
    def output_directory(self, meta):
        return os.path.join(self.__outdir, meta['tile_name'])
    def build_step_output_filename(self, meta):
        filename = meta['basename']
        return os.path.join(self.output_directory(meta), filename)
        # logging.debug("meta is: %s", meta)
        # im0 = in_filename(meta)
        # output_image = im0[:-10]+"xxxxxx"+im0[-4:]
        # return output_image
    def parameters(self, meta):
        return {
                'ram'              : str(self.__ram_per_process),
                # 'progress'       : 'false',
                self.param_in      : in_filename(meta),
                # self.param_out     : out_filename(meta),
                }


class BuildBorderMask(StepFactory):
    """
    Factory that prepares the first step that generates border maks.

    Requires the following information from the configuration object:
    - `ram_per_process`
    Requires the following information from the metadata dictionary
    - input filename
    - output filename
    """
    def __init__(self, cfg):
        super().__init__('BandMath', param_in='il', param_out='out')
        self.__ram_per_process    = cfg.ram_per_process
        self.__tmpdir             = cfg.tmpdir
    def output_directory(self, meta):
        tile_name = meta['tile_name']
        return os.path.join(self.__tmpdir, tile_name)
    def build_step_output_filename(self, meta):
        filename = meta['basename'].replace(".tif", "_BorderMask_TMP.tif")
        return os.path.join(self.output_directory(meta), filename)
    def set_output_pixel_type(self, app, meta):
        # logging.debug('SetParameterOutputImagePixelType(%s, %s)', self.param_out, otb.ImagePixelType_uint8)
        app.SetParameterOutputImagePixelType(self.param_out, otb.ImagePixelType_uint8)
    def parameters(self, meta):
        return {
                'ram'              : str(self.__ram_per_process),
                # 'progress'       : 'false',
                self.param_in      : [in_filename(meta)],
                # self.param_out     : out_filename(meta),
                'exp'              : 'im1b1==0?0:1'
                }


class SmoothBorderMask(StepFactory):
    """
    Factory that prepares the first step that smoothes border maks.

    Requires the following information from the configuration object:
    - `ram_per_process`
    Requires the following information from the metadata dictionary
    - input filename
    - output filename
    """
    def __init__(self, cfg):
        super().__init__('BinaryMorphologicalOperation', param_in='in', param_out='out')
        self.__ram_per_process    = cfg.ram_per_process
        self.__outdir             = cfg.output_preprocess
    def output_directory(self, meta):
        return os.path.join(self.__outdir, meta['tile_name'])
    def build_step_output_filename(self, meta):
        filename = meta['basename'].replace(".tif", "_BorderMask.tif")
        return os.path.join(self.output_directory(meta), filename)
    def set_output_pixel_type(self, app, meta):
        # logging.debug('SetParameterOutputImagePixelType(%s, %s)', self.param_out, otb.ImagePixelType_uint8)
        app.SetParameterOutputImagePixelType(self.param_out, otb.ImagePixelType_uint8)
    def parameters(self, meta):
        return {
                'ram'                   : str(self.__ram_per_process),
                # 'progress'            : 'false',
                self.param_in           : in_filename(meta),
                self.param_out          : out_filename(meta),
                'structype'             : 'ball',
                # 'structype.ball.xradius': 5,
                # 'structype.ball.yradius': 5 ,
                'xradius'               : 5,
                'yradius'               : 5 ,
                'filter'                : 'opening'
                }


