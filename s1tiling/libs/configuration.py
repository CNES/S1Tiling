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
#
# =========================================================================

"""
Sub-module that manages decoding of S1Processor options.
"""

import configparser
import copy
import logging
import logging.handlers
# import multiprocessing
import os
from pathlib import Path
import re
import sys
import yaml

from s1tiling.libs import exceptions
from .otbpipeline import otb_version

resource_dir = Path(__file__).parent.parent.absolute() / 'resources'


# Helper functions for extracting configuration options

def _get_opt(getter, config_filename, section, name, **kwargs):
    """
    Root helper function to report errors while extracting configuration
    options.

    The default exception returned won't always report the invalid
    section+optionname pair. Also, we prefer to obtain
    :class:`s1tiling.libs.exceptions.ConfigurationError` exception objects in
    case of error.
    """
    try:
        value = getter(section, name, **kwargs)
        return value
    except configparser.NoOptionError as e:
        # Convert the exception type
        raise exceptions.ConfigurationError(e, config_filename)
    except ValueError as e:
        # Convert the exception type, and give more context to the error.
        raise exceptions.ConfigurationError(f"Cannot decode '{name}' option '{section}' section: {e}", config_filename)


def get_opt(cfg, config_filename, section, name, **kwargs):
    """
    Helper function to report errors while extracting string configuration options
    """
    return _get_opt(cfg.get, config_filename, section, name, **kwargs)


def getint_opt(cfg, config_filename, section, name, **kwargs):
    """
    Helper function to report errors while extracting int configuration options
    """
    return _get_opt(cfg.getint, config_filename, section, name, **kwargs)


def getfloat_opt(cfg, config_filename, section, name, **kwargs):
    """
    Helper function to report errors while extracting floatting point configuration options
    """
    return _get_opt(cfg.getfloat, config_filename, section, name, **kwargs)


def getboolean_opt(cfg, config_filename, section, name, **kwargs):
    """
    Helper function to report errors while extracting boolean configuration options
    """
    return _get_opt(cfg.getboolean, config_filename, section, name, **kwargs)


# Helper functions related to logs
def init_logger(mode, paths):
    """
    Initializes logging service.
    """
    # Add the dirname where the current script is
    paths += [Path(__file__).parent.parent.absolute()]
    paths = [p / 'logging.conf.yaml' for p in paths]
    cfgpaths = [p for p in paths if p.is_file()]
    # print("from %s, keep %s" % (paths, cfgpaths))

    verbose   = 'debug'   in mode
    log2files = 'logging' in mode
    # print("verbose: ", verbose)
    # print("log2files: ", log2files)
    if cfgpaths:
        with open(cfgpaths[0], 'r') as stream:
            # FullLoader requires yaml 5.1
            # And it SHALL be used, see https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
            if hasattr(yaml, 'FullLoader'):
                config = yaml.load(stream, Loader=yaml.FullLoader)
            else:
                print("WARNING - upgrade pyyaml to version 5.1 at least!!")
                config = yaml.load(stream)
        if verbose:
            # Control the maximum global verbosity level
            config["root"]["level"] = "DEBUG"

            # Control the local console verbosity level
            config["handlers"]["console"]["level"] = "DEBUG"
        if log2files:
            if 'file' not in config["root"]["handlers"]:
                config["root"]["handlers"] += ['file']
            if 'important' not in config["root"]["handlers"]:
                config["root"]["handlers"] += ['important']
        main_config = copy.deepcopy(config)
        for _, cfg in main_config['handlers'].items():
            if 'filename' in cfg and '%' in cfg['filename']:
                cfg['filename'] = cfg['filename'] % ('main',)
        logging.config.dictConfig(main_config)
        return config
    else:
        # This situation should not happen
        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # os.environ["OTB_LOGGER_LEVEL"]="DEBUG"
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
        return None


# The configuration decoding specific to S1Tiling application
class Configuration():
    """This class handles the parameters from the cfg file"""
    def __init__(self, configFile, do_show_configuration=True):
        config = configparser.ConfigParser(os.environ)
        config.read(configFile)

        # Logs
        self.Mode = get_opt(config, configFile, 'Processing', 'mode')
        self.log_config = init_logger(self.Mode, [Path(configFile).parent.absolute()])
        # self.log_queue = multiprocessing.Queue()
        # self.log_queue_listener = logging.handlers.QueueListener(self.log_queue)
        if "debug" in self.Mode and self.log_config and self.log_config['loggers']['s1tiling.OTB']['level'] == 'DEBUG':
            # OTB DEBUG logs are displayed iff level is DEBUG and configFile mode
            # is debug as well.
            os.environ["OTB_LOGGER_LEVEL"] = "DEBUG"

        # Other options
        self.output_preprocess = get_opt(config, configFile, 'Paths', 'output')
        self.lia_directory     = get_opt(config, configFile, 'Paths', 'lia', fallback=os.path.join(self.output_preprocess, '_LIA'))
        self.raw_directory     = get_opt(config, configFile, 'Paths', 's1_images')
        self.srtm              = get_opt(config, configFile, 'Paths', 'srtm')
        self.tmpdir            = get_opt(config, configFile, 'Paths', 'tmp')
        if not os.path.isdir(self.tmpdir) and not os.path.isdir(os.path.dirname(self.tmpdir)):
            # Even if tmpdir doesn't exist we should still be able to create it
            raise exceptions.ConfigurationError(f"tmpdir={self.tmpdir} is not a valid path", configFile)
        self.GeoidFile         = get_opt(config, configFile, 'Paths', 'geoid_file', fallback=str(resource_dir/'Geoid/egm96.grd'))
        if config.has_section('PEPS'):
            raise exceptions.ConfigurationError('Since version 0.2, S1Tiling use [DataSource] instead of [PEPS] in config files. Please update your configuration!', configFile)
        self.eodagConfig               = get_opt(config, configFile, 'DataSource', 'eodagConfig', fallback=None)
        self.download                  = getboolean_opt(config, configFile, 'DataSource', 'download')
        self.ROI_by_tiles              = get_opt(config, configFile, 'DataSource', 'roi_by_tiles')
        self.first_date                = get_opt(config, configFile, 'DataSource', 'first_date')
        self.last_date                 = get_opt(config, configFile, 'DataSource', 'last_date')
        self.orbit_direction           = get_opt(config, configFile, 'DataSource', 'orbit_direction', fallback=None)
        if self.orbit_direction and self.orbit_direction not in ['ASC', 'DES']:
            raise exceptions.ConfigurationError("Parameter [orbit_direction] must be either unset or DES, or ASC", configFile)
        relative_orbit_list_str        = get_opt(config, configFile, 'DataSource', 'relative_orbit_list', fallback='')
        self.relative_orbit_list       = [int(o) for o in re.findall(r'\d+', relative_orbit_list_str)]
        self.polarisation              = get_opt(config, configFile, 'DataSource', 'polarisation')
        if   self.polarisation == 'VV-VH':
            self.polarisation = 'VV VH'
        elif self.polarisation == 'HH-HV':
            self.polarisation = 'HH HV'
        elif self.polarisation not in ['VV', 'VH', 'HH', 'HV']:
            raise exceptions.ConfigurationError("Parameter [polarisation] must be either HH-HV, VV-VH, HH, HV, VV or VH", configFile)

        # 0 => no filter
        self.tile_to_product_overlap_ratio = getint_opt(config, configFile, 'DataSource', 'tile_to_product_overlap_ratio', fallback=0)
        if self.tile_to_product_overlap_ratio > 100:
            raise exceptions.ConfigurationError("Parameter [tile_to_product_overlap_ratio] must be a percentage in [1, 100]", configFile)

        if self.download:
            self.nb_download_processes = getint_opt(config, configFile, 'DataSource', 'nb_parallel_downloads', fallback=1)

        self.type_image         = "GRD"
        self.mask_cond          = getboolean_opt(config, configFile, 'Mask', 'generate_border_mask')
        self.cache_srtm_by      = get_opt(config, configFile, 'Processing', 'cache_srtm_by', fallback='symlink')
        if self.cache_srtm_by not in ['symlink', 'copy']:
            raise exceptions.ConfigurationError(f"Unexpected value for Processing.cache_srtm_by option: '{self.cache_srtm_by}' is neither 'copy' nor 'symlink'", configFile)

        self.calibration_type   = get_opt(config, configFile, 'Processing', 'calibration')
        self.removethermalnoise = getboolean_opt(config, configFile, 'Processing', 'remove_thermal_noise')
        if self.removethermalnoise and otb_version() < '7.4.0':
            raise exceptions.InvalidOTBVersionError(f"OTB {otb_version()} does not support noise removal. Please upgrade OTB to version 7.4.0 or disable 'remove_thermal_noise' in '{configFile}'")

        self.lower_signal_value = getfloat_opt(config, configFile, 'Processing', 'lower_signal_value', fallback=1e-7)
        if self.lower_signal_value <= 0:  # TODO test nan, and >= 1e-3 ?
            raise exceptions.ConfigurationError("'lower_signal_value' parameter shall be a positive (small value) aimed at replacing null value produced by denoising.", configFile)

        self.out_spatial_res    = getfloat_opt(config, configFile, 'Processing', 'output_spatial_resolution')

        self.output_grid        = get_opt(config, configFile, 'Processing', 'tiles_shapefile', fallback=str(resource_dir/'shapefile/Features.shp'))
        if not os.path.isfile(self.output_grid):
            raise exceptions.ConfigurationError(f"output_grid={self.output_grid} is not a valid path", configFile)

        self._SRTMShapefile       = resource_dir / 'shapefile' / 'srtm_tiles.gpkg'

        self.grid_spacing         = getfloat_opt(config, configFile, 'Processing', 'orthorectification_gridspacing')
        self.interpolation_method = get_opt(config, configFile, 'Processing', 'orthorectification_interpolation_method', fallback='nn')
        try:
            tiles_file = get_opt(config, configFile, 'Processing', 'tiles_list_in_file')
            with open(tiles_file, 'r') as tiles_file_handle:
                self.tile_list = tiles_file_handle.readlines()
            self.tile_list = [s.rstrip() for s in self.tile_list]
            logging.info("The following tiles will be processed: %s", self.tile_list)
        except Exception:  # pylint: disable=broad-except
            tiles = get_opt(config, configFile, 'Processing', 'tiles')
            self.tile_list = [s.strip() for s in re.split(r'\s*,\s*', tiles)]

        self.nb_procs                      = getint_opt(config, configFile, 'Processing', 'nb_parallel_processes')
        self.ram_per_process               = getint_opt(config, configFile, 'Processing', 'ram_per_process')
        self.OTBThreads                    = getint_opt(config, configFile, 'Processing', 'nb_otb_threads')

        self.produce_lia_map               = getboolean_opt(config, configFile, 'Processing', 'produce_lia_map', fallback=False)

        self.filter = get_opt(config, configFile, 'Filtering', 'filter', fallback='').lower()
        if self.filter and self.filter == 'none':
            self.filter = ''
        if self.filter:
            self.keep_non_filtered_products = getboolean_opt(config, configFile, 'Filtering', 'keep_non_filtered_products')
            # if generate_border_mask, override this value
            if self.mask_cond:
                logging.warning('As masks are produced, Filtering.keep_non_filtered_products value will be ignored')
                self.keep_non_filtered_products = True

            self.filter_options = {'rad': getint_opt(config, configFile, 'Filtering', 'window_radius')}
            if self.filter == 'frost':
                self.filter_options['deramp']  = getfloat_opt(config, configFile, 'Filtering', 'deramp')
            elif self.filter in ['lee', 'gammamap', 'kuan']:
                self.filter_options['nblooks'] = getfloat_opt(config, configFile, 'Filtering', 'nblooks')
            else:
                raise exceptions.ConfigurationError(f"Invalid despeckling filter value '{self.filter}'. Select one among none/lee/frost/gammamap/kuan", configFile)

        try:
            self.override_azimuth_cut_threshold_to = getboolean_opt(config, configFile, 'Processing', 'override_azimuth_cut_threshold_to')
        except Exception:  # pylint: disable=broad-except
            # We cannot use "fallback=None" to handle ": None" w/ getboolean()
            self.override_azimuth_cut_threshold_to = None

        # Permit to override default file name formats
        fname_fmt_keys = ['calibration', 'correct_denoising', 'cut_borders',
                'orthorectification', 'concatenation', 'dem_s1_agglomeration',
                's1_on_dem', 'xyz', 'normals', 's1_lia', 's1_sin_lia',
                'lia_orthorectification', 'lia_concatenation', 'lia_product',
                's2_lia_corrected', 'filtered']
        self.fname_fmt = {}
        for key in fname_fmt_keys:
            fmt = get_opt(config, configFile, 'Processing', f'fname_fmt.{key}', fallback=None)
            # Default value is defined in associated StepFactories
            if fmt:
                self.fname_fmt[key] = fmt

        if do_show_configuration:
            self.show_configuration()

    def show_configuration(self):
        logging.debug("Running S1Tiling with:")
        logging.debug("[Paths]")
        logging.debug("- geoid_file                     : %s",     self.GeoidFile)
        logging.debug("- s1_images                      : %s",     self.raw_directory)
        logging.debug("- output                         : %s",     self.output_preprocess)
        logging.debug("- LIA                            : %s",     self.lia_directory)
        logging.debug("- srtm                           : %s",     self.srtm)
        logging.debug("- tmp                            : %s",     self.tmpdir)
        logging.debug("[DataSource]")
        logging.debug("- download                       : %s",     self.download)
        logging.debug("- first_date                     : %s",     self.first_date)
        logging.debug("- last_date                      : %s",     self.last_date)
        logging.debug("- polarisation                   : %s",     self.polarisation)
        logging.debug("- orbit_direction                : %s",     self.orbit_direction)
        logging.debug("- relative_orbit_list            : %s",     self.relative_orbit_list)
        logging.debug("- tile_to_product_overlap_ratio  : %s%%",   self.tile_to_product_overlap_ratio)
        logging.debug("- roi_by_tiles                   : %s",     self.ROI_by_tiles)
        if self.download:
            logging.debug("- nb_parallel_downloads          : %s", self.nb_download_processes)
        logging.debug("[Processing]")
        logging.debug("- calibration                    : %s",     self.calibration_type)
        logging.debug("- mode                           : %s",     self.Mode)
        logging.debug("- nb_otb_threads                 : %s",     self.OTBThreads)
        logging.debug("- nb_parallel_processes          : %s",     self.nb_procs)
        logging.debug("- orthorectification_gridspacing : %s",     self.grid_spacing)
        logging.debug("- output_spatial_resolution      : %s",     self.out_spatial_res)
        logging.debug("- ram_per_process                : %s",     self.ram_per_process)
        logging.debug("- remove_thermal_noise           : %s",     self.removethermalnoise)
        logging.debug("- srtm_shapefile                 : %s",     self._SRTMShapefile)
        logging.debug("- tiles                          : %s",     self.tile_list)
        logging.debug("- tiles_shapefile                : %s",     self.output_grid)
        logging.debug("- produce LIA° map               : %s",     self.produce_lia_map)
        logging.debug("[Mask]")
        logging.debug("- generate_border_mask           : %s",     self.mask_cond)
        logging.debug("[Filter]")
        logging.debug("- Speckle filtering method       : %s",     self.filter or "none")
        if self.filter:
            logging.debug("- Keeping previous products      : %s",     self.keep_non_filtered_products)
            logging.debug("- Window radius                  : %s",     self.filter_options['rad'])
            if   self.filter in ['lee', 'gammamap', 'kuan']:
                logging.debug("- nblooks                        : %s",     self.filter_options['nblooks'])
            elif self.filter in ['frost']:
                logging.debug("- deramp                         : %s",     self.filter_options['deramp'])

        logging.debug('File formats')
        for k, fmt in self.fname_fmt.items():
            logging.debug(' - %s --> %s', k, fmt)

    @property
    def srtm_db_filepath(self):
        """
        Get the SRTMShapefile databe filepath
        """
        return str(self._SRTMShapefile)

    @property
    def fname_fmt_concatenation(self):
        """
        Helper method to return the ``Processing.fnmatch.concatenation` actual value
        """
        calibration_is_done_in_S1 = self.calibration_type in ['sigma', 'beta', 'gamma', 'dn']
        if calibration_is_done_in_S1:
            # logger.debug('Concatenation in legacy mode: fname_fmt without "_%s"', cfg.calibration_type)
            # Legacy mode: the default final filename won't contain the calibration_type
            fname_fmt = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}.tif'
        else:
            # logger.debug('Concatenation in NORMLIM mode: fname_fmt with "_beta" for %s', cfg.calibration_type)
            # Let the default force the "beta" calibration_type in the filename
            fname_fmt = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_{calibration_type}.tif'
        fname_fmt = self.fname_fmt.get('concatenation') or fname_fmt
        return fname_fmt

    @property
    def fname_fmt_filtered(self):
        """
        Helper method to return the ``Processing.fnmatch.filtered` actual value
        """
        calibration_is_done_in_S1 = self.calibration_type in ['sigma', 'beta', 'gamma', 'dn']
        if calibration_is_done_in_S1:
            # logger.debug('Concatenation in legacy mode: fname_fmt without "_%s"', cfg.calibration_type)
            # Legacy mode: the default final filename won't contain the calibration_type
            fname_fmt = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_filtered.tif'
        else:
            # logger.debug('Concatenation in NORMLIM mode: fname_fmt with "_beta" for %s', cfg.calibration_type)
            # Let the default force the "beta" calibration_type in the filename
            fname_fmt = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_{calibration_type}_filtered.tif'
        fname_fmt = self.fname_fmt.get('filtered') or fname_fmt
        return fname_fmt
