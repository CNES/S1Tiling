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

from s1tiling.libs import exits
from .otbpipeline import otb_version

resource_dir = Path(__file__).parent.parent.absolute() / 'resources'

SPLIT_PATTERN = re.compile("^\s+|\s*,\s*|\s+$")

def load_log_config(cfgpaths):
    """
    Take care of loading a log configuration file expressed in YAML
    """
    with open(cfgpaths[0], 'r') as stream:
        # FullLoader requires yaml 5.1
        # And it SHALL be used, see https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
        if hasattr(yaml, 'FullLoader'):
            config = yaml.load(stream, Loader=yaml.FullLoader)
        else:
            print("WARNING - upgrade pyyaml to version 5.1 at least!!")
            config = yaml.load(stream)
    return config


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
        config = load_log_config(cfgpaths)
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
            if verbose:
                config["handlers"]["file"]["level"] = "DEBUG"
        # Update all filenames with debug mode info.
        filename_opts = {
                "mode": ".debug" if verbose else "",
                "kind": "{kind}",
                }
        for _, cfg in config['handlers'].items():
            if 'filename' in cfg and '{mode}' in cfg['filename']:
                cfg['filename'] = cfg['filename'].format_map(filename_opts)
        # Update main filename with... "main"
        main_config = copy.deepcopy(config)
        for _, cfg in main_config['handlers'].items():
            if 'filename' in cfg and '{kind}' in cfg['filename']:
                cfg['filename'] = cfg['filename'].format(kind="main")
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


class Configuration():
    """This class handles the parameters from the cfg file"""
    def __init__(self, configFile, do_show_configuration=True):
        config = configparser.ConfigParser(os.environ)
        config.read(configFile)

        # Logs
        self.Mode = config.get('Processing', 'mode')
        self.log_config = init_logger(self.Mode, [Path(configFile).parent.absolute()])
        # self.log_queue = multiprocessing.Queue()
        # self.log_queue_listener = logging.handlers.QueueListener(self.log_queue)
        if "debug" in self.Mode and self.log_config and self.log_config['loggers']['s1tiling.OTB']['level'] == 'DEBUG':
            # OTB DEBUG logs are displayed iff level is DEBUG and configFile mode
            # is debug as well.
            os.environ["OTB_LOGGER_LEVEL"] = "DEBUG"

        # Other options
        self.output_preprocess = config.get('Paths', 'output')
        self.lia_directory     = config.get('Paths', 'lia', fallback=os.path.join(self.output_preprocess, '_LIA'))
        self.raw_directory     = config.get('Paths', 's1_images')
        self.srtm              = config.get('Paths', 'srtm')
        self.tmpdir            = config.get('Paths', 'tmp')
        if not os.path.isdir(self.tmpdir) and not os.path.isdir(os.path.dirname(self.tmpdir)):
            # Even if tmpdir doesn't exist we should still be able to create it
            logging.critical("ERROR: tmpdir=%s is not a valid path", self.tmpdir)
            sys.exit(exits.CONFIG_ERROR)
        self.GeoidFile         = config.get('Paths', 'geoid_file', fallback=str(resource_dir/'Geoid/egm96.grd'))
        if config.has_section('PEPS'):
            logging.critical('Since version 0.2, S1Tiling use [DataSource] instead of [PEPS] in config files. Please update your configuration!')
            sys.exit(exits.CONFIG_ERROR)
        self.eodag_config              = config.get('DataSource', 'eodag_config', fallback=None) or config.get('DataSource', 'eodagConfig', fallback=None)
        self.download                  = config.getboolean('DataSource', 'download')
        self.ROI_by_tiles              = config.get('DataSource', 'roi_by_tiles')
        self.first_date                = config.get('DataSource', 'first_date')
        self.last_date                 = config.get('DataSource', 'last_date')

        platform_list_str              = config.get('DataSource', 'platform_list', fallback='')
        platform_list                  = [x for x in SPLIT_PATTERN.split(platform_list_str) if x]
        unsupported_platforms = [p for p in platform_list if p and not p.startswith("S1")]
        if unsupported_platforms:
            logging.critical("Non supported requested platforms: %s", ", ".join(unsupported_platforms))
            logging.critical("Please correct the config file")
            sys.exit(exits.CONFIG_ERROR)
        self.platform_list             = platform_list

        self.orbit_direction           = config.get('DataSource', 'orbit_direction', fallback=None)
        if self.orbit_direction and self.orbit_direction not in ['ASC', 'DES']:
            logging.critical("Parameter [orbit_direction] must be either unset or DES, or ASC")
            logging.critical("Please correct the config file")
            sys.exit(exits.CONFIG_ERROR)
        relative_orbit_list_str        = config.get('DataSource', 'relative_orbit_list', fallback='')
        self.relative_orbit_list       = [int(o) for o in re.findall(r'\d+', relative_orbit_list_str)]
        self.polarisation              = config.get('DataSource', 'polarisation')
        if   self.polarisation == 'VV-VH':
            self.polarisation = 'VV VH'
        elif self.polarisation == 'HH-HV':
            self.polarisation = 'HH HV'
        elif self.polarisation not in ['VV', 'VH', 'HH', 'HV']:
            logging.critical("Parameter [polarisation] must be either HH-HV, VV-VH, HH, HV, VV or VH")
            logging.critical("Please correct the config file")
            sys.exit(exits.CONFIG_ERROR)

        # 0 => no filter
        self.tile_to_product_overlap_ratio = config.getint('DataSource', 'tile_to_product_overlap_ratio', fallback=0)
        if self.tile_to_product_overlap_ratio > 100:
            logging.critical("Parameter [tile_to_product_overlap_ratio] must be a percentage in [1, 100]")
            logging.critical("Please correct the config file")
            sys.exit(exits.CONFIG_ERROR)

        if self.download:
            self.nb_download_processes = config.getint('DataSource', 'nb_parallel_downloads', fallback=1)

        self.type_image         = "GRD"
        self.mask_cond          = config.getboolean('Mask', 'generate_border_mask')
        self.cache_srtm_by      = config.get('Processing', 'cache_srtm_by', fallback='symlink')
        if self.cache_srtm_by not in ['symlink', 'copy']:
            logging.critical("Unexpected value for Processing.cache_srtm_by option: '%s' is neither 'copy' no 'symlink'", self.cache_srtm_by)
            sys.exit(exits.CONFIG_ERROR)

        self.calibration_type   = config.get('Processing', 'calibration')
        self.removethermalnoise = config.getboolean('Processing', 'remove_thermal_noise')
        if self.removethermalnoise and otb_version() < '7.4.0':
            logging.critical("ERROR: OTB %s does not support noise removal. Please upgrade OTB to version 7.4.0 or disable 'remove_thermal_noise' in '%s'", otb_version(), configFile)
            sys.exit(exits.CONFIG_ERROR)

        self.lower_signal_value = config.getfloat('Processing', 'lower_signal_value', fallback=1e-7)
        if self.lower_signal_value <= 0:  # TODO test nan, and >= 1e-3 ?
            logging.critical("ERROR: 'lower_signal_value' parameter shall be a positive (small value) aimed at replacing null value produced by denoising. Please fix '%s'", configFile)
            sys.exit(exits.CONFIG_ERROR)

        self.out_spatial_res    = config.getfloat('Processing', 'output_spatial_resolution')

        self.output_grid        = config.get('Processing', 'tiles_shapefile', fallback=str(resource_dir/'shapefile/Features.shp'))
        if not os.path.isfile(self.output_grid):
            logging.critical("ERROR: output_grid=%s is not a valid path", self.output_grid)
            sys.exit(exits.CONFIG_ERROR)

        self._SRTMShapefile       = resource_dir / 'shapefile' / 'srtm_tiles.gpkg'

        self.grid_spacing         = config.getfloat('Processing', 'orthorectification_gridspacing')
        self.interpolation_method = config.get('Processing', 'orthorectification_interpolation_method', fallback='nn')
        try:
            tiles_file = config.get('Processing', 'tiles_list_in_file')
            with open(tiles_file, 'r') as tiles_file_handle:
                self.tile_list = tiles_file_handle.readlines()
            self.tile_list = [s.rstrip() for s in self.tile_list]
            logging.info("The following tiles will be processed: %s", self.tile_list)
        except Exception:  # pylint: disable=broad-except
            tiles = config.get('Processing', 'tiles')
            self.tile_list = [s.strip() for s in re.split(r'\s*,\s*', tiles)]

        self.nb_procs                      = config.getint('Processing', 'nb_parallel_processes')
        self.ram_per_process               = config.getint('Processing', 'ram_per_process')
        self.OTBThreads                    = config.getint('Processing', 'nb_otb_threads')

        self.produce_lia_map               = config.getboolean('Processing', 'produce_lia_map', fallback=False)

        self.filter = config.get('Filtering', 'filter', fallback='').lower()
        if self.filter and self.filter == 'none':
            self.filter = ''
        if self.filter:
            self.keep_non_filtered_products = config.getboolean('Filtering', 'keep_non_filtered_products')
            # if generate_border_mask, override this value
            if self.mask_cond:
                logging.warning('As masks are produced, Filtering.keep_non_filtered_products value will be ignored')
                self.keep_non_filtered_products = True

            self.filter_options = {'rad': config.getint('Filtering', 'window_radius')}
            if self.filter == 'frost':
                self.filter_options['deramp']  = config.getfloat('Filtering', 'deramp')
            elif self.filter in ['lee', 'gammamap', 'kuan']:
                self.filter_options['nblooks'] = config.getfloat('Filtering', 'nblooks')
            else:
                logging.critical("ERROR: Invalid despeckling filter value '%s'. Select one among none/lee/frost/gammamap/kuan", self.filter)
                sys.exit(exits.CONFIG_ERROR)

        try:
            self.override_azimuth_cut_threshold_to = config.getboolean('Processing', 'override_azimuth_cut_threshold_to')
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
            fmt = config.get('Processing', f'fname_fmt.{key}', fallback=None)
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
        logging.debug("- platform_list                  : %s",     self.platform_list)
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

    # def check_date(self):
    #     """
    #     DEPRECATED
    #     """
    #     import datetime
    #
    #     fd = self.first_date
    #     ld = self.last_date
    #
    #     try:
    #         F_Date = datetime.date(int(fd[0:4]), int(fd[5:7]), int(fd[8:10]))
    #         L_Date = datetime.date(int(ld[0:4]), int(ld[5:7]), int(ld[8:10]))
    #         return F_Date, L_Date
    #     except Exception:  # pylint: disable=broad-except
    #         logging.critical("Invalid date")
    #         sys.exit(exits.CONFIG_ERROR)
