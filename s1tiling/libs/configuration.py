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
#
# =========================================================================

"""
Sub-module that manages decoding of S1Processor options.
"""

import configparser
import copy
from string import Formatter
import logging
import logging.handlers
# import multiprocessing
import os
from pathlib import Path
import re
import yaml

from s1tiling.libs import exceptions
from .otbpipeline import otb_version
from ..__meta__ import __version__ as s1tiling_version

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
def _init_logger(mode, paths):
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


# The configuration decoding specific to S1Tiling application
class Configuration():
    """This class handles the parameters from the cfg file"""
    def __init__(self, configFile, do_show_configuration=True):
        config = configparser.ConfigParser(os.environ)
        config.read(configFile)

        # Logs
        #: Logging mode
        self.Mode = get_opt(config, configFile, 'Processing', 'mode', fallback=None)
        self.__log_config = None
        if self.Mode is not None:
            self.init_logger(Path(configFile).parent.absolute())

        # Other options
        #: Destination directory where product will be generated: :ref:`[PATHS.output] <paths.output>`
        self.output_preprocess   = get_opt(config, configFile, 'Paths', 'output')
        #: Destination directory where LIA maps products are generated:  :ref:`[PATHS.lia] <paths.lia>`
        self.lia_directory       = get_opt(config, configFile, 'Paths', 'lia', fallback=os.path.join(self.output_preprocess, '_LIA'))
        #: Where S1 images are downloaded: See :ref:`[PATHS.s1_images] <paths.s1_images>`!
        self.raw_directory       = get_opt(config, configFile, 'Paths', 's1_images')

        # "dem_dir" or Fallback to old deprecated key: "srtm"
        #: Where DEM files are expected to be found: See :ref:`[PATHS.dem_dir] <paths.dem_dir>`!
        self.dem                 = get_opt(config, configFile, 'Paths', 'dem_dir', fallback='') or get_opt(config, configFile, 'Paths', 'srtm')
        self._DEMShapefile       = get_opt(config, configFile, 'Paths', 'dem_database', fallback='')
        # TODO: Inject resource_dir/'shapefile' if relative dir and not existing
        #: Path to the internal DEM tiles database: automatically set
        self._DEMShapefile       = self._DEMShapefile or resource_dir / 'shapefile' / 'srtm_tiles.gpkg'
        #: Filename format string to locate the DEM file associated to an *identifier*: See :ref:`[PATHS.dem_format] <paths.dem_format>`
        self.dem_filename_format = get_opt(config, configFile, 'Paths', 'dem_format', fallback='{id}.hgt')
        # List of keys/ids to extract from DEM database ; deduced from the keys
        # used in the filename format; see https://stackoverflow.com/a/22830468/15934
        #: List of keys/ids to extract from DEM DB
        self.dem_field_ids       = [fn for _, fn, _, _ in Formatter().parse(self.dem_filename_format) if fn is not None]
        # extract the ID that will be use as the reference for names
        main_ids = list(filter(lambda f: 'id' in f or 'ID' in f, self.dem_field_ids))
        #: Main DEM field id.
        self.dem_main_field_id   = (main_ids or self.dem_field_ids)[0]
        # logger.debug('Using %s as DEM tile main id for name', main_id)

        #: Where tmp files are produced: See :ref:`[PATHS.tmp] <paths.tmp>`
        self.tmpdir            = get_opt(config, configFile, 'Paths', 'tmp')
        if not os.path.isdir(self.tmpdir) and not os.path.isdir(os.path.dirname(self.tmpdir)):
            # Even if tmpdir doesn't exist we should still be able to create it
            raise exceptions.ConfigurationError(f"tmpdir={self.tmpdir} is not a valid path", configFile)
        #: Path to Geoid model. :ref:`[PATHS.geoid_file] <paths.geoid_file>`
        self.GeoidFile         = get_opt(config, configFile, 'Paths', 'geoid_file', fallback=str(resource_dir/'Geoid/egm96.grd'))

        if config.has_section('PEPS'):
            raise exceptions.ConfigurationError('Since version 0.2, S1Tiling use [DataSource] instead of [PEPS] in config files. Please update your configuration!', configFile)
        #: Path to EODAG configuration file: :ref:`[DataSource.eodag_config] <DataSource.eodag_config>`
        self.eodag_config               = get_opt(config, configFile, 'DataSource', 'eodag_config', fallback=None) or get_opt(config, configFile, 'DataSource', 'eodagConfig', fallback=None)
        #: Boolean flag that enables/disables download of S1 input images: :ref:`[DataSource.download] <DataSource.download>`
        self.download                  = getboolean_opt(config, configFile, 'DataSource', 'download')
        #: Region Of Interest to download: See :ref:`[DataSource.roi_by_tiles] <DataSource.roi_by_tiles>`
        self.ROI_by_tiles              = get_opt(config, configFile, 'DataSource', 'roi_by_tiles')
        #: Start date: :ref:`[DataSource.first_date] <DataSource.first_date>`
        self.first_date                = get_opt(config, configFile, 'DataSource', 'first_date')
        #: End date: :ref:`[DataSource.last_date] <DataSource.last_date>`
        self.last_date                 = get_opt(config, configFile, 'DataSource', 'last_date')

        platform_list_str              = get_opt(config, configFile, 'DataSource', 'platform_list', fallback='')
        platform_list                  = [x for x in SPLIT_PATTERN.split(platform_list_str) if x]
        unsupported_platforms          = [p for p in platform_list if p and not p.startswith("S1")]
        if unsupported_platforms:
            raise exceptions.ConfigurationError(f"Non supported requested platforms: {', '.join(unsupported_platforms)}", configFile)
        #: Filter to restrict platform: See  :ref:`[DataSource.platform_list] <DataSource.platform_list>`
        self.platform_list             = platform_list

        #: Filter to restrict orbit direction: See :ref:`[DataSource.orbit_direction] <DataSource.orbit_direction>`
        self.orbit_direction           = get_opt(config, configFile, 'DataSource', 'orbit_direction', fallback=None)
        if self.orbit_direction and self.orbit_direction not in ['ASC', 'DES']:
            raise exceptions.ConfigurationError("Parameter [orbit_direction] must be either unset or DES, or ASC", configFile)
        relative_orbit_list_str        = get_opt(config, configFile, 'DataSource', 'relative_orbit_list', fallback='')
        #: Filter to restrict relative orbits: See :ref:`[DataSource.relative_orbit_list] <DataSource.relative_orbit_list>`
        self.relative_orbit_list       = [int(o) for o in re.findall(r'\d+', relative_orbit_list_str)]
        #: Filter to polarisation: See :ref:`[DataSource.polarisation] <DataSource.polarisation>`
        self.polarisation              = get_opt(config, configFile, 'DataSource', 'polarisation')
        if   self.polarisation == 'VV-VH':
            self.polarisation = 'VV VH'
        elif self.polarisation == 'HH-HV':
            self.polarisation = 'HH HV'
        elif self.polarisation not in ['VV', 'VH', 'HH', 'HV']:
            raise exceptions.ConfigurationError("Parameter [polarisation] must be either HH-HV, VV-VH, HH, HV, VV or VH", configFile)

        # 0 => no filter
        #: Filter to ensure minimum S2 tile coverage by S1 input products: :ref:`[DataSource.tile_to_product_overlap_ratio] <DataSource.tile_to_product_overlap_ratio>`
        self.tile_to_product_overlap_ratio = getint_opt(config, configFile, 'DataSource', 'tile_to_product_overlap_ratio', fallback=0)
        if self.tile_to_product_overlap_ratio > 100:
            raise exceptions.ConfigurationError("Parameter [tile_to_product_overlap_ratio] must be a percentage in [1, 100]", configFile)

        if self.download:
            #: Number of downloads that can be done in parallel: :ref:`[DataSource.nb_parallel_downloads] <DataSource.nb_parallel_downloads>`
            self.nb_download_processes = getint_opt(config, configFile, 'DataSource', 'nb_parallel_downloads', fallback=1)

        #: Type of images handled
        self.type_image         = "GRD"
        #: Shall we generate mask products? :ref:`[Mask.generate_border_mask] <Mask.generate_border_mask>`
        self.mask_cond          = getboolean_opt(config, configFile, 'Mask', 'generate_border_mask')

        #:Tells whether DEM files are copied in a temporary directory, or if symbolic links are to be created. See :ref:`[Processing.cache_dem_by] <Processing.cache_dem_by>`
        self.cache_dem_by      = get_opt(config, configFile, 'Processing', 'cache_dem_by', fallback='symlink')
        if self.cache_dem_by not in ['symlink', 'copy']:
            raise exceptions.ConfigurationError(f"Unexpected value for Processing.cache_dem_by option: '{self.cache_dem_by}' is neither 'copy' nor 'symlink'", configFile)

        #: SAR Calibration applied: See :ref:`[Processing.calibration] <Processing.calibration>`
        self.calibration_type   = get_opt(config, configFile, 'Processing', 'calibration')
        #: Shall we remove thermal noise: :ref:`[Processing.remove_thermal_noise] <Processing.remove_thermal_noise>`
        self.removethermalnoise = getboolean_opt(config, configFile, 'Processing', 'remove_thermal_noise')
        if self.removethermalnoise and otb_version() < '7.4.0':
            raise exceptions.InvalidOTBVersionError(f"OTB {otb_version()} does not support noise removal. Please upgrade OTB to version 7.4.0 or disable 'remove_thermal_noise' in '{configFile}'")

        #: Minimal signal value to set after on "denoised" pixels: See :ref:`[Processing.lower_signal_value] <Processing.lower_signal_value>`
        self.lower_signal_value = getfloat_opt(config, configFile, 'Processing', 'lower_signal_value', fallback=1e-7)
        if self.lower_signal_value <= 0:  # TODO test nan, and >= 1e-3 ?
            raise exceptions.ConfigurationError("'lower_signal_value' parameter shall be a positive (small value) aimed at replacing null value produced by denoising.", configFile)

        #: Pixel size (in meters) of the output images: :ref:`[Processing.output_spatial_resolution] <Processing.output_spatial_resolution>`
        self.out_spatial_res    = getfloat_opt(config, configFile, 'Processing', 'output_spatial_resolution')


        #: Path to the tiles shape definition. See :ref:`[Processing.tiles_shapefile] <Processing.tiles_shapefile>`
        self.output_grid        = get_opt(config, configFile, 'Processing', 'tiles_shapefile', fallback=str(resource_dir/'shapefile/Features.shp'))
        if not os.path.isfile(self.output_grid):
            raise exceptions.ConfigurationError(f"output_grid={self.output_grid} is not a valid path", configFile)

        #: Grid spacing (in meters) for the interpolator in the orthorectification: See :ref:`[Processing.orthorectification_gridspacing] <Processing.orthorectification_gridspacing>`
        self.grid_spacing         = getfloat_opt(config, configFile, 'Processing', 'orthorectification_gridspacing')
        #: Orthorectification interpolation methode: See :ref:`[Processing.orthorectification_interpolation_method] <Processing.orthorectification_interpolation_method>`
        self.interpolation_method = get_opt(config, configFile, 'Processing', 'orthorectification_interpolation_method', fallback='nn')
        try:
            tiles_file = get_opt(config, configFile, 'Processing', 'tiles_list_in_file')
            with open(tiles_file, 'r') as tiles_file_handle:
                self.tile_list = tiles_file_handle.readlines()
            self.tile_list = [s.rstrip() for s in self.tile_list]
            logging.info("The following tiles will be processed: %s", self.tile_list)
        except Exception:  # pylint: disable=broad-except
            tiles = get_opt(config, configFile, 'Processing', 'tiles')
            #: List of S2 tiles to process: See :ref:`[Processing.tiles] <Processing.tiles>`
            self.tile_list = [s.strip() for s in re.split(r'\s*,\s*', tiles)]

        #: Number of tasks executed in parallel: See :ref:`[Processing.nb_parallel_processes] <Processing.nb_parallel_processes>`
        self.nb_procs                      = getint_opt(config, configFile, 'Processing', 'nb_parallel_processes')
        #: RAM allocated to OTB applications: See :ref:`[Processing.ram_per_process] <Processing.ram_per_process>`
        self.ram_per_process               = getint_opt(config, configFile, 'Processing', 'ram_per_process')
        #: Number of threads allocated to each OTB application: See :ref:`[Processing.nb_otb_threads] <Processing.nb_otb_threads>`
        self.OTBThreads                    = getint_opt(config, configFile, 'Processing', 'nb_otb_threads')

        #: Tells whether LIA map in degrees * 100 shall be produced alongside the sine map: See :ref:`[Processing.produce_lia_map] <Processing.produce_lia_map>`
        self.produce_lia_map               = getboolean_opt(config, configFile, 'Processing', 'produce_lia_map', fallback=False)

        #: Despeckle filter to apply, if any: See :ref:`[Filtering.filter] <Filtering.filter>`
        self.filter = get_opt(config, configFile, 'Filtering', 'filter', fallback='').lower()
        if self.filter and self.filter == 'none':
            self.filter = ''
        if self.filter:
            #: Shall we keep non-filtered products? See :ref:`[Filtering.keep_non_filtered_products] <Filtering.keep_non_filtered_products>`
            self.keep_non_filtered_products = getboolean_opt(config, configFile, 'Filtering', 'keep_non_filtered_products')
            # if generate_border_mask, override this value
            if self.mask_cond:
                logging.warning('As masks are produced, Filtering.keep_non_filtered_products value will be ignored')
                self.keep_non_filtered_products = True

            #: Dictionary of filter options: {'rad': :ref:`[Filtering.window_radius] <Filtering.window_radius>`, 'deramp': :ref:`[Filtering.deramp] <Filtering.deramp>`, 'nblooks': :ref:`[Filtering.nblooks] <Filtering.nblooks>`}
            self.filter_options = {
                    'rad': getint_opt(config, configFile, 'Filtering', 'window_radius')
            }
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
            #: Internal to override analysing of top/bottom cutting: See :ref:`[Processing.override_azimuth_cut_threshold_to] <Processing.override_azimuth_cut_threshold_to>`
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
        logging.debug("Running S1Tiling %s with:", s1tiling_version)
        logging.debug("[Paths]")
        logging.debug("- geoid_file                     : %s",     self.GeoidFile)
        logging.debug("- s1_images                      : %s",     self.raw_directory)
        logging.debug("- output                         : %s",     self.output_preprocess)
        logging.debug("- LIA                            : %s",     self.lia_directory)
        logging.debug("- dem directory                  : %s",     self.dem)
        logging.debug("- dem filename format            : %s",     self.dem_filename_format)
        logging.debug("- dem field ids (from shapefile) : %s",     self.dem_field_ids)
        logging.debug("- main ID for DEM names deduced  : %s",     self.dem_main_field_id)
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
        logging.debug("- dem_shapefile                  : %s",     self._DEMShapefile)
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

    def init_logger(self, config_log_dir, mode=None):
        """
        Deported logger initialization function for project that use their own
        logger, and S1Tiling through its API only.

        :param mode: Option to override logging mode, if not found/expected in
                     the configuration file.
        """
        if not self.__log_config:
            self.__log_config = _init_logger(self.Mode, [config_log_dir])
            self.Mode = mode or self.Mode
            assert self.Mode, "Please set a valid logging mode!"
            # self.log_queue = multiprocessing.Queue()
            # self.log_queue_listener = logging.handlers.QueueListener(self.log_queue)
            if "debug" in self.Mode and self.__log_config and self.__log_config['loggers']['s1tiling.OTB']['level'] == 'DEBUG':
                # OTB DEBUG logs are displayed iff level is DEBUG and configFile mode
                # is debug as well.
                os.environ["OTB_LOGGER_LEVEL"] = "DEBUG"

    @property
    def log_config(self):
        """
        Property log
        """
        assert self.__log_config, "Please initialize logger"
        return self.__log_config

    @property
    def dem_db_filepath(self):
        """
        Get the DEMShapefile databe filepath
        """
        return str(self._DEMShapefile)

    @property
    def fname_fmt_concatenation(self):
        """
        Helper method to return the ``Processing.fnmatch.concatenation`` actual value
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
        Helper method to return the ``Processing.fnmatch.filtered`` actual value
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
