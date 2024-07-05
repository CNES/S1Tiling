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
import logging.config
# import multiprocessing
import os
from pathlib import Path
import re
from typing import Callable, Dict, List, NoReturn, Optional, Union, Tuple, TypeVar
import otbApplication as otb
import yaml

from s1tiling.libs import exceptions
from .otbtools import otb_version
from ..__meta__ import __version__ as s1tiling_version

resource_dir = Path(__file__).parent.parent.absolute() / 'resources'

SPLIT_PATTERN = re.compile(r"^\s+|\s*,\s*|\s+$")

PIXEL_TYPES = {
        'uint8'   : otb.ImagePixelType_uint8,
        'int16'   : otb.ImagePixelType_int16,
        'uint16'  : otb.ImagePixelType_uint16,
        'int32'   : otb.ImagePixelType_int32,
        'uint32'  : otb.ImagePixelType_uint32,
        'float'   : otb.ImagePixelType_float,
        'double'  : otb.ImagePixelType_double,
        'cint16'  : otb.ImagePixelType_cint16,
        'cint32'  : otb.ImagePixelType_cint32,
        'cfloat'  : otb.ImagePixelType_cfloat,
        'cdouble' : otb.ImagePixelType_cdouble,
}


def _load_log_config(cfgpaths: Path) -> Dict:
    """
    Take care of loading a log configuration file expressed in YAML
    """
    with open(cfgpaths, 'r', encoding='UTF-8') as stream:
        # FullLoader requires yaml 5.1
        # And it SHALL be used, see https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
        assert hasattr(yaml, 'FullLoader'), "Please upgrade pyyaml to version 5.1+"
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


# Helper functions for extracting configuration options

Opt = TypeVar('Opt', int, float, bool)


def _get_opt(getter: Callable, config_filename: Path, section: str, name: str, **kwargs):
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
        raise exceptions.ConfigurationError(str(e), config_filename) from e
    except ValueError as e:
        # Convert the exception type, and give more context to the error.
        raise exceptions.ConfigurationError(f"Cannot decode '{name}' option '{section}' section: {e}", config_filename) from e


def get_opt(cfg, config_filename: Path, section: str, name: str, **kwargs) -> str:
    """
    Helper function to report errors while extracting string configuration options
    """
    return _get_opt(cfg.get, config_filename, section, name, **kwargs)


def getint_opt(cfg, config_filename: Path, section: str, name: str, **kwargs) -> int:
    """
    Helper function to report errors while extracting int configuration options
    """
    return _get_opt(cfg.getint, config_filename, section, name, **kwargs)


def getfloat_opt(cfg, config_filename: Path, section: str, name: str, **kwargs) -> float:
    """
    Helper function to report errors while extracting floatting point configuration options
    """
    return _get_opt(cfg.getfloat, config_filename, section, name, **kwargs)


def getboolean_opt(cfg, config_filename: Path, section: str, name: str, **kwargs) -> bool:
    """
    Helper function to report errors while extracting boolean configuration options
    """
    return _get_opt(cfg.getboolean, config_filename, section, name, **kwargs)


# Helper functions related to logs
def add_missing(dst: List[str], entry: str):
    """ Add entry to list if not already there """
    if entry not in dst:
        dst.append(entry)


def _init_logger(mode, paths: List[Path]) -> Tuple[Optional[Dict], Optional[Path]]:
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
        config = _load_log_config(cfgpaths[0])
        if verbose:
            # Control the maximum global verbosity level
            config["root"]["level"] = "DEBUG"

            # Control the local console verbosity level
            config["handlers"]["console"]["level"] = "DEBUG"
        if log2files:
            for handler in ["file", "important"]:
                add_missing(config["root"]["handlers"], handler)
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
        return config, cfgpaths[0]
    else:
        # This situation should not happen
        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # os.environ["OTB_LOGGER_LEVEL"]="DEBUG"
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
        return None, None


class _ConfigAccessor:
    """
    Helper class to access options and return high level error messages
    """
    def __init__(
            self, config, configFile: Path
    ) -> None:
        self.__config = config
        self.__config_file = configFile

    @property
    def config_file(self):
        """ Property config_file """
        return self.__config_file

    def has_section(self, section: str) -> bool:
        """ Tells whether the configuration has the requested section """
        return self.__config.has_section(section)

    def throw(self, message: str, e : Optional[BaseException] = None) -> NoReturn:
        """
        Raises a :class:`exceptions.ConfigurationError` filles with everything

        :param e: Optional exception that is the direct cause of the exception raised.
        """
        if e:
            raise exceptions.ConfigurationError(message, self.config_file) from e
        else:
            raise exceptions.ConfigurationError(message, self.config_file)

    def get(self, section: str, name: str, **kwargs) -> str:
        """ Helper function to report errors while extracting string configuration options """
        return get_opt(self.__config, self.config_file, section, name, **kwargs)

    def getint(self, section: str, name: str, **kwargs) -> int:
        """ Helper function to report errors while extracting int configuration options """
        return getint_opt(self.__config, self.config_file, section, name, **kwargs)

    def getfloat(self, section: str, name: str, **kwargs) -> float:
        """ Helper function to report errors while extracting floatting point configuration options """
        return getfloat_opt(self.__config, self.config_file, section, name, **kwargs)

    def getboolean(self, section: str, name: str, **kwargs) -> bool:
        """ Helper function to report errors while extracting boolean configuration options """
        return getboolean_opt(self.__config, self.config_file, section, name, **kwargs)


# The configuration decoding specific to S1Tiling application
class Configuration():  # pylint: disable=too-many-instance-attributes
    """This class handles the parameters from the cfg file"""
    def __init__(
            self, config_file : Union[str, Path], do_show_configuration=True
    ) -> None:
        #: Cache of DEM information covering S2 tiles
        self.__dems_by_s2_tiles : Dict[str, Dict] = {}

        config = configparser.ConfigParser(os.environ)
        config.read(config_file)

        self.__config_file = config_file
        accessor = _ConfigAccessor(config, Path(config_file))

        # Load configuration by topics
        self.__init_log(accessor)
        self.__init_paths(accessor)
        self.__init_data_source(accessor)
        self.__init_mask(accessor)
        self.__init_processing(accessor)
        self.__init_filtering(accessor)
        self.__init_fname_fmt(accessor)
        self.__init_dname_fmt(accessor)
        self.__init_creation_options(accessor)

        # Other options
        #: Type of images handled
        self.type_image         = "GRD"

        # Extra checks
        all_requested = self.tile_list[0] == "ALL"
        if all_requested and self.download and "ALL" in self.roi_by_tiles:
            accessor.throw("Can not request to download 'ROI_by_tiles : ALL' if 'Tiles : ALL'."
                    + " Change either value or deactivate download instead")

        if do_show_configuration:
            self.show_configuration()

    # ----------------------------------------------------------------------
    def __init_log(self, accessor: _ConfigAccessor) -> None:
        # Logs
        #: Logging mode
        self.Mode = accessor.get('Processing', 'mode', fallback=None)
        self.__log_config : Optional[Dict] = None
        if self.Mode is not None:
            self.init_logger(accessor.config_file.parent.absolute())

    # ----------------------------------------------------------------------
    def __init_paths(self, accessor: _ConfigAccessor) -> None:
        #: Destination directory where product will be generated: :ref:`[PATHS.output] <paths.output>`
        self.output_preprocess   = accessor.get('Paths', 'output')
        #: Destination directory where LIA maps products are generated:  :ref:`[PATHS.lia] <paths.lia>`
        self.lia_directory       = accessor.get('Paths', 'lia', fallback=os.path.join(self.output_preprocess, '_LIA'))
        #: Where S1 images are downloaded: See :ref:`[PATHS.s1_images] <paths.s1_images>`!
        self.raw_directory       = accessor.get('Paths', 's1_images')

        # "dem_dir" or Fallback to old deprecated key: "srtm"
        #: Where DEM files are expected to be found: See :ref:`[PATHS.dem_dir] <paths.dem_dir>`!
        self.dem                 = accessor.get('Paths', 'dem_dir', fallback='') or accessor.get('Paths', 'srtm')
        dem_database             = accessor.get('Paths', 'dem_database', fallback='')
        # TODO: Inject resource_dir/'shapefile' if relative dir and not existing
        #: Path to the internal DEM tiles database: automatically set
        self._DEMShapefile       = Path(dem_database or resource_dir / 'shapefile' / 'srtm_tiles.gpkg')
        #: Filename format string to locate the DEM file associated to an *identifier*: See :ref:`[PATHS.dem_format] <paths.dem_format>`
        self.dem_filename_format = accessor.get('Paths', 'dem_format', fallback='{id}.hgt')
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
        self.tmpdir              = accessor.get('Paths', 'tmp')
        if not os.path.isdir(self.tmpdir) and not os.path.isdir(os.path.dirname(self.tmpdir)):
            # Even if tmpdir doesn't exist we should still be able to create it
            accessor.throw(f"tmpdir={self.tmpdir} is not a valid path")
        #: Path to Geoid model. :ref:`[PATHS.geoid_file] <paths.geoid_file>`
        self.GeoidFile           = accessor.get('Paths', 'geoid_file', fallback=str(resource_dir / 'Geoid/egm96.grd'))
        #: Path to directory of temp DEMs
        self.tmp_dem_dir: str    = ""

    # ----------------------------------------------------------------------
    def __init_data_source(self, accessor: _ConfigAccessor) -> None:
        if accessor.has_section('PEPS'):
            accessor.throw(
                    'Since version 0.2, S1Tiling use [DataSource] instead of [PEPS] in config files. Please update your configuration!')
        #: Path to EODAG configuration file: :ref:`[DataSource.eodag_config] <DataSource.eodag_config>`
        self.eodag_config        = accessor.get('DataSource', 'eodag_config', fallback=None) or \
                                   accessor.get('DataSource', 'eodagConfig',  fallback=None)
        #: Boolean flag that enables/disables download of S1 input images: :ref:`[DataSource.download] <DataSource.download>`
        self.download            = accessor.getboolean('DataSource', 'download')
        #: Region Of Interest to download: See :ref:`[DataSource.roi_by_tiles] <DataSource.roi_by_tiles>`
        self.roi_by_tiles        = accessor.get('DataSource', 'roi_by_tiles')
        #: Start date: :ref:`[DataSource.first_date] <DataSource.first_date>`
        self.first_date          = accessor.get('DataSource', 'first_date')
        #: End date: :ref:`[DataSource.last_date] <DataSource.last_date>`
        self.last_date           = accessor.get('DataSource', 'last_date')

        platform_list_str        = accessor.get('DataSource', 'platform_list', fallback='')
        platform_list            = [x for x in SPLIT_PATTERN.split(platform_list_str) if x]
        unsupported_platforms    = [p for p in platform_list if p and not p.startswith("S1")]
        if unsupported_platforms:
            accessor.throw(f"Non supported requested platforms: {', '.join(unsupported_platforms)}")
        #: Filter to restrict platform: See  :ref:`[DataSource.platform_list] <DataSource.platform_list>`
        self.platform_list       = platform_list

        #: Filter to restrict orbit direction: See :ref:`[DataSource.orbit_direction] <DataSource.orbit_direction>`
        self.orbit_direction : Optional[str] = accessor.get('DataSource', 'orbit_direction', fallback=None)
        if self.orbit_direction and self.orbit_direction not in ['ASC', 'DES']:
            accessor.throw("Parameter [orbit_direction] must be either unset or DES, or ASC")
        relative_orbit_list_str  = accessor.get('DataSource', 'relative_orbit_list', fallback='')
        #: Filter to restrict relative orbits: See :ref:`[DataSource.relative_orbit_list] <DataSource.relative_orbit_list>`
        self.relative_orbit_list = [int(o) for o in re.findall(r'\d+', relative_orbit_list_str)]
        #: Filter to polarisation: See :ref:`[DataSource.polarisation] <DataSource.polarisation>`
        self.polarisation        = accessor.get('DataSource', 'polarisation')
        if   self.polarisation   == 'VV-VH':
            self.polarisation    = 'VV VH'
        elif self.polarisation   == 'HH-HV':
            self.polarisation    = 'HH HV'
        elif self.polarisation not in ['VV', 'VH', 'HH', 'HV']:
            accessor.throw("Parameter [polarisation] must be either HH-HV, VV-VH, HH, HV, VV or VH")

        # 0 => no filter
        #: Filter to ensure minimum S2 tile coverage by S1 input products: :ref:`[DataSource.tile_to_product_overlap_ratio] <DataSource.tile_to_product_overlap_ratio>`
        self.tile_to_product_overlap_ratio = accessor.getint('DataSource', 'tile_to_product_overlap_ratio', fallback=0)
        if self.tile_to_product_overlap_ratio > 100:
            accessor.throw("Parameter [tile_to_product_overlap_ratio] must be a percentage in [1, 100]")

        if self.download:
            #: Number of downloads that can be done in parallel: :ref:`[DataSource.nb_parallel_downloads] <DataSource.nb_parallel_downloads>`
            self.nb_download_processes = accessor.getint('DataSource', 'nb_parallel_downloads', fallback=1)

    # ----------------------------------------------------------------------
    def __init_mask(self, accessor: _ConfigAccessor) -> None:
        #: Shall we generate mask products? :ref:`[Mask.generate_border_mask] <Mask.generate_border_mask>`
        self.mask_cond = accessor.getboolean('Mask', 'generate_border_mask')

    # ----------------------------------------------------------------------
    def __init_processing(self, accessor: _ConfigAccessor) -> None:
        #: Tells whether DEM files are copied in a temporary directory, or if symbolic links are to be created. See :ref:`[Processing.cache_dem_by] <Processing.cache_dem_by>`
        self.cache_dem_by         = accessor.get('Processing', 'cache_dem_by', fallback='symlink')
        if self.cache_dem_by not in ['symlink', 'copy']:
            accessor.throw(
                    f"Unexpected value for Processing.cache_dem_by option: '{self.cache_dem_by}' is neither 'copy' nor 'symlink'")

        # - - - - - - - - - -[ Cut margins
        try:
            self.override_azimuth_cut_threshold_to : Optional[bool] = accessor.getboolean(
                    'Processing', 'override_azimuth_cut_threshold_to')
        except Exception:  # pylint: disable=broad-except
            # We cannot use "fallback=None" to handle ": None" w/ getboolean()
            #: Internal to override analysing of top/bottom cutting: See :ref:`[Processing.override_azimuth_cut_threshold_to] <Processing.override_azimuth_cut_threshold_to>`
            self.override_azimuth_cut_threshold_to = None

        # - - - - - - - - - -[ Calibration
        #: SAR Calibration applied: See :ref:`[Processing.calibration] <Processing.calibration>`
        self.calibration_type     = accessor.get('Processing', 'calibration')
        #: Shall we remove thermal noise: :ref:`[Processing.remove_thermal_noise] <Processing.remove_thermal_noise>`
        self.removethermalnoise   = accessor.getboolean('Processing', 'remove_thermal_noise')
        if self.removethermalnoise and otb_version() < '7.4.0':
            raise exceptions.InvalidOTBVersionError(
                f"OTB {otb_version()} does not support noise removal. "
                f"Please upgrade OTB to version 7.4.0 or disable 'remove_thermal_noise' in '{accessor.config_file}'")

        #: Minimal signal value to set after on "denoised" pixels: See :ref:`[Processing.lower_signal_value] <Processing.lower_signal_value>`
        self.lower_signal_value   = accessor.getfloat('Processing', 'lower_signal_value', fallback=1e-7)
        if self.lower_signal_value <= 0:  # TODO test nan, and >= 1e-3 ?
            accessor.throw(
                "'lower_signal_value' parameter shall be a positive (small value) aimed at replacing null value produced by denoising.")

        # - - - - - - - - - -[ Orthorectification
        #: Pixel size (in meters) of the output images: :ref:`[Processing.output_spatial_resolution] <Processing.output_spatial_resolution>`
        self.out_spatial_res      = accessor.getfloat('Processing', 'output_spatial_resolution')

        #: Grid spacing (in meters) for the interpolator in the orthorectification: See :ref:`[Processing.orthorectification_gridspacing] <Processing.orthorectification_gridspacing>`
        self.grid_spacing         = accessor.getfloat('Processing', 'orthorectification_gridspacing')
        #: Orthorectification interpolation methode: See :ref:`[Processing.orthorectification_interpolation_method] <Processing.orthorectification_interpolation_method>`
        self.interpolation_method = accessor.get('Processing', 'orthorectification_interpolation_method', fallback='nn')

        # - - - - - - - - - -[ Tiles
        #: Path to the tiles shape definition. See :ref:`[Processing.tiles_shapefile] <Processing.tiles_shapefile>`
        self.output_grid          = accessor.get('Processing', 'tiles_shapefile', fallback=str(resource_dir / 'shapefile/Features.shp'))
        if not os.path.isfile(self.output_grid):
            accessor.throw(f"output_grid={self.output_grid} is not a valid path")

        # IF tiles_list_in_file is set, use the option, and throw if there is an error
        # ELSE: if unset, then use "tiles" option
        tiles_file = accessor.get('Processing', 'tiles_list_in_file', fallback=None)
        if tiles_file:
            try:
                with open(tiles_file, 'r', encoding='utf-8') as tiles_file_handle:
                    tile_list = tiles_file_handle.readlines()
                self.tile_list: List[str] = [s.rstrip() for s in tile_list]
                logging.info("The following tiles will be processed: %s", self.tile_list)
            except Exception as e:  # pylint: disable=broad-exception-caught
                accessor.throw(f"Cannot read tile list file {tiles_file!r}", e)
        else:
            tiles = accessor.get('Processing', 'tiles')
            #: List of S2 tiles to process: See :ref:`[Processing.tiles] <Processing.tiles>`
            self.tile_list = [s.strip() for s in re.split(r'\s*,\s*', tiles)]

        # - - - - - - - - - -[ Parallelization & RAM
        #: Number of tasks executed in parallel: See :ref:`[Processing.nb_parallel_processes] <Processing.nb_parallel_processes>`
        self.nb_procs             = accessor.getint('Processing', 'nb_parallel_processes')
        #: RAM allocated to OTB applications: See :ref:`[Processing.ram_per_process] <Processing.ram_per_process>`
        self.ram_per_process      = accessor.getint('Processing', 'ram_per_process')
        #: Number of threads allocated to each OTB application: See :ref:`[Processing.nb_otb_threads] <Processing.nb_otb_threads>`
        self.OTBThreads           = accessor.getint('Processing', 'nb_otb_threads')

        # - - - - - - - - - -[ LIA
        #: Tells whether LIA map in degrees * 100 shall be produced alongside the sine map: See :ref:`[Processing.produce_lia_map] <Processing.produce_lia_map>`
        self.produce_lia_map      = accessor.getboolean('Processing', 'produce_lia_map', fallback=False)

        #: Resampling method used by :external:std:doc:`gdalwarp <programs/gdalwarp>` to project DEM on S2 tiles for LIA computation purposes
        resamplings = ['near', 'bilinear', 'cubic', 'cubicspline', 'lanczos', 'average', 'rms', 'mode', 'max', 'min', 'med', 'q1', 'q3', 'qum']
        self.dem_warp_resampling_method = accessor.get('Processing', 'dem_warp_resampling_method', fallback="cubic")
        if self.dem_warp_resampling_method not in resamplings:
            accessor.throw(f"{self.dem_warp_resampling_method} is an invalid choice for `dem_warp_resampling_method`. Choose one among {resamplings}")

        #: no-data value used for various processings
        self.nodatas : Dict[str, Union[int,float,str,None]] = {}
        self.nodatas['SAR'] = accessor.get('Processing', 'nodata.SAR', fallback=0)  # undocumented => best avoided!!!
        self.nodatas['LIA'] = accessor.get('Processing', 'nodata.LIA', fallback=None)

    # ----------------------------------------------------------------------
    def __init_filtering(self, accessor: _ConfigAccessor) -> None:
        #: Despeckle filter to apply, if any: See :ref:`[Filtering.filter] <Filtering.filter>`
        self.filter = accessor.get('Filtering', 'filter', fallback='').lower()
        if self.filter and self.filter == 'none':
            self.filter = ''
        if self.filter:
            #: Shall we keep non-filtered products? See :ref:`[Filtering.keep_non_filtered_products] <Filtering.keep_non_filtered_products>`
            self.keep_non_filtered_products = accessor.getboolean('Filtering', 'keep_non_filtered_products')
            # if generate_border_mask, override this value
            if self.mask_cond:
                logging.warning('As masks are produced, Filtering.keep_non_filtered_products value will be ignored')
                self.keep_non_filtered_products = True

            #: Dictionary of filter options: {'rad': :ref:`[Filtering.window_radius] <Filtering.window_radius>`, 'deramp': :ref:`[Filtering.deramp] <Filtering.deramp>`, 'nblooks': :ref:`[Filtering.nblooks] <Filtering.nblooks>`}
            self.filter_options : Dict = {
                    'rad': accessor.getint('Filtering', 'window_radius')
            }
            if self.filter == 'frost':
                self.filter_options['deramp']  = accessor.getfloat('Filtering', 'deramp')
            elif self.filter in ['lee', 'gammamap', 'kuan']:
                self.filter_options['nblooks'] = accessor.getfloat('Filtering', 'nblooks')
            else:
                accessor.throw(
                    f"Invalid despeckling filter value '{self.filter}'. Select one among none/lee/frost/gammamap/kuan")

    # ----------------------------------------------------------------------
    def __init_fname_fmt(self, accessor: _ConfigAccessor) -> None:
        # Permit to override default file name formats
        fname_fmt_keys = [
                'calibration', 'correct_denoising', 'cut_borders',
                'orthorectification', 'concatenation', 'filtered',
                'dem_on_s2', 'geoid_on_s2', 'height_on_s2', 'ground_and_sat_s2',
                'normals_on_s2', 's1_lia',  's1_sin_lia', 'lia_product', 's2_lia_corrected',
                # Keys to deprecated workflow
                'dem_s1_agglomeration', 's1_on_dem', 'xyz', 'normals_on_s1',
                'lia_orthorectification', 'lia_concatenation',
        ]
        self.fname_fmt = {}
        for key in fname_fmt_keys:
            fmt = accessor.get('Processing', f'fname_fmt.{key}', fallback=None)
            # Default values are defined in associated StepFactories, or below
            if fmt:
                self.fname_fmt[key] = fmt

    # ----------------------------------------------------------------------
    def __init_dname_fmt(self, accessor: _ConfigAccessor) -> None:
        # Permit to override default file name formats
        dname_fmt_keys = [
                'tiled', 'filtered', 'mask',
                's1_lia',  's1_sin_lia', 'lia_product',
        ]
        self.dname_fmt = {}
        for key in dname_fmt_keys:
            fmt = accessor.get('Processing', f'dname_fmt.{key}', fallback=None)
            # Default values are defined below
            if fmt:
                self.dname_fmt[key] = fmt

    # ----------------------------------------------------------------------
    def __init_creation_options(self, accessor: _ConfigAccessor) -> None:
        # Permit to override default file name formats
        creation_options_keys = [
                'tiled', 'filtered', 'mask',
                's1_lia',  's1_sin_lia', 'lia_deg', 'lia_sin',
        ]
        self.creation_options = {}
        for key in creation_options_keys:
            s_cos = accessor.get('Processing', f'creation_options.{key}', fallback=None)
            # Default value is defined in associated StepFactories
            if s_cos:
                l_cos = [x for x in SPLIT_PATTERN.split(s_cos) if x]
                cos = {}
                if l_cos[0] in PIXEL_TYPES:
                    cos['pixel_type'] = l_cos[0]  # OTB_pixel_type
                    cos['gdal_options'] = l_cos[1:]
                else:
                    cos['gdal_options'] = l_cos[0:]
                for co in cos['gdal_options']:
                    KEY_PATTERN = re.compile(r'[A-Z]+=')
                    if not KEY_PATTERN.match(co):
                        # The only validation used is UPPERCASE=value
                        # We don't check against a list that may change over time. In that case the error will be caught later.
                        accessor.throw(f"{co} is not a valid GDAL creation option for {key}. Expected syntax is `<OPTIONNAME>=<value>`")

                self.creation_options[key] = cos

    # ----------------------------------------------------------------------
    def show_configuration(self) -> None:  # pylint: disable=too-many-statements
        """
        Displays the configuration
        """
        logging.info("Running S1Tiling %s with:", s1tiling_version)
        logging.info("From request file: %s", self.__config_file or "(some string)")
        logging.info("[Paths]")
        logging.info("- geoid_file                       : %s",     self.GeoidFile)
        logging.info("- s1_images                        : %s",     self.raw_directory)
        logging.info("- output                           : %s",     self.output_preprocess)
        logging.info("- LIA                              : %s",     self.lia_directory)
        logging.info("- dem directory                    : %s",     self.dem)
        logging.info("- dem filename format              : %s",     self.dem_filename_format)
        logging.info("- dem field ids (from shapefile)   : %s",     self.dem_field_ids)
        logging.info("- main ID for DEM names deduced    : %s",     self.dem_main_field_id)
        logging.info("- tmp                              : %s",     self.tmpdir)
        logging.info("[DataSource]")
        logging.info("- download                         : %s",     self.download)
        logging.info("- first_date                       : %s",     self.first_date)
        logging.info("- last_date                        : %s",     self.last_date)
        logging.info("- platform_list                    : %s",     self.platform_list)
        logging.info("- polarisation                     : %s",     self.polarisation)
        logging.info("- orbit_direction                  : %s",     self.orbit_direction)
        logging.info("- relative_orbit_list              : %s",     self.relative_orbit_list)
        logging.info("- tile_to_product_overlap_ratio    : %s%%",   self.tile_to_product_overlap_ratio)
        logging.info("- roi_by_tiles                     : %s",     self.roi_by_tiles)
        if self.download:
            logging.info("- nb_parallel_downloads            : %s", self.nb_download_processes)
        logging.info("[Processing]")
        logging.info("- calibration                      : %s",     self.calibration_type)
        logging.info("- mode                             : %s",     self.Mode)
        logging.info("- nb_otb_threads                   : %s",     self.OTBThreads)
        logging.info("- nb_parallel_processes            : %s",     self.nb_procs)
        logging.info("- orthorectification interpolation : %s",     self.interpolation_method)
        logging.info("- orthorectification_gridspacing   : %s",     self.grid_spacing)
        logging.info("- output_spatial_resolution        : %s",     self.out_spatial_res)
        logging.info("- ram_per_process                  : %s",     self.ram_per_process)
        logging.info("- remove_thermal_noise             : %s",     self.removethermalnoise)
        logging.info("- dem_shapefile                    : %s",     self._DEMShapefile)
        logging.info("- tiles                            : %s",     self.tile_list)
        logging.info("- tiles_shapefile                  : %s",     self.output_grid)
        logging.info("- produce LIA° map                 : %s",     self.produce_lia_map)
        logging.info("- warping method for DEM on S2     : %s",     self.dem_warp_resampling_method)
        logging.info("- superimpose interpol Geoid on S2 : %s",     self.interpolation_method)
        logging.info("[Mask]")
        logging.info("- generate_border_mask             : %s",     self.mask_cond)
        logging.info("[Filter]")
        logging.info("- Speckle filtering method         : %s",     self.filter or "none")
        if self.filter:
            logging.info("- Keeping previous products        : %s", self.keep_non_filtered_products)
            logging.info("- Window radius                    : %s", self.filter_options['rad'])
            if   self.filter in ['lee', 'gammamap', 'kuan']:
                logging.info("- nblooks                          : %s", self.filter_options['nblooks'])
            elif self.filter in ['frost']:
                logging.info("- deramp                           : %s", self.filter_options['deramp'])

        logging.info('Output directories:')
        for k, fmt in self.dname_fmt.items():
            logging.info(' - %s --> %s', k, fmt)
        logging.info('Filename formats:')
        for k, fmt in self.fname_fmt.items():
            logging.info(' - %s --> %s', k, fmt)
        logging.info('Creation options:')
        for k, co in self.creation_options.items():
            logging.info(' - %s --> %s', k, co)

    def init_logger(self, config_log_dir: Path, mode=None) -> None:
        """
        Deported logger initialization function for projects that use their own
        logger, and S1Tiling through its API only.

        :param mode: Option to override logging mode, if not found/expected in
                     the configuration file.
        """
        if not self.__log_config:
            # pylint: disable=attribute-defined-outside-init
            self.__log_config, config_file = _init_logger(self.Mode, [config_log_dir])
            self.Mode = mode or self.Mode
            assert self.Mode, "Please set a valid logging mode!"
            logging.debug("S1 tiling configuration initialized from '%s'", config_file)
            # self.log_queue = multiprocessing.Queue()
            # self.log_queue_listener = logging.handlers.QueueListener(self.log_queue)
            if "debug" in self.Mode and self.__log_config and self.__log_config['loggers']['s1tiling.OTB']['level'] == 'DEBUG':
                # OTB DEBUG logs are displayed iff level is DEBUG and configFile mode
                # is debug as well.
                os.environ["OTB_LOGGER_LEVEL"] = "DEBUG"

    # ======================================================================
    @property
    def log_config(self):
        """
        Property log
        """
        assert self.__log_config, "Please initialize logger"
        return self.__log_config

    @property
    def dem_db_filepath(self) -> str:
        """
        Get the DEMShapefile databe filepath
        """
        return str(self._DEMShapefile)

    # ======================================================================
    # Things stored for later use
    def register_dems_related_to_S2_tiles(self, dems_by_s2_tiles: Dict[str, Dict]) -> None:
        """
        Workaround that helps caching DEM related information for later use.
        """
        self.__dems_by_s2_tiles = dems_by_s2_tiles

    def get_dems_covering_s2_tile(self, tile_name: str) -> Dict:
        """
        Retrieve the DEM associated to the specified S2 tile.
        """
        if tile_name not in self.__dems_by_s2_tiles:
            raise AssertionError(f"No DEM information has been associated to {tile_name}. Only the following tiles have known information: {self.__dems_by_s2_tiles.keys()}")
        return self.__dems_by_s2_tiles[tile_name]


def fname_fmt_concatenation(cfg: Configuration) -> str:
    """
    Helper function that returns the ``Processing.fnmatch.concatenation`` actual
    value, or its default value according to the calibration kind.
    """
    calibration_is_done_in_S1 = cfg.calibration_type in ['sigma', 'beta', 'gamma', 'dn']
    if calibration_is_done_in_S1:
        # logger.debug('Concatenation in legacy mode: fname_fmt without "_%s"', cfg.calibration_type)
        # Legacy mode: the default final filename won't contain the calibration_type
        fname_fmt = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}.tif'
    else:
        # logger.debug('Concatenation in NORMLIM mode: fname_fmt with "_beta" for %s', cfg.calibration_type)
        # Let the default force the "beta" calibration_type in the filename
        fname_fmt = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_{calibration_type}.tif'
    fname_fmt = cfg.fname_fmt.get('concatenation', fname_fmt)
    return fname_fmt


def fname_fmt_filtered(cfg: Configuration) -> str:
    """
    Helper function that returns the ``Processing.fnmatch.filtered`` actual value,
    or its default value according to the calibration kind.
    """
    calibration_is_done_in_S1 = cfg.calibration_type in ['sigma', 'beta', 'gamma', 'dn']
    if calibration_is_done_in_S1:
        # logger.debug('Concatenation in legacy mode: fname_fmt without "_%s"', cfg.calibration_type)
        # Legacy mode: the default final filename won't contain the calibration_type
        fname_fmt = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_filtered.tif'
    else:
        # logger.debug('Concatenation in NORMLIM mode: fname_fmt with "_beta" for %s', cfg.calibration_type)
        # Let the default force the "beta" calibration_type in the filename
        fname_fmt = '{flying_unit_code}_{tile_name}_{polarisation}_{orbit_direction}_{orbit}_{acquisition_stamp}_{calibration_type}_filtered.tif'
    fname_fmt = cfg.fname_fmt.get('filtered', fname_fmt)
    return fname_fmt


def dname_fmt_tiled(cfg: Configuration) -> str:
    """
    Helper function that returns the ``Processing.dname.tiled`` actual
    value, or its default value.
    """
    return cfg.dname_fmt.get('tiled', '{out_dir}/{tile_name}')


def dname_fmt_mask(cfg: Configuration) -> str:
    """
    Helper function that returns the ``Processing.dname.mask`` actual value,
    or its default value.
    """
    return cfg.dname_fmt.get('mask', '{out_dir}/{tile_name}')


def dname_fmt_filtered(cfg: Configuration) -> str:
    """
    Helper function that returns the ``Processing.dname.filtered`` actual value,
    or its default value.
    """
    return cfg.dname_fmt.get('filtered', '{out_dir}/filtered/{tile_name}')


def dname_fmt_lia_product(cfg: Configuration) -> str:
    """
    Helper function that returns the ``Processing.dname.lia_product`` actual value,
    or its default value.
    """
    return cfg.dname_fmt.get('lia_product', '{lia_dir}')


def pixel_type(cfg: Configuration, product: str, default: Optional[str] = None):  # -> PixelType:
    """
    Helper function that returns the chosen pixel type in the configuration.
    """
    cos = cfg.creation_options.get(product, {})
    assert (not default) or default in PIXEL_TYPES, f"Invalid default pixel_type {default!r} for {product!r}"
    return PIXEL_TYPES.get(cos.get('pixel_type', default), None)


def _extended_filename(cfg: Configuration, product: str, default: List[str]) -> str:
    """
    Internal helper function that returns GDAL creation options through
    :external:std:doc:`OTB Extended Filename <ExtendedFilenames>`.

    This function takes care of fetching the right information and of
    reformatting it as an `extended filename option`.
    """
    cos = cfg.creation_options.get(product, {})
    gdal_options = cos.get('gdal_options', default)
    return '?' + ''.join([f"&gdal:co:{kv}" for kv in gdal_options])


def extended_filename_tiled(cfg: Configuration) -> str:
    """
    Helper function that returns GDAL creation options through
    :external:std:doc:`OTB Extended Filename <ExtendedFilenames>` for S2 tiled
    products.
    """
    return _extended_filename(cfg, 'tiled', ['COMPRESS=DEFLATE', 'PREDICTOR=3'])


def extended_filename_filtered(cfg: Configuration) -> str:
    """
    Helper function that returns GDAL creation options through
    :external:std:doc:`OTB Extended Filename <ExtendedFilenames>` for filetered
    products.
    """
    return _extended_filename(cfg, 'filtered', ['COMPRESS=DEFLATE', 'PREDICTOR=3'])


def extended_filename_mask(cfg: Configuration) -> str:
    """
    Helper function that returns GDAL creation options through
    :external:std:doc:`OTB Extended Filename <ExtendedFilenames>` for masks.
    """
    return _extended_filename(cfg, 'mask', ['COMPRESS=DEFLATE'])


def extended_filename_lia_degree(cfg: Configuration) -> str:
    """
    Helper function that returns GDAL creation options through
    :external:std:doc:`OTB Extended Filename <ExtendedFilenames>` for LIA
    in degrees (*100) products.
    """
    return _extended_filename(cfg, 'filtered', ['COMPRESS=DEFLATE'])


def extended_filename_lia_sin(cfg: Configuration) -> str:
    """
    Helper function that returns GDAL creation options through
    :external:std:doc:`OTB Extended Filename <ExtendedFilenames>` for sin(LIA)
    products.
    """
    return _extended_filename(cfg, 'filtered', ['COMPRESS=DEFLATE', 'PREDICTOR=3'])


def _get_nodata(dict: Dict[str, Optional[Union[str,int,float]]], key: str, default_value: Union[str,int,float]):
    """
    Internal helper to extract nodata value from configuration directionaries.

    :return: if the key exists in the dict, return its value if not None.
    :return: ``default_value`` otherwise

    >>> _get_nodata({'LIA': None, 'SAR': 0, 'DEM': -32768}, 'LIA', 42)
    42
    >>> _get_nodata({'LIA': None, 'SAR': 0, 'DEM': -32768}, 'SAR', 42)
    0
    >>> _get_nodata({'LIA': None, 'SAR': 0, 'DEM': -32768}, 'DEM', 42)
    -32768
    >>> _get_nodata({'LIA': None, 'SAR': 0, 'DEM': -32768}, 'H2G2', 42)
    42
    """
    v = dict.get(key, None)
    return v if v is not None else default_value


def nodata_SAR(cfg: Configuration) -> Union[str, int, float]:
    """
    Helper function that returns typical nodata value used in Sentinel-1 raw
    products and in S1Tiling SAR products.
    """
    return _get_nodata(cfg.nodatas, 'SAR', 0)


def nodata_LIA(cfg: Configuration) -> Union[str, int, float]:
    """
    Helper function that returns typical nodata value used in intermediary
    images generated for LIA normlim correction.
    """
    return _get_nodata(cfg.nodatas, 'LIA', 'nan')


def nodata_DEM(cfg: Configuration) -> Union[str, int, float]:
    """
    Helper function that returns typical nodata value used in intermediary
    DEM images generated for LIA normlim correction.
    """
    return _get_nodata(cfg.nodatas, 'DEM', -32768)


def nodata_XYZ(cfg: Configuration) -> Union[str, int, float]:
    """
    Helper function that returns typical nodata value used in intermediary
    XYZ images generated for LIA normlim correction.
    """
    return _get_nodata(cfg.nodatas, 'XYZ', 'nan')
