#!/usr/bin/env python
#-*- coding: utf-8 -*-
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
#
# =========================================================================

import configparser
import logging
import logging.handlers
import pathlib
import os
import multiprocessing

def init_logger(mode, paths):
    import logging.config
    import yaml
    # Add the dirname where the current script is
    paths += [pathlib.Path(__file__).parent.parent.absolute()]
    paths = [p/'logging.conf.yaml' for p in paths]
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
            if not 'file' in config["root"]["handlers"]:
                config["root"]["handlers"] += ['file']
            if not 'important' in config["root"]["handlers"]:
                config["root"]["handlers"] += ['important']
        logging.config.dictConfig(config)
    else:
        # This situation should not happen
        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            os.environ["OTB_LOGGER_LEVEL"]="DEBUG"
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    queue = multiprocessing.Queue()
    queue_listener = logging.handlers.QueueListener(queue)
    return queue, queue_listener


class Configuration():
    """This class handles the parameters from the cfg file"""
    def __init__(self,configFile):
        config = configparser.ConfigParser(os.environ)
        config.read(configFile)

        # Logs
        self.Mode=config.get('Processing','Mode')
        self.log_queue, self.log_queue_listener = init_logger(self.Mode, [pathlib.Path(configFile).parent.absolute()])
        if "debug" in self.Mode:
            os.environ["OTB_LOGGER_LEVEL"]="DEBUG"
            pass
        ##self.stdoutfile = open("/dev/null", 'w')
        ##self.stderrfile = open("S1ProcessorErr.log", 'a')
        ##if "debug" in self.Mode:
        ##    self.stdoutfile = None
        ##    self.stderrfile = None
        ##    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ##    os.environ["OTB_LOGGER_LEVEL"]="DEBUG"
        ##else:
        ##    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

        ##if "logging" in self.Mode:
        ##    self.stdoutfile = open("S1ProcessorOut.log", 'a')
        ##    self.stderrfile = open("S1ProcessorErr.log", 'a')

        # Other options
        self.region           = config.get('DEFAULT','region')
        self.output_preprocess= config.get('Paths','Output')
        self.raw_directory    = config.get('Paths','S1Images')
        self.srtm             = config.get('Paths','SRTM')
        self.tmpdir           = config.get('Paths', 'tmp')
        if not os.path.exists(self.tmpdir):
            logging.critical("ERROR: tmpdir=%s is not a valid path", self.tmpdir)
            exit(1)
        self.GeoidFile         = config.get('Paths','GeoidFile')
        self.pepsdownload      = config.getboolean('PEPS','Download')
        self.ROI_by_tiles      = config.get('PEPS','ROI_by_tiles')
        self.first_date        = config.get('PEPS','first_date')
        self.last_date         = config.get('PEPS','last_date')
        self.polarisation      = config.get('PEPS','Polarisation')
        self.type_image        = "GRD"
        self.mask_cond         = config.getboolean('Mask','Generate_border_mask')
        self.calibration_type  = config.get('Processing','Calibration')
        self.removethermalnoise= config.getboolean('Processing','Remove_thermal_noise')

        self.out_spatial_res   = config.getfloat('Processing','OutputSpatialResolution')

        self.output_grid       = config.get('Processing','TilesShapefile')
        if not os.path.exists(self.output_grid):
            logging.critical("ERROR: output_grid=%s is not a valid path", self.output_grid)
            exit(1)

        self.SRTMShapefile=config.get('Processing','SRTMShapefile')
        if not os.path.exists(self.SRTMShapefile):
            logging.critical("ERROR: srtm_shapefile=%s is not a valid path", self.srtm_shapefile)
            exit(1)
        self.grid_spacing=config.getfloat('Processing','Orthorectification_gridspacing')
        self.border_threshold=config.getfloat('Processing','BorderThreshold')
        try:
           tiles_file=config.get('Processing','TilesListInFile')
           self.tiles_list=open(tiles_file,'r').readlines()
           self.tiles_list = [s.rstrip() for s in self.tiles_list]
           logging.info("The following tiles will be processed: %s", self.tiles_list)
        except:
           tiles=config.get('Processing','Tiles')
           self.tiles_list = [s.strip() for s in tiles.split(", ")]

        self.TileToProductOverlapRatio= config.getfloat('Processing','TileToProductOverlapRatio')
        self.nb_procs                 = config.getint('Processing','NbParallelProcesses')
        self.ram_per_process          = config.getint('Processing','RAMPerProcess')
        self.OTBThreads               = config.getint('Processing','OTBNbThreads')
        self.filtering_activated      = config.getboolean('Filtering','Filtering_activated')
        self.Reset_outcore            = config.getboolean('Filtering','Reset_outcore')
        self.Window_radius            = config.getint('Filtering','Window_radius')

        self.cluster                  = config.getboolean('HPC-Cluster','Parallelize_tiles')

        def check_date (self):
            import datetime
            import sys

            fd=self.first_date
            ld=self.last_date

            try:
                F_Date = datetime.date(int(fd[0:4]),int(fd[5:7]),int(fd[8:10]))
                L_Date = datetime.date(int(ld[0:4]),int(ld[5:7]),int(ld[8:10]))
            except:
                logging.critical("Invalid date")
                sys.exit()

