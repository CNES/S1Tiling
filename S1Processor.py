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
"""
This module contains a script to build temporal series of S1 images by tiles
It performs the following steps:
  1- Download S1 images from PEPS server
  2- Calibrate the S1 images to gamma0
  3- Orthorectify S1 images and cut their on geometric tiles
  4- Concatenante images from the same orbit on the same tile
  5- Build mask files
  6- Filter images by using a multiimage filter

 Parameters have to be set by the user in the S1Processor.cfg file
"""

import os
import pathlib
import sys
import glob
import shutil
import numpy as np
from PIL import Image
from subprocess import Popen
import multiprocessing
from s1tiling import S1FileManager
from s1tiling import S1FilteringProcessor
from s1tiling import Utils
import configparser
import gdal, rasterio
from rasterio.windows import Window
import subprocess
import datetime
import logging
import logging.handlers
from contextlib import redirect_stdout
import otbApplication as otb

def execute(cmd):
    try:
        logging.debug(cmd)
        logger = logging.getLogger('root')
        logger.write = lambda msg: logger.info(msg) if msg != '\n' else None
        with redirect_stdout(logger):
            subprocess.check_call(cmd, shell = True)
    except subprocess.CalledProcessError as e:
        logging.warning('WARNING : Erreur dans la commande : ')
        logging.warning(e.returncode)
        logging.warning(e.cmd)
        logging.warning(e.output)

def worker_config(q):
    """
    Worker configuration function called by Pool().

    It takes care of initializing the queue handler in the subprocess.

    Params:
        :q: multiprocessing.Queue used for passing logging messages from worker to main process.
    """
    qh = logging.handlers.QueueHandler(q)
    logger = logging.getLogger()
    logger.addHandler(qh)

def execute_command(params):
    """
    Main worker function executed by Sentinel1PreProcess.run_processing()

    Params:
        :p: list of two subparameters: job title + command to execute
    """
    title, cmd = params
    logging.debug('%s # Starting %s', title, cmd)
    proc = Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    log, _ = proc.communicate()
    level = logging.ERROR if proc.returncode else logging.DEBUG
    for line in log.decode().split('\n'):
        logging.log(level, line)
    return title

def remove_files(files):
    """
    Removes the files from the disk
    """
    logging.debug("Remove %s", files)
    for file_it in files:
        if os.path.exists(file_it):
            os.remove(file_it)

class Configuration():
    """This class handles the parameters from the cfg file"""
    def __init__(self,configFile):
        config = configparser.ConfigParser(os.environ)
        config.read(configFile)

        # Logs
        self.Mode=config.get('Processing','Mode')
        self.log_queue, self.log_queue_listener = init_logger(self.Mode, [pathlib.Path(configFile).parent.absolute()])
        if "debug" in self.Mode:
            # os.environ["OTB_LOGGER_LEVEL"]="DEBUG"
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
        self.region=config.get('DEFAULT','region')
        self.output_preprocess=config.get('Paths','Output')
        self.raw_directory=config.get('Paths','S1Images')
        self.srtm=config.get('Paths','SRTM')
        self.tmpdir = config.get('Paths', 'tmp')
        if not os.path.exists(self.tmpdir):
            logging.critical("ERROR: tmpdir=%s is not a valid path", self.tmpdir)
            exit(1)
        self.GeoidFile=config.get('Paths','GeoidFile')
        self.pepsdownload=config.getboolean('PEPS','Download')
        self.ROI_by_tiles=config.get('PEPS','ROI_by_tiles')
        self.first_date=config.get('PEPS','first_date')
        self.last_date=config.get('PEPS','last_date')
        self.polarisation=config.get('PEPS','Polarisation')
        self.type_image="GRD"
        self.mask_cond=config.getboolean('Mask','Generate_border_mask')
        self.calibration_type=config.get('Processing','Calibration')
        self.removethermalnoise=config.getboolean('Processing','Remove_thermal_noise')

        self.out_spatial_res=config.getfloat('Processing','OutputSpatialResolution')

        self.output_grid=config.get('Processing','TilesShapefile')
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

        self.TileToProductOverlapRatio=config.getfloat('Processing','TileToProductOverlapRatio')
        self.nb_procs=config.getint('Processing','NbParallelProcesses')
        self.ram_per_process=config.getint('Processing','RAMPerProcess')
        self.OTBThreads=config.getint('Processing','OTBNbThreads')
        self.filtering_activated=config.getboolean('Filtering','Filtering_activated')
        self.Reset_outcore=config.getboolean('Filtering','Reset_outcore')
        self.Window_radius=config.getint('Filtering','Window_radius')

        self.cluster=config.getboolean('HPC-Cluster','Parallelize_tiles')

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

class Pipeline(object):
    def __init__(self, do_measure):
        self.__pipeline = []
        self.__do_measure = do_measure
    def push(self, name, parameters):
        self.__pipeline += [ {'appname': name, 'parameters': parameters}]
    def do_execute(self):
        assert(self.__pipeline) # shall not be empty!
        app_names = []
        last_app = None
        for crt in self.__pipeline:
            app_names += [crt['appname']]
            app = otb.Registry.CreateApplication(crt['appname'])
            assert(app)
            crt['app'] = app
            app.SetParameters(crt['parameters'])
            if last_app:
                app.ConnectImage('in', last_app, 'out')
                in_memory = True
                app.PropagateConnectMode(in_memory)
            last_app = app

        pipeline_name = '|'.join(app_names)
        with Utils.ExecutionTimer('-> '+pipeline_name, self.__do_measure) as t:
            assert(last_app)
            last_app.ExecuteAndWriteOutput()
        for crt in self.__pipeline:
            del(crt['app']) # Make sure to release application memory
            crt['app'] = None
        return pipeline_name + ' > ' + crt['parameters']['out']

def execute1(pipeline):
    return pipeline.do_execute()

class PoolOfOTBExecutions(object):
    def __init__(self, title, do_measure, nb_procs, nb_threads, log_queue, log_queue_listener):
        """
        constructor
        """
        self.__pool = []
        self.__title               = title
        self.__do_measure          = do_measure
        self.__nb_procs            = nb_procs
        self.__nb_threads          = nb_threads
        self.__log_queue           = log_queue
        self.__log_queue_listener  = log_queue_listener

    def new_pipeline(self):
        pipeline = Pipeline(self.__do_measure)
        self.__pool += [pipeline]
        return pipeline

    def process(self):

        import time
        nb_cmd = len(self.__pool)

        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(self.__nb_threads)
        with multiprocessing.Pool(self.__nb_procs, worker_config, [self.__log_queue]) as pool:
            self.__log_queue_listener.start()
            for count, result in enumerate(pool.imap_unordered(execute1, self.__pool), 1):
                logging.info("%s correctly finished", result)
                logging.info(' --> %s... %s%%', self.__title, count*100./nb_cmd)

            pool.close()
            pool.join()
            self.__log_queue_listener.stop()


def needToBeCrop(image_name, thr):
    """
    Deprecated in favor of has_to_many_NoData
    """
    imgpil = Image.open(image_name)
    ima = np.asarray(imgpil)
    nbNan = len(np.argwhere(ima==0))
    return nbNan>thr


def has_to_many_NoData(image, threshold, nodata):
    """
    Analyses whether an image contains NO DATA.

        :param image:     np.array image to analyse
        :param threshold: number of NoData searched
        :param nodata:    no data value
        :return:          whether the number of no-data pixel > threshold
    """
    nbNoData = len(np.argwhere(image==nodata))
    return nbNoData>threshold

class Sentinel1PreProcess():
    """ This class handles the processing for Sentinel1 ortho-rectification """
    def __init__(self,cfg):
        try:
            os.remove("S1ProcessorErr.log.log")
            os.remove("S1ProcessorOut.log")
        except os.error:
            pass
        self.cfg=cfg

    def generate_border_mask(self, all_ortho):
                """
                This method generate the border mask files from the
                orthorectified images.

                Args:
                  all_ortho: A list of ortho-rectified S1 images
                  """
                cmd_bandmath = []
                cmd_morpho = []
                files_to_remove = []
                logging.info("Generate Mask ...")
                for current_ortho in all_ortho:
                    if "vv" not in current_ortho:
                        continue
                    working_directory, basename = os.path.split(current_ortho)
                    name_border_mask            = basename.replace(".tif", "_BorderMask.tif")
                    name_border_mask_tmp        = basename.replace(".tif", "_BorderMask_TMP.tif")
                    pathname_border_mask_tmp    = os.path.join(working_directory, name_border_mask_tmp)

                    files_to_remove.append(pathname_border_mask_tmp)
                    cmd_bandmath.append(['    Mask building of '+name_border_mask_tmp,
                        'export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={};'.format(self.cfg.OTBThreads)+'otbcli_BandMath -ram '\
                        +str(self.cfg.ram_per_process)\
                        +' -il '+current_ortho\
                        +' -out '+pathname_border_mask_tmp\
                        +' uint8 -exp "im1b1==0?0:1"'])

                    #due to threshold approximation

                    cmd_morpho.append(['    Mask smoothing of '+name_border_mask,
                        'export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={};'.format(self.cfg.OTBThreads)+"otbcli_BinaryMorphologicalOperation -ram "\
                        +str(self.cfg.ram_per_process)+" -progress false -in "\
                        +pathname_border_mask_tmp\
                        +" -out "\
                        +os.path.join(working_directory, name_border_mask)\
                        +" uint8 -structype ball"\
                        +" -structype.ball.xradius 5"\
                        +" -structype.ball.yradius 5 -filter opening"])

                self.run_processing(cmd_bandmath, title="   Mask building")
                self.run_processing(cmd_morpho,   title="   Mask smoothing")
                remove_files(files_to_remove)
                logging.info("Generate Mask done")

    def do_calibrate_and_cut(self, raw_raster):
        """
        This method:
        - performs radiometric calibration of raw S1 images,
        - and remove pixels on the image borders.
        Args:
          :raw_raster: list of raw S1 raster file to calibrate
        """
        cut_overlap_range = 1000 # Number of columns to cut on the sides. Here 500pixels = 5km
        cut_overlap_azimuth = 1600 # Number of lines to cut at top or bottom
        thr_nan_for_cropping = cut_overlap_range*2 #Quand on fait les tests, on a pas encore couper les nan sur le cote, d'ou l'utilisatoin de ce thr

        logging.info("Calibrate and Cut %s", raw_raster)

        pool = PoolOfOTBExecutions('calibrate & cut', True, self.cfg.nb_procs, self.cfg.OTBThreads, self.cfg.log_queue, self.cfg.log_queue_listener)

        for i in range(len(raw_raster)):
            images = raw_raster[i][0].get_images_list()

            # First we check whether the first and/or the last lines need to be cropped (set to 0 actually)
            # TODO: no need to test this if none of the images need to be cut...
            with rasterio.open(images[0]) as ds_reader:
                xsize = ds_reader.width
                ysize = ds_reader.height
                north = ds_reader.read(1, window=Window(0, 100, xsize+1, 1))
                south = ds_reader.read(1, window=Window(0, ysize-100, xsize+1, 1))

            crop1 = has_to_many_NoData(north, thr_nan_for_cropping, 0)
            crop2 = has_to_many_NoData(south, thr_nan_for_cropping, 0)
            logging.debug("   => need to crop north: %s", crop1)
            logging.debug("   => need to crop south: %s", crop2)

            # Then we process of the images in the list
            for image in images:
                logging.debug(" - %s", image)
                # TODO: makes sure it goes into the temp directory...
                image_out = image.replace(".tiff", "_OrthoReady.tiff")
                # First, skip the step if the associated OrthoReady image already exists
                if os.path.exists(image_out):
                    logging.info('   %s already exists => calibration|cut skipped', image_out)
                    continue

                pipeline = pool.new_pipeline()
                # Parameters for the calibration application
                params_calibration = {
                        'ram'     : str(self.cfg.ram_per_process),
                        # 'progress': 'false',
                        'in'      : image,
                        'lut'     : self.cfg.calibration_type,
                        'noise'   : str(self.cfg.removethermalnoise).lower()
                        }
                pipeline.push('SARCalibration', params_calibration)
                # Parameters for the cutting application
                params_cut = {
                        'ram'              : str(self.cfg.ram_per_process),
                        # 'progress'         : False,
                        'out'              : image_out,
                        'threshold.x'      : cut_overlap_range,
                        'threshold.y.start': cut_overlap_azimuth if crop1 else 0,
                        'threshold.y.end'  : cut_overlap_azimuth if crop2 else 0,
                        }
                pipeline.push('ClampROI', params_cut)

        logging.debug('Launch pipelines')
        pool.process()

    def cut_image_cmd(self, raw_raster):
        """
        This method removes pixels on the image borders.
        Args:
          raw_raster: list of raw S1 raster file to calibrate
        """
        logging.info("Cutting %s", raw_raster)

        for i in range(len(raw_raster)):
            logging.debug("    Cutting:"+str(int(float(i)/len(raw_raster)*100.))+"%")
            # Check if all OrthoReady files have been already generated
            images = raw_raster[i][0].get_images_list()
            completed=True
            for image in images:
                completed = completed and os.path.exists(image.replace(".tiff","_OrthoReady.tiff"))
            if completed:
                logging.debug('    Cutting step not required => abort')
                continue

            image=images[0]
            image = image.replace(".tiff","_calOk.tiff")
            image_ok = image.replace("_calOk.tiff", "_OrthoReady.tiff")
            ##image_mask=image.replace("_calOk.tiff","_mask.tiff")
            im1_name = image.replace(".tiff","test_nord.tiff")
            im2_name = image.replace(".tiff","test_sud.tiff")
            raster = gdal.Open(image)
            xsize = raster.RasterXSize
            ysize = raster.RasterYSize
            ## npmask= np.ones((ysize,xsize), dtype=bool)

            cut_overlap_range = 1000 # Nombre de pixel a couper sur les cotes. ici 500 = 5km
            cut_overlap_azimuth = 1600 # Nombre de pixels a couper sur le haut ou le bas
            thr_nan_for_cropping = cut_overlap_range*2 #Quand on fait les tests, on a pas encore couper les nan sur le cote, d'ou l'utilisatoin de ce thr

            execute('gdal_translate -srcwin 0 100 '+str(xsize)+' 1 '+image+' '+im1_name)
            execute('gdal_translate -srcwin 0 '+str(ysize-100)+' '+str(xsize)+' 1 '+image+' '+im2_name)

            crop1 = needToBeCrop(im1_name, thr_nan_for_cropping)
            crop2 = needToBeCrop(im2_name, thr_nan_for_cropping)

            ##npmask[:,0:cut_overlap_range]=0 # Coupe W
            ##npmask[:,(xsize-cut_overlap_range):]=0 # Coupe E
            ##if crop1 : npmask[0:cut_overlap_azimuth,:]=0 # Coupe N
            ##if crop2 : npmask[ysize-cut_overlap_azimuth:,:]=0 # Coupe S

            ##driver = gdal.GetDriverByName("GTiff")
            ##outdata = driver.Create(image_mask, xsize, ysize, 1, gdal.GDT_Byte)
            ##outdata.SetGeoTransform(raster.GetGeoTransform())##sets same geotransform as input
            ##outdata.SetProjection(raster.GetProjection())##sets same projection as input
            ##outdata.GetRasterBand(1).WriteArray(npmask)
            ##outdata.SetGCPs(raster.GetGCPs(),raster.GetGCPProjection())
            ##outdata.FlushCache() ##saves to disk!!
            ##outdata = None

            ##files_to_remove = [image_mask, im1_name, im2_name]
            files_to_remove = [im1_name, im2_name]
            all_cmd=[]
            for image in images:
                im_calok = image.replace(".tiff", "_calOk.tiff")
                im_ortho = image.replace(".tiff", "_OrthoReady.tiff")
                ##cmd='export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={};'.format(self.cfg.OTBThreads)+'otbcli_BandMath ' \
                ##               +"-ram "+str(self.cfg.ram_per_process)\
                ##               +" -progress false " \
                ##               +'-il {} {} -out {} -exp "im1b1*im2b1"'.format(im_calok, image_mask,im_ortho)
                cmd='export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={};'.format(self.cfg.OTBThreads)+'otbcli_ClampROI' \
                               +" -ram "+str(self.cfg.ram_per_process)\
                               +(" -threshold.x {}" \
                               +" -threshold.y.start {}" \
                               +" -threshold.y.end {}" \
                               +" -progress false" \
                               +' -in {} -out {}').format(cut_overlap_range,
                                      cut_overlap_azimuth if crop1 else 0,
                                      cut_overlap_azimuth if crop2 else 0,
                                      im_calok, im_ortho)
                all_cmd.append(['    Cutting of '+im_ortho, cmd])
                files_to_remove += [image, im_calok]

            self.run_processing(all_cmd, title="   Cutting: Apply mask")

            # TODO: add geoms
            remove_files(files_to_remove)

        logging.info("Cutting done ")

    def do_calibration_cmd(self, raw_raster):
        """
        This method performs radiometric calibration of raw S1 images.

        Args:
          raw_raster: list of raw S1 raster file to calibrate
        """
        files_to_remove = []
        all_cmd = []
        logging.info("Calibration %s",raw_raster)

        for i in range(len(raw_raster)):

            # Check if all OrthoReady files have been already generated
            images = raw_raster[i][0].get_images_list()
            completed=True
            for image in images:
                logging.debug('* Check calibration for %s', image)
                completed = completed and os.path.exists(image.replace(".tiff","_OrthoReady.tiff"))
            if completed:
                logging.debug('    Calibration step not required => abort')
                continue

            for image in images:
                image_ok = image.replace(".tiff", "_calOk.tiff")
                #UNCOMMENT TO DELETE RAW DATA
                # files_to_remove += [image]
                all_cmd.append(['    Calibration of '+image,
                    'export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={};'.format(self.cfg.OTBThreads)+"otbcli_SARCalibration"\
                    +" -ram "+str(self.cfg.ram_per_process)\
                    +" -progress false -in "+image\
                    +" -out "+image_ok+' -lut '+self.cfg.calibration_type \
                    +" -noise "+str(self.cfg.removethermalnoise).lower()])

        self.run_processing(all_cmd, title="   Calibration "+self.cfg.calibration_type)

        # TODO: add geoms
        remove_files(files_to_remove)

        logging.info("Calibration done")

    def do_ortho_by_tile(self, raster_list, tile_name, tmp_srtm_dir):
        """
        This method performs ortho-rectification of a list of
        s1 images on given tile.

        Args:
          raster_list: list of raw S1 raster file to orthorectify
          tile_name: Name of the MGRS tile to generate
        """
        all_cmd = []
        output_files_list = []
        logging.info("Start orthorectification of %s",tile_name)
        for raster, tile_origin in raster_list:
            logging.debug("- Otho: raster: %s; tile_origin: %s", raster, tile_origin)
            manifest = raster.get_manifest()
            logging.debug("  -> manifest: %s", manifest)
            logging.debug("  -> images: %s", raster.get_images_list())

            for image in raster.get_images_list():
                image_ok = image.replace(".tiff", "_OrthoReady.tiff")
                current_date            = Utils.get_date_from_s1_raster(image)
                current_polar           = Utils.get_polar_from_s1_raster(image)
                current_platform        = Utils.get_platform_from_s1_raster(image)
                current_orbit_direction = Utils.get_orbit_direction(manifest)
                current_relative_orbit  = Utils.get_relative_orbit(manifest)
                out_utm_zone            = tile_name[0:2]
                out_utm_northern        = (tile_name[2] >= 'N')
                working_directory       = os.path.join(self.cfg.output_preprocess, tile_name)
                if os.path.exists(working_directory) == False:
                    os.makedirs(working_directory)

                in_epsg = 4326
                out_epsg = 32600+int(out_utm_zone)
                if not out_utm_northern:
                    out_epsg = out_epsg+100

                x_coord, y_coord, _  = Utils.convert_coord([tile_origin[0]], in_epsg, out_epsg)[0]
                lrx, lry, _          = Utils.convert_coord([tile_origin[2]], in_epsg, out_epsg)[0]

                if not out_utm_northern and y_coord < 0:
                    y_coord = y_coord+10000000.
                    lry = lry+10000000.

                ortho_image_name = current_platform\
                                   +"_"+tile_name\
                                   +"_"+current_polar\
                                   +"_"+current_orbit_direction\
                                   +'_{:0>3d}'.format(current_relative_orbit)\
                                   +"_"+current_date\
                                   +".tif"

                ortho_image_pathname = os.path.join(working_directory, ortho_image_name)
                if not os.path.exists(ortho_image_pathname) and not os.path.exists(os.path.join(working_directory,ortho_image_name[:-11]+"txxxxxx.tif")):
                    cmd = 'export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={};'.format(self.cfg.OTBThreads)+"otbcli_OrthoRectification -opt.ram "\
                      +str(self.cfg.ram_per_process)\
                      +" -progress false -io.in "+image_ok\
                      +" -io.out \""+ortho_image_pathname\
                      +"?&writegeom=false&gdal:co:COMPRESS=DEFLATE\" -interpolator nn -outputs.spacingx "\
                      +str(self.cfg.out_spatial_res)\
                      +" -outputs.spacingy -"+str(self.cfg.out_spatial_res)\
                      +" -outputs.sizex "\
                      +str(int(round(abs(lrx-x_coord)/self.cfg.out_spatial_res)))\
                      +" -outputs.sizey "\
                      +str(int(round(abs(lry-y_coord)/self.cfg.out_spatial_res)))\
                      +" -opt.gridspacing "+str(self.cfg.grid_spacing)\
                      +" -map utm -map.utm.zone "+str(out_utm_zone)\
                      +" -map.utm.northhem "+str(out_utm_northern).lower()\
                      +" -outputs.ulx "+str(x_coord)\
                      +" -outputs.uly "+str(y_coord)\
                      +" -elev.dem "+tmp_srtm_dir+" -elev.geoid "+self.cfg.GeoidFile

                    all_cmd.append(['    Orthorectification of '+ortho_image_name, cmd])
                    output_files_list.append(ortho_image_pathname)

        self.run_processing(all_cmd, title="   Orthorectification")

        # Writing the metadata
        for f in os.listdir(working_directory):
            fullpath = os.path.join(working_directory, f)
            if os.path.isfile(fullpath) and f.startswith('s1') and f.endswith('.tif'):
                dst = gdal.Open(fullpath, gdal.GA_Update)
                oin = f.split('_')

                dst.SetMetadataItem('S2_TILE_CORRESPONDING_CODE', tile_name)
                dst.SetMetadataItem('PROCESSED_DATETIME', str(datetime.datetime.now().strftime('%Y:%m:%d')))
                dst.SetMetadataItem('ORTHORECTIFIED', 'true')
                dst.SetMetadataItem('CALIBRATION', str(self.cfg.calibration_type))
                dst.SetMetadataItem('SPATIAL_RESOLUTION', str(self.cfg.out_spatial_res))
                dst.SetMetadataItem('IMAGE_TYPE', 'GRD')
                dst.SetMetadataItem('FLYING_UNIT_CODE', oin[0])
                dst.SetMetadataItem('POLARIZATION', oin[2])
                dst.SetMetadataItem('ORBIT', oin[4])
                dst.SetMetadataItem('ORBIT_DIRECTION', oin[3])
                if oin[5][9] == 'x':
                    date = oin[5][0:4]+':'+oin[5][4:6]+':'+oin[5][6:8]+' 00:00:00'
                else:
                    date = oin[5][0:4]+':'+oin[5][4:6]+':'+oin[5][6:8]+' '+oin[5][9:11]+':'+oin[5][11:13]+':'+oin[5][13:15]
                dst.SetMetadataItem('ACQUISITION_DATETIME', date)

        return output_files_list

    def concatenate_images(self,tile):
        """
        This method concatenates images sub-swath for all generated tiles.
        """
        logging.info("Start concatenation of %s",tile)
        cmd_list = []
        files_to_remove = []

        image_list = [i.name for i in Utils.list_files(os.path.join(self.cfg.output_preprocess, tile))
                if (len(i.name) == 40 and "xxxxxx" not in i.name)]
        image_list.sort()

        while len(image_list) > 1:
            image_sublist=[i for i in image_list if (image_list[0][:29] in i)]

            if len(image_sublist) >1 :
                images_to_concatenate=[os.path.join(self.cfg.output_preprocess, tile,i) for i in image_sublist]
                files_to_remove += images_to_concatenate
                output_image = images_to_concatenate[0][:-10]+"xxxxxx"+images_to_concatenate[0][-4:]

                ### build the expression for BandMath for concanetation of many images
                ### for each pixel, the concatenation consists in selecting the first non-zero value in the time serie
                ##expression="(im%sb1!=0 ? im%sb1 : 0)" % (str(len(images_to_concatenate)),str(len(images_to_concatenate)))
                ##for i in range(len(images_to_concatenate)-1,0,-1):
                ##    expression="(im%sb1!=0 ? im%sb1 : %s)" % (str(i),str(i),expression)
                ##cmd_list.append(['    Concatenation of '+output_image,
                ##    'export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={};'.format(self.cfg.OTBThreads)+'otbcli_BandMath -progress false -ram '\
                ##    +str(self.cfg.ram_per_process)\
                ##    +' -il '+' '.join(images_to_concatenate)\
                ##    +' -out '+output_image\
                ##    +' -exp "'+expression+'"'])
                cmd_list.append(['    Concatenation of '+output_image,
                    'export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={};'.format(self.cfg.OTBThreads)+'otbcli_Synthetize -progress false -ram '\
                    +str(self.cfg.ram_per_process)\
                    +' -il '+' '.join(images_to_concatenate)\
                    +' -out '+output_image])

                if self.cfg.mask_cond:
                    if "vv" in image_list[0]:
                        images_msk_to_concatenate = [i.replace(".tif", "_BorderMask.tif") for i in images_to_concatenate]
                        files_to_remove += images_msk_to_concatenate
                        cmd_list.append(['    Concatenation of '+output_image+' mask',
                            'export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={};'.format(self.cfg.OTBThreads)+'otbcli_BandMath -progress false -ram '\
                            +str(self.cfg.ram_per_process)\
                            +' -il '+' '.join(images_msk_to_concatenate)\
                            +' -out '+output_image.replace(".tif", "_BorderMask.tif")\
                            +' -exp "'+expression+'"'])

            for i in image_sublist:
                image_list.remove(i)

        self.run_processing(cmd_list, "   Concatenation")

        remove_files(files_to_remove)

    def run_processing(self, cmd_list, title=""):
        """
        This method executes a given command.
        Args:
          cmd_list: the command to run
          title: optional title
        """
        import time
        nb_cmd = len(cmd_list)

        with multiprocessing.Pool(self.cfg.nb_procs, worker_config, [self.cfg.log_queue]) as pool:
            self.cfg.log_queue_listener.start()
            for count, result in enumerate(pool.imap_unordered(execute_command, cmd_list), 1):
                logging.info("%s correctly finished", result)
                logging.info(' --> %s... %s%%', title, count*100./nb_cmd)

            pool.close()
            pool.join()
            self.cfg.log_queue_listener.stop()

        logging.info("%s done", title)


def init_logger(mode, paths):
    import logging.config
    import yaml
    # Add the dirname where the current script is
    paths += [pathlib.Path(__file__).parent.absolute()]
    paths = [p/'logging.conf.yaml' for p in paths]
    cfgpaths = [p for p in paths if p.is_file()]
    # print("from %s, keep %s" % (paths, cfgpaths))

    verbose   = 'debug'   in mode
    log2files = 'logging' in mode
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


# Main code

if len(sys.argv) != 2:
    print("Usage: "+sys.argv[0]+" config.cfg")
    sys.exit(1)

CFG = sys.argv[1]
Cg_Cfg=Configuration(CFG)
S1_CHAIN = Sentinel1PreProcess(Cg_Cfg)
S1_FILE_MANAGER = S1FileManager.S1FileManager(Cg_Cfg)


TILES_TO_PROCESS = []

ALL_REQUESTED = False

for tile_it in Cg_Cfg.tiles_list:
    logging.info('Requesting to process tile %s', tile_it)
    if tile_it == "ALL":
        ALL_REQUESTED = True
        break
    elif True:  #S1_FILE_MANAGER.tile_exists(tile_it):
        TILES_TO_PROCESS.append(tile_it)
    else:
        logging.info("Tile %s does not exist, skipping ...", tile_it)

# We can not require both to process all tiles covered by downloaded products
# and and download all tiles


if ALL_REQUESTED:
    if Cg_Cfg.pepsdownload and "ALL" in Cg_Cfg.roi_by_tiles:
        logging.critical("Can not request to download ROI_by_tiles : ALL if Tiles : ALL."\
            +" Use ROI_by_coordinates or deactivate download instead")
        sys.exit(1)
    else:
        TILES_TO_PROCESS = S1_FILE_MANAGER.get_tiles_covered_by_products()
        logging.info("All tiles for which more than "\
            +str(100*Cg_Cfg.TileToProductOverlapRatio)\
            +"% of the surface is covered by products will be produced: "\
            +str(TILES_TO_PROCESS))

if len(TILES_TO_PROCESS) == 0:
    logging.critical("No existing tiles found, exiting ...")
    sys.exit(1)

# Analyse SRTM coverage for MGRS tiles to be processed
SRTM_TILES_CHECK = S1_FILE_MANAGER.check_srtm_coverage(TILES_TO_PROCESS)

NEEDED_SRTM_TILES = []
TILES_TO_PROCESS_CHECKED = []
# For each MGRS tile to process
for tile_it in TILES_TO_PROCESS:
    logging.info("Check SRTM coverage for %s",tile_it)
    # Get SRTM tiles coverage statistics
    srtm_tiles = SRTM_TILES_CHECK[tile_it]
    current_coverage = 0
    current_NEEDED_SRTM_TILES = []
    # Compute global coverage
    for (srtm_tile, coverage) in srtm_tiles:
        current_NEEDED_SRTM_TILES.append(srtm_tile)
        current_coverage += coverage
    # If SRTM coverage of MGRS tile is enough, process it
    if current_coverage >= 1.:
        NEEDED_SRTM_TILES += current_NEEDED_SRTM_TILES
        TILES_TO_PROCESS_CHECKED.append(tile_it)
    else:
        # Skip it
        logging.warning("Tile %s has insuficient SRTM coverage (%s%%)",
                tile_it, 100*current_coverage)
        NEEDED_SRTM_TILES += current_NEEDED_SRTM_TILES
        TILES_TO_PROCESS_CHECKED.append(tile_it)


# Remove duplicates
NEEDED_SRTM_TILES = list(set(NEEDED_SRTM_TILES))

logging.info("%s images to process on %s tiles",
        S1_FILE_MANAGER.nb_images, TILES_TO_PROCESS_CHECKED)

if len(TILES_TO_PROCESS_CHECKED) == 0:
    logging.critical("No tiles to process, exiting ...")
    sys.exit(1)

logging.info("Required SRTM tiles: %s", NEEDED_SRTM_TILES)

SRTM_OK = True

for srtm_tile in NEEDED_SRTM_TILES:
    tile_path = os.path.join(Cg_Cfg.srtm, srtm_tile)
    if not os.path.exists(tile_path):
        SRTM_OK = False
        logging.critical(tile_path+" is missing")

if not SRTM_OK:
    logging.critical("Some SRTM tiles are missing, exiting ...")
    sys.exit(1)

# copy all needed SRTM file in a temp directory for orthorectification processing
for srtm_tile in NEEDED_SRTM_TILES:
    os.symlink(os.path.join(Cg_Cfg.srtm,srtm_tile),os.path.join(S1_FILE_MANAGER.tmpsrtmdir,srtm_tile))


if not os.path.exists(Cg_Cfg.GeoidFile):
    logging.critical("Geoid file does not exists (%s), exiting ...", Cg_Cfg.GeoidFile)
    sys.exit(1)

filteringProcessor=S1FilteringProcessor.S1FilteringProcessor(Cg_Cfg)

for idx, tile_it in enumerate(TILES_TO_PROCESS_CHECKED):

    logging.info("Tile: "+tile_it+" ("+str(idx+1)+"/"+str(len(TILES_TO_PROCESS_CHECKED))+")")

    # keep only the 500's newer files
    safeFileList=sorted(glob.glob(os.path.join(Cg_Cfg.raw_directory,"*")),key=os.path.getctime)
    if len(safeFileList)> 1000	:
        for f in safeFileList[:len(safeFileList)-1000]:
            logging.debug("Remove : ",os.path.basename(f))
            shutil.rmtree(f,ignore_errors=True)
        S1_FILE_MANAGER.get_s1_img()

    with Utils.ExecutionTimer("Downloading tiles", True) as t:
        S1_FILE_MANAGER.download_images(tiles=tile_it)

    with Utils.ExecutionTimer("Intersecting raster list", True) as t:
        intersect_raster_list = S1_FILE_MANAGER.get_s1_intersect_by_tile(tile_it)

    if len(intersect_raster_list) == 0:
        logging.info("No intersections with tile %s",tile_it)
        continue

    Horizontal = False
    if Horizontal:
        with Utils.ExecutionTimer("Calibration", True) as t:
            S1_CHAIN.do_calibration_cmd(intersect_raster_list)
        with Utils.ExecutionTimer("Cut Images", True) as t:
            S1_CHAIN.cut_image_cmd(intersect_raster_list)
    else:
        with Utils.ExecutionTimer("Calibrate & Cut Images", True) as t:
            S1_CHAIN.do_calibrate_and_cut(intersect_raster_list)

    with Utils.ExecutionTimer("Ortho", True) as t:
        raster_tiles_list = S1_CHAIN.do_ortho_by_tile(\
                intersect_raster_list, tile_it,S1_FILE_MANAGER.tmpsrtmdir)
    if Cg_Cfg.mask_cond:
        with Utils.ExecutionTimer("Generate Border Mask", True) as t:
            S1_CHAIN.generate_border_mask(raster_tiles_list)

    with Utils.ExecutionTimer("Concatenate", True) as t:
        S1_CHAIN.concatenate_images(tile_it)

    """
    if Cg_Cfg.filtering_activated:
        with Utils.ExecutionTimer("MultiTemp Filter", True) as t:
            filteringProcessor.process(tile_it)
    """
