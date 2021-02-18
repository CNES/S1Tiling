#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2021 (c) CESBIO. All rights reserved.
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
#
# =========================================================================

""" This module contains the multitemporal speckle filtering processor """

import datetime
import glob
import os
from subprocess import Popen
import pickle
import time
from osgeo import gdal


class S1FilteringProcessor():
    def __init__(self, cfg):
        self.Cg_Cfg = cfg

    def process(self, tile):
        """Main function for speckle filtering script"""
        directory = os.path.join(self.Cg_Cfg.output_preprocess, tile.upper())
        print("Start speckle filtering: " + tile.upper())
        year_outcore_list = ["2019", "2018"]
        year_filter_list = ["2019", "2018"]

        year_outcore_str = "-".join(year_outcore_list)    # pour les noms de fichiers

        filelist_s1des = []
        filelist_s1asc = []
        filelist_s1des_updateoutcore = []
        filelist_s1asc_updateoutcore = []
        # Build the lists of files :
        #    - for computing outcores
        #    - for filtering

        for y in year_outcore_list:
            for file_it in glob.glob(os.path.join(directory, "s1?_?????_??_DES_???_" + y + "????t??????.tif")):
                filelist_s1des_updateoutcore.append(file_it)

            for file_it in glob.glob(os.path.join(directory, "s1?_?????_??_ASC_???_" + y + "????t??????.tif")):
                filelist_s1asc_updateoutcore.append(file_it)

        # Select only 100 images for the outcore dataset (for both ASC and DES outcores)
        filelist_s1des_updateoutcore = filelist_s1des_updateoutcore[:100]
        filelist_s1asc_updateoutcore = filelist_s1asc_updateoutcore[:100]

        for y in year_filter_list:
            for file_it in glob.glob(os.path.join(directory, "s1?_?????_??_DES_???_" + y + "????t??????.tif")):
                filelist_s1des.append(file_it)

            for file_it in glob.glob(os.path.join(directory, "s1?_?????_??_ASC_???_" + y + "????t??????.tif")):
                filelist_s1asc.append(file_it)

        print(filelist_s1des)
        print()
        print(filelist_s1asc)
        print()

        if self.Cg_Cfg.Reset_outcore:
            processed_files = []
            try:
                os.remove(os.path.join(directory, "outcore" + year_filter + ".txt"))
            except:
                pass
        else:
            try:
                processed_files = \
                        pickle.load(open(os.path.join(directory, "outcore" + year_filter + ".txt")))
            except pickle.PickleError:
                processed_files = []

        # Compute the outcores for ASC and DES images

        for file_it in processed_files:
            try:
                filelist_s1des_updateoutcore.remove(file_it)
                filelist_s1asc_updateoutcore.remove(file_it)
            except ValueError:
                pass

        # Build the strings containing the filenames to be processed
        filelist_s1des_updateoutcore_str = " ".join(filelist_s1des_updateoutcore)
        filelist_s1asc_updateoutcore_str = " ".join(filelist_s1asc_updateoutcore)
        filelist_s1des_str = " ".join(filelist_s1des)
        filelist_s1asc_str = " ".join(filelist_s1asc)

        pids = []

        # Adapts the processing ressources to only two processes

        ram_per_process = int(self.Cg_Cfg.ram_per_process * self.Cg_Cfg.nb_procs / 2)
        OTBThreads = int(self.Cg_Cfg.OTBThreads * self.Cg_Cfg.nb_procs / 2)

        ####### TK
        # On vide la liste des fichiers ASC pour eviter de calculer l'outcore
        filelist_s1asc_updateoutcore = []
        filelist_s1asc = []
        #

        if filelist_s1des_updateoutcore:
            command = 'export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={};'.format(OTBThreads)\
                  + "otbcli_MultitempFilteringOutcore -progress false -inl "\
                  + filelist_s1des_updateoutcore_str + " -oc "\
                  + os.path.join(directory, "outcore" + year_outcore_str + "_S1DES.tif")\
                  + " -wr {}".format(self.Cg_Cfg.Window_radius)\
                  + " -ram {}".format(str(ram_per_process))
            pids.append([Popen(command, stdout=self.Cg_Cfg.stdoutfile,
                stderr=self.Cg_Cfg.stderrfile, shell=True), command])
        if filelist_s1asc_updateoutcore:
            command = 'export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={};'.format(OTBThreads)\
                  + "otbcli_MultitempFilteringOutcore -progress false -inl "\
                  + filelist_s1asc_updateoutcore_str + " -oc "\
                  + os.path.join(directory, "outcore" + year_outcore_str + "_S1ASC.tif")\
                  + " -wr " + str(self.Cg_Cfg.Window_radius)\
                  + " -ram {}".format(str(ram_per_process))
            pids.append([Popen(command, stdout=self.Cg_Cfg.stdoutfile,
                stderr=self.Cg_Cfg.stderrfile, shell=True), command])
        try:
            os.makedirs(os.path.join(directory, "filtered"))
        except os.error:
            pass

        title = "Compute outcore"
        nb_cmd = len(pids)
        print(title + "... 0%")
        while len(pids) > 0:

            for i, pid in enumerate(pids):
                status = pid[0].poll()
                if status:
                    print("Error in pid #" + str(i) + " id = " + str(pid[0]))
                    print(pid[1])
                    del pids[i]
                    break

                elif status == 0:
                    del pids[i]
                    print(title + "... " + str(int((nb_cmd - len(pids)) * 100. / nb_cmd)) + "%")
                    time.sleep(0.2)
                    break
            time.sleep(2)

        processed_files = processed_files + filelist_s1des_updateoutcore\
                + filelist_s1asc_updateoutcore

        pickle.dump(processed_files, open(os.path.join(directory, "outcore.txt"), 'w'))

        # Compute the filtered images using the outcores

        pids = []
        if filelist_s1des:
            command = 'export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={};'.format(OTBThreads)\
                  + "otbcli_MultitempFilteringFilter -progress false -inl "\
                  + filelist_s1des_str + " -oc "\
                  + os.path.join(directory, "outcore" + year_outcore_str + "_S1DES.tif")\
                  + " -wr " + str(self.Cg_Cfg.Window_radius) + " -enl "\
                  + os.path.join(directory, "filtered", "enl_" + year_outcore_str + "_S1DES.tif")\
                  + " -ram {}".format(str(ram_per_process))
            pids.append([Popen(command, stdout=self.Cg_Cfg.stdoutfile,
                stderr=self.Cg_Cfg.stderrfile, shell=True), command])

        if filelist_s1asc:
            command = 'export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={};'.format(OTBThreads)\
                  + "otbcli_MultitempFilteringFilter -progress false -inl "\
                  + filelist_s1asc_str + " -oc "\
                  + os.path.join(directory, "outcore" + year_outcore_str + "_S1ASC.tif")\
                  + " -wr " + str(self.Cg_Cfg.Window_radius) + " -enl "\
                  + os.path.join(directory, "filtered", "enl_" + year_outcore_str + "_S1ASC.tif")\
                  + " -ram {}".format(str(ram_per_process))
            pids.append([Popen(command, stdout=self.Cg_Cfg.stdoutfile,
                stderr=self.Cg_Cfg.stderrfile, shell=True), command])

        title = "Compute filtered images"
        nb_cmd = len(pids)
        print(title + "... 0%")
        while len(pids) > 0:

            for i, pid in enumerate(pids):
                status = pid[0].poll()
                if status:
                    print("Error in pid #" + str(i) + " id = " + str(pid[0]))
                    print(pid[1])
                    del pids[i]
                    break

                elif status == 0:
                    del pids[i]
                    print(title + "... " + str(int((nb_cmd - len(pids)) * 100. / nb_cmd)) + "%")
                    time.sleep(0.2)
                    break
            time.sleep(2)

        filtering_directory = os.path.join(directory, 'filtered/')
        for f in os.listdir(filtering_directory):
            fullpath = os.path.join(filtering_directory, f)
            if os.path.isfile(fullpath) and f.startswith('s1') and f.endswith('filtered.tif'):
                dst = gdal.Open(fullpath, gdal.GA_Update)
                dst.SetMetadataItem('FILTERED', 'true')
                dst.SetMetadataItem('FILTERING_WINDOW_RADIUS', str(self.Cg_Cfg.Window_radius))
                dst.SetMetadataItem('FILTERING_PROCESSINGDATE', str(datetime.datetime.now()))
