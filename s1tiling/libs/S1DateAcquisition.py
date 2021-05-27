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

""" This module contains the S1DateAcquisition class"""


class S1DateAcquisition:
    """This class handles the list of images for one S1 product"""
    def __init__(self, manifest, image_filenames_list):
        self.manifest = manifest
        self.image_filenames_list = image_filenames_list


    def __repr__(self):
        return "S1DateAcquisition('%s', %s)" % (self.manifest, self.image_filenames_list)

    def get_manifest(self):
        """ Get the manifest file """
        return self.manifest

    def add_image(self, image_list):
        """ Add an image to the image list """
        self.image_filenames_list.append(image_list)

    def get_images_list(self):
        """ Get the image list"""
        return self.image_filenames_list
