#!/usr/bin/python
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
#
# =========================================================================

""" This module contains the S1DateAcquisition class"""

class S1DateAcquisition(object):
    """This class handles the list of images for one S1 product"""
    def __init__(self, manifest, image_filenames_list):
        self.manifest = manifest
        self.image_filenames_list = image_filenames_list

    def get_manifest(self):
        """ Get the manifest file """
        return self.manifest

    def add_image(self, image_list):
        """ Add an image to the image list """
        self.image_filenames_list.append(image_list)

    def get_images_list(self):
        """ Get the image list"""
        return self.image_filenames_list
