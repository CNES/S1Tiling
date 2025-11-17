#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program: generate-DEM-gpkg.py
#
#   All rights reserved.
#   Copyright 2017-2025 (c) CNES.
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
#
# This code is used to generate a GEOPACKAGE (gpkg) that contains the
# footprint and filename of Copernicus Digital Elevation Model tiles.
# It should be adapted to fit with the organisation of the tiles in the user context.
#
# Starting from the SHAPEFILE provide by Copernicus, we need first to convert it to a gpkg file
# Ex: ogr2ogr GEO1988-CopernicusDEM-RP-002_GridFile_I4.0.gpkg GEO1988-CopernicusDEM-RP-002_GridFile_I4.0.shp
#
# The SHAPEFILE is available here:
# https://s3.waw3-1.cloudferro.com/swift/v1/portal_uploads_prod/GEO1988-CopernicusDEM-RP-002_GridFile_I6.0.shp_08.2024.zip
#

import sys
from typing import NoReturn
from osgeo import ogr,osr
import os

def _die(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    sys.exit(127)


def select_columns(input_path, output_path, columns_to_keep) -> None:
    # Ouvrir le fichier GPKG en lecture seule
    input_ds = ogr.Open(input_path, 0)
    if input_ds is None:
        _die(f"Cannot open GPKG file {input_path!r} in read-only mode!")

    # Créer un nouveau fichier GPKG en sortie
    driver = ogr.GetDriverByName("GPKG")
    output_ds = driver.CreateDataSource(output_path)
    if output_ds is None:
        _die(f"Cannot create new GPKG file {output_path!r}!")

    # Créer le CRS du DEM Copernicus
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    # Parcourir chaque couche du fichier d'entrée
    nb_layers = input_ds.GetLayerCount()
    print(f"{nb_layers} layers will be converted")
    for layer_index in range(nb_layers):
        input_layer = input_ds.GetLayerByIndex(layer_index)

        ## Create the output layer
        output_layer = output_ds.CreateLayer(
            input_layer.GetName(),
            geom_type=input_layer.GetGeomType(),
            srs=srs,
            options=["SPATIAL_INDEX=YES"]
        )

        ## Determine output layer definition
        # Filter input fields among those to keep, and add them to the definition
        layer_definition = input_layer.GetLayerDefn()
        nb_fields = layer_definition.GetFieldCount()

        fields_to_copy = []
        for field_index in range(nb_fields):
            field_defn = layer_definition.GetFieldDefn(field_index)
            field_name = field_defn.GetName()

            if field_name in columns_to_keep:
                fields_to_copy.append(field_name)
                output_layer.CreateField(field_defn)

        print(f"Layer {layer_index}: {nb_fields} found, {len(fields_to_copy)} kept.")

        # TODO: Warn for columns_to_keep not in fields_to_copy

        # Extra field: FileID: name of the file to expect
        file_id_field = ogr.FieldDefn("FileID", ogr.OFTString)
        file_id_field.SetWidth(150)  # Ajustez la largeur en fonction de vos besoins
        output_layer.CreateField(file_id_field)

        ## Copy kept input elements into output layer
        output_layer_definition = output_layer.GetLayerDefn()
        for feature in input_layer:
            output_feature = ogr.Feature(output_layer_definition)

            for field_name in fields_to_copy:
                output_feature.SetField(field_name, feature.GetField(field_name))
            CellID = feature.GetField("GeoCellID")
            print(CellID)
            latitudeID  = CellID[0:3]
            longitudeID = CellID[3:]
            fileID=f"Copernicus_DSM_10_{latitudeID}_00_{longitudeID}_00/DEM/Copernicus_DSM_10_{latitudeID}_00_{longitudeID}_00_DEM.tif"
            if not os.path.exists(os.path.join(pathDEM,fileID)):
                _die(f"{fileID!r} does not exist")
            
            output_feature.SetField(
                    "FileID",
                    fileID
            )

            output_feature.SetGeometry(feature.GetGeometryRef())
            output_layer.CreateFeature(output_feature)

            output_feature = None

    # Close datasets
    input_ds = None
    output_ds = None
    print(f"{list(columns_to_keep)} columns selected and written with success\n-> {output_file}")


# Chemin vers le fichier GPKG d'entrée
input_file = "/home/il/koleckt/s1tiling/s1tiling/resources/generate-gpkg/GEO1988-CopernicusDEM-RP-002_GridFile_I6.0.gpkg"

# Chemin vers le fichier GPKG de sortie
output_file = "/home/il/koleckt/s1tiling/s1tiling/resources/shapefile/CopernicusDEM2023-CNES.gpkg"

# Chemin du DEM
pathDEM="/work/datalake/static_aux/MNT/COP-DEM_GLO-30-DGED_2023_1"

# List of fields/columns to keep from the input dataset
columns_to_keep = {"GeoCellID"}

# Appeler la fonction pour conserver les colonnes sélectionnées
select_columns(input_file, output_file, columns_to_keep)
