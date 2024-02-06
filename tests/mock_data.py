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

import logging
from typing import List, Tuple

# from .mock_otb import compute_coverage

def tmp_suffix(tmp) -> str:
    return '.tmp' if tmp else ''


class FileDB:
    FILE_FMTS = {
            's1file'              : '{s1_basename}.tiff',
            'cal_ok'              : '{s1_basename}{tmp}.tiff',
            'ortho_ready'         : '{s1_basename}_OrthoReady{tmp}.tiff',
            'orthofile'           : '{s2_basename}{calibration}{tmp}',
            'sigma0_normlim_file' : '{s2_basename}_NormLim{tmp}',
            'border_mask_tmp'     : '{s2_basename}{calibration}_BorderMaskTmp{tmp}.tif',
            'border_mask'         : '{s2_basename}{calibration}_BorderMask{tmp}.tif',

            'vrt'                 : 'DEM_{s1_polarless}{tmp}.vrt',
            'sardemprojfile'      : 'S1_on_DEM_{s1_polarless}{tmp}.tiff',
            'xyzfile'             : 'XYZ_{s1_polarless}{tmp}.tiff',
            'normalsfile'         : 'Normals_{s1_polarless}{tmp}.tiff',
            'LIAfile'             : 'LIA_{s1_polarless}{tmp}.tiff',
            'sinLIAfile'          : 'sin_LIA_{s1_polarless}{tmp}.tiff',
            'orthoLIAfile'        : 'LIA_{s2_polarless}{tmp}',
            'orthosinLIAfile'     : 'sin_LIA_{s2_polarless}{tmp}',
            }
    FILES = [
            # 08 jan 2020
            {
                'start_time'      : '2020:01:08 04:41:50',
                's1dir'           : 'S1A_IW_GRDH_1SDV_20200108T044150_20200108T044215_030704_038506_C7F5',
                's1_basename'     : 's1a-iw-grd-{polarity}-20200108t044150-20200108t044215-030704-038506-{nr}',
                's2_basename'     : 's1a_33NWB_{polarity}_DES_007_20200108t044150',
                's1_polarless'    : 's1a-iw-grd-20200108t044150-20200108t044215-030704-038506',
                's2_polarless'    : 's1a_33NWB_DES_007_20200108t044150',
                'dem_coverage'    : ['N00E014', 'N00E015', 'N00E016', 'N01E014', 'N01E015', 'N01E016', 'N02E014', 'N02E015', 'N02E016'],
                'polygon'         : [(1.137156, 14.233953), (0.660935, 16.461103), (2.173307, 16.77552), (2.645077, 14.545785), (1.137156, 14.233953)],
                'srsname'         : 'epsg:4326',
                'orbit_direction' : 'DES',
                'relative_orbit'  : 7,
                },
            {
                'start_time'      : '2020:01:08 04:42:15',
                's1dir'           : 'S1A_IW_GRDH_1SDV_20200108T044215_20200108T044240_030704_038506_D953',
                's1_basename'     : 's1a-iw-grd-{polarity}-20200108t044215-20200108t044240-030704-038506-{nr}',
                's2_basename'     : 's1a_33NWB_{polarity}_DES_007_20200108t044215',
                's1_polarless'    : 's1a-iw-grd-20200108t044215-20200108t044240-030704-038506',
                's2_polarless'    : 's1a_33NWB_DES_007_20200108t044215',
                'dem_coverage'    : ['N00E013', 'N00E014', 'N00E015', 'N00E016', 'N01E014', 'S01E013', 'S01E014', 'S01E015', 'S01E016'],
                'polygon'         : [(-0.370174, 13.917268), (-0.851051, 16.143845), (0.660845, 16.461084), (1.137179, 14.233407), (-0.370174, 13.917268)],
                'srsname'         : 'epsg:4326',
                'orbit_direction' : 'DES',
                'relative_orbit'  : 7,
                },
            # 20 jan 2020
            {
                'start_time'      : '2020:01:20 04:41:49',
                's1dir'           : 'S1A_IW_GRDH_1SDV_20200120T044149_20200120T044214_030879_038B2D_5671',
                's1_basename'     : 's1a-iw-grd-{polarity}-20200120t044149-20200120t044214-030879-038B2D-{nr}',
                's2_basename'     : 's1a_33NWB_{polarity}_DES_007_20200120t044149',
                's1_polarless'    : 's1a-iw-grd-20200120t044149-20200120t044214-030879-038B2D',
                's2_polarless'    : 's1a_33NWB_DES_007_20200120t044149',
                'dem_coverage'    : ['N00E014', 'N00E015', 'N00E016', 'N01E014', 'N01E015', 'N01E016', 'N02E014', 'N02E015', 'N02E016'],
                'polygon'         : [(1.137292, 14.233942), (0.661038, 16.461086), (2.173408, 16.775522), (2.645211, 14.545794), (1.137292, 14.233942)],
                'srsname'         : 'epsg:4326',
                'orbit_direction' : 'DES',
                'relative_orbit'  : 7,
                },
            {
                'start_time'      : '2020:01:20 04:42:14',
                's1dir'           : 'S1A_IW_GRDH_1SDV_20200120T044214_20200120T044239_030879_038B2D_FDB0',
                's1_basename'     : 's1a-iw-grd-{polarity}-20200120t044214-20200120t044239-030879-038B2D-{nr}',
                's2_basename'     : 's1a_33NWB_{polarity}_DES_007_20200120t044214',
                's1_polarless'    : 's1a-iw-grd-20200120t044214-20200120t044239-030879-038B2D',
                's2_polarless'    : 's1a_33NWB_DES_007_20200120t044214',
                'dem_coverage'    : ['N00E013', 'N00E014', 'N00E015', 'N00E016', 'N01E014', 'S01E013', 'S01E014', 'S01E015', 'S01E016'],
                'polygon'         : [(-0.370036, 13.917237), (-0.850946, 16.143806), (0.660948, 16.461067), (1.137315, 14.233396), (-0.370036, 13.917237)],
                'srsname'         : 'epsg:4326',
                'orbit_direction' : 'DES',
                'relative_orbit'  : 7,
                },
            # 02 feb 2020
            {
                'start_time'      : '2020:02:01 04:41:49',
                's1dir'           : 'S1A_IW_GRDH_1SDV_20200201T044149_20200201T044214_031054_039149_ED12',
                's1_basename'     : 's1a-iw-grd-{polarity}-20200201t044149-20200201t044214-031054-039149-{nr}',
                's2_basename'     : 's1a_33NWB_{polarity}_DES_007_20200201t044149',
                's1_polarless'    : 's1a-iw-grd-20200201t044149-20200201t044214-031054-039149-{nr}',
                's2_polarless'    : 's1a_33NWB_DES_007_20200201t044149',
                'dem_coverage'    : ['N00E014', 'N00E015', 'N00E016', 'N01E014', 'N01E015', 'N01E016', 'N02E014', 'N02E015', 'N02E016'],
                'polygon'         : [(1.137385, 14.233961), (0.661111, 16.461193), (2.173392, 16.775606), (2.645215, 14.54579), (1.137385, 14.233961)],
                'srsname'         : 'epsg:4326',
                'orbit_direction' : 'DES',
                'relative_orbit'  : 7,
                },
            {
                'start_time'      : '2020:02:01 04:42:14',
                's1dir'           : 'S1A_IW_GRDH_1SDV_20200201T044214_20200201T044239_031054_039149_CC58',
                's1_basename'     : 's1a-iw-grd-{polarity}-20200201t044214-20200201t044239-031054-039149-{nr}',
                's2_basename'     : 's1a_33NWB_{polarity}_DES_007_20200201t044214',
                's1_polarless'    : 's1a-iw-grd-20200201t044214-20200201t044239-031054-039149-{nr}',
                's2_polarless'    : 's1a_33NWB_DES_007_20200201t044214',
                'dem_coverage'    : ['N00E013', 'N00E014', 'N00E015', 'N00E016', 'N01E014', 'S01E013', 'S01E014', 'S01E015', 'S01E016'],
                'polygon'         : [(-0.370053, 13.91733), (-0.850965, 16.1439), (0.661021, 16.461174), (1.137389, 14.233503), (-0.370053, 13.91733)],
                'srsname'         : 'epsg:4326',
                'orbit_direction' : 'DES',
                'relative_orbit'  : 7,
                },
            ]
    CONCATS = [
            # 08 jan 2020
            {
                's2_basename' : 's1a_33NWB_{polarity}_DES_007_20200108txxxxxx',
                's2_polarless': 's1a_33NWB_DES_007_20200108txxxxxx',
                'start_time'  : '2020:01:08 04:41:50',
                'first_date'  : '2020-01-01',
                'last_date'   : '2020-01-10',
                },
            # 20 jan 2020
            {
                's2_basename' : 's1a_33NWB_{polarity}_DES_007_20200120txxxxxx',
                's2_polarless': 's1a_33NWB_DES_007_20200120txxxxxx',
                'start_time'  : '2020:01:20 04:41:49',
                'first_date'  : '2020-01-10',
                'last_date'   : '2020-01-21',
                },
            # 02 feb 2020
            {
                's2_basename' : 's1a_33NWB_{polarity}_DES_007_20200201txxxxxx',
                's2_polarless': 's1a_33NWB_DES_007_20200201txxxxxx',
                'start_time'  : '2020:02:01 04:41:49',
                'first_date'  : '2020-02-01',
                'last_date'   : '2020-02-05',
                },
            ]
    TILE = '33NWB'
    extended_geom_compress = '?&writegeom=false&gdal:co:COMPRESS=DEFLATE'
    extended_compress      = '?&gdal:co:COMPRESS=DEFLATE'

    def __init__(self, inputdir, tmpdir, outputdir, liadir, tile, demdir, geoid_file) -> None:
        self.__input_dir  = inputdir
        self.__tmp_dir    = tmpdir
        self.__output_dir = outputdir
        self.__lia_dir    = liadir
        self.__tmp_dir    = tmpdir
        self.__tile       = tile
        self.__dem_dir    = demdir
        self.__GeoidFile  = geoid_file

        NFiles   = len(self.FILES)
        NConcats = len(self.CONCATS)

        self.nb_S1_products = NFiles
        self.nb_S2_products = NConcats

        names_to_map = [
                # function_reference,               [indices...]
                [self.cal_ok,                       NFiles],
                [self.ortho_ready,                  NFiles],
                [self.orthofile,                    NFiles],
                [self.concatfile_from_one,          NFiles],
                [self.concatfile_from_two,          NConcats],
                [self.masktmp_from_one,             NFiles],
                [self.masktmp_from_two,             NConcats],
                [self.maskfile_from_one,            NFiles],
                [self.maskfile_from_two,            NConcats],

                [self.vrtfile,                      NFiles],
                [self.sardemprojfile,               NFiles],
                [self.xyzfile,                      NFiles],
                [self.normalsfile,                  NFiles],
                [self.LIAfile,                      NFiles],
                [self.sinLIAfile,                   NFiles],
                [self.orthoLIAfile,                 NFiles],
                [self.orthosinLIAfile,              NFiles],
                [self.concatLIAfile_from_two,       NConcats],
                [self.concatsinLIAfile_from_two,    NConcats],
                [self.sigma0_normlim_file_from_one, NFiles],
                [self.sigma0_normlim_file_from_two, NConcats],
                ]
        names_to_map_for_beta_calib = [
                [self.orthofile,                    NFiles],
                [self.concatfile_from_one,          NFiles],
                [self.concatfile_from_two,          NConcats],
                [self.masktmp_from_one,             NFiles],
                [self.masktmp_from_two,             NConcats],
                [self.maskfile_from_one,            NFiles],
                [self.maskfile_from_two,            NConcats],
                ]
        self.__tmp_to_out_map = {}
        for func, nb in names_to_map:
            for idx in range(nb):
                self.__tmp_to_out_map[func(idx, True)] = func(idx, False)
        # coded beta-calibration cases...
        for func, nb in names_to_map_for_beta_calib:
            for idx in range(nb):
                self.__tmp_to_out_map[func(idx, True, calibration='_beta')] = func(idx, False, calibration='_beta')
        # for idx in range(NFiles):
        #     self.__tmp_to_out_map[self.orthofile(idx, True, calibration='_beta')] = self.orthofile(idx, False, calibration='_beta')
        #     self.__tmp_to_out_map[self.concatfile_from_one(idx, True, calibration='_beta')] = self.concatfile_from_one(idx, False, calibration='_beta')

    @property
    def tmp_to_out_map(self):
        """
        Property tmp_to_out_map
        """
        return self.__tmp_to_out_map

    @property
    def inputdir(self):
        """
        Property inputdir
        """
        return self.__input_dir

    @property
    def outputdir(self):
        """
        Property outputdir
        """
        return self.__output_dir

    @property
    def demdir(self):
        """
        Property demdir
        """
        return self.__dem_dir

    @property
    def dem_files(self) -> List[str]:
        """
        Return list of all DEM files.
        """
        dem_tiles = []
        for idx in range(len(self.FILES)):
            dem_tiles.extend(self.dem_coverage(idx))
        # TODO: adapt it to any DEM support
        return [f"{self.__dem_dir}/{tile}.hgt" for tile in set(dem_tiles)]

    @property
    def GeoidFile(self):
        """
        Property GeoidFile
        """
        return self.__GeoidFile

    def all_products(self) -> List[str]:
        return [self.product_name(idx) for idx in range(len(self.FILES))]

    def all_files(self) -> List[str]:
        return [self.input_file(idx) for idx in range(len(self.FILES))]

    def all_vvvh_files(self) -> List[str]:
        # return [f'{idx} {pol}' for idx in range(len(self.FILES)) for pol in ['vv', 'vh']]
        return [self.input_file(idx, polarity=pol) for idx in range(len(self.FILES)) for pol in ['vv', 'vh']]

    def start_time(self, idx) -> str:
        return self.FILES[idx]['start_time']

    def start_time_for_two(self, idx) -> str:
        return self.CONCATS[idx]['start_time']

    def product_name(self, idx) -> str:
        s1dir  = self.FILES[idx]['s1dir']
        return s1dir

    def safe_dir(self, idx) -> str:
        s1dir  = self.FILES[idx]['s1dir']
        return f'{self.__input_dir}/{s1dir}/{s1dir}.SAFE'

    def input_file(self, idx, polarity='vv') -> str:
        crt    = self.FILES[idx]
        s1dir  = crt['s1dir']
        s1file = self.FILE_FMTS['s1file'].format(**crt).format(polarity=polarity, nr="001" if polarity == "vv" else "002")
        # return f'{self.__tmp_dir}/S1/{self.FILE_FMTS["cal_ok"]}'.format(**crt, tmp=tmp_suffix(tmp))
        return f'{self.__input_dir}/{s1dir}/{s1dir}.SAFE/measurement/{s1file}'

    def input_file_vv(self, idx) -> str:
        assert idx < 6
        return self.input_file(idx, polarity='vv')

    def tile_origins(self, tile_name) -> List[Tuple[float, float]]:
        origins = {
                '33NWB': [(14.9998201759, 1.8098185887), (15.9870050338, 1.8095484335), (15.9866155411, 0.8163071941), (14.9998202469, 0.8164290331000001)],
                }
        return origins[tile_name]

    def raster_vv(self, idx) -> dict:
        S2_tile_origin = self.tile_origins(self.TILE)
        s1dir  = self.FILES[idx]['s1dir']
        coverage = compute_coverage(FILES[idx]['polygon'], S2_tile_origin)
        logging.debug("coverage of %s by %s = %s", self.TILE, s1dir, coverage)
        return {
                'raster': S1DateAcquisition(f'{self.safe_dir(idx)}/manifest.safe', [self.input_file_vv(idx)]),
                'tile_origin': S2_tile_origin,
                'tile_coverage': coverage
                }

    def _find_image(self, manifest_path) -> int:
        manifest_path = str(manifest_path)  # manifest_path is either a str or a PosixPath
        for idx in range(len(self.FILES)):
            if self.FILES[idx]['s1dir'] in manifest_path:
                return idx
        raise AssertionError(f'{manifest_path} cannot be found in input list {[f["s1dir"] for f in self.FILES]}')

    def get_origin(self, id) -> Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float], Tuple[float,float], str]:
        """
        Mock alternative for Utils.get_origin
        """
        # str => id == manifest_path
        idx = id if isinstance(id, int) else self._find_image(id)
        assert idx < len(self.FILES)
        origin  = self.FILES[idx]['polygon'][1:]
        srsname = self.FILES[idx]['srsname']
        logging.debug('  mock.get_origin(%s) -> %s', self.FILES[idx]['s1dir'], origin)
        return *origin, srsname

    def get_orbit_direction(self, id) -> str:
        # str => id == manifest_path
        idx = id if isinstance(id, int) else self._find_image(id)
        assert idx < len(self.FILES)
        dir = self.FILES[idx]['orbit_direction']
        return dir

    def get_relative_orbit(self, id) -> int:
        # str => id == manifest_path
        idx = id if isinstance(id, int) else self._find_image(id)
        assert idx < len(self.FILES)
        dir = self.FILES[idx]['relative_orbit']
        return dir

    def cal_ok(self, idx, tmp, polarity='vv') -> str:
        crt = self.FILES[idx]
        return f'{self.__tmp_dir}/S1/{self.FILE_FMTS["cal_ok"]}'.format(**crt, tmp=tmp_suffix(tmp))

    def ortho_ready(self, idx, tmp, polarity='vv') -> str:
        crt = self.FILES[idx]
        return f'{self.__tmp_dir}/S1/{self.FILE_FMTS["ortho_ready"]}'.format(**crt, tmp=tmp_suffix(tmp))

    def orthofile(self, idx, tmp, polarity='vv', calibration='_sigma') -> str:
        crt = self.FILES[idx]
        ext = self.extended_geom_compress if tmp else ''
        return f'{self.__tmp_dir}/S2/{self.__tile}/{self.FILE_FMTS["orthofile"]}.tif{ext}'.format(**crt, tmp=tmp_suffix(tmp), calibration=calibration).format(polarity=polarity)

    def _concatfile_for_all(self, crt, tmp, polarity, calibration) -> str:
        if tmp or (calibration == '_beta'):
            # logging.error('concatfile_for_all(tmp=%s, calibration=%s) ==> TMP', tmp, calibration)
            dir = f'{self.__tmp_dir}/S2/{self.__tile}'
        else:
            # logging.error('concatfile_for_all(tmp=%s, calibration=%s) ==> OUT', tmp, calibration)
            dir = f'{self.__output_dir}/{self.__tile}'
        if tmp:
            ext = self.extended_compress
        else:
            ext = ''
        return f'{dir}/{self.FILE_FMTS["orthofile"]}.tif{ext}'.format(**crt, tmp=tmp_suffix(tmp), calibration=calibration).format(polarity=polarity, nr="001" if polarity == "vv" else "002")
    def concatfile_from_one(self, idx, tmp, polarity='vv', calibration='_sigma') -> str:
        crt = self.FILES[idx]
        return self._concatfile_for_all(crt, tmp, polarity, calibration)
    def concatfile_from_two(self, idx, tmp, polarity='vv', calibration='_sigma') -> str:
        crt = self.CONCATS[idx]
        return self._concatfile_for_all(crt, tmp, polarity, calibration)

    def _masktmp_for_all(self, crt, tmp, polarity, calibration) -> str:
        dir = f'{self.__tmp_dir}/S2/{self.__tile}'
        return f'{dir}/{self.FILE_FMTS["border_mask_tmp"]}.tif'.format(**crt, tmp=tmp_suffix(tmp), calibration=calibration).format(polarity=polarity)
    def masktmp_from_one(self, idx, tmp, polarity='vv', calibration='_sigma') -> str:
        crt = self.FILES[idx]
        return self._masktmp_for_all(crt, tmp, polarity, calibration)
    def masktmp_from_two(self, idx, tmp, polarity='vv', calibration='_sigma') -> str:
        crt = self.CONCATS[idx]
        return self._masktmp_for_all(crt, tmp, polarity, calibration)

    def _maskfile_for_all(self, crt, tmp, polarity, calibration) -> str:
        if tmp:
            dir = f'{self.__tmp_dir}/S2/{self.__tile}'
        else:
            dir = f'{self.__output_dir}/{self.__tile}'
        return f'{dir}/{self.FILE_FMTS["border_mask"]}'.format(**crt, tmp=tmp_suffix(tmp), calibration=calibration).format(polarity=polarity)
    def maskfile_from_one(self, idx, tmp, polarity='vv', calibration='_sigma') -> str:
        crt = self.FILES[idx]
        return self._maskfile_for_all(crt, tmp, polarity, calibration)
    def maskfile_from_two(self, idx, tmp, polarity='vv', calibration='_sigma') -> str:
        crt = self.CONCATS[idx]
        return self._maskfile_for_all(crt, tmp, polarity, calibration)

    def dem_file(self) -> str:
        return f'{self.__tmp_dir}/TMP'

    def vrtfile(self, idx, tmp) -> str:
        crt = self.FILES[idx]
        return f'{self.__tmp_dir}/S1/{self.FILE_FMTS["vrt"]}'.format(**crt, tmp=tmp_suffix(tmp))
    def dem_coverage(self, idx) -> List[str]:
        return self.FILES[idx]['dem_coverage']
    def sardemprojfile(self, idx, tmp) -> str:
        crt = self.FILES[idx]
        return f'{self.__tmp_dir}/S1/{self.FILE_FMTS["sardemprojfile"]}'.format(**crt, tmp=tmp_suffix(tmp))
    def xyzfile(self, idx, tmp) -> str:
        crt = self.FILES[idx]
        return f'{self.__tmp_dir}/S1/{self.FILE_FMTS["xyzfile"]}'.format(**crt, tmp=tmp_suffix(tmp))
    def normalsfile(self, idx, tmp) -> str:
        crt = self.FILES[idx]
        return f'{self.__tmp_dir}/S1/{self.FILE_FMTS["normalsfile"]}'.format(**crt, tmp=tmp_suffix(tmp))
    def LIAfile(self, idx, tmp) -> str:
        crt = self.FILES[idx]
        return f'{self.__tmp_dir}/S1/{self.FILE_FMTS["LIAfile"]}'.format(**crt, tmp=tmp_suffix(tmp))
    def sinLIAfile(self, idx, tmp) -> str:
        crt = self.FILES[idx]
        return f'{self.__tmp_dir}/S1/{self.FILE_FMTS["sinLIAfile"]}'.format(**crt, tmp=tmp_suffix(tmp))

    def orthoLIAfile(self, idx, tmp) -> str:
        crt = self.FILES[idx]
        ext = self.extended_geom_compress if tmp else ''
        return f'{self.__tmp_dir}/S2/{self.__tile}/{self.FILE_FMTS["orthoLIAfile"]}.tif{ext}'.format(**crt, tmp=tmp_suffix(tmp))

    def orthosinLIAfile(self, idx, tmp) -> str:
        crt = self.FILES[idx]
        ext = self.extended_geom_compress if tmp else ''
        return f'{self.__tmp_dir}/S2/{self.__tile}/{self.FILE_FMTS["orthosinLIAfile"]}.tif{ext}'.format(**crt, tmp=tmp_suffix(tmp))

    def _concatLIAfile_for_all(self, crt, tmp) -> str:
        dir = f'{self.__tmp_dir}/S2/{self.__tile}'
        ext = self.extended_compress if tmp else ''
        return f'{dir}/{self.FILE_FMTS["orthoLIAfile"]}.tif{ext}'.format(**crt, tmp=tmp_suffix(tmp))
    def concatLIAfile_from_one(self, idx, tmp) -> str:
        crt = self.FILES[idx]
        return self._concatLIAfile_for_all(crt, tmp)
    def concatLIAfile_from_two(self, idx, tmp) -> str:
        crt = self.CONCATS[idx]
        return self._concatLIAfile_for_all(crt, tmp)

    def _concatsinLIAfile_for_all(self, crt, tmp) -> str:
        dir = f'{self.__tmp_dir}/S2/{self.__tile}'
        ext = self.extended_compress if tmp else ''
        return f'{dir}/{self.FILE_FMTS["orthosinLIAfile"]}.tif{ext}'.format(**crt, tmp=tmp_suffix(tmp))
    def concatsinLIAfile_from_one(self, idx, tmp) -> str:
        crt = self.FILES[idx]
        return self._concatsinLIAfile_for_all(crt, tmp)
    def concatsinLIAfile_from_two(self, idx, tmp) -> str:
        crt = self.CONCATS[idx]
        return self._concatsinLIAfile_for_all(crt, tmp)

    def selectedLIAfile(self) -> str:
        return f'{self.__lia_dir}/LIA_s1a_33NWB_DES_007.tif'

    def selectedsinLIAfile(self) -> str:
        return f'{self.__lia_dir}/sin_LIA_s1a_33NWB_DES_007.tif'

    def _sigma0_normlim_file_for_all(self, crt, tmp, polarity) -> str:
        if tmp:
            dir = f'{self.__tmp_dir}/S2/{self.__tile}'
            ext = self.extended_compress
        else:
            dir = f'{self.__output_dir}/{self.__tile}'
            ext = ''
        return f'{dir}/{self.FILE_FMTS["sigma0_normlim_file"]}.tif{ext}'.format(**crt, tmp=tmp_suffix(tmp)).format(polarity=polarity)
    def sigma0_normlim_file_from_one(self, idx, tmp, polarity='vv') -> str:
        crt = self.FILES[idx]
        return self._sigma0_normlim_file_for_all(crt, tmp, polarity)

    def sigma0_normlim_file_from_two(self, idx, tmp, polarity='vv') -> str:
        crt = self.CONCATS[idx]
        return self._sigma0_normlim_file_for_all(crt, tmp, polarity)

    # def geoid_file(self):
    #     return f'resources/Geoid/egm96.grd'
