#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import fnmatch
import logging
import os
import pathlib
import shutil
import subprocess

import otbApplication as otb

from unittest.mock import patch

# import pytest_check
from .helpers import otb_compare, comparable_metadata
from .mock_otb import OTBApplicationsMockContext, isfile, isdir, list_dirs, glob, dirname, makedirs
import s1tiling.S1Processor
import s1tiling.libs.configuration

from s1tiling.libs.otbpipeline import _fetch_input_data


# ======================================================================
# Full processing versions
# ======================================================================

def remove_dirs(dir_list):
    for dir in dir_list:
        if os.path.isdir(dir):
            logging.info("rm -r '%s'", dir)
            shutil.rmtree(dir)


def process(tmpdir, outputdir, baseline_reference_outputs, test_file, watch_ram, dirs_to_clean=None):
    '''
    Executes the S1Processor
    '''
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    src_dir       = crt_dir.parent.absolute()
    dirs_to_clean = dirs_to_clean or [outputdir, tmpdir/'S1', tmpdir/'S2']

    logging.info('$S1TILING_TEST_DATA_INPUT  -> %s', os.environ['S1TILING_TEST_DATA_INPUT'])
    logging.info('$S1TILING_TEST_DATA_OUTPUT -> %s', os.environ['S1TILING_TEST_DATA_OUTPUT'])
    logging.info('$S1TILING_TEST_SRTM        -> %s', os.environ['S1TILING_TEST_SRTM'])
    logging.info('$S1TILING_TEST_TMPDIR      -> %s', os.environ['S1TILING_TEST_TMPDIR'])
    logging.info('$S1TILING_TEST_DOWNLOAD    -> %s', os.environ['S1TILING_TEST_DOWNLOAD'])
    logging.info('$S1TILING_TEST_RAM         -> %s', os.environ['S1TILING_TEST_RAM'])

    remove_dirs(dirs_to_clean)

    args = ['python3', src_dir / 's1tiling/S1Processor.py', test_file]
    if watch_ram:
        args.append('--watch-ram')
    # args.append('--cache-before-ortho')
    logging.info('Running: %s', args)
    return subprocess.call(args, cwd=crt_dir)


def test_33NWB_202001_NR_execute_OTB(baselinedir, outputdir, tmpdir, srtmdir, ram, download, watch_ram):
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)
    # In all cases, the baseline is required for the reference outputs
    # => We need it
    assert os.path.isdir(baselinedir), \
        ("No baseline found in '%s', please run minio-client to fetch it with:\n"+\
        "?> mc cp --recursive minio-otb/s1-tiling/baseline '%s'") % (baselinedir, baselinedir.absolute(),)

    if download:
        inputdir                                   = (tmpdir/'data_raw').absolute()
        os.environ['S1TILING_TEST_DOWNLOAD']       = 'True'
        os.environ['S1TILING_TEST_OVERRIDE_CUT_Y'] = 'None' # Use default
    else:
        inputdir                                   = str((baselinedir/'inputs').absolute())
        os.environ['S1TILING_TEST_DOWNLOAD']       = 'False'
        os.environ['S1TILING_TEST_OVERRIDE_CUT_Y'] = 'False' # keep everything

    os.environ['S1TILING_TEST_DATA_INPUT']         = str(inputdir)
    os.environ['S1TILING_TEST_DATA_OUTPUT']        = str(outputdir.absolute())
    os.environ['S1TILING_TEST_SRTM']               = str(srtmdir.absolute())
    os.environ['S1TILING_TEST_TMPDIR']             = str(tmpdir.absolute())
    os.environ['S1TILING_TEST_RAM']                = str(ram)

    images = [
            '33NWB/s1a_33NWB_vh_DES_007_20200108txxxxxx.tif',
            '33NWB/s1a_33NWB_vv_DES_007_20200108txxxxxx.tif',
            '33NWB/s1a_33NWB_vh_DES_007_20200108txxxxxx_BorderMask.tif',
            '33NWB/s1a_33NWB_vv_DES_007_20200108txxxxxx_BorderMask.tif',
            ]
    baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    logging.info("Full test")
    EX = process(tmpdir, outputdir, baseline_path, test_file, watch_ram)
    assert EX == 0
    for im in images:
        expected = baseline_path / im
        produced = outputdir / im
        assert os.path.isfile(produced)
        assert otb_compare(expected, produced) == 0, \
                ("Comparison of %s against %s failed" % (produced, expected))
        assert comparable_metadata(expected) == comparable_metadata(produced)
    # The following line permits to test otb_compare correctly detect differences when
    # called from pytest.
    # assert otb_compare(baseline_path+images[0], result_path+images[1]) == 0


def test_33NWB_202001_NR_masks_only_execute_OTB(baselinedir, outputdir, tmpdir, srtmdir, ram, download, watch_ram):
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)
    # In all cases, the baseline is required for the reference outputs
    # => We need it
    assert os.path.isdir(baselinedir), \
        ("No baseline found in '%s', please run minio-client to fetch it with:\n"+\
        "?> mc cp --recursive minio-otb/s1-tiling/baseline '%s'") % (baselinedir, baselinedir.absolute(),)

    if download:
        inputdir                                   = (tmpdir/'data_raw').absolute()
        os.environ['S1TILING_TEST_DOWNLOAD']       = 'True'
        os.environ['S1TILING_TEST_OVERRIDE_CUT_Y'] = 'None' # Use default
    else:
        inputdir                                   = str((baselinedir/'inputs').absolute())
        os.environ['S1TILING_TEST_DOWNLOAD']       = 'False'
        os.environ['S1TILING_TEST_OVERRIDE_CUT_Y'] = 'False' # keep everything

    os.environ['S1TILING_TEST_DATA_INPUT']         = str(inputdir)
    os.environ['S1TILING_TEST_DATA_OUTPUT']        = str(outputdir.absolute())
    os.environ['S1TILING_TEST_SRTM']               = str(srtmdir.absolute())
    os.environ['S1TILING_TEST_TMPDIR']             = str(tmpdir.absolute())
    os.environ['S1TILING_TEST_RAM']                = str(ram)

    images = [
            '33NWB/s1a_33NWB_vh_DES_007_20200108txxxxxx_BorderMask.tif',
            '33NWB/s1a_33NWB_vv_DES_007_20200108txxxxxx_BorderMask.tif',
            ]
    baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'

    logging.info("Mask only test")
    # Fake remaining things to do
    remove_dirs([outputdir])
    os.makedirs(outputdir / '33NWB')
    start_points = [
            '33NWB/s1a_33NWB_vh_DES_007_20200108txxxxxx.tif',
            '33NWB/s1a_33NWB_vv_DES_007_20200108txxxxxx.tif'
            ]
    for pt in start_points:
        os.symlink(baseline_path/pt, outputdir/pt)


    dirs_to_clean = [tmpdir/'S1', tmpdir/'S2'] # do not clear outputdir in that case
    EX = process(tmpdir, outputdir, baseline_path, test_file, watch_ram, dirs_to_clean)
    assert EX == 0
    for im in images:
        expected = baseline_path / im
        produced = outputdir / im
        assert os.path.isfile(produced)
        assert otb_compare(expected, produced) == 0, \
                ("Comparison of %s against %s failed" % (produced, expected))
        assert comparable_metadata(expected) == comparable_metadata(produced)


# ======================================================================
# Mocked versions
# ======================================================================

def tmp_suffix(tmp):
    return '.tmp' if tmp else ''


class FileDB:
    FILE_FMTS = {
            's1file'              : '{s1_basename}.tiff',
            'cal_ok'              : '{s1_basename}{tmp}.tiff',
            'ortho_ready'         : '{s1_basename}_OrthoReady{tmp}.tiff',
            'orthofile'           : '{s2_basename}{tmp}',
            'sigma0_normlim_file' : '{s2_basename}_NormLim{tmp}',
            'border_mask_tmp'     : '{s2_basename}_BorderMaskTmp{tmp}.tif',
            'border_mask'         : '{s2_basename}_BorderMask{tmp}.tif',

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
                's1dir'               : 'S1A_IW_GRDH_1SDV_20200108T044150_20200108T044215_030704_038506_C7F5',
                's1_basename'         : 's1a-iw-grd-vv-20200108t044150-20200108t044215-030704-038506-001',
                's2_basename'         : 's1a_33NWB_vv_DES_007_20200108t044150',
                's1_polarless'        : 's1a-iw-grd-20200108t044150-20200108t044215-030704-038506',
                's2_polarless'        : 's1a_33NWB_DES_007_20200108t044150',
                'dem_coverage'        : ['N00E014', 'N00E015', 'N00E016', 'N01E014', 'N01E015', 'N01E016', 'N02E014', 'N02E015', 'N02E016'],
                'polygon'             : [(1.137156, 14.233953), (0.660935, 16.461103), (2.173307, 16.77552), (2.645077, 14.545785), (1.137156, 14.233953)],
                'orbit_direction'     : 'DES',
                'relative_orbit'      : 7,
                },
            {
                's1dir'               : 'S1A_IW_GRDH_1SDV_20200108T044215_20200108T044240_030704_038506_D953',
                's1_basename'         : 's1a-iw-grd-vv-20200108t044215-20200108t044240-030704-038506-001',
                's2_basename'         : 's1a_33NWB_vv_DES_007_20200108t044215',
                's1_polarless'        : 's1a-iw-grd-20200108t044215-20200108t044240-030704-038506',
                's2_polarless'        : 's1a_33NWB_DES_007_20200108t044215',
                'dem_coverage'        : ['N00E013', 'N00E014', 'N00E015', 'N00E016', 'N01E014', 'S01E013', 'S01E014', 'S01E015', 'S01E016'],
                'polygon'             : [(-0.370174, 13.917268), (-0.851051, 16.143845), (0.660845, 16.461084), (1.137179, 14.233407), (-0.370174, 13.917268)],
                'orbit_direction'     : 'DES',
                'relative_orbit'      : 7,
                },
            # 20 jan 2020
            {
                's1dir'               : 'S1A_IW_GRDH_1SDV_20200120T044149_20200120T044214_030879_038B2D_5671',
                's1_basename'         : 's1a-iw-grd-{polarity}-20200120t044149-20200120t044214-030879-038B2D-{nr}',
                's2_basename'         : 's1a_33NWB_{polarity}_DES_007_20200120t044149',
                's1_polarless'        : 's1a-iw-grd-20200120t044149-20200120t044214-030879-038B2D',
                's2_polarless'        : 's1a_33NWB_DES_007_20200120t044149',
                'dem_coverage'        : ['N00E014', 'N00E015', 'N00E016', 'N01E014', 'N01E015', 'N01E016', 'N02E014', 'N02E015', 'N02E016'],
                'polygon'             : [(1.137292, 14.233942), (0.661038, 16.461086), (2.173408, 16.775522), (2.645211, 14.545794), (1.137292, 14.233942)],
                'orbit_direction'     : 'DES',
                'relative_orbit'      : 7,
                },
            {
                's1dir'               : 'S1A_IW_GRDH_1SDV_20200120T044214_20200120T044239_030879_038B2D_FDB0',
                's1_basename'         : 's1a-iw-grd-{polarity}-20200120t044214-20200120t044239-030879-038B2D-{nr}',
                's2_basename'         : 's1a_33NWB_{polarity}_DES_007_20200120t044214',
                's1_polarless'        : 's1a-iw-grd-20200120t044214-20200120t044239-030879-038B2D',
                's2_polarless'        : 's1a_33NWB_DES_007_20200120t044214',
                'dem_coverage'        : ['N00E013', 'N00E014', 'N00E015', 'N00E016', 'N01E014', 'S01E013', 'S01E014', 'S01E015', 'S01E016'],
                'polygon'             : [(-0.370036, 13.917237), (-0.850946, 16.143806), (0.660948, 16.461067), (1.137315, 14.233396), (-0.370036, 13.917237)],
                'orbit_direction'     : 'DES',
                'relative_orbit'      : 7,
                },
            ]
    CONCATS = [
            # 08 jan 2020
            {
                's2_basename' :  's1a_33NWB_{polarity}_DES_007_20200108txxxxxx',
                's2_polarless': 's1a_33NWB_DES_007_20200108txxxxxx',
                'first_date'  : '2020-01-01',
                'last_date'   : '2020-01-10',
                },
            # 20 jan 2020
            {
                's2_basename' :  's1a_33NWB_{polarity}_DES_007_20200120txxxxxx',
                's2_polarless': 's1a_33NWB_DES_007_20200120txxxxxx',
                'first_date'  : '2020-01-10',
                'last_date'   : '2020-01-21',
                },
            ]
    TILE = '33NWB'
    extended_geom_compress = '?&writegeom=false&gdal:co:COMPRESS=DEFLATE'
    extended_compress      = '?&gdal:co:COMPRESS=DEFLATE'

    def __init__(self, inputdir, tmpdir, outputdir, tile, srtmdir, geoid_file):
        self.__input_dir      = inputdir
        self.__tmp_dir        = tmpdir
        self.__output_dir     = outputdir
        self.__tile           = tile
        self.__srtm_dir       = srtmdir
        self.__GeoidFile      = geoid_file

        NFiles   = len(self.FILES)
        NConcats = len(self.CONCATS)
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
        self.__tmp_to_out_map = {}
        for func, nb in names_to_map:
            for idx in range(nb):
                self.__tmp_to_out_map[func(idx, True)] = func(idx, False)

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
    def srtmdir(self):
        """
        Property srtmdir
        """
        return self.__srtm_dir

    @property
    def GeoidFile(self):
        """
        Property GeoidFile
        """
        return self.__GeoidFile

    def all_files(self):
        return [self.input_file(idx) for idx in range(len(self.FILES))]

    def safe_dir(self, idx):
        s1dir  = self.FILES[idx]['s1dir']
        return f'{self.__input_dir}/{s1dir}/{s1dir}.SAFE'

    def input_file(self, idx, polarity='vv'):
        crt    = self.FILES[idx]
        s1dir  = crt['s1dir']
        s1file = self.FILE_FMTS['s1file'].format(**crt).format(polarity='vv', nr="001" if polarity == "vv" else "002")
        return f'{self.__input_dir}/{s1dir}/{s1dir}.SAFE/measurement/{s1file}'

    def input_file_vv(self, idx):
        assert idx < 4
        return self.input_file(idx, polarity='vv')

    def tile_origins(self, tile_name):
        origins = {
                '33NWB': [(14.9998201759, 1.8098185887), (15.9870050338, 1.8095484335), (15.9866155411, 0.8163071941), (14.9998202469, 0.8164290331000001)],
                }
        return origins[tile_name]

    def raster_vv(self, idx):
        S2_tile_origin = self.tile_origins(self.TILE)
        s1dir  = self.FILES[idx]['s1dir']
        coverage = compute_coverage(FILES[idx]['polygon'], S2_tile_origin)
        logging.debug("coverage of %s by %s = %s", self.TILE, s1dir, coverage)
        return {
                'raster': S1DateAcquisition(f'{self.safe_dir(idx)}/manifest.safe', [self.input_file_vv(idx)]),
                'tile_origin': S2_tile_origin,
                'tile_coverage': coverage
                }

    def _find_image(self, manifest_path):
        manifest_path = str(manifest_path)  # manifest_path is either a str or a PosixPath
        for idx in range(len(self.FILES)):
            if self.FILES[idx]['s1dir'] in manifest_path:
                return idx
        raise AssertionError(f'{manifest_path} cannot be found in input list {[f["s1dir"] for f in self.FILES]}')

    def get_origin(self, manifest_path):
        """
        Mock alternative for Utils.get_origin
        """
        idx = self._find_image(manifest_path)
        origin = self.FILES[idx]['polygon'][1:]
        logging.debug('  mock.get_origin(%s) -> %s', self.FILES[idx]['s1dir'], origin)
        return origin

    def get_orbit_direction(self, manifest_path):
        idx = self._find_image(manifest_path)
        dir = self.FILES[idx]['orbit_direction']
        return dir

    def get_relative_orbit(self, manifest_path):
        idx = self._find_image(manifest_path)
        dir = self.FILES[idx]['relative_orbit']
        return dir

    def cal_ok(self, idx, tmp, polarity='vv'):
        crt = self.FILES[idx]
        return f'{self.__tmp_dir}/S1/{self.FILE_FMTS["cal_ok"]}'.format(**crt, tmp=tmp_suffix(tmp))

    def ortho_ready(self, idx, tmp, polarity='vv'):
        crt = self.FILES[idx]
        return f'{self.__tmp_dir}/S1/{self.FILE_FMTS["ortho_ready"]}'.format(**crt, tmp=tmp_suffix(tmp))

    def orthofile(self, idx, tmp, polarity='vv'):
        crt = self.FILES[idx]
        ext = self.extended_geom_compress if tmp else ''
        return f'{self.__tmp_dir}/S2/{self.__tile}/{self.FILE_FMTS["orthofile"]}.tif{ext}'.format(**crt, tmp=tmp_suffix(tmp)).format(polarity=polarity)

    def _concatfile_for_all(self, crt, tmp, polarity):
        if tmp:
            dir = f'{self.__tmp_dir}/S2/{self.__tile}'
            ext = self.extended_compress
        else:
            dir = f'{self.__output_dir}/{self.__tile}'
            ext = ''
        return f'{dir}/{self.FILE_FMTS["orthofile"]}.tif{ext}'.format(**crt, tmp=tmp_suffix(tmp)).format(polarity='vv', nr="001" if polarity == "vv" else "002")
    def concatfile_from_one(self, idx, tmp, polarity='vv'):
        crt = self.FILES[idx]
        return self._concatfile_for_all(crt, tmp, polarity)
    def concatfile_from_two(self, idx, tmp, polarity='vv'):
        crt = self.CONCATS[idx]
        return self._concatfile_for_all(crt, tmp, polarity)

    def _masktmp_for_all(self, crt, tmp, polarity):
        dir = f'{self.__tmp_dir}/S2/{self.__tile}'
        return f'{dir}/{self.FILE_FMTS["border_mask_tmp"]}.tif'.format(**crt, tmp=tmp_suffix(tmp)).format(polarity=polarity)
    def masktmp_from_one(self, idx, tmp, polarity='vv'):
        crt = self.FILES[idx]
        return self._masktmp_for_all(crt, tmp, polarity)
    def masktmp_from_two(self, idx, tmp, polarity='vv'):
        crt = self.CONCATS[idx]
        return self._masktmp_for_all(crt, tmp, polarity)

    def _maskfile_for_all(self, crt, tmp, polarity):
        if tmp:
            dir = f'{self.__tmp_dir}/S2/{self.__tile}'
        else:
            dir = f'{self.__output_dir}/{self.__tile}'
        return f'{dir}/{self.FILE_FMTS["border_mask"]}'.format(**crt, tmp=tmp_suffix(tmp)).format(polarity=polarity)
    def maskfile_from_one(self, idx, tmp, polarity='vv'):
        crt = self.FILES[idx]
        return self._maskfile_for_all(crt, tmp, polarity)
    def maskfile_from_two(self, idx, tmp, polarity='vv'):
        crt = self.CONCATS[idx]
        return self._maskfile_for_all(crt, tmp, polarity)

    def dem_file(self):
        return f'{self.__tmp_dir}/TMP'

    def vrtfile(self, idx, tmp):
        crt = self.FILES[idx]
        return f'{self.__tmp_dir}/S1/{self.FILE_FMTS["vrt"]}'.format(**crt, tmp=tmp_suffix(tmp))
    def dem_coverage(self, idx):
        return self.FILES[idx]['dem_coverage']
    def sardemprojfile(self, idx, tmp):
        crt = self.FILES[idx]
        return f'{self.__tmp_dir}/S1/{self.FILE_FMTS["sardemprojfile"]}'.format(**crt, tmp=tmp_suffix(tmp))
    def xyzfile(self, idx, tmp):
        crt = self.FILES[idx]
        return f'{self.__tmp_dir}/S1/{self.FILE_FMTS["xyzfile"]}'.format(**crt, tmp=tmp_suffix(tmp))
    def normalsfile(self, idx, tmp):
        crt = self.FILES[idx]
        return f'{self.__tmp_dir}/S1/{self.FILE_FMTS["normalsfile"]}'.format(**crt, tmp=tmp_suffix(tmp))
    def LIAfile(self, idx, tmp):
        crt = self.FILES[idx]
        return f'{self.__tmp_dir}/S1/{self.FILE_FMTS["LIAfile"]}'.format(**crt, tmp=tmp_suffix(tmp))
    def sinLIAfile(self, idx, tmp):
        crt = self.FILES[idx]
        return f'{self.__tmp_dir}/S1/{self.FILE_FMTS["sinLIAfile"]}'.format(**crt, tmp=tmp_suffix(tmp))

    def orthoLIAfile(self, idx, tmp):
        crt = self.FILES[idx]
        ext = self.extended_geom_compress if tmp else ''
        return f'{self.__tmp_dir}/S2/{self.__tile}/{self.FILE_FMTS["orthoLIAfile"]}.tif{ext}'.format(**crt, tmp=tmp_suffix(tmp))

    def orthosinLIAfile(self, idx, tmp):
        crt = self.FILES[idx]
        ext = self.extended_geom_compress if tmp else ''
        return f'{self.__tmp_dir}/S2/{self.__tile}/{self.FILE_FMTS["orthosinLIAfile"]}.tif{ext}'.format(**crt, tmp=tmp_suffix(tmp))

    def _concatLIAfile_for_all(self, crt, tmp):
        dir = f'{self.__tmp_dir}/S2/{self.__tile}'
        ext = self.extended_compress if tmp else ''
        return f'{dir}/{self.FILE_FMTS["orthoLIAfile"]}.tif{ext}'.format(**crt, tmp=tmp_suffix(tmp))
    def concatLIAfile_from_one(self, idx, tmp):
        crt = self.FILES[idx]
        return self._concatLIAfile_for_all(crt, tmp)
    def concatLIAfile_from_two(self, idx, tmp):
        crt = self.CONCATS[idx]
        return self._concatLIAfile_for_all(crt, tmp)

    def _concatsinLIAfile_for_all(self, crt, tmp):
        dir = f'{self.__tmp_dir}/S2/{self.__tile}'
        ext = self.extended_compress if tmp else ''
        return f'{dir}/{self.FILE_FMTS["orthosinLIAfile"]}.tif{ext}'.format(**crt, tmp=tmp_suffix(tmp))
    def concatsinLIAfile_from_one(self, idx, tmp):
        crt = self.FILES[idx]
        return self._concatsinLIAfile_for_all(crt, tmp)
    def concatsinLIAfile_from_two(self, idx, tmp):
        crt = self.CONCATS[idx]
        return self._concatsinLIAfile_for_all(crt, tmp)

    def selectedsinLIAfile(self):
        return f'{self.__output_dir}/{self.__tile}/sin_LIA_s1a_33NWB_DES_007.tif'

    def _sigma0_normlim_file_for_all(self, crt, tmp, polarity):
        if tmp:
            dir = f'{self.__tmp_dir}/S2/{self.__tile}'
            ext = self.extended_compress
        else:
            dir = f'{self.__output_dir}/{self.__tile}'
            ext = ''
        return f'{dir}/{self.FILE_FMTS["sigma0_normlim_file"]}.tif{ext}'.format(**crt, tmp=tmp_suffix(tmp)).format(polarity=polarity)
    def sigma0_normlim_file_from_one(self, idx, tmp, polarity='vv'):
        crt = self.FILES[idx]
        return self._sigma0_normlim_file_for_all(crt, tmp, polarity)

    def sigma0_normlim_file_from_two(self, idx, tmp, polarity='vv'):
        crt = self.CONCATS[idx]
        return self._sigma0_normlim_file_for_all(crt, tmp, polarity)

    # def geoid_file(self):
    #     return f'resources/Geoid/egm96.grd'


def _declare_know_files(mocker, known_files, known_dirs, patterns, file_db):
    # logging.debug('_declare_know_files(%s)', patterns)
    all_files = file_db.all_files()
    # logging.debug('- all_files: %s', all_files)
    files = []
    for pattern in patterns:
        files += [fn for fn in all_files if fnmatch.fnmatch(fn, '*'+pattern+'*')]
    known_files.extend(files)
    known_dirs.update([dirname(fn, 3) for fn in known_files])
    known_dirs.update([dirname(fn, 2) for fn in known_files])
    # known_dirs.update([dirname(fn, 1) for fn in known_files])
    logging.debug('Mocking w/ %s --> %s', patterns, files)
    # Utils.list_dirs has been imported in S1FileManager. This is the one that needs patching!
    mocker.patch('s1tiling.libs.S1FileManager.list_dirs', lambda dir, pat : list_dirs(dir, pat, known_dirs, file_db.inputdir))
    mocker.patch('glob.glob',        lambda pat  : glob(pat, known_files))
    mocker.patch('os.path.isfile',   lambda file : isfile(file, known_files))
    mocker.patch('os.path.isdir',    lambda dir  : isdir(dir, known_dirs))
    mocker.patch('os.makedirs',      lambda dir, **kw  : makedirs(dir, known_dirs))
    mocker.patch('os.path.getctime', lambda file : 0)
    # TODO: Test written meta data as well
    mocker.patch('s1tiling.libs.otbwrappers.OrthoRectify.add_ortho_metadata',    lambda slf, mt, app : True)
    mocker.patch('s1tiling.libs.otbwrappers.OrthoRectifyLIA.add_ortho_metadata', lambda slf, mt, app : True)
    mocker.patch('s1tiling.libs.otbpipeline.commit_execution',                   lambda tmp, out : True)
    mocker.patch('s1tiling.libs.Utils.get_origin',          lambda manifest : file_db.get_origin(manifest))
    mocker.patch('s1tiling.libs.Utils.get_orbit_direction', lambda manifest : file_db.get_orbit_direction(manifest))
    mocker.patch('s1tiling.libs.Utils.get_relative_orbit',  lambda manifest : file_db.get_relative_orbit(manifest))

    def mock_commit_execution_for_SelectLIA(inp, out):
        logging.debug('mock.mv %s %s', inp, out)
        assert os.path.isfile(inp)
        known_files.append(out)
        known_files.remove(inp)
    mocker.patch('s1tiling.libs.otbwrappers.commit_execution', mock_commit_execution_for_SelectLIA)

    def mock_add_image_metadata(slf, mt, *args, **kwargs):
        # TODO: Problem: how can we pass around meta from different pipelines???
        fullpath = mt.get('out_filename')
        logging.debug('Mock Set metadata in %s', fullpath)
        assert 'inputs' in mt, f'Looking for "inputs" in {mt.keys()}'
        inputs = mt['inputs']
        # indem = _fetch_input_data('indem', inputs)
        # assert 'srtms' in indem.meta, f"Metadata don't contain 'srtms', only: {indem.meta.keys()}"
        assert 'srtms' in mt, f"Metadata don't contain 'srtms', only: {mt.keys()}"
        return mt
    mocker.patch('s1tiling.libs.otbwrappers.SARDEMProjection.add_image_metadata', mock_add_image_metadata)

    def mock_direction_to_scan(slf, meta):
        logging.debug('Mocking direction to scan')
        meta['directiontoscandeml'] = 12
        meta['directiontoscandemc'] = 24
        meta['gain']                = 42
        return meta
    mocker.patch('s1tiling.libs.otbwrappers.SARCartesianMeanEstimation.fetch_direction', lambda slf, ip, mt : mock_direction_to_scan(slf, mt))


def set_environ_mocked(inputdir, outputdir, srtmdir, tmpdir, ram):
    os.environ['S1TILING_TEST_DOWNLOAD']       = 'False'
    os.environ['S1TILING_TEST_OVERRIDE_CUT_Y'] = 'False' # keep everything

    os.environ['S1TILING_TEST_DATA_INPUT']         = str(inputdir)
    os.environ['S1TILING_TEST_DATA_OUTPUT']        = str(outputdir.absolute())
    os.environ['S1TILING_TEST_SRTM']               = str(srtmdir.absolute())
    os.environ['S1TILING_TEST_TMPDIR']             = str(tmpdir.absolute())
    os.environ['S1TILING_TEST_RAM']                = str(ram)


def mock_upto_concat_S2(application_mocker, file_db, calibration, N=2):
    raw_calibration = 'beta' if calibration == 'normlim' else calibration
    for i in range(N):
        input_file = file_db.input_file_vv(i)
        expected_ortho_file = file_db.orthofile(i, False)

        application_mocker.set_expectations('SARCalibration', {
            'ram'        : '2048',
            'in'         : input_file,
            'lut'        : raw_calibration,
            'removenoise': False,
            'out'        : 'ResetMargin|>OrthoRectification|>'+file_db.orthofile(i, True),
            }, None)

        application_mocker.set_expectations('ResetMargin', {
            'in'               : input_file+'|>SARCalibration',
            'ram'              : '2048',
            'threshold.x'      : 1000,
            'threshold.y.start': 0,
            'threshold.y.end'  : 0,
            'mode'             : 'threshold',
            'out'              : 'OrthoRectification|>'+file_db.orthofile(i, True),
            }, None)

        application_mocker.set_expectations('OrthoRectification', {
            'io.in'           : input_file+'|>SARCalibration|>ResetMargin',
            'opt.ram'         : '2048',
            'interpolator'    : 'nn',
            'outputs.spacingx': 10.0,
            'outputs.spacingy': -10.0,
            'outputs.sizex'   : 10980,
            'outputs.sizey'   : 10980,
            'opt.gridspacing' : 40.0,
            'map'             : 'utm',
            'map.utm.zone'    : 33,
            'map.utm.northhem': True,
            'outputs.ulx'     : 499979.99999484676,
            'outputs.uly'     : 200040.0000009411,
            'elev.dem'        : file_db.dem_file(),
            'elev.geoid'      : file_db.GeoidFile,
            'io.out'          : file_db.orthofile(i, True),
            }, None)

    for i in range(N//2):
        application_mocker.set_expectations('Synthetize', {
            'ram'      : '2048',
            'il'       : [file_db.orthofile(2*i, False), file_db.orthofile(2*i+1, False)],
            'out'      : file_db.concatfile_from_two(i, True),
            }, None)


def mock_masking(application_mocker, file_db, calibration, N=2):
    if calibration == 'normlim':
        infile = file_db.sigma0_normlim_file_from_two
    else:
        infile = file_db.concatfile_from_two

    for i in range(N // 2):
        application_mocker.set_expectations('BandMath', {
            'ram'      : '2048',
            'il'       : [infile(i, False)],
            'exp'      : 'im1b1==0?0:1',
            'out'      : 'BinaryMorphologicalOperation|>'+file_db.maskfile_from_two(i, True),
            }, {'out': otb.ImagePixelType_uint8})
        application_mocker.set_expectations('BinaryMorphologicalOperation', {
            'in'       : [infile(i, False)+'|>BandMath'],
            'ram'      : '2048',
            'structype': 'ball',
            'xradius'  : 5,
            'yradius'  : 5,
            'filter'   : 'opening',
            'out'      : file_db.maskfile_from_two(i, True),
            }, {'out': otb.ImagePixelType_uint8})


def mock_LIA(application_mocker, file_db):
    srtmdir = file_db.srtmdir
    for idx in range(2):
        cov               = file_db.dem_coverage(idx)
        exp_srtm_names    = sorted(cov)
        exp_out_vrt       = file_db.vrtfile(idx, False)
        exp_out_dem       = file_db.sardemprojfile(idx, False)
        exp_in_srtm_files = [f"{srtmdir}/{srtm}.hgt" for srtm in exp_srtm_names]

        application_mocker.set_expectations('gdalbuildvrt', [file_db.vrtfile(idx, True)] + exp_in_srtm_files, None)

        application_mocker.set_expectations('SARDEMProjection', {
            'ram'        : '2048',
            'insar'      : file_db.input_file_vv(idx),
            'indem'      : exp_out_vrt,
            'withxyz'    : True,
            'nodata'     : -32768,
            'out'        : file_db.sardemprojfile(idx, True),
            }, None)

        application_mocker.set_expectations('SARCartesianMeanEstimation', {
            'ram'             : '2048',
            'insar'           : file_db.input_file_vv(idx),
            'indem'           : exp_out_vrt,
            'indemproj'       : exp_out_dem,
            'indirectiondemc' : 24,
            'indirectiondeml' : 12,
            'mlran'           : 1,
            'mlazi'           : 1,
            'out'             : file_db.xyzfile(idx, True),
            }, None)

        application_mocker.set_expectations('ExtractNormalVector', {
            'ram'             : '2048',
            'xyz'             : file_db.xyzfile(idx, False),
            'out'             : 'SARComputeLocalIncidenceAngle|>'+file_db.LIAfile(idx, True),
            }, None)

        application_mocker.set_expectations('SARComputeLocalIncidenceAngle', {
            'ram'             : '2048',
            'in.normals'      : file_db.xyzfile(idx, False)+'|>ExtractNormalVector', #'ComputeNormals|>'+file_db.normalsfile(idx),
            'in.xyz'          : file_db.xyzfile(idx, False),
            'out.lia'         : file_db.LIAfile(idx, True),
            'out.sin'         : file_db.sinLIAfile(idx, True),
            }, None)

        application_mocker.set_expectations('OrthoRectification', {
            'opt.ram'         : '2048',
            'io.in'           : file_db.LIAfile(idx, False),
            'interpolator'    : 'nn',
            'outputs.spacingx': 10.0,
            'outputs.spacingy': -10.0,
            'outputs.sizex'   : 10980,
            'outputs.sizey'   : 10980,
            'opt.gridspacing' : 40.0,
            'map'             : 'utm',
            'map.utm.zone'    : 33,
            'map.utm.northhem': True,
            'outputs.ulx'     : 499979.99999484676,
            'outputs.uly'     : 200040.0000009411,
            'elev.dem'        : file_db.dem_file(),
            'elev.geoid'      : file_db.GeoidFile,
            'io.out'          : file_db.orthoLIAfile(idx, True),
            }, None)

        application_mocker.set_expectations('OrthoRectification', {
            'opt.ram'         : '2048',
            'io.in'           : file_db.sinLIAfile(idx, False),
            'interpolator'    : 'nn',
            'outputs.spacingx': 10.0,
            'outputs.spacingy': -10.0,
            'outputs.sizex'   : 10980,
            'outputs.sizey'   : 10980,
            'opt.gridspacing' : 40.0,
            'map'             : 'utm',
            'map.utm.zone'    : 33,
            'map.utm.northhem': True,
            'outputs.ulx'     : 499979.99999484676,
            'outputs.uly'     : 200040.0000009411,
            'elev.dem'        : file_db.dem_file(),
            'elev.geoid'      : file_db.GeoidFile,
            'io.out'          : file_db.orthosinLIAfile(idx, True),
            }, None)

    # endfor on 2 consecutive images

    application_mocker.set_expectations('Synthetize', {
        'ram'      : '2048',
        'il'       : [file_db.orthoLIAfile(0, False), file_db.orthoLIAfile(1, False)],
        'out'      : file_db.concatLIAfile_from_two(0, True),
        }, None)

    application_mocker.set_expectations('Synthetize', {
        'ram'      : '2048',
        'il'       : [file_db.orthosinLIAfile(0, False), file_db.orthosinLIAfile(1, False)],
        'out'      : file_db.concatsinLIAfile_from_two(0, True),
        }, None)


def test_33NWB_202001_NR_core_mocked(baselinedir, outputdir, tmpdir, srtmdir, ram, download, watch_ram, mocker):
    """
    Mocked test of production of S2 sigma0 calibrated images.
    """
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)

    inputdir = str((baselinedir/'inputs').absolute())
    set_environ_mocked(inputdir, outputdir, srtmdir, tmpdir, ram)

    baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    configuration = s1tiling.libs.configuration.Configuration(test_file)
    logging.info("Full mocked test")

    file_db = FileDB(inputdir, tmpdir.absolute(), outputdir.absolute(), '33NWB', srtmdir, configuration.GeoidFile)
    mocker.patch('s1tiling.libs.otbwrappers.otb_version', lambda : '7.4.0')

    application_mocker = OTBApplicationsMockContext(configuration, mocker, file_db.tmp_to_out_map)
    known_files = application_mocker.known_files
    known_dirs = set()
    _declare_know_files(mocker, known_files, known_dirs, ['vv'], file_db)
    assert os.path.isfile(file_db.input_file_vv(0))  # Check mocking
    assert os.path.isfile(file_db.input_file_vv(1))
    mock_upto_concat_S2(application_mocker, file_db, 'sigma')
    mock_masking(application_mocker, file_db, 'sigma')
    s1tiling.S1Processor.s1_process(config_opt=configuration, searched_items_per_page=0,
            dryrun=False, debug_otb=True, watch_ram=False,
            debug_tasks=False, cache_before_ortho=False)
    application_mocker.assert_all_have_been_executed()


def test_33NWB_202001_lia_mocked(baselinedir, outputdir, tmpdir, srtmdir, ram, download, watch_ram, mocker):
    """
    Mocked test of production of LIA and sin LIA files
    """
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)

    inputdir = str((baselinedir/'inputs').absolute())

    set_environ_mocked(inputdir, outputdir, srtmdir, tmpdir, ram)

    tile_name = '33NWB'
    baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    configuration = s1tiling.libs.configuration.Configuration(test_file)
    configuration.calibration_type = 'normlim'
    logging.info("Sigma0 NORMLIM mocked test")

    file_db = FileDB(inputdir, tmpdir.absolute(), outputdir.absolute(), tile_name, srtmdir, configuration.GeoidFile)
    mocker.patch('s1tiling.libs.otbwrappers.otb_version', lambda : '7.4.0')

    application_mocker = OTBApplicationsMockContext(configuration, mocker, file_db.tmp_to_out_map)
    known_files = application_mocker.known_files
    known_dirs = set()
    _declare_know_files(mocker, known_files, known_dirs, ['vv'], file_db)
    assert os.path.isfile(file_db.input_file_vv(0))  # Check mocking
    assert os.path.isfile(file_db.input_file_vv(1))

    mock_LIA(application_mocker, file_db)

    s1tiling.S1Processor.s1_process_lia(config_opt=configuration, searched_items_per_page=0,
            dryrun=False, debug_otb=True, watch_ram=False,
            debug_tasks=False)
    application_mocker.assert_all_have_been_executed()


def test_33NWB_202001_normlim_mocked_one_date(baselinedir, outputdir, tmpdir, srtmdir, ram, download, watch_ram, mocker):
    """
    Mocked test of production of S2 normlim calibrated images.
    """
    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)

    inputdir = str((baselinedir/'inputs').absolute())

    set_environ_mocked(inputdir, outputdir, srtmdir, tmpdir, ram)

    tile_name = '33NWB'
    baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    configuration = s1tiling.libs.configuration.Configuration(test_file, do_show_configuration=False)
    configuration.calibration_type = 'normlim'
    configuration.show_configuration()
    logging.info("Sigma0 NORMLIM mocked test")

    file_db = FileDB(inputdir, tmpdir.absolute(), outputdir.absolute(), tile_name, srtmdir, configuration.GeoidFile)
    mocker.patch('s1tiling.libs.otbwrappers.otb_version', lambda : '7.4.0')

    application_mocker = OTBApplicationsMockContext(configuration, mocker, file_db.tmp_to_out_map)
    known_files = application_mocker.known_files
    known_dirs = set()
    _declare_know_files(mocker, known_files, known_dirs, ['vv'], file_db)
    assert os.path.isfile(file_db.input_file_vv(0))  # Check mocking
    assert os.path.isfile(file_db.input_file_vv(1))

    mock_upto_concat_S2(application_mocker, file_db, 'normlim')
    mock_LIA(application_mocker, file_db)
    mock_masking(application_mocker, file_db, 'normlim')

    application_mocker.set_expectations('BandMath', {
        'ram'      : '2048',
        'il'       : [file_db.concatfile_from_two(0, False), file_db.selectedsinLIAfile()],
        'exp'      : 'im1b1*im2b1',
        'out'      : file_db.sigma0_normlim_file_from_two(0, True),
        }, None)

    s1tiling.S1Processor.s1_process(config_opt=configuration, searched_items_per_page=0,
            dryrun=False, debug_otb=True, watch_ram=False,
            debug_tasks=False)
    application_mocker.assert_all_have_been_executed()


def test_33NWB_202001_normlim_mocked_all_dates(baselinedir, outputdir, tmpdir, srtmdir, ram, download, watch_ram, mocker):
    """
    Mocked test of production of S2 normlim calibrated images.
    """
    number_dates = 2

    crt_dir       = pathlib.Path(__file__).parent.absolute()
    logging.info("Baseline expected in '%s'", baselinedir)

    inputdir = str((baselinedir/'inputs').absolute())

    set_environ_mocked(inputdir, outputdir, srtmdir, tmpdir, ram)

    tile_name = '33NWB'
    baseline_path = baselinedir / 'expected'
    test_file     = crt_dir / 'test_33NWB_202001.cfg'
    configuration = s1tiling.libs.configuration.Configuration(test_file, do_show_configuration=False)
    configuration.calibration_type = 'normlim'
    logging.info("Sigma0 NORMLIM mocked test")

    file_db = FileDB(inputdir, tmpdir.absolute(), outputdir.absolute(), tile_name, srtmdir, configuration.GeoidFile)
    configuration.first_date = file_db.CONCATS[0]['first_date']
    configuration.last_date  = file_db.CONCATS[number_dates-1]['last_date']
    configuration.show_configuration()

    mocker.patch('s1tiling.libs.otbwrappers.otb_version', lambda : '7.4.0')

    application_mocker = OTBApplicationsMockContext(configuration, mocker, file_db.tmp_to_out_map)
    known_files = application_mocker.known_files
    known_dirs = set()
    _declare_know_files(mocker, known_files, known_dirs, ['vv'], file_db)
    for i in range(number_dates):
        assert os.path.isfile(file_db.input_file_vv(i))  # Check mocking

    mock_upto_concat_S2(application_mocker, file_db, 'normlim', number_dates*2)  # 2x2 inputs images
    mock_LIA(application_mocker, file_db)  # always N=2
    mock_masking(application_mocker, file_db, 'normlim', number_dates*2)  # 2x2 inputs images

    for idx in range(number_dates):
        application_mocker.set_expectations('BandMath', {
            'ram'      : '2048',
            'il'       : [file_db.concatfile_from_two(idx, False), file_db.selectedsinLIAfile()],
            'exp'      : 'im1b1*im2b1',
            'out'      : file_db.sigma0_normlim_file_from_two(idx, True),
            }, None)

    s1tiling.S1Processor.s1_process(config_opt=configuration, searched_items_per_page=0,
            dryrun=False, debug_otb=True, watch_ram=False,
            debug_tasks=False)
    application_mocker.assert_all_have_been_executed()

