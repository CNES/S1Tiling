#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from s1tiling.libs.otbpipeline import PipelineDescriptionSequence, Pipeline, MergeStep, FirstStep, to_dask_key
from s1tiling.libs.otbwrappers import (
        ExtractSentinel1Metadata, AnalyseBorders, Calibrate, CutBorders, OrthoRectify, Concatenate, BuildBorderMask, SmoothBorderMask,
        AgglomerateDEM, SARDEMProjection, SARCartesianMeanEstimation, ComputeNormals, ComputeLIA,
        OrthoRectifyLIA, ConcatenateLIA, SelectBestCoverage)
from s1tiling.libs.S1DateAcquisition import S1DateAcquisition
from s1tiling.libs.Utils import get_shape_from_polygon

# ======================================================================
# Scenarios
scenarios('../features/build_dependencies_and_tasks.feature', '../features/normlim.feature')

# ======================================================================
# Test Data

DEBUG_OTB = False
FILES = [
        # 08 jan 2020
        {
            's1dir'    : 'S1A_IW_GRDH_1SDV_20200108T044150_20200108T044215_030704_038506_C7F5',
            's1file'   : 's1a-iw-grd-{polarity}-20200108t044150-20200108t044215-030704-038506-001.tiff',
            'orthofile': 's1a_33NWB_{polarity}_DES_007_20200108t044150',
            'root'     : '{kind}_s1a-iw-grd-20200108t044150-20200108t044215-030704-038506-001',
            'orthoLIA' : 'LIA_s1a_33NWB_DES_007_20200108t044150',
            'polygon'  : [(14.233953, 1.137156), (16.461103, 0.660935), (16.77552, 2.173307), (14.545785, 2.645077), (14.233953, 1.137156)]
            },
        {
            's1dir'    : 'S1A_IW_GRDH_1SDV_20200108T044215_20200108T044240_030704_038506_D953',
            's1file'   : 's1a-iw-grd-{polarity}-20200108t044215-20200108t044240-030704-038506-001.tiff',
            'orthofile': 's1a_33NWB_{polarity}_DES_007_20200108t044215',
            'root'     : '{kind}_s1a-iw-grd-20200108t044215-20200108t044240-030704-038506-001',
            'orthoLIA' : 'LIA_s1a_33NWB_DES_007_20200108t044215',
            'polygon'  : [(13.917268, -0.370174), (16.143845, -0.851051), (16.461084, 0.660845), (14.233407, 1.137179), (13.917268, -0.370174)]
            },
        # 20 jan 2020
        {
            's1dir': 'S1A_IW_GRDH_1SDV_20200120T044214_20200120T044239_030879_038B2D_FDB0',
            's1file': 's1a-iw-grd-{polarity}-20200120t044214-20200120t044239-030879-038B2D-001.tiff',
            'orthofile': 's1a_33NWB_{polarity}_DES_007_20200120t044214',
            'polygon' : [(13.917237, -0.370036), (16.143806, -0.850946), (16.461067, 0.660948), (14.233396, 1.137315), (13.917237, -0.370036)]
            },
        {
            's1dir': 'S1A_IW_GRDH_1SDV_20200120T044149_20200120T044214_030879_038B2D_5671',
            's1file': 's1a-iw-grd-{polarity}-20200120t044149-20200120t044214-030879-038B2D-001.tiff',
            'orthofile': 's1a_33NWB_{polarity}_DES_007_20200120T044149',
            'polygon' : [(14.233942, 1.137292), (16.461086, 0.661038), (16.775522, 2.173408), (14.545794, 2.645211), (14.233942, 1.137292)]
            },
        # 02 feb 2020
        {
            's1dir': 'S1A_IW_GRDH_1SDV_20200201T044214_20200201T044239_031054_039149_CC58',
            's1file': 's1a-iw-grd-{polarity}-20200201t044214-20200201t044239-031054-039149-001.tiff',
            'orthofile': 's1a_33NWB_{polarity}_DES_007_20200201t044214',
            'polygon' : [(13.91733, -0.370053), (16.1439, -0.850965), (16.461174, 0.661021), (14.233503, 1.137389), (13.91733, -0.370053)]
            },
        {
            's1dir': 'S1A_IW_GRDH_1SDV_20200201T044149_20200201T044214_031054_039149_ED12',
            's1file': 's1a-iw-grd-{polarity}-20200201t044149-20200201t044214-031054-039149-001.tiff',
            'orthofile': 's1a_33NWB_{polarity}_DES_007_20200201t044149',
            'polygon' : [(14.233961, 1.137385), (16.461193, 0.661111), (16.775606, 2.173392), (14.54579, 2.645215), (14.233961, 1.137385)]
            },
        ]

TMPDIR = 'TMP'
INPUT  = 'data_raw'
OUTPUT = 'OUTPUT'
TILE   = '33NWB'

def compute_coverage(image_footprint_polygon, reference_tile_footprint_polygon):
    image_footprint          = get_shape_from_polygon(image_footprint_polygon[:4])
    reference_tile_footprint = get_shape_from_polygon(reference_tile_footprint_polygon[:4])
    intersection = image_footprint.Intersection(reference_tile_footprint)
    coverage = intersection.GetArea() / reference_tile_footprint.GetArea()
    assert coverage > 0   # We wouldn't have selected this pair S2 tile + S1 image otherwise
    assert coverage <= 1  # the ratio intersection / S2 tile should be <= 1!!
    return coverage

def tile_origins(tile_name):
    origins = {
            '33NWB': [(14.9998201759, 1.8098185887), (15.9870050338, 1.8095484335), (15.9866155411, 0.8163071941), (14.9998202469, 0.8164290331000001)],
            }
    return origins[tile_name]

def polarization(idx):
    return ['vv', 'vh'][idx]

def input_file(idx, polarity):
    s1dir  = FILES[idx]['s1dir']
    s1file = FILES[idx]['s1file'].format(polarity=polarity)
    return f'{INPUT}/{s1dir}/{s1dir}.SAFE/measurement/{s1file}'

def raster(idx, polarity):
    S2_tile_origin = tile_origins(TILE)
    s1dir  = FILES[idx]['s1dir']
    coverage = compute_coverage(FILES[idx]['polygon'], S2_tile_origin)
    logging.debug("coverage of %s by %s = %s", TILE, FILES[idx]['s1dir'], coverage)
    return {
            'raster': S1DateAcquisition(f'{INPUT}/{s1dir}/{s1dir}.SAFE/manifest.safe', [input_file(idx, 'vv')]),
            'tile_origin': S2_tile_origin,
            'tile_coverage': coverage
        }

def raster_vv(idx):
    return raster(idx, 'vv')
def raster_vh(idx):
    return raster(idx, 'vh')

def orthofile(idx, polarity):
    file = FILES[idx]["orthofile"].format(polarity=polarity)
    return f'{TMPDIR}/S2/{TILE}/{file}.tif'

def concatfile(idx, polarity):
    if idx is None:
        return f'{OUTPUT}/{TILE}/s1a_33NWB_{polarity}_DES_007_20200108txxxxxx.tif'
    else:
        file = FILES[idx]["orthofile"].format(polarity=polarity)
        return f'{OUTPUT}/{TILE}/{file}.tif'

def maskfile(idx, polarity):
    if idx is None:
        return f'{OUTPUT}/{TILE}/s1a_33NWB_{polarity}_DES_007_20200108txxxxxx_BorderMask.tif'
    else:
        file = FILES[idx]["orthofile"].format(polarity=polarity)
        return f'{OUTPUT}/{TILE}/{file}_BorderMask.tif'

def DEM_file(idx=None):
    if idx is None:
        return f'{TMPDIR}/S1/DEM_s1a-iw-grd-20200108t044150-20200108t044215-030704-038506-001.vrt'
    else:
        file = FILES[idx]["root"].format(kind='DEM')
        return f'{TMPDIR}/S1/{file}.vrt'

def DEMPROJ_file(idx=None):
    if idx is None:
        return f'{TMPDIR}/S1/S1_on_DEM_s1a-iw-grd-20200108t044150-20200108t044215-030704-038506-001.tiff'
    else:
        file = FILES[idx]["root"].format(kind='S1_on_DEM')
        return f'{TMPDIR}/S1/{file}.tiff'

def XYZ_file(idx=None):
    if idx is None:
        return f'{TMPDIR}/S1/XYZ_s1a-iw-grd-20200108t044150-20200108t044215-030704-038506-001.tiff'
    else:
        file = FILES[idx]["root"].format(kind='XYZ')
        return f'{TMPDIR}/S1/{file}.tiff'

def LIA_file(idx=None):
    if idx is None:
        return f'{TMPDIR}/S1/LIA_s1a-iw-grd-20200108t044150-20200108t044215-030704-038506-001.tiff'
    else:
        file = FILES[idx]["root"].format(kind='LIA')
        return f'{TMPDIR}/S1/{file}.tiff'

def ortho_LIA_file(idx):
    file  = FILES[idx]['orthoLIA']
    return f'{TMPDIR}/S2/{TILE}/{file}.tif'

def S2_LIA_file():
    return f'{OUTPUT}/{TILE}/LIA_s1a_33NWB_DES_007.tif'

def S2_LIA_preselect_file():
    return f'{TMPDIR}/S2/{TILE}/LIA_s1a_33NWB_DES_007_20200108txxxxxx.tif'

# ======================================================================
# Mocks

resource_dir = Path(__file__).parent.parent.parent.absolute() / 's1tiling/resources'

class Configuration():
    def __init__(self, tmpdir, outputdir, *argv):
        """
        constructor
        """
        self.GeoidFile                         = 'UNUSED HERE'
        self.calibration_type                  = 'sigma'
        self.grid_spacing                      = 40
        self.interpolation_method              = 'nn'
        self.out_spatial_res                   = 10
        self.output_preprocess                 = outputdir
        self.override_azimuth_cut_threshold_to = None
        self.ram_per_process                   = 4096
        self.removethermalnoise                = True
        self.tmp_srtm_dir                      = 'UNUSED HERE'
        self.tmpdir                            = tmpdir
        self.srtm_db_filepath                  = resource_dir / 'shapefile' / 'srtm_tiles.gpkg'
        self.cache_srtm_by                     = 'symlink'
        assert self.srtm_db_filepath.is_file()

def isfile(filename, existing_files):
    # assert False
    res = filename in existing_files
    logging.debug("isfile(%s) = %s ∈ %s", filename, res, existing_files)
    return res

# ======================================================================
# Fixtures

@pytest.fixture
def known_file_ids():
    fn = []
    return fn

@pytest.fixture
def known_files():
    kf = []
    return kf

@pytest.fixture()
def expected_files_id():
    ex = []
    return ex

@pytest.fixture
def pipelines():
    # TODO: propagate --tmpdir to scenario runners
    config = Configuration(tmpdir=TMPDIR, outputdir=OUTPUT)
    pd = PipelineDescriptionSequence(config, dryrun=True)
    return pd

@pytest.fixture
def last_pipeline():
    lp = []
    return lp

@pytest.fixture
def raster_list():
    rl = []
    return rl

@pytest.fixture
def dependencies():
    deps = []
    return deps

@pytest.fixture
def tasks():
    t = {}
    return t

# ======================================================================
# Given steps

@given('A pipeline that calibrates and orthorectifies')
def given_pipeline_ortho(pipelines, last_pipeline):
    pipeline = pipelines.register_pipeline([ExtractSentinel1Metadata, AnalyseBorders, Calibrate, CutBorders, OrthoRectify],
            'FullOrtho', product_required=False, is_name_incremental=True
            # , inputs={'in': 'basename'}
            )
    last_pipeline.append(pipeline)

@given('that concatenates')
def given_pipeline_concat(pipelines, last_pipeline):
    pipeline = pipelines.register_pipeline([Concatenate], product_required=True
            # , inputs={'in': last_pipeline[-1]}
            )
    last_pipeline.append(pipeline)

@given(parsers.parse('that {builds} masks'))
def given_pipeline_concat(pipelines, builds, last_pipeline):
    if builds == 'builds':
        # logging.info('REGISTER MASKS')
        pipeline = pipelines.register_pipeline([BuildBorderMask, SmoothBorderMask], 'GenerateMask',    product_required=True
            # , inputs={'in': last_pipeline[-1]}
                )
        last_pipeline.append(pipeline)

@given('A pipeline that computes LIA')
def given_pipeline_ortho(pipelines):
    dem = pipelines.register_pipeline([AgglomerateDEM], 'AgglomerateDEM', product_required=False,
            inputs={'insar': 'basename'})
    demproj = pipelines.register_pipeline([SARDEMProjection], 'SARDEMProjection', product_required=False,
            inputs={'insar': 'basename', 'indem': dem})
    xyz = pipelines.register_pipeline([SARCartesianMeanEstimation], 'SARCartesianMeanEstimation', product_required=False,
            inputs={'insar': 'basename', 'indem': dem, 'indemproj': demproj})
    lia = pipelines.register_pipeline([ComputeNormals, ComputeLIA], 'Normals|LIA', product_required=True, is_name_incremental=True,
            inputs={'xyz': xyz})

@given('A pipeline that fully computes in LIA S2 geometry')
def given_pipeline_ortho_n_concat_LIA(pipelines):
    dem = pipelines.register_pipeline([AgglomerateDEM], 'AgglomerateDEM', product_required=False,
            inputs={'insar': 'basename'})
    demproj = pipelines.register_pipeline([ExtractSentinel1Metadata, SARDEMProjection], 'SARDEMProjection', product_required=False, is_name_incremental=True,
            inputs={'insar': 'basename', 'indem': dem})
    xyz = pipelines.register_pipeline([SARCartesianMeanEstimation], 'SARCartesianMeanEstimation', product_required=False,
            inputs={'insar': 'basename', 'indem': dem, 'indemproj': demproj})
    lia = pipelines.register_pipeline([ComputeNormals, ComputeLIA], 'Normals|LIA', product_required=False, is_name_incremental=True,
            inputs={'xyz': xyz})
    ortho = pipelines.register_pipeline([OrthoRectifyLIA], 'OrthoLIA', product_required=False, is_name_incremental=True,
            inputs={'in': lia})
    concat = pipelines.register_pipeline([ConcatenateLIA], 'ConcatLIA', product_required=False, is_name_incremental=True,
            inputs={'in': ortho})
    select = pipelines.register_pipeline([SelectBestCoverage], 'SelectLIA', product_required=True, is_name_incremental=True,
            inputs={'in': concat})

@given('a single S1 image')
def given_one_S1_image(raster_list, known_files, known_file_ids):
    known_files.append(input_file(0, 'vv'))
    raster_list.append(raster_vv(0))
    known_file_ids.append((0, 'vv'))
    return raster_list

@given('a pair of VV + VH S1 images')
def given_one_VV_and_one_VH_S1_images(raster_list, known_files, known_file_ids, expected_files_id):
    known_files.append(input_file(0, 'vv'))
    known_files.append(input_file(0, 'vh'))
    raster_list.append(raster_vv(0))
    raster_list.append(raster_vh(0))
    known_file_ids.append((0, 'vv'))
    known_file_ids.append((0, 'vh'))
    expected_files_id.append(0)
    return raster_list

@given('a series of S1 VV images')
def given_one_VV_and_one_VH_S1_images(raster_list, known_files, known_file_ids, expected_files_id):
    for i in range(6):
        known_files.append(input_file(i, 'vv'))
        raster_list.append(raster_vv(i))
        known_file_ids.append((i, 'vv'))
    for i in [0, 1]:  # Only the first 2 should be kept. TODO: support coverage
        expected_files_id.append(i)
    return raster_list

@given('two S1 images')
def given_two_S1_images(raster_list, known_files, known_file_ids):
    known_files.extend([input_file(0, 'vv'), input_file(1, 'vv')])
    known_file_ids.extend([(0, 'vv'), (1, 'vv')])
    raster_list.append(raster_vv(0))
    raster_list.append(raster_vv(1))
    return raster_list

@given('a FullOrtho tmp image')
def given_one_FullOrtho_tmp_image(raster_list, known_files, known_file_ids):
    known_file_ids.append((1, 'vv'))
    known_files.append(orthofile(1, 'vv'))
    raster_list.append(raster_vv(1))
    return raster_list

@given('two FullOrtho tmp images')
def given_two_FullOrtho_tmp_images(raster_list, known_files, known_file_ids):
    known_file_ids.extend([(0, 'vv'), (1, 'vv')])
    known_files.append(orthofile(0, 'vv'))
    known_files.append(orthofile(1, 'vv'))
    raster_list.append(raster_vv(0))
    raster_list.append(raster_vv(1))
    return raster_list

# ======================================================================
# When steps

@when('dependencies are analysed')
def when_analyse_dependencies(pipelines, raster_list, dependencies, mocker, known_files):
    logging.debug("raster_list: %s" % (raster_list,))
    mocker.patch('s1tiling.libs.Utils.get_orbit_direction', return_value='DES')
    mocker.patch('s1tiling.libs.Utils.get_relative_orbit',  return_value=7)
    mocker.patch('os.path.isfile', lambda f: isfile(f, known_files))
    dependencies.extend(pipelines._build_dependencies(TILE, raster_list))

@when('tasks are generated')
def when_tasks_are_generated(pipelines, dependencies, tasks, mocker):
    # mocker.patch('os.path.isfile', lambda f: isfile(f, [input_file(0), input_file(1)]))
    required, previous, task2outfile_map = dependencies
    res = pipelines._build_tasks_from_dependencies(required=required, previous=previous, task_names_to_output_files_table=task2outfile_map,
            debug_otb=DEBUG_OTB, do_watch_ram=False)
    assert isinstance(res, dict)
    # logging.info("tasks (%s) = %s", type(res), res)
    tasks.update(res)

# ======================================================================
# Then steps

@then(parsers.parse('a txxxxxx S2 file is required, and {a} mask is required'))
def then_require_txxxxxx_and_mask(dependencies, a):
    expected_fn = [concatfile(None, 'vv')]
    if a != 'no':
        expected_fn += [maskfile(None, 'vv')]

    required, previous, task2outfile_map = dependencies
    # logging.info("required (%s) = %s", type(required), required)
    # logging.info("expected_fn (%s) = %s", type(expected_fn), expected_fn)
    assert isinstance(required, set)
    assert len(required) == len(expected_fn)
    for fn in expected_fn:
        assert fn in required, f'Expected {fn} not find in computed requirements {required}'
    assert concatfile(0, 'vv') not in required
    assert concatfile(1, 'vv') not in required
    assert maskfile(0, 'vv')   not in required
    assert maskfile(1, 'vv')   not in required

@then(parsers.parse('it depends on 2 ortho files (and two S1 inputs), and {a} mask on a concatenated product'))
def then_depends_on_2_ortho_files(dependencies, a):
    required, previous, task2outfile_map = dependencies
    # logging.info("previous (%s) = %s", type(previous), previous)

    if a == 'a':
        expected_fn = maskfile(None, 'vv')
        prev_expected = previous[expected_fn]
        expected_input_groups = prev_expected.inputs
        assert len(expected_input_groups) == 1
        for key, inputs in expected_input_groups.items():
            assert key == 'in'  # May change in the future...
            assert set([inp['out_filename'] for inp in inputs]) == set([concatfile(None, 'vv')])

    expected_fn = concatfile(None, 'vv')
    assert expected_fn in required
    prev_expected = previous[expected_fn]
    expected_input_groups = prev_expected.inputs
    assert len(expected_input_groups) == 1
    for key, inputs in expected_input_groups.items():
        assert key == 'in'  # May change in the future...
        assert set([inp['out_filename'] for inp in inputs]) == set([orthofile(0, 'vv'), orthofile(1, 'vv')])

    for i in range(2):
        expected_fn = orthofile(i, 'vv')
        prev_expected = previous[expected_fn]
        expected_input_groups = prev_expected.inputs
        assert len(expected_input_groups) == 1
        for key, inputs in expected_input_groups.items():
            assert key == 'in'  # May change in the future...
            assert len(inputs) == 1
            assert [inp['out_filename'] for inp in inputs][0] == input_file(i, 'vv')


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@then(parsers.parse('a t-chrono S2 file is required, and {a} mask is required'))
def then_require_tchrono_outline(dependencies, known_file_ids, a):
    assert len(known_file_ids) == 1
    known_file_id            = known_file_ids[0]
    known_file_number, polar = known_file_id
    expected_fn = [concatfile(*known_file_id)]
    if a != 'no':
        expected_fn += [maskfile(*known_file_id)]

    required, previous, task2outfile_map = dependencies
    req_files = [task2outfile_map[t] for t in required]
    # logging.info("required (%s) = %s", type(required), required)
    assert isinstance(required, set)
    assert len(required) == len(expected_fn)
    assert concatfile(None, polar)                not in req_files
    assert maskfile(None, polar)                  not in req_files
    assert concatfile(1-known_file_number, polar) not in req_files
    assert maskfile(1-known_file_number, polar)   not in req_files
    assert concatfile(known_file_number, polar)       in req_files
    if a == 'a':
        assert maskfile(known_file_number, polar)     in req_files
    else:
        assert maskfile(known_file_number, polar) not in req_files


@then(parsers.parse('it depends on one ortho file (and one S1 input), and {a} mask on a concatenated product'))
def then_depends_on_first_ortho_file(dependencies, known_file_ids, a):
    __then_depends_on_a_single_ortho_file(dependencies, known_file_ids, a)

@then(parsers.parse('it depends on second ortho file (and second S1 input), and {a} mask on a concatenated product'))
def then_depends_on_second_ortho_file(dependencies, a, known_file_ids):
    __then_depends_on_a_single_ortho_file(dependencies, known_file_ids, a)

def __then_depends_on_a_single_ortho_file(dependencies, known_file_ids, a):
    required, previous, task2outfile_map = dependencies
    # logging.info("previous (%s) = %s", type(previous), previous)
    assert len(known_file_ids) == 1
    known_file_number, polar = known_file_ids[0]
    if a == 'a':
        expected_fn = maskfile(known_file_number, polar)
        prev_expected = previous[expected_fn]
        expected_input_groups = prev_expected.inputs
        assert len(expected_input_groups) == 1
        for key, inputs in expected_input_groups.items():
            assert key == 'in'  # May change in the future...
            assert set([inp['out_filename'] for inp in inputs]) == set([concatfile(known_file_number, polar)])

    expected_fn = concatfile(None, polar)
    assert expected_fn in required
    prev_expected = previous[expected_fn]
    expected_input_groups = prev_expected.inputs
    assert len(expected_input_groups) == 1
    for key, inputs in expected_input_groups.items():
        assert key == 'in'  # May change in the future...
        assert set([inp['out_filename'] for inp in inputs]) == set([orthofile(known_file_number, polar)])

    for i in [known_file_number]:
        expected_fn = orthofile(i, polar)
        prev_expected = previous[expected_fn]
        expected_input_groups = prev_expected.inputs
        assert len(expected_input_groups) == 1
        for key, inputs in expected_input_groups.items():
            assert key == 'in'  # May change in the future...
            assert len(inputs) == 1
            assert [inp['out_filename'] for inp in inputs][0] == input_file(i, polar)


# ----------------------------------------------------------------------
# Helpers
def assert_orthorectify_product_number(idx, tasks, task2outfile_map):
    expectations = {
            orthofile(idx, 'vv'): {'pipeline': 'FullOrtho',
                'input_steps': {
                    input_file(idx, 'vv'): ['in', FirstStep],
                    }}
            }
    _check_registered_task(expectations, tasks, [orthofile(idx, 'vv')], task2outfile_map)

def assert_dont_orthorectify_product_number(idx, tasks):
    ortho = to_dask_key(orthofile(idx, 'vv'))
    assert (ortho not in tasks) or isinstance(tasks[ortho], FirstStep)

def assert_start_from_s1_image_number(idx, tasks):
    input = to_dask_key(input_file(idx, 'vv'))
    assert input in tasks
    task = tasks[input]
    assert isinstance(task, FirstStep)

def assert_dont_start_from_s1_image_number(idx, tasks):
    input = to_dask_key(input_file(idx, 'vv'))
    assert input not in tasks

def _check_registered_task(expectations, tasks, task_names, task2outfile_map):
    for req_taskname in task_names:
        ex_output        = req_taskname
        assert ex_output in expectations, f"Task {ex_output} isn't expected (expectations: {list(expectations.keys())})"
        ex               = expectations[ex_output]
        ex_pipeline_name = ex['pipeline']
        ex_in_steps      = ex['input_steps']
        logging.debug("TASKS: %s", tasks.keys())
        req_task_key = to_dask_key(req_taskname)
        assert req_task_key in tasks
        req_task = tasks[req_task_key]
        logging.debug("req_task: %s", req_task)

        req_pipeline = req_task[1]
        assert req_pipeline.output == task2outfile_map[ex_output]
        assert isinstance(req_pipeline, Pipeline)
        assert req_pipeline._Pipeline__name == ex_pipeline_name
        req_inputs = req_pipeline._Pipeline__inputs
        logging.debug("inputs: %s", req_inputs)
        for ex_in_file, ex_in_info in ex_in_steps.items():
            ex_in_key, ex_in_step = ex_in_info
            matching_input = [inp[ex_in_key] for inp in req_inputs if ex_in_key in inp]
            assert len(matching_input) == 1, f"No {ex_in_key} key in {[inp.keys() for inp in req_inputs]}"
            assert isinstance(matching_input[0], ex_in_step)
            assert ex_in_file in matching_input[0].out_filename

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@then(parsers.parse('a concatenation task is registered and produces txxxxxxx S2 file and {a} mask'))
def then_concatenate_2_files_(tasks, dependencies, a):
    expectations = {
            # MergeStep as there are two inputs
            concatfile(None, 'vv'): {'pipeline': 'Concatenation',
                'input_steps': {
                    orthofile(0, 'vv'): ['in', MergeStep],
                    orthofile(1, 'vv'): ['in', MergeStep],
                    }}
            }
    if a != 'no':
        expectations[maskfile(None, 'vv')] = {'pipeline': 'GenerateMask',
                'input_steps': {
                    concatfile(None, 'vv'): ['in', FirstStep]}}
    required, previous, task2outfile_map = dependencies
    # logging.info("tasks (type: %s) = %s", type(tasks), tasks)
    assert isinstance(tasks, dict)
    assert len(tasks) >= 3
    assert len(required) == len(expectations)
    assert concatfile(None, 'vv') in required
    _check_registered_task(expectations, tasks, required, task2outfile_map)


@then('two orthorectification tasks are registered')
def then_orthorectify_two_products(tasks, dependencies):
    required, previous, task2outfile_map = dependencies
    # logging.info("tasks (%s) = %s", type(tasks), tasks)
    assert len(tasks) >= 5

    for i in (0, 1):
        assert_orthorectify_product_number(i, tasks, task2outfile_map)

    for i in (0, 1):
        assert_start_from_s1_image_number(i, tasks)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@then(parsers.parse('a concatenation task is registered and produces t-chrono S2 file, and {a} mask'))
def then_concatenate_1_files(tasks, dependencies, known_file_ids, a):
    assert len(known_file_ids) == 1
    known_file_number, polar = known_file_ids[0]
    expectations = {
            # Task name is in txxxxxx, but file name is not
            concatfile(None, 'vv'): {'pipeline': 'Concatenation',
                'input_steps': {
                    # FirstStep as there is only one input
                    orthofile(known_file_number, 'vv'): ['in', FirstStep],
                    }}
            }
    if a != 'no':
        expectations[maskfile(known_file_number, 'vv')] = {'pipeline': 'GenerateMask',
                'input_steps': {
                    concatfile(known_file_number, 'vv'): ['in', FirstStep]}}

    required, previous, task2outfile_map = dependencies
    assert task2outfile_map[concatfile(None, 'vv')] == concatfile(known_file_number, 'vv')
    # logging.info("tasks (type: %s) = %s", type(tasks), tasks)
    assert isinstance(tasks, dict)
    assert len(tasks) >= 1, f'Only {len(tasks)} tasks are registered instead of 0+ : {list(tasks.keys())}'
    assert len(required) == len(expectations)
    assert concatfile(None, 'vv') in required
    _check_registered_task(expectations, tasks, required, task2outfile_map)


@then('a single orthorectification task is registered')
def then_orthorectify_one_product(tasks, dependencies):
    required, previous, task2outfile_map = dependencies
    # logging.info("tasks (%s) = %s", type(tasks), tasks)
    assert len(tasks) >= 3
    assert_orthorectify_product_number(0, tasks, task2outfile_map)
    assert_start_from_s1_image_number(0, tasks)

@then('no orthorectification tasks is registered')
def then_dont_orthorectify_any_product(tasks, dependencies):
    required, previous, task2outfile_map = dependencies
    # logging.info("tasks (%s) = %s", type(tasks), tasks)
    for i in (0, 1):
        assert_dont_orthorectify_product_number(i, tasks)
        assert_dont_start_from_s1_image_number(i, tasks)

@then('dont orthorectify the second product')
def but_dont_orthorectify_the_second_product(tasks, dependencies):
    required, previous, task2outfile_map = dependencies
    assert_dont_orthorectify_product_number(1, tasks)
    assert_dont_start_from_s1_image_number(1, tasks)

@then('it depends on the existing FullOrtho tmp product')
def depend_on_the_existing_fullortho_product(tasks, dependencies):
    required, previous, task2outfile_map = dependencies
    # logging.info("tasks (%s) = %s", type(tasks), tasks)

    ortho = to_dask_key(orthofile(1, 'vv'))
    assert ortho in tasks
    task = tasks[ortho]
    assert isinstance(task, FirstStep)

    assert_dont_start_from_s1_image_number(1, tasks)

@then('it depends on two existing FullOrtho tmp products')
def depend_on_two_existing_fullortho_products(tasks, dependencies):
    required, previous, task2outfile_map = dependencies
    # logging.info("tasks (%s) = %s", type(tasks), tasks)

    for i in (0, 1):
        ortho = to_dask_key(orthofile(i, 'vv'))
        assert ortho in tasks
        task = tasks[ortho]
        assert isinstance(task, FirstStep)

        assert_dont_start_from_s1_image_number(i, tasks)

# ----------------------------------------------------------------------
# -- LIA tests
# ----------------------------------------------------------------------

@then('a single LIA image is required')
def then_LIA_image_is_required(dependencies):
    required, previous, task2outfile_map = dependencies

    expected_fn = [LIA_file()]

    logging.info("required (%s) = %s", type(required), required)
    assert isinstance(required, set)
    assert len(required) == len(expected_fn), f'Expecting {expected_fn}, but requirements found are: {required}'
    for fn in expected_fn:
        assert fn in required, f'Expected {fn} not find in computed requirements {required}'

@then('a single S2 LIA image is required')
def then_S2_LIA_image_is_required(dependencies):
    required, previous, task2outfile_map = dependencies

    expected_fn = [S2_LIA_file()]

    logging.info("required (%s) = %s", type(required), required)
    assert isinstance(required, set)
    assert len(required) == len(expected_fn), f'Expecting {expected_fn}, but requirements found are: {required}'
    for fn in expected_fn:
        assert fn in required, f'Expected {fn} not find in computed requirements {required}'

@then('final LIA image has been selected from one concat LIA')
def final_LIA_image_has_been_selected_from_one_concat_LIA(dependencies):
    required, previous, task2outfile_map = dependencies
    expected_fn = S2_LIA_file()
    assert expected_fn in required
    prev_expected = previous[expected_fn]
    expected_input_groups = prev_expected.inputs
    assert len(expected_input_groups) == 1
    for key, inputs in expected_input_groups.items():
        assert key == 'in'  # May change in the future...
        assert set([inp['out_filename'] for inp in inputs]) == set([S2_LIA_preselect_file()])

@then('concat LIA depends on 2 ortho LIA images')
def concat_LIA_depends_on_2_ortho_LIA_images(dependencies):
    required, previous, task2outfile_map = dependencies

    expected_fn = S2_LIA_preselect_file()
    assert expected_fn not in required
    prev_expected = previous[expected_fn]
    expected_input_groups = prev_expected.inputs
    assert len(expected_input_groups) == 1
    for key, inputs in expected_input_groups.items():
        assert key == 'in'  # May change in the future...
        assert set([inp['out_filename'] for inp in inputs]) == set([ortho_LIA_file(0), ortho_LIA_file(1)])


@then('2 ortho LIA images depend on two LIA images')
def two_ortho_LIA_depend_on_two_LIA_images(dependencies):
    required, previous, task2outfile_map = dependencies

    for i in [0, 1]:  # Only the first 2 dates should be used
        expected_fn = ortho_LIA_file(i)
        prev_expected = previous[expected_fn]
        expected_input_groups = prev_expected.inputs
        assert len(expected_input_groups) == 1
        for key, inputs in expected_input_groups.items():
            assert key == 'in'  # May change in the future...
            assert len(inputs) == 1
            assert [inp['out_filename'] for inp in inputs][0] == LIA_file(i)


@then('LIA images depend on XYZ images')
def LIA_images_depend_on_two_XYZ_images(dependencies, expected_files_id):
    required, previous, task2outfile_map = dependencies

    for i in expected_files_id:
        expected_fn = LIA_file(i)
        # assert expected_fn not in required  # Depends on the scenario...
        prev_expected = previous[expected_fn]
        expected_inputs = prev_expected.inputs
        assert len(expected_inputs) == 1
        for key, inputs in expected_inputs.items():
            assert key == 'xyz'
            assert len(inputs) == 1
            input = inputs[0]
            # logging.info('Inputs from %s: %s', expected_inputs, input)
            assert 'out_filename' in input
            xyz_file = input["out_filename"]
            assert xyz_file == XYZ_file(i)

@then('XYZ images depend on DEM, DEMPROJ and BASE images')
def XYZ_depend_on_DEM_DEMPROJ_and_BASE(dependencies, expected_files_id):
    required, previous, task2outfile_map = dependencies

    for i in expected_files_id:
        expected_fn = XYZ_file(i)
        prev_expected = previous[expected_fn]
        expected_inputs = prev_expected.inputs
        assert len(expected_inputs) == 3
        assert {'indem', 'insar', 'indemproj'} == set(expected_inputs.keys())

        insar_as_inputs = expected_inputs['insar']
        assert len(insar_as_inputs) == 1, f"{len(insar_as_inputs)} in SAR input founds, only 1 expected.\nFound: {insar_as_inputs}"
        insar_as_input = insar_as_inputs[0]
        assert insar_as_input['out_filename'] == input_file(i, 'vv')

        indem_as_inputs = expected_inputs['indem']
        assert len(indem_as_inputs) == 1
        indem_as_input = indem_as_inputs[0]
        assert indem_as_input['out_filename'] == DEM_file(i)

        indemproj_as_inputs = expected_inputs['indemproj']
        assert len(indemproj_as_inputs) == 1
        indemproj_as_input = indemproj_as_inputs[0]
        assert indemproj_as_input['out_filename'] == DEMPROJ_file(i)

@then('DEMPROJ images depend on DEM and BASE images')
def DEMPROJ_depends_on_DEM_and_BASE(dependencies, expected_files_id):
    required, previous, task2outfile_map = dependencies

    for i in expected_files_id:
        expected_fn = DEMPROJ_file(i)
        prev_expected = previous[expected_fn]
        expected_inputs = prev_expected.inputs
        assert len(expected_inputs) == 2
        assert {'indem', 'insar'} == set(expected_inputs.keys())

        insar_as_inputs = expected_inputs['insar']
        assert len(insar_as_inputs) == 1, f"{len(insar_as_inputs)} in SAR input founds, only 1 expected.\nFound: {insar_as_inputs}"
        insar_as_input = insar_as_inputs[0]
        assert insar_as_input['out_filename'] == input_file(i, 'vv')

        indem_as_inputs = expected_inputs['indem']
        assert len(indem_as_inputs) == 1
        indem_as_input = indem_as_inputs[0]
        assert indem_as_input['out_filename'] == DEM_file(i)

@then('DEM images depend on BASE images')
def DEM_depends_on_BASE(dependencies, expected_files_id):
    required, previous, task2outfile_map = dependencies

    for i in expected_files_id:
        expected_fn = DEM_file(i)
        prev_expected = previous[expected_fn]
        expected_inputs = prev_expected.inputs
        assert len(expected_inputs) == 1
        assert {'insar'} == set(expected_inputs.keys())

        insar_as_inputs = expected_inputs['insar']
        assert len(insar_as_inputs) == 1, f"{len(insar_as_inputs)} in SAR input founds, only 1 expected.\nFound: {insar_as_inputs}"
        insar_as_input = insar_as_inputs[0]
        assert insar_as_input['out_filename'] == input_file(i, 'vv')


@then('a select LIA task is registered')
def then_a_select_LIA_task_is_registered(tasks, dependencies, expected_files_id):
    out = S2_LIA_file()
    expectations = {
            out: {'pipeline': 'SelectLIA',
                'input_steps': {
                    S2_LIA_preselect_file(): ['in', FirstStep]
                    }}
            }

    required, previous, task2outfile_map = dependencies
    # logging.info("tasks (%s) = %s", type(tasks), tasks)
    assert isinstance(tasks, dict)
    assert len(tasks) >= 3
    assert len(required) == len(expectations)
    assert S2_LIA_preselect_file() not in required
    _check_registered_task(expectations, tasks, required, task2outfile_map)


@then('a concat LIA task is registered')
def then_a_concat_LIA_task_is_registered(tasks, dependencies, expected_files_id):
    out = S2_LIA_preselect_file()
    expectations = {
            out: {'pipeline': 'ConcatLIA',
                'input_steps': {}}
            }
    dest = [out]
    for i in expected_files_id:
        expectations[out]['input_steps'][ortho_LIA_file(i)] = ['in', MergeStep]

    required, previous, task2outfile_map = dependencies
    logging.info("tasks (%s) = %s", type(tasks), tasks)
    assert isinstance(tasks, dict)
    assert len(tasks) >= 2 + len(expected_files_id)
    assert len(required) == len(expectations)
    for i in expected_files_id:
        assert ortho_LIA_file(i) not in required
    _check_registered_task(expectations, tasks, dest, task2outfile_map)


@then('ortho LIA task(s) is(/are) registered')
def then_ortho_LIA_task_is_registered(tasks, dependencies, expected_files_id):
    expectations = {}
    dest = []
    for i in expected_files_id:
        out = ortho_LIA_file(i)
        dest.append(out)
        expectations[out] = {
                'pipeline': 'OrthoLIA',
                'input_steps': {LIA_file(i) : ['in', FirstStep]}
                }

    required, previous, task2outfile_map = dependencies
    # logging.info("tasks (%s) = %s", type(tasks), tasks)
    assert isinstance(tasks, dict)
    assert len(tasks) >= 2 + len(expected_files_id)
    for i in expected_files_id:
        assert ortho_LIA_file(i) not in required
        assert LIA_file(i) not in required
    _check_registered_task(expectations, tasks, dest, task2outfile_map)

@then('LIA task(s) is(/are) registered')
def then_a_LIA_task_is_registered(tasks, dependencies, expected_files_id):
    expectations = {}
    dest = []
    for i in expected_files_id:
        out = LIA_file(i)
        dest.append(out)
        expectations[out] = {
                'pipeline': 'Normals|LIA',
                'input_steps': {
                    XYZ_file(i): ['xyz', FirstStep]
                    }
                }
    required, previous, task2outfile_map = dependencies
    # logging.info("tasks (%s) = %s", type(tasks), tasks)
    assert isinstance(tasks, dict)
    assert len(tasks) >= 3
    # assert len(required) == len(expectations)
    # assert LIA_file() not in required
    _check_registered_task(expectations, tasks, dest, task2outfile_map)

@then('XYZ task(s) is(/are) registered')
def then_a_XYZ_task_is_registered(tasks, dependencies, expected_files_id):
    expectations = {}
    dest = []
    for i in expected_files_id:
        out = XYZ_file(i)
        dest.append(out)
        expectations[out] = {
                'pipeline': 'SARCartesianMeanEstimation',
                'input_steps': {
                    DEM_file(i):          ['indem',     FirstStep],
                    DEMPROJ_file(i):      ['indemproj', FirstStep],
                    input_file(i, 'vv'):  ['insar',     FirstStep],
                    }
                }
    required, previous, task2outfile_map = dependencies
    # logging.info("tasks (%s) = %s", type(tasks), tasks)
    assert isinstance(tasks, dict)
    _check_registered_task(expectations, tasks, dest, task2outfile_map)

@then('DEMPROJ task(s) is(/are) registered')
def then_a_DEMPROJ_task_is_registered(tasks, dependencies, expected_files_id):
    expectations = {}
    dest = []
    for i in expected_files_id:
        out = DEMPROJ_file(i)
        dest.append(out)
        expectations[out] = {
                'pipeline': 'SARDEMProjection',
                'input_steps': {
                    DEM_file(i):          ['indem',     FirstStep],
                    input_file(i, 'vv'):  ['insar',     FirstStep],
                    }
                }
    required, previous, task2outfile_map = dependencies
    # logging.info("tasks (%s) = %s", type(tasks), tasks)
    assert isinstance(tasks, dict)
    _check_registered_task(expectations, tasks, dest, task2outfile_map)

@then('DEM task(s) is(/are) registered')
def then_a_DEM_task_is_registered(tasks, dependencies, expected_files_id):
    expectations = {}
    dest = []
    for i in expected_files_id:
        out = DEM_file(i)
        dest.append(out)
        expectations[out] = {
                'pipeline': 'AgglomerateDEM',
                'input_steps': {
                    input_file(i, 'vv'):  ['insar',     FirstStep],
                    }
                }
    required, previous, task2outfile_map = dependencies
    # logging.info("tasks (%s) = %s", type(tasks), tasks)
    assert isinstance(tasks, dict)
    _check_registered_task(expectations, tasks, dest, task2outfile_map)

