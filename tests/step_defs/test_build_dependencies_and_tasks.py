#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from s1tiling.libs.otbpipeline import PipelineDescriptionSequence, Pipeline, MergeStep, FirstStep, to_dask_key
from s1tiling.libs.otbwrappers import (
        ExtractSentinel1Metadata, AnalyseBorders, Calibrate, CutBorders, OrthoRectify, Concatenate, BuildBorderMask, SmoothBorderMask,
        AgglomerateDEM, SARDEMProjection, SARCartesianMeanEstimation, ComputeNormals, ComputeLIA)
from s1tiling.libs.S1DateAcquisition import S1DateAcquisition

# ======================================================================
# Scenarios
scenarios('../features/build_dependencies_and_tasks.feature', '../features/normlim.feature')

# ======================================================================
# Test Data

DEBUG_OTB = False
FILES = [
        {
            's1dir': 'S1A_IW_GRDH_1SDV_20200108T044150_20200108T044215_030704_038506_C7F5',
            's1file': 's1a-iw-grd-vv-20200108t044150-20200108t044215-030704-038506-001.tiff',
            'orthofile': 's1a_33NWB_vv_DES_007_20200108t044150'
            },
        {
            's1dir': 'S1A_IW_GRDH_1SDV_20200108T044215_20200108T044240_030704_038506_D953',
            's1file': 's1a-iw-grd-vv-20200108t044215-20200108t044240-030704-038506-001.tiff',
            'orthofile': 's1a_33NWB_vv_DES_007_20200108t044215'
            }
        ]

TMPDIR = 'TMP'
INPUT  = 'data_raw'
OUTPUT = 'OUTPUT'
TILE   = '33NWB'

def input_file(idx):
    s1dir  = FILES[idx]['s1dir']
    s1file = FILES[idx]['s1file']
    return f'{INPUT}/{s1dir}/{s1dir}.SAFE/measurement/{s1file}'

def raster_vv(idx):
    s1dir  = FILES[idx]['s1dir']
    return (S1DateAcquisition(
        f'{INPUT}/{s1dir}/{s1dir}.SAFE/manifest.safe',
        [input_file(idx)]),
        [(14.9998201759, 1.8098185887), (15.9870050338, 1.8095484335), (15.9866155411, 0.8163071941), (14.9998202469, 0.8164290331000001)])

def orthofile(idx):
    return f'{TMPDIR}/S2/{TILE}/{FILES[idx]["orthofile"]}.tif'

def concatfile(idx):
    if idx is None:
        return f'{OUTPUT}/{TILE}/s1a_33NWB_vv_DES_007_20200108txxxxxx.tif'
    else:
        return f'{OUTPUT}/{TILE}/{FILES[idx]["orthofile"]}.tif'

def maskfile(idx):
    if idx is None:
        return f'{OUTPUT}/{TILE}/s1a_33NWB_vv_DES_007_20200108txxxxxx_BorderMask.tif'
    else:
        return f'{OUTPUT}/{TILE}/{FILES[idx]["orthofile"]}_BorderMask.tif'

def DEM_file():
    return f'{TMPDIR}/S1/DEM_s1a-iw-grd-vv-20200108t044150-20200108t044215-030704-038506-001.vrt'

def DEMPROJ_file():
    return f'{TMPDIR}/S1/S1_on_DEM_s1a-iw-grd-vv-20200108t044150-20200108t044215-030704-038506-001.tiff'

def XYZ_file():
    return f'{TMPDIR}/S1/XYZ_s1a-iw-grd-vv-20200108t044150-20200108t044215-030704-038506-001.tiff'

def LIA_file():
    return f'{TMPDIR}/S1/LIA_s1a-iw-grd-vv-20200108t044150-20200108t044215-030704-038506-001.tiff'

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
def known_file_numbers():
    fn = []
    return fn

@pytest.fixture
def known_files():
    kf = []
    return kf

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
    # pipelines.register_pipeline([ExtractSentinel1Metadata], 'ExtractS1Meta', product_required=False)
    pipeline = pipelines.register_pipeline([ExtractSentinel1Metadata, AnalyseBorders, Calibrate, CutBorders, OrthoRectify],
    # pipelines.register_pipeline([AnalyseBorders, Calibrate, CutBorders, OrthoRectify],
            'FullOrtho', product_required=False, is_name_incremental=True
            , inputs={'in': 'basename'}
            )
    last_pipeline.append(pipeline)

@given('that concatenates')
def given_pipeline_concat(pipelines, last_pipeline):
    pipeline = pipelines.register_pipeline([Concatenate], product_required=True
            , inputs={'in': last_pipeline[-1]}
            )
    last_pipeline.append(pipeline)

@given('that <builds> masks')
def given_pipeline_concat(pipelines, builds, last_pipeline):
    if builds == 'builds':
        # logging.info('REGISTER MASKS')
        pipeline = pipelines.register_pipeline([BuildBorderMask, SmoothBorderMask], 'GenerateMask',    product_required=True
            , inputs={'in': last_pipeline[-1]}
                )
        last_pipeline.append(pipeline)

@given('A pipeline that computes LIA')
def given_pipeline_ortho(pipelines):
    # pipelines.register_pipeline([ExtractSentinel1Metadata], 'ExtractS1Meta', product_required=False)
    dem = pipelines.register_pipeline([AgglomerateDEM], 'AgglomerateDEM', product_required=False,
            inputs={'insar': 'basename'})
    demproj = pipelines.register_pipeline([SARDEMProjection], 'SARDEMProjection', product_required=False,
            inputs={'insar': 'basename', 'indem': dem})
    xyz = pipelines.register_pipeline([SARCartesianMeanEstimation], 'SARCartesianMeanEstimation', product_required=False,
            inputs={'insar': 'basename', 'indem': dem, 'indemproj': demproj})
    lia = pipelines.register_pipeline([ComputeNormals, ComputeLIA], 'Normals|LIA', product_required=True, is_name_incremental=True,
            inputs={'xyz': xyz})

@given('a single S1 image')
def given_one_S1_image(raster_list, known_files, known_file_numbers):
    known_files.append(input_file(0))
    raster_list.append(raster_vv(0))
    known_file_numbers.append(0)
    return raster_list

@given('two S1 images')
def given_two_S1_images(raster_list, known_files, known_file_numbers):
    known_files.extend([input_file(0), input_file(1)])
    known_file_numbers.extend([0, 1])
    raster_list.append(raster_vv(0))
    raster_list.append(raster_vv(1))
    return raster_list

@given('a FullOrtho tmp image')
def given_one_FullOrtho_tmp_image(raster_list, known_files, known_file_numbers):
    known_file_numbers.append(1)
    known_files.append(orthofile(1))
    raster_list.append(raster_vv(1))
    return raster_list

@given('two FullOrtho tmp images')
def given_two_FullOrtho_tmp_images(raster_list, known_files, known_file_numbers):
    known_file_numbers.extend([0, 1])
    known_files.append(orthofile(0))
    known_files.append(orthofile(1))
    raster_list.append(raster_vv(0))
    raster_list.append(raster_vv(1))
    return raster_list

# ======================================================================
# When steps

@when('dependencies are analysed')
def when_analyse_dependencies(pipelines, raster_list, dependencies, mocker, known_files):
    # print("raster_list: %s" % (raster_list,))
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

@then('a txxxxxx S2 file is required, and <a> mask is required')
def then_require_txxxxxx_and_mask(dependencies, a):
    expected_fn = [concatfile(None)]
    if a != 'no':
        expected_fn += [maskfile(None)]

    required, previous, task2outfile_map = dependencies
    # logging.info("required (%s) = %s", type(required), required)
    # logging.info("expected_fn (%s) = %s", type(expected_fn), expected_fn)
    assert isinstance(required, set)
    assert len(required) == len(expected_fn)
    for fn in expected_fn:
        assert fn in required
    assert concatfile(0) not in required
    assert concatfile(1) not in required
    assert maskfile(0)   not in required
    assert maskfile(1)   not in required

@then('it depends on 2 ortho files (and two S1 inputs), and <a> mask on a concatenated product')
def then_depends_on_2_ortho_files(dependencies, a):
    required, previous, task2outfile_map = dependencies
    # logging.info("previous (%s) = %s", type(previous), previous)

    if a == 'a':
        expected_fn = maskfile(None)
        prev_expected = previous[expected_fn]
        expected_input_groups = prev_expected.inputs
        assert len(expected_input_groups) == 1
        for key, inputs in expected_input_groups.items():
            assert key == 'in'  # May change in the future...
            assert set([inp['out_filename'] for inp in inputs]) == set([concatfile(None)])

    expected_fn = concatfile(None)
    assert expected_fn in required
    prev_expected = previous[expected_fn]
    expected_input_groups = prev_expected.inputs
    assert len(expected_input_groups) == 1
    for key, inputs in expected_input_groups.items():
        assert key == 'in'  # May change in the future...
        assert set([inp['out_filename'] for inp in inputs]) == set([orthofile(0), orthofile(1)])

    for i in range(2):
        expected_fn = orthofile(i)
        prev_expected = previous[expected_fn]
        expected_input_groups = prev_expected.inputs
        assert len(expected_input_groups) == 1
        for key, inputs in expected_input_groups.items():
            assert key == 'in'  # May change in the future...
            assert len(inputs) == 1
            assert [inp['out_filename'] for inp in inputs][0] == input_file(i)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@then('a t-chrono S2 file is required, and <a> mask is required')
def then_require_tchrono_outline(dependencies, known_file_numbers, a):
    assert len(known_file_numbers) == 1
    known_file = known_file_numbers[0]
    expected_fn = [concatfile(known_file)]
    if a != 'no':
        expected_fn += [maskfile(known_file)]

    required, previous, task2outfile_map = dependencies
    req_files = [task2outfile_map[t] for t in required]
    # logging.info("required (%s) = %s", type(required), required)
    assert isinstance(required, set)
    assert len(required) == len(expected_fn)
    assert concatfile(None) not in req_files
    assert maskfile(None)   not in req_files
    assert concatfile(1-known_file) not in req_files
    assert maskfile(1-known_file)   not in req_files
    assert concatfile(known_file) in req_files
    if a == 'a':
        assert maskfile(known_file)     in req_files
    else:
        assert maskfile(known_file) not in req_files


@then('it depends on one ortho file (and one S1 input), and <a> mask on a concatenated product')
def then_depends_on_first_ortho_file(dependencies, known_file_numbers, a):
    __then_depends_on_a_single_ortho_file(dependencies, known_file_numbers, a)

@then('it depends on second ortho file (and second S1 input), and <a> mask on a concatenated product')
def then_depends_on_second_ortho_file(dependencies, a, known_file_numbers):
    __then_depends_on_a_single_ortho_file(dependencies, known_file_numbers, a)

def __then_depends_on_a_single_ortho_file(dependencies, known_file_numbers, a):
    required, previous, task2outfile_map = dependencies
    # logging.info("previous (%s) = %s", type(previous), previous)
    assert len(known_file_numbers) == 1
    known_file = known_file_numbers[0]
    if a == 'a':
        expected_fn = maskfile(known_file)
        prev_expected = previous[expected_fn]
        expected_input_groups = prev_expected.inputs
        assert len(expected_input_groups) == 1
        for key, inputs in expected_input_groups.items():
            assert key == 'in'  # May change in the future...
            assert set([inp['out_filename'] for inp in inputs]) == set([concatfile(known_file)])

    expected_fn = concatfile(None)
    assert expected_fn in required
    prev_expected = previous[expected_fn]
    expected_input_groups = prev_expected.inputs
    assert len(expected_input_groups) == 1
    for key, inputs in expected_input_groups.items():
        assert key == 'in'  # May change in the future...
        assert set([inp['out_filename'] for inp in inputs]) == set([orthofile(known_file)])

    for i in [known_file]:
        expected_fn = orthofile(i)
        prev_expected = previous[expected_fn]
        expected_input_groups = prev_expected.inputs
        assert len(expected_input_groups) == 1
        for key, inputs in expected_input_groups.items():
            assert key == 'in'  # May change in the future...
            assert len(inputs) == 1
            assert [inp['out_filename'] for inp in inputs][0] == input_file(i)


# ----------------------------------------------------------------------
# Helpers
def assert_orthorectify_product_number(idx, tasks, task2outfile_map):
    expectations = {
            orthofile(idx): {'pipeline': 'FullOrtho',
                'input_steps': {
                    input_file(idx): ['in', FirstStep],
                    }}
            }
    _check_registered_task(expectations, tasks, [orthofile(idx)], task2outfile_map)

def assert_dont_orthorectify_product_number(idx, tasks):
    ortho = to_dask_key(orthofile(idx))
    assert (ortho not in tasks) or isinstance(tasks[ortho], FirstStep)

def assert_start_from_s1_image_number(idx, tasks):
    input = to_dask_key(input_file(idx))
    assert input in tasks
    task = tasks[input]
    assert isinstance(task, FirstStep)

def assert_dont_start_from_s1_image_number(idx, tasks):
    input = to_dask_key(input_file(idx))
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
@then('a concatenation task is registered and produces txxxxxxx S2 file and <a> mask')
def then_concatenate_2_files_(tasks, dependencies, a):
    expectations = {
            # MergeStep as there are two inputs
            concatfile(None): {'pipeline': 'Concatenation',
                'input_steps': {
                    orthofile(0): ['in', MergeStep],
                    orthofile(1): ['in', MergeStep],
                    }}
            }
    if a != 'no':
        expectations[maskfile(None)] = {'pipeline': 'GenerateMask',
                'input_steps': {
                    concatfile(None): ['in', FirstStep]}}
    required, previous, task2outfile_map = dependencies
    # logging.info("tasks (type: %s) = %s", type(tasks), tasks)
    assert isinstance(tasks, dict)
    assert len(tasks) >= 3
    assert len(required) == len(expectations)
    assert concatfile(None) in required
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
@then('a concatenation task is registered and produces t-chrono S2 file, and <a> mask')
def then_concatenate_1_files(tasks, dependencies, known_file_numbers, a):
    assert len(known_file_numbers) == 1
    known_file = known_file_numbers[0]
    expectations = {
            # Task name is in txxxxxx, but file name is not
            concatfile(None): {'pipeline': 'Concatenation',
                'input_steps': {
                    # FirstStep as there is only one input
                    orthofile(known_file): ['in', FirstStep],
                    }}
            }
    if a != 'no':
        expectations[maskfile(known_file)] = {'pipeline': 'GenerateMask',
                'input_steps': {
                    concatfile(known_file): ['in', FirstStep]}}

    required, previous, task2outfile_map = dependencies
    assert task2outfile_map[concatfile(None)] == concatfile(known_file)
    # logging.info("tasks (type: %s) = %s", type(tasks), tasks)
    assert isinstance(tasks, dict)
    assert len(tasks) >= 1, f'Only {len(tasks)} tasks are registered instead of 0+ : {list(tasks.keys())}'
    assert len(required) == len(expectations)
    assert concatfile(None) in required
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

    ortho = to_dask_key(orthofile(1))
    assert ortho in tasks
    task = tasks[ortho]
    assert isinstance(task, FirstStep)

    assert_dont_start_from_s1_image_number(1, tasks)

@then('it depends on two existing FullOrtho tmp products')
def depend_on_two_existing_fullortho_products(tasks, dependencies):
    required, previous, task2outfile_map = dependencies
    # logging.info("tasks (%s) = %s", type(tasks), tasks)

    for i in (0, 1):
        ortho = to_dask_key(orthofile(i))
        assert ortho in tasks
        task = tasks[ortho]
        assert isinstance(task, FirstStep)

        assert_dont_start_from_s1_image_number(i, tasks)

# ----------------------------------------------------------------------
# -- LIA tests
# ----------------------------------------------------------------------

@then('a LIA image is required')
def then_LIA_image_is_required(dependencies):
    required, previous, task2outfile_map = dependencies

    expected_fn = [LIA_file()]

    # logging.info("required (%s) = %s", type(required), required)
    assert isinstance(required, set)
    assert len(required) == len(expected_fn)
    for fn in expected_fn:
        assert fn in required
    assert concatfile(0) not in required
    assert concatfile(1) not in required
    assert maskfile(0)   not in required
    assert maskfile(1)   not in required

@then('LIA depends on XYZ image')
def LIA_depends_on_XYZ_image(dependencies):
    required, previous, task2outfile_map = dependencies

    expected_fn = LIA_file()
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
        assert xyz_file == XYZ_file()

@then('XYZ depends on DEM, DEMPROJ and BASE')
def XYZ_depends_on_DEM_DEMPROJ_and_BASE(dependencies):
    required, previous, task2outfile_map = dependencies

    expected_fn = XYZ_file()
    prev_expected = previous[expected_fn]
    expected_inputs = prev_expected.inputs
    assert len(expected_inputs) == 3
    assert {'indem', 'insar', 'indemproj'} == set(expected_inputs.keys())

    insar_as_inputs = expected_inputs['insar']
    assert len(insar_as_inputs) == 1
    insar_as_input = insar_as_inputs[0]
    assert insar_as_input['out_filename'] == input_file(0)

    indem_as_inputs = expected_inputs['indem']
    assert len(indem_as_inputs) == 1
    indem_as_input = indem_as_inputs[0]
    assert indem_as_input['out_filename'] == DEM_file()

    indemproj_as_inputs = expected_inputs['indemproj']
    assert len(indemproj_as_inputs) == 1
    indemproj_as_input = indemproj_as_inputs[0]
    assert indemproj_as_input['out_filename'] == DEMPROJ_file()

@then('DEMPROJ depends on DEM and BASE')
def DEMPROJ_depends_on_DEM_and_BASE(dependencies):
    required, previous, task2outfile_map = dependencies

    expected_fn = DEMPROJ_file()
    prev_expected = previous[expected_fn]
    expected_inputs = prev_expected.inputs
    assert len(expected_inputs) == 2
    assert {'indem', 'insar'} == set(expected_inputs.keys())

    insar_as_inputs = expected_inputs['insar']
    assert len(insar_as_inputs) == 1
    insar_as_input = insar_as_inputs[0]
    assert insar_as_input['out_filename'] == input_file(0)

    indem_as_inputs = expected_inputs['indem']
    assert len(indem_as_inputs) == 1
    indem_as_input = indem_as_inputs[0]
    assert indem_as_input['out_filename'] == DEM_file()

@then('DEM depends on BASE')
def DEM_depends_on_BASE(dependencies):
    required, previous, task2outfile_map = dependencies

    expected_fn = DEM_file()
    prev_expected = previous[expected_fn]
    expected_inputs = prev_expected.inputs
    assert len(expected_inputs) == 1
    assert {'insar'} == set(expected_inputs.keys())

    insar_as_inputs = expected_inputs['insar']
    assert len(insar_as_inputs) == 1
    insar_as_input = insar_as_inputs[0]
    assert insar_as_input['out_filename'] == input_file(0)

@then('a LIA task is registered')
def them_a_LIA_task_is_registered(tasks, dependencies):
    expectations = {
            LIA_file(): {'pipeline': 'Normals|LIA',
                'input_steps': {
                    XYZ_file(): ['xyz', FirstStep]
                    }}
            }
    required, previous, task2outfile_map = dependencies
    # logging.info("tasks (%s) = %s", type(tasks), tasks)
    assert isinstance(tasks, dict)
    assert len(tasks) >= 3
    assert len(required) == len(expectations)
    assert LIA_file() in required
    _check_registered_task(expectations, tasks, required, task2outfile_map)

@then('a XYZ task is registered')
def them_a_XYZ_task_is_registered(tasks, dependencies):
    expectations = {
            XYZ_file(): {'pipeline': 'SARCartesianMeanEstimation',
                'input_steps': {
                    DEM_file():     ['indem',     FirstStep],
                    DEMPROJ_file(): ['indemproj', FirstStep],
                    input_file(0):  ['insar',     FirstStep],
                    }}
            }
    required, previous, task2outfile_map = dependencies
    # logging.info("tasks (%s) = %s", type(tasks), tasks)
    assert isinstance(tasks, dict)
    _check_registered_task(expectations, tasks, [XYZ_file()], task2outfile_map)

@then('a DEMPROJ task is registered')
def them_a_DEMPROJ_task_is_registered(tasks, dependencies):
    expectations = {
            DEMPROJ_file(): {'pipeline': 'SARDEMProjection',
                'input_steps': {
                    DEM_file():     ['indem',     FirstStep],
                    input_file(0):  ['insar',     FirstStep],
                    }}
            }
    required, previous, task2outfile_map = dependencies
    # logging.info("tasks (%s) = %s", type(tasks), tasks)
    assert isinstance(tasks, dict)
    _check_registered_task(expectations, tasks, [DEMPROJ_file()], task2outfile_map)

@then('a DEM task is registered')
def them_a_DEM_task_is_registered(tasks, dependencies):
    expectations = {
            DEM_file(): {'pipeline': 'AgglomerateDEM',
                'input_steps': {
                    input_file(0):  ['insar',     FirstStep],
                    }}
            }
    required, previous, task2outfile_map = dependencies
    # logging.info("tasks (%s) = %s", type(tasks), tasks)
    assert isinstance(tasks, dict)
    _check_registered_task(expectations, tasks, [DEM_file()], task2outfile_map)

