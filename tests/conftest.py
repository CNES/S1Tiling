#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pathlib
import argparse

# - ${S1TILING_TEST_DATA_OUTPUT}
# - ${S1TILING_TEST_DATA_INPUT}
# - ${S1TILING_TEST_SRTM}
# - ${S1TILING_TEST_TMPDIR}
# - ${S1TILING_TEST_DOWNLOAD}
# - ${S1TILING_TEST_RAM}

def dir_path(path):
    if os.path.isdir(path):
        return pathlib.Path(path)
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid directory")

def pytest_addoption(parser):
    crt_dir = pathlib.Path(__file__).parent.absolute()
    src_dir = crt_dir.parent.absolute()
    test_dir = (crt_dir.parent.parent / "tests").absolute()

    parser.addoption("--baselinedir", action="store",      default=crt_dir/'baseline',                 type=dir_path, help="Directory where the baseline is")
    parser.addoption("--outputdir",   action="store",      default=crt_dir/'output',                   type=dir_path, help="Directory where the S2 products will be generated. Don't forget to clean it eventually.")
    parser.addoption("--tmpdir",      action="store",      default=crt_dir/'tmp',                      type=dir_path, help="Directory where the temporary files will be generated. Don't forget to clean it eventually.")
    parser.addoption("--srtmdir",     action="store",      default=os.getenv('SRTM_DIR', '$SRTM_DIR'), type=dir_path, help="Directory where SRTM files are - default: $SRTM_DIR")
    parser.addoption("--ram",         action="store",      default='4096'                            , type=int     , help="Available RAM allocated to each OTB process")
    parser.addoption("--download",    action="store_true", default=False, help="Download the input files with eodag instead of using the compressed ones from the baseline. If true, raw S1 products will be downloaded into {tmpdir}/inputs")
    parser.addoption("--watch_ram",   action="store_true", default=False, help="Watch memory usage")

def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_list = ['baselinedir', 'srtmdir', 'download', 'outputdir', 'tmpdir', 'watch_ram', 'ram']
    for option in option_list:
        value = getattr(metafunc.config.option, option)
        # print("%s ===> %s // %s" % (option, value, option in metafunc.fixturenames))
        # value = metafunc.config.option.baselinedir
        if option in metafunc.fixturenames and value is not None:
            metafunc.parametrize(option, [value])
