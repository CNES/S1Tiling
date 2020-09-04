#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess


def otb_compare(baseline, result):
    """
    Compare the images produced by the test
    """
    args = ['otbTestDriver',
            '--compare-image', '1e-12', baseline, result,
            'Execute', 'echo', '"running OTB Compare"',
            '-testenv']
    print(args)
    return subprocess.call(args)
