#!/bin/bash

# export LANG=en_US.utf8
# RuntimeError: Click will abort further execution because Python was
# configured to use ASCII as encoding for the environment. Consult
# https://click.palletsprojects.com/unicode-support/ for mitigation
# steps.
# This system supports the C.UTF-8 locale which is recommended. You
# might be able to resolve your issue by exporting the following
# environment variables:
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
. "${OTB_INSTALL_DIRNAME}/otbenv.profile"
. "${S1TILING_VENV}/bin/activate"
if [ "$1" = "--lia" ] ; then
    shift
    S1LIAMap "$@"
else
    S1Processor "$@"
fi
