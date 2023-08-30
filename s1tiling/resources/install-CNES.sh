#!/bin/bash
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2023 (c) CNES. All rights reserved.
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
# Helper script to install S1Tiling on CNES HPC centres.
#

## Globals {{{1
# Constant parameters {{{2
s1tiling_version=1.0.0rc2
otb_ver=7.4.2
# HAL
# python_ml_dep=python3.8.4-gcc8.2
# TREX
python_ml_dep=python3.8.4

repo_url=https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling.git

# Other constants {{{2

RH_FLAVOR=$(cat /etc/redhat-release)
RH_FLAVOR=${RH_FLAVOR#* release}
RH_FLAVOR=${RH_FLAVOR:1:1}

# Current date is used to install new versions as, for instance:
#     1.0.0rc2-otb7.4-20230620/
# and have a symbolic link to the exact installation path:
#     1.0rc2-otb7.4 -> 1.0rc2-otb7.4-20230620/
# This will ease the update of the version installed if need be without
# destabilising pre-existing versions.
date="$(date "+%Y%m%d")"
public_prefix="${s1tiling_version}-otb${otb_ver}"
env="${public_prefix}-${date}"

# Where it will be installed
# -> HAL
# projets_root="/softs/projets"
# -> TREX
projets_root="/work/softs/projets"

prefix_root="${projets_root}/s1tiling/rh${RH_FLAVOR}"
module_root="${projets_root}/modulefiles/s1tiling"

## Helper functions {{{1
# _verbose                          {{{2
__log_head='\033[36m$>\033[0m '
function _verbose()
{
    # if debug...
    echo -e "\n${__log_head:-}$*"
    # fi
}

# _die                              {{{2
function _die()
{
   local msg=$1
   [ -z "${msg}" ] && msg="Died"
   # echo "BASH_SOURCE: ${#BASH_SOURCE[@]}, BASH_LINENO: ${#BASH_LINENO[@]}, FUNCNAME: ${#FUNCNAME[@]}"
   printf "${BASH_SOURCE[0]:-($0 ??)}:${BASH_LINENO[0]}: ${FUNCNAME[1]}: ${msg}\n" >&2
   for i in $(seq 2 $((${#BASH_LINENO[@]} -1))) ; do
       printf "called from: ${BASH_SOURCE[$i]:-($0 ??)}:${BASH_LINENO[$(($i-1))]}: ${FUNCNAME[$i]}\n" >&2
   done
   # printf "%s\n" "${msg}" >&2
   exit 127
}

# _execute                          {{{2
# Si $noexec est définie à 1, l'exécution ne fait rien
function _execute()
{
    _verbose "$@"
    [ "${noexec:-0}" = "1" ] || "$@"
}


## Main installation script {{{1

_execute cd "${prefix_root}" || _die "Can't cd to installation base directory ${prefix_root}"

_verbose ml "otb/${otb_ver}-${python_ml_dep}"
ml "otb/${otb_ver}-${python_ml_dep}" || _die "Can't load module otb/${otb_ver}-${python_ml_dep}"

[ -f "${env}/bin/activate" ] || _execute python -m venv "${env}" || _die "Can't create virtual environment '${env}'"

_verbose source "${env}/bin/activate"
source "${env}/bin/activate"

_execute python -m pip install --upgrade pip                || _die "Can't upgrade pip"
_execute python -m pip install --upgrade setuptools==57.5.0 || _die "Can't upgrade setuptools to v57.5.0"
_execute python -m pip --no-cache-dir install numpy         || _die "Can't install numpy from scratch"

# Check GDAL fulfills all S1Tiling requirements
echo -e "\n# Check GDAL is compatible with S1Tiling requirements..."

# python -c "from osgeo import gdal ; print('GDAL version:', gdal.__version__)"
_execute python -c "from osgeo import gdal_array" || _die "GDAL (from OTB?) is not correctly installed (_GLIBCXX_USE_CXX11_ABI mismatch)"
# if mismatch, see https://github.com/OSGeo/gdal/issues/4724#issuecomment-953734163

function _test_gdal_gpkg
{
    _verbose "gdalinfo --formats | grep -qi gpkg"
    gdalinfo --formats | grep -qi gpkg
}
_test_gdal_gpkg || _die "GDAL lacks GPKG support"


_execute cd "${env}"
[ -d "repo" ] || _execute git clone "${repo_url}" "repo" || _die "Can't clone S1Tiling repository"
_execute cd repo || _die "Repository hasn't been cloned properly..."
# _execute git checkout develop
_execute git checkout "tags/${s1tiling_version}" || _die "Can't checkout tag ${s1tiling_version}"
_execute python -m pip install .                 || _die "Can't install S1Tiling (from repo)"
_execute S1Processor --version                   || _die "S1Tiling isn't properly installed"

_execute cd "${prefix_root}"
_execute chmod -R go+rX "${env}"
[ -h "${public_prefix}" ] && _execute rm "${public_prefix}"
_execute ln -s "${env}" "${public_prefix}"

# And create the modulefile!
export module_file="${module_root}/${public_prefix}.lua"
_verbose "Create modulefile: ${module_file}"
cat > "${module_file}" << EOF
-- -*- lua -*-
-- Aide du module accessible avec la commande module help
help(
[[
Version disponible sous rh${RH_FLAVOR}
]])

local function is_empty(s)
  return s == nil or s == ''
end

-- Information du modulefile
local os_disponible = "rh${RH_FLAVOR}"
local nom           = "s1tiling"
local version       = "${public_prefix}"
local installation  = "$(date)"
local informations  = system_information()
local rhos          = informations.os

-- Information du module accessible avec la commande module whatis
whatis("Nom     : "..nom)
whatis("Version : "..version)
whatis("Os      : "..os_disponible)
whatis("Date d installation : "..installation)

-- Verification du modulefile
-- check_os(os_disponible) -- on HAL only, not on TREX...

-- Variable du modulefile
local home=pathJoin("/softs/projets/s1tiling",rhos,version)

-- Action du modulefile
setenv("S1TILING_HOME",home)

-- Emule activate des virtualenv python...
pushenv("VIRTUAL_ENV", home)
prepend_path("PATH", pathJoin(home, "bin"))
if is_empty(os.getenv("VIRTUAL_ENV_DISABLE_PROMPT")) then
    pushenv("PS1", "(s1tiling "..version..") ".. os.getenv("PS1"))
end

-- Dependances
depend("otb/${otb_ver}-${python_ml_dep}")
EOF

# TODO: Use ACL when bug is fixed on TREX!
_execute chmod u+rw "${module_file}"
_execute chmod go+r "${module_file}"

# }}}
# vim: set foldmethod=marker:
