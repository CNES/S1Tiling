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
# Helper script to install S1Tiling locally from binary releases.
# Expecting on the system:
# - conda
# - lmod module
# - g++, cmake, git...
#

## ======[ Globals {{{1
LANG=C

# ==[ Colors {{{2
declare -A _colors
_colors[reset]="$(tput sgr0)"
_colors[red]="$(tput setaf 1)"
_colors[green]="$(tput setaf 2)"
_colors[yellow]="$(tput setaf 3)"
_colors[blue]="$(tput setaf 4)"
_colors[magenta]="$(tput setaf 5)"
_colors[cyan]="$(tput setaf 6)"
_colors[white]="$(tput setaf 7)"
_colors[bold]="$(tput bold)"
_colors[blink]="$(tput blink)"

# ==[ Constant parameters {{{2
declare -A py_ver_for_otb
py_ver_for_otb[7.4.0]="3.8"
py_ver_for_otb[7.4.1]="3.8"
py_ver_for_otb[7.4.2]="3.8"
py_ver_for_otb[8.1.0]="3.11"
py_ver_for_otb[8.1.1]="3.11"
py_ver_for_otb[8.1.2]="3.11"
py_ver_for_otb[8.1.3]="3.11"
py_ver_for_otb[8.2.0]="3.11"

# s1tiling_version=1.0.0rc2
# git_node=develop
# s1tiling_version=1.1.0beta
# git_node=develop_worldcereal

# if HAL:
# python_ml_dep=python3.8.4-gcc8.2
# if TREX:
python_ml_dep=python3.8.4

repo_url=https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling.git

# ==[ Other constants {{{2

# Current date is used to install new versions as, for instance:
#     1.0.0rc2-otb7.4-20230620/
# and have a symbolic link to the exact installation path:
#     1.0rc2-otb7.4 -> 1.0rc2-otb7.4-20230620/
# This will ease the update of the version installed if need be without
# destabilising pre-existing versions.
date="$(date "+%Y%m%d")"
# public_prefix="${s1tiling_version}-otb${otb_version}"

## ======[ Helper functions {{{1
# ==[ _ask_yes_no            {{{2
function _ask_Yes_no()
{
    local prompt="$1"
    local response
    read -r -p "${prompt} [Y/O/n] " response
    response=${response,,}    # tolower
    [[ $response =~ ^(yes|y|o|oui|)$ ]]
}

# ==[ _current_dir           {{{2
function _current_dir()
{
    local depth=${1:-0}
    (cd "$(dirname "$(readlink -f "${BASH_SOURCE[${depth}]}")")" > /dev/null && pwd)
}
current_dir="$(_current_dir)"
# ----

# ==[ _extract_OTB_version   {{{2
function _extract_OTB_version()
{
    version="$(echo "${1}" | sed 's#.*OTB-\([0-9][^-_]*\).*#\1#')"
    echo "${version}"
}

# ==[ _die                   {{{2
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

# ==[ _execute               {{{2
# If $noexec is defined to 1, the execution is "dry" and does nothing.
function _execute()
{
    _verbose "$@"
    [ "${noexec:-0}" = "1" ] || "$@"
}

# ==[ _has_executable        {{{2
function _has_executable()
{
    command -v "$1" > /dev/null
}

# ==[ _is_false              {{{2
function _is_false()
{
    _is_unset $1 || [[ $1 -ne 1 ]]
}

# ==[ _is_set                {{{2
function _is_set()
{
    # [[ -v $1 ]] # with bash 4.2+, work with empty arrays
    [[ -n ${!1+x} ]] # doesn't work with empty arrays
}

# ==[ _is_true               {{{2
function _is_true()
{
    _is_set $1 && [[ $1 -eq 1 ]]
}

# ==[ _search_array          {{{2
# search all occurrences of pattern from array elements
# $1:  pattern
# $2+: array to filter
function _search_array()
{
    declare -a res
    local pat=$1
    shift
    for e in "$@"; do
        [[ "${e}" =~ "${pat}" ]] && res+=("$e")
    done
    echo "${res[@]}"
}

# ==[ _split_path            {{{2
function _split_path()
{
    local IFS=:
    local res=( $1 )
    echo "${res[@]}"
}

# ==[ _verbose               {{{2
__log_head='\033[36m$>\033[0m '
function _verbose()
{
    # if debug...
    echo -e "\n${__log_head:-}$*"
    # fi
}

# ==[ _version_Mm            {{{2
# Return Major.minor
function _version_Mm()
{
    separator=${2:- } # Optional; default: "space" to return 2 values
    echo "$1" | gawk -v sep="${separator}" -F. '{ printf("%d%s%d\n", $1,sep,$2); }';
}


## ======[ Restore colors, in all cases {{{1
function _restore_colors
{
    echo -en "${_colors[reset]}"
}
trap _restore_colors EXIT

## ======[ Parse parameters {{{1

usage() {
    echo "USAGE: $0 [OPTIONS] <OTB-X.X.X-Linux64.run> <s1tiling source directory>"
    echo
    echo "  -c|--clean             remove previous conda environment & OTB binaries"
    echo "  -p|--python <version>  override default python version for this OTB version"
    [ -z "$1" ] || {
        echo
        echo "$1"
        exit 127
    }
}

if [ $# -lt 2 ] ; then
    usage
    exit -1
fi

declare -a args
do_clean=0
py_version=
while [ $# -gt 0 ] ; do
    case $1 in
        -c|--clean)
            do_clean=1
            ;;
        -p|--python)
            shift
            [ $# -gt 1 ] || _die "Cannot read python version"
            py_version=$1
            ;;
        *)
            args+=("$1")
    esac
    shift
done

run_script="${args[0]}"
run_script_name="$(basename "${run_script}" ".run")"

s1tiling_src_dir="${args[1]}"

[ -f "${run_script}" ]                       || usage "Error: Non existant OTB binaries (${run_script})"
[[ "${run_script}" =~ OTB-.*-Linux64.*run ]] || usage "Error: Invalid OTB binaries (${run_script})"

[ -d "${s1tiling_src_dir}" ] || usage "Error: Non existant S1Tiling source directory (${s1tiling_src_dir})"
[ -d "${s1tiling_src_dir}/s1tiling" ] || usage "Error: Invalid S1Tiling source directory (${s1tiling_src_dir})"
[ -f "${s1tiling_src_dir}/setup.py" ] || usage "Error: Invalid S1Tiling source directory (${s1tiling_src_dir})"

prefix_root="$(dirname "$(readlink -f "${run_script}")")"
s1tiling_fulldir="$(readlink -f "${s1tiling_src_dir}" )"
module_paths=($(_split_path "${MODULEPATH}"))
module_root="$(_search_array "${HOME}" "${module_paths[@]}")"

otb_version=$(_extract_OTB_version "${run_script_name}")
otb_ver2=$(_version_Mm "${otb_version}" ".")
py_version=${py_version:-${py_ver_for_otb[${otb_version}]}}

short_otb_version=$(echo "${otb_version}" | sed 's#\.##g')
short_py_version=$(echo "${py_version}" | sed 's#\.##g')

otb_basename_prefix="${run_script_name}-py${short_py_version}"
otb_prefix="${prefix_root}/${otb_basename_prefix}"

env_name="s1tiling-otb${short_otb_version}-py${short_py_version}"
mod_name="s1tiling/otb${short_otb_version}-py${short_py_version}"

echo "OTB version:      ${otb_version} -> ${short_otb_version}"
echo "Python version:   ${py_version}  -> ${short_py_version}"
echo "OTB install dir:  ${prefix_root}"
echo "OTB_DIR:          ${otb_prefix}"
echo "Conda environment ${env_name}"
echo "Module root:      ${module_root}"
echo "Module name:      ${mod_name}"

_ask_Yes_no "${_colors[cyan]}Continue?" || exit 0
echo -en "${_colors[reset]}"

## ======[ Main installation script {{{1

# ==[ Activate conda
if ! _has_executable conda ; then
    _verbose "module load conda"
    module load conda || _die "Cannot load conda..."
fi

# ==[ Shall clean?

if _is_true ${do_clean} ; then
    _execute conda remove -n "${env_name}" --all
    _execute rm -r "${otb_prefix}"
fi

# ==[ Create virtual environment
_execute conda create -n "${env_name}" python==${py_version}

_verbose conda activate "${env_name}"
conda activate "${env_name}" || _die "Cannot activate ${env_name}"

_execute python --version

_execute cd "${prefix_root}" || _die "Can't cd to installation base directory ${prefix_root}"

# ==[ Prepare the virtual env
_execute python -m pip install --upgrade pip                || _die "Can't upgrade pip"
_execute python -m pip install --upgrade setuptools==57.5.0 || _die "Can't upgrade setuptools to v57.5.0"
_execute python -m pip --no-cache-dir install numpy         || _die "Can't install numpy from scratch"

# ==[ Extract OTB binaries
[ -d "${otb_prefix}" ] \
    && echo ">> ${run_script_name}.run already extracted into '${otb_prefix}'..."  \
    || { 
    _execute bash "${run_script_name}.run" --nox11 --target "${otb_basename_prefix}"
    _execute cd "${otb_basename_prefix}"
    # # Inject ${CMAKE_PREFIX_PATH}/lib into LD_LIBRARY_PATH 
    # _execute patch -p1 --ignore-whitespace < "${current_dir}/OTB-env.patch"
    echo "# LD_LIBRARY_PATH patch for s1tiling" >> "otbenv.profile"
    echo 'export LD_LIBRARY_PATH="${CMAKE_PREFIX_PATH}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' >> "otbenv.profile" \
    # Make sure to compile with new C++ ABI with OTB 7.4.2
    [[ "${otb_version}" != "7.4.2" ]] || _execute patch -p1 --ignore-whitespace < "${current_dir}/OTB-CXX-ABI.patch"
}   || _die "Cannot extract OTB binaries"

_verbose source "${otb_prefix}/otbenv.profile"
source "${otb_prefix}/otbenv.profile" || _die "Cannot source OTB env"

_execute ctest -VV -S "${otb_prefix}/share/otb/swig/build_wrapping.cmake" -VV \
    || _die "Cannnot recompile OTB bindings for Python ${py_version}"

# ==[ Tune GDAL
_execute cp "${s1tiling_fulldir}/s1tiling/resources/gdal-config" "${otb_prefix}/bin" \
    || _die "Cannot copy gdal-config patch into OTB binaries"
_execute chmod +x "${otb_prefix}/bin/gdal-config" \
    || _die "Cannot make gdal-config executable"

_execute python -m pip --no-cache-dir install "gdal==$(gdal-config --version)" --no-binary :all: \
    || _die "Cannot install GDAL python bindings"


# Check if GDAL fulfils all S1Tiling requirements
echo -e "\n# Check GDAL is compatible with S1Tiling requirements..."

_execute python -c "from osgeo import gdal ; print('GDAL version:', gdal.__version__)" \
    || _die "GDAL is not correctly installed"
_execute python -c "from osgeo import gdal_array" \
    || _die "GDAL (from OTB?) is not correctly installed (_GLIBCXX_USE_CXX11_ABI mismatch)"
# if mismatch, see https://github.com/OSGeo/gdal/issues/4724#issuecomment-953734163

function _test_gdal_gpkg
{
    _verbose "gdalinfo --formats | grep -qi gpkg"
    gdalinfo --formats | grep -qi gpkg
}
_test_gdal_gpkg || _die "GDAL lacks GPKG support"

# ==[ Install S1Tiling
_execute cd "${s1tiling_fulldir}"
# TODO: add options for dev/docs
_execute python -m pip install -e .[dev,docs]

# ==[ And create the modulefile!
if _has_executable module ; then
    export module_file="${module_root}/${mod_name}.lua"
    _verbose "Create modulefile: ${module_file}"
    cat > "${module_file}" << EOF
-- -*- lua -*-
-- Aide du module accessible avec la commande module help
help(
[[
OTB:       ${otb_version}
Python:    ${py_version}
S1Tiling:  ${s1tiling_fulldir}
Conda env: ${env_name}
]])

local function is_empty(s)
  return s == nil or s == ''
end

-- Information du modulefile
local nom           = "s1tiling+otb"
local home          = os.getenv("HOME")
local version       = "${otb_version}"
local installation  = "$(date)"
local pkgName   = myModuleName()
-- TODO: extract reldeb from pkgName
local pkg       = "${otb_prefix}"
local conda_pkg = pathJoin(home,"local","miniconda3")

-- Information du module accessible avec la commande module whatis
whatis("Nom     : "..nom)
whatis("Version : "..version)
whatis("pkgName : "..pkgName)
whatis("Date d installation : "..installation)

-- Action du modulefile
setenv("GDAL_DATA",pathJoin(pkg,"share/gdal"))
setenv("PROJ_LIB", pathJoin(pkg,"share/proj"))
setenv('GDAL_DRIVER_PATH', 'disable')
prepend_path("CPATH",pathJoin(pkg,"include"))
prepend_path("PYTHONPATH",pathJoin(pkg,"lib/python"))
setenv("CMAKE_PREFIX_PATH",pathJoin(pkg, "lib/cmake/OTB-${otb_ver2}"))

setenv("OTB_HOME",pkg)
setenv("OTB_VER", version)

setenv("S1TILING_HOME",pkg)
prepend_path("LD_LIBRARY_PATH", pathJoin(pkg, "lib"))
-- prepend_path("OTB_APPLICATION_PATH", pathJoin(pkg, "lib"))
prepend_path("PATH",pathJoin(pkg,"bin"))
prepend_path("CPATH",pathJoin(pkg,"include", 'OTB-${otb_ver2}'))
prepend_path("LD_LIBRARY_PATH",pathJoin(pkg,"lib"))

prepend_path("OTB_APPLICATION_PATH",pathJoin(pkg,"lib/otb/applications"))
prepend_path("LD_LIBRARY_PATH",pathJoin(conda_pkg,"envs/${env_name}/lib"))

-- Emulate conda activate
execute{cmd='source '..conda_pkg..'/etc/profile.d/conda.sh', modeA={'load'}}
if     (mode() == 'load') then
    depends_on('conda')
    execute{cmd='conda activate ${env_name}',  modeA={'load'}}
elseif (mode() == 'unload') then
    execute{cmd='conda deactivate',     modeA={'unload'}}
    depends_on('conda')
end
EOF

    # ==[ Check S1Tiling
    # we should unload otbenv.profile for the test... but it's not possible...
    _execute conda deactivate
    _verbose module load "${mod_name}"
    module load "${mod_name}" || _die "Cannot load ${mod_name}"
else
    _verbose source "${otb_prefix}/otbenv.profile"
    source "${otb_prefix}/otbenv.profile" || _die "Cannot source OTB env"
    _verbose conda activate "${env_name}"
    conda activate "${env_name}" || _die "Cannot activate ${env_name}"
fi

_execute S1Processor --version      || _die "S1Tiling isn't properly installed"

echo "#> ${_colors[green]}${_colors[blink]}Installation complete"

# }}}
# vim: set foldmethod=marker:
