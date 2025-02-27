#!/usr/bin/bash
# =========================================================================
#   Program:   apply-calibration-map.sh
#
#   Copyright 2017-2025 (c) CNES. All rights reserved.
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

# This program converts S1Tiling sigma0 orthorectified products to beta0 or
# gamma0 calibration

# Parameters
# - --mapdir:     Directory where cosine files are stored
#   		  Typical pattern:
#   		  	cos_IA_{platform}_{tile}_{relobt}.tif
#   		  Example:
#   		  	cos_IA_s1a_31TCH_008.tif
#
# - $sigma_files: list of sigma calibrated files to convert
#   		  Expected pattern:
#   		  	{platform}_{tile}_{polar}_{dir}_{relobt}_{timestamp}[_sigma].tif
#   		  Example:
#   		  	s1a_31TCH_vh_DES_110_20231107t060120_NormLim.tif
#
# Warning: other naming scheme are not supported!
# | If you are using other filename formats, you'll have to change the
# | create-recalibration-target function in consequence.
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

# ==[ _dirname_n <N> <PATH>  {{{2
function _dirname_n()
{
    local nb="$1"
    local left
    left=$((nb - 1))
    local path="$2"
    [ ${nb} -le 0 ] && echo "${path}" || _dirname_n ${left} "$(dirname "${path}")"
}

# ==[ _execute               {{{2
# If $noexec is defined to 1, the execution is "dry" and does nothing.
function _execute()
{
    _verbose "$@"
    [ "${noexec:-0}" = "1" ] || "$@"
}

# ==[ _filter_array          {{{2
# remove all occurrences of pattern from array elements
# $1:  pattern
# $2+: array to filter
function _filter_array()
{
    declare -a res
    local pat=$1
    shift
    for e in "$@"; do
        [[ "${e}" != "${pat}" ]] && res+=("$e")
    done
    echo "${res[@]}"
}

# ==[ _glob_ext              {{{2
# globbing with `eval res=(*(pattern))`  doesn't always work correclty
function _glob_ext()
{
    declare -n result="$1" ; shift
    basename="$1" ; shift
    # printf " # %s\n" "${result[@]}"
    for ext in "$@" ; do
        # printf " && %s\n" "${result[@]}"
        # [ -f "${basename}.${ext}" ] && result+=("${basename}.${ext}")
        for f in ${basename}.${ext} ; do
            [ -f "$f" ] && result+=("$f") # && echo "globbed: $f"
        done
    done
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

# ==[ _is_unset              {{{2
function _is_unset()
{
    # [[ ! -v $1 ]] # with bash 4.2+, work with empty arrays
    [[ -z ${!1+x} ]] # doesn't work with empty arrays
}

# ==[ _join_by               {{{2
# > join_by sep elem1 elem2...
# http://stackoverflow.com/questions/1527049/join-elements-of-an-array/#17841619
# This comment and code Copyleft, selectable license under the GPL2.0 or
# later or CC-SA 3.0 (CreativeCommons Share Alike) or later. (c) 2008.
# All rights reserved. No warranty of any kind. You have been warned.
# http://www.gnu.org/licenses/gpl-2.0.txt
# http://creativecommons.org/licenses/by-sa/3.0/
function _join_by()
{
    local d=$1; shift; echo -n "$1"; shift; printf "%s" "${@/#/$d}";
}
# ----

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

# ==[ _split_into_by         {{{2
function _split_into_by()
{
    local res="$1"
    local IFS="$2"
    read -r -a "${res}" <<< "$3"
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

## ======[ Restore colors, in all cases {{{1
function _restore_colors
{
    echo -en "${_colors[reset]}"
}
trap _restore_colors EXIT

## ======[ Parse parameters {{{1

usage() {
    [ -z "$1" ] || echo
    echo "Usage: $0 [OPTIONS] INPUTS..."
    echo
    echo "  This program converts S1Tiling sigma0 orthorectified products to beta0 or"
    echo "  gamma0 calibration"
    echo
    echo "Parameters:"
    echo "  INPUTS...         List of sigma0 calibrated files to convert, or directory where to search them"
    echo "                    Expected format: {platform}_{tile}_{pol}_{dir}_{obt}_{timestamp}[_sigma].tif"
    echo "                    Example:         s1a_31TCH_vh_DES_110_20231107t060120.tif"
    echo
    echo "Options:"
    echo "  --mapdir <dir>    Directory where sin/cos/tan maps are stored"
    echo "                    Expected format: (sin|cos)_IA_{platform}_{tile}_{relobt}.tif"
    echo "                    Example:         cos_IA_s1a_31TCH_008.tif"
    echo
    echo "  -c|--calibration  (beta|gamma)"
    # echo "  -v|--verbose      Verbose mode"
    echo "  -n|--dryrun       Don't execute anything but display what would have been"
    echo "                    executed by this program"
    echo "  -h|--help         Show this message and exit."

    [ -z "$1" ] && {
        echo
        echo "  This tools is part of S1Tiling. See also: S1IAMap"
        echo
        echo "  Check out our docs at https://s1-tiling.pages.orfeo-toolbox.org/s1tiling/latest/ for more details."
        echo "  Copyright 2017-2025 (c) CNES."
    }|| {
        echo
        echo "${_colors[red]}Error:${_colors[reset]} $1"
        exit 127
    }
}

if [ $# -lt 2 ] ; then
    usage
    exit -1
fi

declare -a sigma_files
ia_dir=
calibration=
# verbose=0
while [ $# -gt 0 ] ; do
    case $1 in
        --mapdir)
            shift
            [ $# -ge 1 ] || usage "Cannot read IA maps directory: no more parameters"
            ia_dir="$1"
            ;;
        -c|--calibration)
            shift
            [ $# -ge 1 ] || usage "Cannot read calibration: no more parameters"
            calibration="$1"
            ;;
        # -v|--verbose)
        #     verbose=1
        #     ;;
        -n|--dryrun)
            noexec=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            usage "Unexpected parameter '$1'"
            ;;
        *)
            if [ -f "$1" ] ; then
                sigma_files+=("$1")
            elif [ -d "$1" ] ; then
                # Filter only the files compatible with the expected
                # pattern
                sigma_files+=($(find "$1" -regextype egrep -regex '.*/s1[a-e]_.{5}_(vv|vh|hv|hh)_(DES|ASC)_[0-9]{3}_[0-9]{8}t([0-9]{6}|x{6})(_sigma)?\.tiff?'))
            else
                # sigma_files+=("$1")
                usage "'$1' is not a valid input file"

            fi
            args+=("$1")
    esac
    shift
done

[ -n "${ia_dir}" ]       || usage "--mapdir hasn't been set"
[ -d "${ia_dir}" ]       || usage "Cannot access specified IA directory '${ia_dir}'"
[ -n "${calibration}" ]  || usage "--calibration hasn't been set"
if [[ ! " beta gamma " =~ " ${calibration} " ]] ; then
    usage "Unexpected value for calibration: '${calibration}'"
fi

echo "IA directory: ${_colors[green]}${ia_dir}${_colors[reset]}"
echo   "Input σ° files:"
printf -- "         - ${_colors[green]}%s${_colors[reset]}\n" "${sigma_files[@]}"

_ask_Yes_no "${_colors[cyan]}Continue?" || exit 0
echo -en "${_colors[reset]}"

## ======[ Domain functions {{{1
# _analyse_sigma_file_name {{{2
# expect:
# - a basename
# return:
# - platform
# - tilename
# - relative orbit number
# - calibration
# - extension
# - name without the calibration nor the extention
function _analyse_sigma_file_name_into()
{
    declare -n res="$2"
    res[ext]=${1#*.}
    basename=${1%.*}
    # echo "basename: ${basename}"

    declare -a parts
    _split_into_by parts _ "$1"

    res[platform]="${parts[0]}"
    res[tilename]="${parts[1]}"
    res[relorbit]="${parts[4]}"
    res[calib]="${parts[6]}"
    res[prefix]="$(_join_by "_" "${parts[@]:0:6}")"
}

## ======[ Main script {{{1

declare -A ia_map_kind
ia_map_kind[beta]="sin"
ia_map_kind[gamma]="cos"

declare -A ia_calc_exp
ia_calc_exp[beta]="im1b1 / im2b1"
ia_calc_exp[gamma]="im1b1 / im2b1"

for sig_file in "${sigma_files[@]}"; do
    basename="$(basename "${sig_file}")"
    declare -A info
    _analyse_sigma_file_name_into "${basename}" info

    dirname="$(dirname "${sig_file}")"
    outfile="${dirname}/${info[prefix]}_${calibration}.${info[ext]}"
    # echo "outfile: ${outfile}"

    info[iaregex]="${ia_map_kind[${calibration}]}_IA_s1?_${info[tilename]}_${info[relorbit]}"
    # info[iaregex]="sin_IA_s1?_${info[tilename]}_${info[relorbit]}"
    # printf "%s\n" "${info[@]@K}"
    IA_files=()
    iapattern="${ia_dir}/${info[iaregex]}"
    _glob_ext IA_files "${iapattern}" tiff tif
    # printf " ~ %s\n" "${IA_files[@]}"
    [ "${#IA_files[@]}" -eq 1 ] || _die "No compatible IA file found for ${_colors[green]}${sig_file}${_colors[reset]}\n--> '${_colors[green]}${iapattern}.{tif,tiff}${_colors[reset]}'"

    if [ -f "${outfile}" ] ; then
        echo "Ignore existing ${_colors[green]}${outfile}${_colors[reset]}"
    else
        _execute otbcli_BandMath -il "${sig_file}" "${IA_files[0]}" -out "${outfile}" -exp "${ia_calc_exp[${calibration}]}"
        _execute gdal_edit.py -mo CALIBRATION=${calibration} "${outfile}"
    fi
done

# }}}
# vim: set foldmethod=marker:
