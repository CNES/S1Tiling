#!/bin/bash
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2021 (c) CESBIO. All rights reserved.
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

# Fail at first error
set -e

## Support functions {{{1

# _current_dir {{{2
_current_dir() {
    (cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" > /dev/null && pwd)
}

# _dir_exists {{{2
function _dir_exists()
{
    [[ -d "$1" ]]
}

# _dir_dont_exists {{{2
function _dir_dont_exists()
{
    [[ ! -d "$1" ]]
}

# _execute {{{2
function _execute()
{
    echo '$> ' "$@"
    [ "${dryrun:-0}" = "1" ] || "$@"
}

# _is_set {{{2
function _is_set()
{
    # [[ -v $1 ]] with bash 4.2+
    [[ -n ${!1+x} ]]
}

# _mkdir {{{2
function _mkdir()
{
    _dir_exists "$1" || _execute mkdir -p "$1"
}

# _rmdir {{{2
function _rmdir()
{
    _dir_dont_exists "$1" || _execute rm -r "$1"
}

## The script {{{1

usage() {
    echo "USAGE: $0 <dest-dir>"
}

if [ $# -lt 1 ] ; then
    usage
    exit -1
fi

public="$1"
_is_set CI_COMMIT_REF_NAME  \
    && version="${CI_COMMIT_REF_NAME}" \
    || version="$(git describe --tags --exact-match 2> /dev/null || git symbolic-ref -q --short HEAD)"

echo "Installing documentation for v:${version} in ${public}"

[[ "${version}" =~ ^[0-9]+(\.[0-9]+)*$ ]] && is_full_tag=1 || is_full_tag=0
echo "Is this a non-release-candidate tag? ${is_full_tag}"

# Generate doc in _build
echo "Build documentation"
_execute sphinx-build -b html -d _build/doctrees docs _build/html -v

# Make sure public exists
echo "Move documentation into '${public}/'" directory
_mkdir "${public}"

# Move result in public/${version]
_rmdir "${public}/${version}"
_execute mv _build/html "${public}/${version}"

# When releasing a new official version, remove release candidates
# expect version tag =~ 'M.m.p'
# expect RC tag =~ 'M.m.p-rcX'
if [ ${is_full_tag} -eq 1 ] ; then
    echo "This is a new relase. Removing release candidates ${version}-* ..."
    _rmdir "${public}/${version}-"*
fi

# Prepare latest/ as a copy of the latest version
# (using a copy instead of a symlonk because gitlab-ci or gitlab-pages don't
# seem to support symlink)
# Automatically symlink 'latest' to latest version, or develop
latest=$(((find "${public}"  -maxdepth 1 -mindepth 1 -type d -name "*.*" -printf "%P\n" | grep  .) 2> /dev/null || echo develop) | sort -V | tail -1)
echo "Latest version found: '${latest}'"
if [ "${latest}" = "${version}" ] ; then
    echo " => update ${public}/latest"
    _rmdir "${public}/latest"
    _execute cp -r "${public}/${latest}" "${public}/latest"
fi

# Generate list of versions
echo "Generate 'versions.html' menu"
_execute "$(_current_dir)"/list_versions.py "${public}"

# Make sure that public/index.html is redirected

# }}}1
# vim:set foldmethod=marker:
