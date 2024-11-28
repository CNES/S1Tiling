#!/bin/bash
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2024 (c) CNES. All rights reserved.
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

# Script meant to be executed in CI dockers to simulate the existence of valid
# credential files.
#

# =====[ _die
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

# =====[ Forge eodag config file
eodag_config="${HOME}/.config/eodag/eodag.yml"
[ -f "${eodag_config}" ] && _die "No previous eodag configuration shall exist"

# The actual l/p doesn't matter as the cassettes will be played on the CI
mkdir -p "${HOME}/.config/eodag"
cat << end_eodag > "${eodag_config}"
cop_dataspace:
  priority: 2 # Lower value means lower priority (Default: 0)
  search:   # Search parameters configuration
  auth:
    credentials:
      username: dummy_login
      password: dummy_password
end_eodag

# =====[ Forge .netrc
netrc="${HOME}/.netrc"
[ -f "${netrc}" ] && _die "No previous .netrc configuration shall exist"
cat << end_netrc > "${netrc}"
machine urs.earthdata.nasa.gov
  login    dummy_password
  password dummy_password
end_netrc
chmod og-rwx "${netrc}"
