#!/bin/bash
s1tiling_version=1.0.0rc2
otb_ver=7.4

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
prefix_root=/softs/projets/s1tiling/rh7

cd ${prefix_root}

ml "otb/${otb_ver}-python3.8.4-gcc8.2"
python -m venv "${env}"

source "${env}/bin/activate"
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools==57.5.0
python -m pip --no-cache-dir install numpy

# Check
# python -c "from osgeo import gdal ; print('GDAL version:', gdal.__version__)"
python -c "from osgeo import gdal_array"

cd "${env}"
git clone https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling.git repo
cd repo
# git checkout develop
git checkout tags/${s1tiling_version}
python -m pip install .
S1Processor --version

cd ${prefix_root}
ln -s "${env}" "${public_prefix}"

# And create the modulefile!
cat > "/softs/projets/modulefiles/s1tiling/${public_prefix}.lua" << EOF
-- -*- lua -*-
-- Aide du module accessible avec la commande module help
help(
[[
Version disponible sous rh7
]])

-- Information du modulefile
local os_disponible = "rh7"
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
check_os(os_disponible)

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
depend("otb/${otb_ver}-python3.8.4-gcc8.2")
EOF

