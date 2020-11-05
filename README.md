# S1Tiling
On demand Ortho-rectification of Sentinel-1 data on Sentinel-2 grid.

Sentinel-1 is currently the only system to provide SAR images regularly on all
lands on the planet. Access to these time series of images opens an
extraordinary range of applications. In order to meet the needs of a large
number of users, including our needs, we have created an automatic processing
chain to generate _"Analysis Ready"_ time series for a very large number of
applications.

Sentinel-1 data is ortho-rectified on the Sentinel-2 grid to promote joint use
of both missions.

S1Tiling was developed within the CNES radar service, in collaboration with
CESBIO, to generate time series of calibrated, ortho-rectified and filtered
Sentinel-1 images on any terrestrial region of the Earth. The tool benefits for
the SAR ortho-rectification application
[from the Orfeo Tool Box](https://www.orfeo-toolbox.org/).

The resulting images are registered to Sentinel-2 optical images, using the
same MGRS geographic reference. You will be able to access Sentinel-1 data
acquired on Sentinel-2 31TCJ or 11SPC tiles.This Python software, is based on
the Orfeo Tool Box (OTB) image processing library, developed by CNES, as well
as on [the PEPS platform](https://peps.cnes.fr/) to
access the Sentinel-1 data. It can be used on any type of platform, from a
large computing cluster to a laptop (the fan will make some noise during
processing). It is considerably faster than the ortho-rectification tool in
SNAP, and can be easily used in script form.

S1Tiling is currently used for many applications, such deforestation detection
in the Amazon, monitoring of rice crops in Southeast Asia or monitoring of
water stocks in India.In addition, this software is accessible as an on-demand
processing service on the French PEPS collaborative ground segment, in order to
make it easier for users to use.

# Installation

TBC

## Requirements

* OTB
* OTB external modules
  * SARMutitempFiltering
  * Synthetize
  * ClampROI
* GDAL with python bindings as well
* Python 3
  * pickle
  * json
  * zipfile
  * xml
  * timeit
  * numpy
  * osgeo.gdal, osgeo.gdalconst
  * ogr
  * rasterio
  * yaml
  * ConfigParser
  * Dask"distributed"
  * bokeh (to display Dask dashboard)
  * graphviz (to generate task graphs)

## Setup

### On HAL cluster

First we need to create a conda environment 

`module load conda`

`conda create -n s1tiling python==3.7.2`

`conda activate s1tiling`

`pip install gdal==3.1.0`

Define the following env variables

`export GDAL_DATA=/home/il/koleck/.conda/envs/s1tiling-hpc/lib/python3.7/site-packages/rasterio/gdal_data`

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/il/koleck/OTB-7.2.0-Linux64/lib/`

Then we have to clone S1tiling git repository and install S1tiling packages

`git clone https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling.git`

`cd s1tiling`

`pip install --use-features=2020-resolver -e .`

Source the OTB environement

`source ~/OTB-7.2.0-Linux64/otbenv.profile`


### Dask
> Dask does not require any setup if you only want to use it on a single computer.
> -- https://docs.dask.org/en/latest/setup.html

https://docs.dask.org/en/latest/setup/single-distributed.html

# Design Notes

S1Tiling processes the requested S2 tiles one after the other.

For each S2 tile:

1. It first retrieves the S1 images that intersect the S2 tile at the given
   time range
2. Then it builds a graph of tasks to realize (Calibration+Cutting,
   Orthorectification, Concatenation, Mask generation). Each node of the graph
   corresponds to a file that is meant to be produced. The graph is trimmed
   from the edges that produce the expected files that already exist.
3. Finally, all the remaining tasks are executed in parallel, and in order,
   thanks to Dask.

# Copyright

>   Copyright (c) CESBIO. All rights reserved.
>
>   Licensed under the Apache License, Version 2.0 (the "License");
>   you may not use this file except in compliance with the License.
>   You may obtain a copy of the License at
>
>       http://www.apache.org/licenses/LICENSE-2.0
>
>   Unless required by applicable law or agreed to in writing, software
>   distributed under the License is distributed on an "AS IS" BASIS,
>   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
>   See the License for the specific language governing permissions and
>   limitations under the License.

## Contributors (according to master branch on http://tully.ups-tlse.fr/koleckt/s1tiling/commits/master/s1tiling)
- Thierry KOLECK (CNES)
- Luc HERMITTE (CS Group)
- Guillaume EYNARD-BONTEMPS (CNES)
- Julien MICHEL (CNES)
- Lesly SYLVESTRE (CNES)
- Wenceslas SAINTE MARIE (CESBIO)
- Arthur VINCENT (CESBIO)
