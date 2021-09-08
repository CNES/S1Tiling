# S1Tiling
Read the full documentation at
https://s1-tiling.pages.orfeo-toolbox.org/s1tiling/latest

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

S1Tiling installation has a few traps. Please read the [relevant documentation](https://s1-tiling.pages.orfeo-toolbox.org/s1tiling/latest/install.html)
regarding OTB and GDAL installation.

## Requirements

* OTB 7.2
* GDAL with python bindings as well
* Python 3
  * click
  * eodag
  * numpy
  * gdal
  * yaml
  * Dask"distributed"
  * bokeh (to display Dask dashboard)
  * graphviz (to generate task graphs)

### Dask
> Dask does not require any setup if you only want to use it on a single computer.
> -- https://docs.dask.org/en/latest/setup.html

https://docs.dask.org/en/latest/setup/single-distributed.html

# Community

[![S1Tiling Discourse (status)](https://img.shields.io/discourse/status?server=https%3A%2F%2Fforum.orfeo-toolbox.org%2F)](https://forum.orfeo-toolbox.org/c/otb-chains/s1-tiling/11)

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

## Contributors

(according to master branch on http://tully.ups-tlse.fr/koleckt/s1tiling/commits/master/s1tiling)

- Thierry KOLECK (CNES)
- Luc HERMITTE (CS Group FRANCE)
- Guillaume EYNARD-BONTEMPS (CNES)
- Julien MICHEL (CNES)
- Lesly SYLVESTRE (CNES)
- Wenceslas SAINTE MARIE (CESBIO)
- Arthur VINCENT (CESBIO)
