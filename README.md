# S1Tiling

[![Docs (latest)](https://img.shields.io/badge/docs-passing-brightgreen)](https://s1-tiling.pages.orfeo-toolbox.org/s1tiling/latest/)
[![S1Tiling Discourse (status)](https://img.shields.io/discourse/status?server=https%3A%2F%2Fforum.orfeo-toolbox.org%2F)](https://forum.orfeo-toolbox.org/c/otb-chains/s1-tiling/11)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17237358.svg)](https://doi.org/10.5281/zenodo.17237358)
<br>
[![Docker Image Version](https://img.shields.io/docker/v/cnes/s1tiling?logo=docker)](https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/container_registry/87)
[![Fury](https://badge.fury.io/py/S1Tiling.svg)](https://badge.fury.io/py/S1Tiling)
<br>
[![Pipeline](https://img.shields.io/gitlab/pipeline-status/s1-tiling%2Fs1tiling?gitlab_url=https%3A%2F%2Fgitlab.orfeo-toolbox.org&branch=develop&label=CI%20develop&logo=gitlab)](https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/pipelines?page=1&scope=branches&ref=develop)
[![Coverage](https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/badges/develop/coverage.svg)](https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/pipelines?page=1&scope=branches&ref=develop)
[![Issues](https://img.shields.io/gitlab/issues/open/s1-tiling/s1tiling?gitlab_url=https%3A%2F%2Fgitlab.orfeo-toolbox.org&labels=Bug&logo=gitlab)](https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues)
<br>
[![Sources](https://img.shields.io/badge/sources-gitlab.OTB-informational)](https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling)
[![Licence](https://img.shields.io/pypi/l/s1tiling.svg)](https://pypi.org/project/s1tiling/)

(Warning, if you're reading this on github, please note that S1Tiling is
maintained on [gitlab.orfeo-toolbox.org](https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling))


On demand Ortho-rectification of Sentinel-1 data on Sentinel-2 grid.

Sentinel-1 is currently the only system to provide SAR images regularly on all
lands on the planet. Access to these time series of images opens an
extraordinary range of applications. In order to meet the needs of a large
number of users, including our needs, we have created an automatic processing
chain to generate _"Analysis Ready"_ time series for a very large number of
applications.

With __S1Tiling__, Sentinel-1 data is ortho-rectified on the Sentinel-2 grid to
promote joint use of both missions.

__S1Tiling__ was developed within the CNES radar service, in collaboration with
CESBIO, to generate time series of calibrated, ortho-rectified and filtered
Sentinel-1 images on any lands on the Earth. The tool benefits for the SAR
ortho-rectification application from the
[Orfeo Tool Box](https://www.orfeo-toolbox.org/).

The resulting images are gridded to Sentinel-2 MGRS geographic reference grid
[S2 tiling system - zipped kml file](https://sentiwiki.copernicus.eu/__attachments/1692737/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.zip).
Thanks to [EODAG](https://eodag.readthedocs.io/), different Sentinel-1 data
providers can be used like [Geodes](https://geodes-portal.cnes.fr/) or
[Copernicus Data Space](https://dataspace.copernicus.eu/).
It can be used on any type of platform, from a large computing cluster to a
laptop (the fan will make some noise during processing). It is considerably
faster than the ortho-rectification tool in ESA SNAP software with similar
results and can be easily used in script form.

S1Tiling is currently used as Sentinel-1 data pre-processing for many
applications, such deforestation detection in the Amazon, monitoring of rice
crops in Southeast Asia or monitoring of water stocks in India. In addition,
this software will be implemented in GEODES, the French portal for Earth
Observation data, in order to provide SAR Ready Analysis Data to users.

The reference documentation is provided at
https://s1-tiling.pages.orfeo-toolbox.org/s1tiling/latest. And the source code
is always available at https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling.

# Installation

S1Tiling installation has a few traps. Please read the [relevant documentation](https://s1-tiling.pages.orfeo-toolbox.org/s1tiling/latest/install.html)
regarding OTB and GDAL installation.

We highly recommand the usage of [dockers](https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/container_registry/87), or of S1Tiling module for TREX users.

# Community

[![S1Tiling Discourse (status)](https://img.shields.io/discourse/status?server=https%3A%2F%2Fforum.orfeo-toolbox.org%2F)](https://forum.orfeo-toolbox.org/c/otb-chains/s1-tiling/11)

# Copyright

```
All rights reserved.
Copyright 2017-2025 (c) CNES.
Copyright 2022-2025 (c) CS GROUP France.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Contributors

- Thierry KOLECK (CNES)
- Luc HERMITTE (CS Group FRANCE)
- Guillaume EYNARD-BONTEMPS (CNES)
- Julien MICHEL (CNES)
- Lesly SYLVESTRE (CNES)
- Wenceslas SAINTE MARIE (CESBIO)
- Arthur VINCENT (CESBIO)
