# S1Tiling
On demand Ortho-rectification of Sentinel-1 data on Sentinel-2 grid

# Installation

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

## Setup

### Dask
> Dask does not require any setup if you only want to use it on a single computer.
> -- https://docs.dask.org/en/latest/setup.html

https://docs.dask.org/en/latest/setup/single-distributed.html

# Design Notes

S1Tiling processes the requested S2 tiles one after the other.

For each tile:

1. It first retrieves the S1 images that intersect the S2 tile at the given
   time range
2. Then it builds a graph of tasks to realize (Calibration+Cutting,
   Orthorectification, Concatenation, Mask generation). Each node of the graph
   corresponds to a file that is meant to be produced. The graph is trimmed
   from the edges that produce the expected files that already exist.
3. Finally, all the remaining tasks are executed in parallel, and in order,
   thanks to Dask.
