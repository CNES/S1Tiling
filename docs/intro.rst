.. include:: <isoamsa.txt>

.. _intro:

Introduction
============


Sentinel-1 is currently the only system to provide SAR images regularly on all
lands on the planet. Access to these time series of images opens an
extraordinary range of applications. In order to meet the needs of a large
number of users, including our needs, we have created an automatic processing
chain to generate *"Analysis Ready"* time series for a very large number of
applications.

Sentinel-1 data is ortho-rectified on the Sentinel-2 grid to promote joint use
of both missions.

.. list-table::
  :widths: auto
  :header-rows: 0
  :stub-columns: 0

  * - .. image:: _static/inputs.jpeg
           :scale: 50%
           :alt:   From Sentinel-1 images to Sentinel-2 images
           :align: right

    - |Rarrtl|

    - .. image:: _static/s1a_33NWB_vh_DES_007_20200108txxxxxx.jpeg
           :scale: 50%
           :alt:   The orthorectified result
           :align: left

S1Tiling was developed within the CNES radar service, in collaboration with
CESBIO, to generate time series of calibrated, ortho-rectified and filtered
Sentinel-1 images on any terrestrial region of the Earth. The tool benefits for
the SAR ortho-rectification application
`from the Orfeo Tool Box <https://www.orfeo-toolbox.org/>`_.

The resulting images are registered to Sentinel-2 optical images, using the
same MGRS geographic reference. You will be able to access Sentinel-1 data
acquired on Sentinel-2 31TCJ or 11SPC tiles.This Python software, is based on
the Orfeo Tool Box (OTB) image processing library, developed by CNES, as well
as on `the PEPS platform <https://peps.cnes.fr/>`_ to access the Sentinel-1
data. It can be used on any type of platform, from a large computing cluster to
a laptop (the fan will make some noise during processing). It is considerably
faster than the ortho-rectification tool in SNAP, and can be easily used in
script form.

S1Tiling is currently used for many applications, such deforestation detection
in the Amazon, monitoring of rice crops in Southeast Asia or monitoring of
water stocks in India.In addition, this software is accessible as an on-demand
processing service on the French PEPS collaborative ground segment, in order to
make it easier for users to use.
