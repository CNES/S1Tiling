.. include:: <isoamsa.txt>

.. _intro:

Introduction
============


`Sentinel-1
<https://sentinels.copernicus.eu/web/sentinel/copernicus/sentinel-1>`_ is
currently the only mission to provide long time series of Synthetic Aperture
Radar (SAR) data over land. In order to generate *"Analysis Ready Data"* for
various related applications, CNES, in collaboration with CESBIO, have created
and developed since several years an operational processor to generate
calibrated and ortho-rectified Sentinel-1 data on the Sentinel-2 grid:
**S1Tiling**.

With this Python software, users can easily use jointly Sentinel-1 and
Sentinel-2 time series.

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

**S1Tiling** was developed as an open source project by CNES based on the
existing CNES open source project `Orfeo Tool Box <https://www.orfeo-toolbox.org/>`_.

The resulting images are gridded to Sentinel-2 MGRS geographic reference grid
(`S2 tiling system - zipped kml file
<https://sentiwiki.copernicus.eu/__attachments/1692737/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.zip>`_).
Thanks to `EODAG <https://eodag.readthedocs.io/>`_, different Sentinel-1 data
providers can be used like `Geodes <https://geodes-portal.cnes.fr/>`_ or
`Copernicus Data Space <https://dataspace.copernicus.eu/>`_.
It can be used on any type of platform, from a large computing cluster to a
laptop (the fan will make some noise during processing). It is considerably
faster than the ortho-rectification tool in ESA SNAP software with similar
results and can be easily used in script form.

S1Tiling is currently used as Sentinel-1 data pre-processing for many
applications, such deforestation detection in the Amazon, monitoring of rice
crops in Southeast Asia or monitoring of water stocks in India. In addition,
this software will be implemented in GEODES, the French portal for Earth
Observation data, in order to provide SAR Ready Analysis Data to users.
