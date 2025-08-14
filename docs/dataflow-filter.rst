.. include:: <isoamsa.txt>

.. _dataflow-filter:

.. index:: S1Tiling data flow for speckle filtering

======================================================================
S1Tiling data flow for speckle filtering
======================================================================

.. contents:: Contents:
   :local:
   :depth: 3

Independently of the two other dataflows, S1Tiling permits generating
despeckled images from the S2 products.

S1Tiling filters
----------------

.. _spatial-despeckle:
.. index:: Spatial despeckle

Spatial Despeckle filter
++++++++++++++++++++++++

:Inputs:         Any :ref:`final S2 tiles <full-S2-tiles>`
:Output:         A :ref:`filtered file <filtered-files>`
:OTBApplication: :external+OTB:std:doc:`Despeckle <Applications/app_Despeckle>`
:StepFactory:    :class:`s1tiling.libs.otbwrappers.SpatialDespeckle`

This step applies any of the 3 spatial despeckling filters supported by
:external+OTB:std:doc:`OTB Despeckle <Applications/app_Despeckle>`. See the
documentation of this OTB application for more precise information.
