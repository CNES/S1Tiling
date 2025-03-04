.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. _api:

.. index:: Developer documentation

======================================================================
S1Tiling API
======================================================================

This section is intended for people who want to directly call S1Tiling code
from their project instead of using the external programs :ref:`S1Processor`,
:ref:`S1LIAMap`, :ref:`S1IAMap`, and :ref:`S1GammaAreaMap`.


.. contents:: Contents:
   :local:
   :depth: 4


Entry points
============

``s1_process``
--------------

.. autofunction:: s1tiling.libs.api.s1_process


``s1_process_lia``
------------------

.. autofunction:: s1tiling.libs.api.s1_process_lia


``s1_process_ia``
------------------

.. autofunction:: s1tiling.libs.api.s1_process_ia

``s1_process_gamma_area``
-------------------------

.. autofunction:: s1tiling.libs.api.s1_process_gamma_area


Configuration object
====================

.. autoclass:: s1tiling.libs.configuration.Configuration
   :members:


Result (``Situation``) object
=============================

.. autoclass:: s1tiling.libs.exits.Situation
   :members:


Exceptions
==========

S1Tiling may raise the following exceptions:

.. inheritance-diagram:: s1tiling.libs.exceptions.CorruptedDataSAFEError
   s1tiling.libs.exceptions.DownloadS1FileError
   s1tiling.libs.exceptions.NoS2TileError
   s1tiling.libs.exceptions.NoS1ImageError
   s1tiling.libs.exceptions.MissingDEMError
   s1tiling.libs.exceptions.MissingGeoidError
   s1tiling.libs.exceptions.InvalidOTBVersionError
   s1tiling.libs.exceptions.MissingApplication
   :parts: 1
   :top-classes: s1tiling.libs.exceptions.Error
   :private-bases:

.. autoclass:: s1tiling.libs.exceptions.Error
.. autoclass:: s1tiling.libs.exceptions.CorruptedDataSAFEError
.. autoclass:: s1tiling.libs.exceptions.DownloadS1FileError
.. autoclass:: s1tiling.libs.exceptions.NoS2TileError
.. autoclass:: s1tiling.libs.exceptions.NoS1ImageError
.. autoclass:: s1tiling.libs.exceptions.MissingDEMError
.. autoclass:: s1tiling.libs.exceptions.MissingGeoidError
.. autoclass:: s1tiling.libs.exceptions.InvalidOTBVersionError
.. autoclass:: s1tiling.libs.exceptions.MissingApplication
