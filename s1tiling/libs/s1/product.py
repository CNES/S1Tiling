#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2025 (c) CNES.
#
#   This file is part of S1Tiling project
#       https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# =========================================================================
#
# Authors:
# - Thierry KOLECK (CNES)
# - Luc HERMITTE (CSGROUP)
#
# =========================================================================

"""S1 product information"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from eodag.utils       import get_geometry_from_various

try:
    from shapely.errors import TopologicalError
except ImportError:
    from shapely.geos   import TopologicalError

from .. import Utils
from ..orbit._direction import Direction

if TYPE_CHECKING:
    from eodag.api.product import EOProduct
    from osgeo.ogr         import Geometry


logger = logging.getLogger('s1tiling.s1.product')


# =====[ Abstract class ================================================
class ProductInformation(ABC):  # pylint: disable=too-many-instance-attributes
    """
    Abstract class for all S1 product information
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        identifier      : str,
        absolute_orbit  : int,
        relative_orbit  : int,
        orbit_direction : Direction|str,
        platform        : str,
        polarization    : str,  # should be an enum
        start_time      : str,  # date object?
        completion_time : str,  # date object?
    ) -> None:
        """
        Constructor. Make sure all properties are initialized.
        """
        prod_re = re.compile(r'S1._IW_...._...._(\d{8})T(\d{6})_(\d{8}T\d{6}).*')
        match = prod_re.match(identifier)
        assert match
        start_date, start_sec, completion_stamp = match.groups()

        self.__start_stamp      = f"{start_date}T{start_sec}"
        self.__completion_stamp = completion_stamp
        self.__start_date       = start_date

        self.__identifier       = identifier
        self.__absolute_orbit   = absolute_orbit
        self.__relative_orbit   = relative_orbit
        self.__orbit_direction  = Direction.create(orbit_direction)
        self.__platform         = platform
        self.__polarization     = polarization
        self.__start_time       = start_time
        self.__completion_time  = completion_time

        #:> Associated product to which it'll be concatenated
        self.__associated_product : Optional[ProductInformation] = None

    @property
    def identifier(self) -> str:
        """
        Return S1 product identifier
        """
        return self.__identifier

    @property
    def absolute_orbit(self) -> int:
        """
        Return S1 product absolute orbit
        """
        return self.__absolute_orbit

    @property
    def relative_orbit(self) -> int:
        """
        Return S1 product relative orbit
        """
        return self.__relative_orbit

    @property
    def orbit_direction(self) -> Direction:
        """
        Return S1 product orbit direction
        """
        return self.__orbit_direction

    @property
    def platform(self) -> str:
        """
        Return S1 product platform identifier
        """
        return self.__platform

    @property
    def polarization(self) -> str:
        """
        Return S1 product polarization
        """
        return self.__polarization

    @property
    def start_time(self) -> str:
        """
        Return S1 product start_time as in "2017-12-23T17:30:32.108Z"
        """
        return self.__start_time

    @property
    def start_date(self) -> str:
        """
        Return S1 product start_date as in "20171223"

        ..note:: there is no ``completion_date`` as it's expected to be the same.
        """
        return self.__start_date

    @property
    def completion_time(self) -> str:
        """
        Return S1 product completion_time as in "2017-12-23T17:30:32.108Z"
        """
        return self.__completion_time

    @property
    def start_stamp(self) -> str:
        """
        Return S1 product start_stamp as in "20171223T173032"
        """
        return self.__start_stamp

    @property
    def completion_stamp(self) -> str:
        """
        Return S1 product completion_stamp as in "20171223T173032"
        """
        return self.__completion_stamp

    @property
    def is_appaired(self) -> bool:
        """
        Tells whether the product has an associated S1 product to which it'll be concatenated for
        the current output tile.
        """
        return bool(self.__associated_product)

    def associate_with(self, product: "ProductInformation") -> None:
        """
        Sets the S1 product with which the current (self) is associated
        """
        self.__associated_product = product

    def get_associated_product(self) -> Optional["ProductInformation"]:
        """
        Returns the associated S1 product to which it'll be concatenated for the current output
        tile, if any.
        """
        return self.__associated_product

    def __repr__(self):
        return f"{self.identifier}"
        # return f"id={self.identifier} -> {self.relative_orbit:03}/{self.orbit_direction}"

    @abstractmethod
    def compute_relative_cover_of(self, footprint) -> float:
        """
        Compute the coverage of the intersection of the product and the target geometry
        relativelly to the target geometry.
        Return a percentage in the range [0..100].
        """
        pass


# =====[ EOProduct =====================================================

def product_property(prod: EOProduct, key: str, default=None):
    """
    Returns the required (EODAG) product property, or default in the property isn't found.
    """
    res = prod.properties.get(key, default)
    return res


class EOProductInformation(ProductInformation):
    """
    Information on found EODag S1 :class:`EOProduct`.

    ..note::

        EODAG takes care of harmonizing the various properties, even if they are encoded differently
        by the various providers.
        - orbitDirection           ∈ {'ascending', 'descending'}
        - platformSerialIdentifier ∈ {'S1A', 'S1B', 'S1C'...}
        - polarizationChannels     ∈ {'VV+VH', 'HH+HV'}
    """
    def __init__(self, product: EOProduct):
        """
        constructor
        """
        super().__init__(
            identifier      = product.as_dict()['id'],
            absolute_orbit  = product_property(product, 'orbitNumber'),
            relative_orbit  = product_property(product, 'relativeOrbitNumber'),
            orbit_direction = product_property(product, "orbitDirection", ""),
            platform        = product_property(product, "platformSerialIdentifier", ""),
            polarization    = product_property(product, "polarizationChannels", ""),
            start_time      = product_property(product, "startTimeFromAscendingNode", ""),
            completion_time = product_property(product, "completionTimeFromAscendingNode", ""),
        )
        self.__product = product

    @property
    def product(self):
        """
        Return the :class:`EOProduct`
        """
        return self.__product

    def compute_relative_cover_of(self, footprint: Dict[str, float]) -> float:
        """
        Compute the coverage of the intersection of the product and the target geometry
        relativelly to the target geometry.
        Return a percentage in the range [0..100].

        This function has been extracted and adapted from
        :func:`eodag.plugins.crunch.filter_overlap.FilterOverlap.proceed`, which is
        under the Apache Licence 2.0.

        Unlike the original function, the actual filtering is done differently and we
        only need the computed coverage. Also, we are not interrested in the
        coverage of the intersection relativelly to the input product.
        """
        search_geom = get_geometry_from_various(geometry=footprint)
        assert search_geom, "Let's suppose eodag returns a geometry"
        if self.product.search_intersection:
            intersection = self.product.search_intersection
            product_geometry = self.product.geometry
        elif self.product.geometry.is_valid:
            product_geometry = self.product.geometry
            intersection = search_geom.intersection(product_geometry)
        else:
            logger.debug(
                "Trying our best to deal with invalid geometry on product: %r",
                self.product,
            )
            product_geometry = self.product.geometry.buffer(0)
            try:
                intersection = search_geom.intersection(product_geometry)
            except TopologicalError:
                logger.debug("Product geometry still invalid. Force its acceptance")
            return 100

        ipos = (intersection.area / search_geom.area) * 100
        return ipos


# =====[ Filename ======================================================

# Or content_info...

class FileProductInformation(ProductInformation):
    """
    Information on disk-local S1 product.
    """
    MANIFEST_PATTERN = "manifest.safe"

    @classmethod
    def filter_valid_products(
        cls,
        entries: Sequence[os.DirEntry]
    ) -> Sequence["FileProductInformation"]:
        """
        First level factory method that returns a list of :class:`FileProductInformation`
        """
        # Build tuples of {product_dir, safe_dir, manifest_path, orbit_direction, relative_orbit}
        products_info_eodag2 = [ {
            'product':  p,
            # EODAG v2 saves SAFEs into {rawdir}/{prod}/{prod}.SAFE
            'safe_dir': os.path.join(p.path, p.name + '.SAFE'),
        } for p in entries]
        products_info_eodag3 = [ {
            'product':  p,
            # EODAG v3 saves SAFEs into {rawdir}/{prod}
            'safe_dir': p.path,
        } for p in entries]
        products_info = products_info_eodag2 + products_info_eodag3

        # Filter according to valid manifests
        for ci in products_info:
            manifest = os.path.join(ci['safe_dir'], cls.MANIFEST_PATTERN)
            ci['manifest'] = manifest
        products_info = list(filter(lambda ci: os.path.isfile(ci['manifest']), products_info))
        logger.debug('%s local products remaining after filtering valid manifests', len(products_info))

        # Return as FileProductInformation list
        return [fpi for ci in products_info if (fpi := FileProductInformation.create(ci)) is not None]

    def __init__(self, ci: Dict):
        """
        Constructor

        :raise RuntimeError: if expected S1 information cannot be extracted from the manifest file.

        See also: :meth:`FileProductInformation.create`
        """
        assert 'product' in ci
        product : os.DirEntry = ci['product']
        assert product.is_dir(), f"{product} is not a valid S1 product"

        assert 'manifest' in ci
        manifest = ci['manifest']
        assert os.path.isfile(manifest), f"{manifest} is not a valid S1 manifest"

        orbit_info = Utils.get_orbit_information(manifest)
        acq_time = Utils.extract_product_start_time(product.name)

        super().__init__(
            identifier      = product.name,
            absolute_orbit  = orbit_info['absolute_orbit'],
            relative_orbit  = orbit_info['relative_orbit'],
            orbit_direction = orbit_info['orbit_direction'],
            platform        = product.name[:3],
            polarization    = None,
            start_time      = '{YYYY}:{MM}:{DD}T{hh}:{mm}:{ss}Z'.format_map(acq_time) if acq_time else '????',
            completion_time = "None-yet",
        )
        self.__product  = product
        self.__manifest = manifest
        self.__safe_dir = ci['safe_dir']
        self.__product_shape         : Optional[Geometry] = None
        self.__tile_origin           : Sequence[Tuple[float, float]] = ()
        self.__current_tile_coverage : Optional[float] = None

    @staticmethod
    def create(ci: Dict) -> Optional["FileProductInformation"]:
        """
        Factory method meant to be used to create :class:`FileProductInformation` for comprehension
        lists.
        """
        try:
            if not os.path.isfile(ci['manifest']):
                logger.debug('DISCARD non-existing %r', ci['manifest'])
                return None
            logger.debug('USE existing %r', ci['manifest'])
            return FileProductInformation(ci)
        except BaseException as e:  # pylint: disable=broad-except
            logger.critical(e, exc_info=True)
            logger.debug("Cannot extract product information from %r: %s", ci['product'], e)
            return None

    def __repr__(self):
        return str(self.__product)

    @property
    def product(self) -> os.DirEntry:
        """
        Product accessor
        """
        return self.__product

    @property
    def manifest(self) -> Path:
        """
        manifest path accessor
        """
        return self.__manifest

    @property
    def safe_dir(self) -> os.PathLike:
        """
        SAFE directory accessor
        """
        return self.__safe_dir

    def get_shape(self) -> Geometry:
        """
        Shape accessor.
        :return: the shape according to the manifest. If unset, it'll be extracted on-the-fly.
        """
        if self.__product_shape is None:
            self.__product_shape = Utils.get_shape(self.__manifest)
        return self.__product_shape

    def set_tile_origin(self, origin: List[Tuple[float, float]]) -> None:
        """
        Setter for the footprint of the current targetted S2 MGRS tile.
        """
        self.__tile_origin = origin

    def get_tile_origin(self) -> Sequence[Tuple[float, float]]:
        """
        Getter on the footprint of the current targetted S2 MGRS tile.
        """
        return self.__tile_origin

    def compute_relative_cover_of(self, footprint: Geometry) -> float:
        """
        Compute the coverage of the intersection of the product and the target geometry
        relativelly to the target geometry.
        Return a percentage in the range [0..100].
        """
        poly = self.get_shape()
        intersection = poly.Intersection(footprint)
        coverage = intersection.GetArea() / footprint.GetArea() * 100
        logger.debug('%s -> %s %% (inter %s / tile %s)',
                     self.identifier, coverage, intersection.GetArea(), footprint.GetArea())
        self.__current_tile_coverage = coverage
        return coverage

    def get_current_tile_coverage(self) -> Optional[float]:
        """
        Getter on the last known coverage of the targetted S2 MGRS tile
        """
        return self.__current_tile_coverage
