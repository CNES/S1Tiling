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
#       http://www.apache.org/licenses/LICENSE-2.0
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

from abc import ABC, abstractmethod
import logging
import os
import re
from typing import Dict, Optional

from eodag.api.product import EOProduct
from eodag.utils       import get_geometry_from_various

try:
    from shapely.errors import TopologicalError
except ImportError:
    from shapely.geos   import TopologicalError


logger = logging.getLogger('s1tiling.s1.products')

K_DIR_ASSOC = { 'ascending': 'ASC', 'descending': 'DES' }


# =====[ Abstract class ================================================
class ProductInformation(ABC):
    """
    Abstract class for all S1 product information
    """

    def __init__(
        self,
        *,
        identifier      : str,
        absolute_orbit  : int,
        relative_orbit  : int,
        orbit_direction : str,  # should be an enum
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
        self.__orbit_direction  = orbit_direction
        self.__platform         = platform
        self.__polarization     = polarization
        self.__start_time       = start_time
        self.__completion_time  = completion_time

        #:> Associated product to which it'll be concatenated
        self.__associated_product : Optional[ProductInformation] = None

        assert self.__identifier
        assert self.__absolute_orbit
        assert self.__relative_orbit
        assert self.__orbit_direction
        assert self.__platform
        assert self.__polarization
        assert self.__start_time
        assert self.__completion_time

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
    def orbit_dir(self) -> str:
        """
        Return S1 product orbit direction as "ASC"/"DES"
        """
        return K_DIR_ASSOC[self.__orbit_direction]

    @property
    def orbit_direction(self) -> str:
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


    @abstractmethod
    def get_relative_cover_of(self, geometry: Dict[str, float]) -> float:
        """
        Compute the coverage of the intersection of the product and the target geometry relativelly
        to the target geometry.
        Return a percentage in the range [0..100].
        """
        pass

    def __repr__(self):
        return f"{self.identifier}"
        # return f"id={self.identifier} -> {self.relative_orbit:03}/{self.orbit_direction}"


# =====[ EOProduct =====================================================

def product_property(prod: EOProduct, key: str, default=None):
    """
    Returns the required (EODAG) product property, or default in the property isn't found.
    """
    res = prod.properties.get(key, default)
    return res


def get_relative_cover_of(product: EOProduct, geometry: Dict[str, float]) -> float:
    """
    Compute the coverage of the intersection of the product and the target geometry
    relativelly to the target geometry.
    Return a percentage in the range [0..100].

    This function has been extracted and adapted from
    :func:`eodag.plugins.crunch.filter_overlap.FilterOverlap.proceed`, which is
    under the Apache Licence 2.0.

    Unlike the original function, the actual filtering is done differenty and we
    only need the computed coverage. Also, we are not interrested in the
    coverage of the intersection relativelly to the input product.
    """
    search_geom = get_geometry_from_various(geometry=geometry)
    assert search_geom, "Let's suppose eodag returns a geometry"
    if product.search_intersection:
        intersection = product.search_intersection
        product_geometry = product.geometry
    elif product.geometry.is_valid:
        product_geometry = product.geometry
        intersection = search_geom.intersection(product_geometry)
    else:
        logger.debug(
            "Trying our best to deal with invalid geometry on product: %r",
            product,
        )
        product_geometry = product.geometry.buffer(0)
        try:
            intersection = search_geom.intersection(product_geometry)
        except TopologicalError:
            logger.debug("Product geometry is still invalid. Force its acceptance")
        return 100

    ipos = (intersection.area / search_geom.area) * 100
    return ipos


class EOProductInformation(ProductInformation):
    """
    Information on found EODag S1 :class:`EOProduct`.
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
            polarization    = product_property(product, "polarizationMode", ""),
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

    def get_relative_cover_of(self, geometry: Dict[str, float]) -> float:
        return get_relative_cover_of(self.product, geometry)


# =====[ Filename ======================================================

# Or content_info...

class FileProductInformation(ProductInformation):
    """
    Information on disk-local S1 product.
    """
    def __init__(self, filename: os.DirEntry):
        """
        constructor
        """
        assert os.path.exists(filename), f"{filename} is not a valid S1 filename"

        super().__init__(
            identifier      = product.as_dict()['id'],
            absolute_orbit  = product_property(product, 'orbitNumber'),
            relative_orbit  = product_property(product, 'relativeOrbitNumber'),
            orbit_direction = product_property(product, "orbitDirection", ""),
            platform        = product_property(product, "platformSerialIdentifier", ""),
            polarization    = product_property(product, "polarizationMode", ""),
            start_time      = product_property(product, "startTimeFromAscendingNode", ""),
            completion_time = product_property(product, "completionTimeFromAscendingNode", ""),
        )
        self.__product = product

    def __repr__(self):
        return self.__product
