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

"""Filters that apply on S1 product information"""


import logging
from typing import Callable, Dict, List, Optional, Sequence, TypeVar

from ..orbit._direction import Direction
from .product import ProductInformation


logger = logging.getLogger('s1tiling.s1.filters')

TProductInformation = TypeVar('TProductInformation', bound=ProductInformation)


# =====[ Helpers =======================================================
# =====[ Filters =======================================================
def discard_small_redundant(
    products: Sequence[TProductInformation],
) -> Sequence[TProductInformation]:
    """
    Sometimes there are several S1 product with the same start date, but a different end-date.
    Let's discard the smallest products

    Can be tested on

    dag.search(page=1, items_per_page=200, productType='S1_SAR_GRD', start='2017-12-23T17:30:00',
    end='2017-12-23T17:31:00', sensorMode='IW')

    See #47
    """
    if not products:
        return products

    ordered_products = sorted(products, key=lambda p: p.identifier)
    # logger.debug("all products before clean: %s", ordered_products)
    res = [ordered_products[0]]
    last = res[0].start_stamp
    for product in ordered_products[1:]:
        if last == product.start_stamp:
            # We can suppose the new end date to be >
            # => let's replace
            logger.warning('Discarding %s that is smaller than %s', res[-1], product)
            res[-1] = product
        else:
            res.append(product)
            last = product.start_stamp
    return res


def find_paired_products(
    products: Sequence[TProductInformation],
) -> Dict[str, List[TProductInformation]]:
    """
    Associate products of the same date and orbit into pairs (at most).

    :return: a dictionary with key="date:relobt", value=list matching products
    """
    date_grouped_products : Dict[str, List[TProductInformation]] = {}
    logger.debug('Search group of S1 products that work in pair to be concatenated')
    for p in products:
        pid   = p.identifier
        date  = p.start_date
        ron   = p.relative_orbit
        dron  = f'{date}#{ron:03}'
        logger.debug('* @ %s, -> %s', dron, pid)
        if dron not in date_grouped_products:
            date_grouped_products[dron] = []
        date_grouped_products[dron].append(p)

    for group in date_grouped_products.values():
        if len(group) > 1:
            assert len(group) == 2, f"S1 products should not be associated with more than one other product: {group=}"
            p1, p2 = group
            logger.debug("Associating %r with %r", p1, p2)
            p1.associate_with(p2)
            p2.associate_with(p1)
        else:
            assert len(group) > 0
            logger.info("No S1 product found to be concatenated with %r", group[0])

    return date_grouped_products


def filter_image_groups_providing_enough_cover_by_pair(
    date_grouped_products: Dict[str, List[TProductInformation]],
    target_cover         : float,
    get_cover            : Callable[[TProductInformation], float],
) -> Sequence[TProductInformation]:
    """
    Given associated products of the same date and orbit into pairs (at most), compute the total
    coverage of the target zone.

    If the total coverage is inferior to the target coverage, the products are filtered out.
    """
    assert target_cover
    kept_products : List[TProductInformation] = []
    logger.debug('Checking coverage for each pair of products:')
    for dron, group in date_grouped_products.items():
        covers = []
        for p in group:
            cover = get_cover(p)
            logger.debug('* @ %s, %s%% coverage for %r', dron, round(cover, 2), p)
            covers.append(cover)
        cover_sum      = round(sum(covers), 2)
        str_cov_to_sum = '+'.join((str(round(c, 2)) for c in covers))
        if cover_sum < target_cover:
            logger.warning('Reject products @ %s for insufficient coverage: %s=%s%% < %s%% %s',
                    dron, str_cov_to_sum, cover_sum, target_cover, group)
        else:
            kept_products.extend(group)

    return kept_products


def filter_images_providing_enough_cover_by_pair(  # pylint: disable=too-many-locals
    products:     Sequence[TProductInformation],
    target_cover: float,
    get_cover:    Callable[[TProductInformation], float],
) -> Sequence[TProductInformation]:
    """
    DEPRECATED!!

    Associate products of the same date and orbit into pairs (at most),
    to compute the total coverage of the target zone.
    If the total coverage is inferior to the target coverage, the products
    are filtered out.

    This function can be used on product information returned by EODAG as well
    as product information extracted from existing files. It's acheived thanks
    to the `ident`, `get_cover` and `get_orbit` variation points.
    """
    if not products or not target_cover:
        return products
    kept_products : List[TProductInformation] = []
    date_grouped_products : Dict[str, Dict[float, TProductInformation]] = {}
    logger.debug('Checking coverage for each product:')
    for p in products:
        pid   = p.identifier
        date  = p.start_date
        cover = get_cover(p)
        ron   = p.relative_orbit
        dron  = f'{date}#{ron:03}'
        logger.debug('* @ %s, %s%% coverage for %s', dron, round(cover, 2), pid)
        if dron not in date_grouped_products:
            date_grouped_products[dron] = {}
        date_grouped_products[dron].update({cover : p})

    logger.debug('Checking coverage for each date (and # relative orbit number)')
    for dron, cov_prod in date_grouped_products.items():
        covers         = cov_prod.keys()
        cov_sum        = round(sum(covers), 2)
        str_cov_to_sum = '+'.join((str(round(c, 2)) for c in covers))
        logger.debug('* @ %s -> %s%% = %s', dron, cov_sum, str_cov_to_sum)
        if cov_sum < target_cover:
            logger.warning('Reject products @ %s for insufficient coverage: %s=%s%% < %s%% %s',
                    dron, str_cov_to_sum, cov_sum, target_cover,
                    [p.identifier for p in cov_prod.values()])
        else:
            if len(cov_prod.values()) > 1:
                assert len(cov_prod)==2, f"At most S1 products may be assembled in pairs, but we have: {cov_prod.values()}"
                first, second = cov_prod.values()
                first.associate_with(second)
                second.associate_with(first)
            kept_products.extend(cov_prod.values())
    return kept_products


def keep_requested_orbits(
    content_info:           Sequence[TProductInformation],
    rq_orbit_direction:     Optional[str],
    rq_relative_orbit_list: List[int],
) -> Sequence[TProductInformation]:
    """
    Takes care of discarding products that don't match the requested orbit specification.

    Note: Beware that specifications could be contradictory and end up discarding everything.
    """
    if not rq_orbit_direction and not rq_relative_orbit_list:
        return content_info
    kept_products = []
    requested_orbit_direction = Direction.create(rq_orbit_direction) if rq_orbit_direction else None
    for ci in content_info:
        name      = ci.identifier
        direction = ci.orbit_direction
        orbit     = ci.relative_orbit
        # logger.debug('CHECK orbit: %s / %s / %s', p, safe_dir, manifest)

        # if rq_orbit_direction:
        #     if direction != rq_orbit_direction:
        #         logger.debug('Discard %s as its direction (%s) differs from the requested %s',
        #                 name, direction, rq_orbit_direction)
        #         continue
        if requested_orbit_direction:
            if direction != requested_orbit_direction:
                logger.debug('Discard %s as its direction (%s) differs from the requested %s',
                        name, direction, requested_orbit_direction)
                continue
        if rq_relative_orbit_list:
            if orbit not in rq_relative_orbit_list:
                logger.debug('Discard %s as its orbit (%s) differs from the requested ones %s',
                        name, orbit, rq_relative_orbit_list)
                continue
        kept_products.append(ci)
    return kept_products


def keep_requested_platforms(
    content_info:     Sequence[TProductInformation],
    rq_platform_list: List[str]
) ->  Sequence[TProductInformation]:
    """
    Takes care of discarding products that don't match the requested platform specification.

    Note: Beware that specifications could be contradictory and end up discarding everything.
    """
    if not rq_platform_list:
        return content_info
    kept_products = []
    for ci in content_info:
        name     = ci.identifier
        platform = ci.platform
        logger.debug('CHECK platform: %s / %s', name, platform)

        if rq_platform_list:
            if platform not in rq_platform_list:
                logger.debug('Discard %s as its platform (%s) differs from the requested ones %s',
                             name, platform, rq_platform_list)
                continue
        kept_products.append(ci)
    return kept_products
