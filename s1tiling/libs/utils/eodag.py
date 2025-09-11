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

"""Centralizes EODAG heper functions"""

from functools import partial
import logging
import logging.handlers
import multiprocessing
import os
from typing import List, Optional, Protocol

from requests.exceptions import ReadTimeout
from eodag.api.core      import EODataAccessGateway
from eodag.api.product   import EOProduct

from ..             import exceptions
from ..outcome      import S1DownloadOutcome
from ..otbpipeline  import mp_worker_config


logger = logging.getLogger('s1tiling.utils.eodag')


class EODAGConfiguration(Protocol):
    """
    Specialized protocol for configuration information related to function:`EODataAccessGateway
    factory<create>`.

    Can be seen an a ISP compliant concept for Configuration object regarding eodag object
    construction.
    """
    download     : bool
    raw_directory: str
    eodag_config : str


def create(cfg: EODAGConfiguration) -> Optional[EODataAccessGateway]:
    """
    :class:`EODataAccessGateway` factory from S1Tiling configuration obejct.
    """
    if not cfg.download:
        return None

    logger.debug('Using %s EODAG configuration file', cfg.eodag_config or 'user default')
    dag = EODataAccessGateway(cfg.eodag_config)
    # TODO: update once eodag directly offers "DL directory setting" feature v1.7? +?
    dest_dir = os.path.abspath(cfg.raw_directory)
    logger.debug('Override EODAG output directory to %s', dest_dir)
    for provider in dag.providers_config.keys():
        if hasattr(dag.providers_config[provider], 'download'):
            dag.providers_config[provider].download.update({'output_dir': dest_dir})
            logger.debug(' - for %s', provider)
        else:
            logger.debug(' - NOT for %s', provider)
    return dag


def _download_and_extract_one_product(
    dag:           EODataAccessGateway,
    raw_directory: str,
    dl_wait:       int,
    dl_timeout:    int,
    product:       EOProduct
) -> S1DownloadOutcome[str, EOProduct]:
    """
    Takes care of downloading exactly one remote product and unzipping it, if required.

    Some products are already unzipped on the fly by eodag.
    """
    logging.info("Starting download of %s...", product)
    ok_msg = f"Successful download (and extraction) of {product}"  # because eodag'll clear product
    prod_id = product.as_dict()['id']
    zip_file = os.path.join(raw_directory, prod_id) + '.zip'
    path: S1DownloadOutcome[str, EOProduct]
    try:
        path = S1DownloadOutcome(
            dag.download(
                product,            # EODAG will clear this variable
                extract=True,       # Let's eodag do the job
                wait=dl_wait,       # Wait time in minutes between two download tries
                timeout=dl_timeout  # Maximum time in mins before stop retrying to download (default=20’)
            ),
            product)
        logging.debug(ok_msg)
        if os.path.exists(zip_file) :
            try:
                logger.debug('Removing downloaded ZIP: %s', zip_file)
                os.remove(zip_file)
            except OSError:
                pass
        # eodag may say the product is correctly downloaded while it failed to do so
        # => let's do a quick sanity check

        # eodag2 product naming scheme
        manifest = os.path.join(raw_directory, prod_id, f'{prod_id}.SAFE', 'manifest.safe')
        if not os.path.exists(manifest):
            # eodag3 product naming scheme
            manifest = os.path.join(raw_directory, prod_id, 'manifest.safe')
            if not os.path.exists(manifest):
                logger.error('Actually download of %s failed, the expected manifest could not be found in the product (%s)', prod_id, manifest)
                e = exceptions.CorruptedDataSAFEError(prod_id, f"no manifest file named {manifest!r} found")
                path = S1DownloadOutcome(e, product)
    except BaseException as e:  # pylint: disable=broad-except
        logger.warning('%s while attempting download of %s', e, prod_id)  # EODAG error message is good and precise enough, just use it!
        # logger.error('Product is %s', product_property(product, 'storageStatus', 'online?'))
        logger.debug('Exception type is: %s', e.__class__.__name__)
        ## ERROR - Product is OFFLINE
        ## ERROR - Exception type is: NotAvailableError
        # logger.error('======================')
        # logger.exception(e)
        ## Traceback (most recent call last):
        ##   File "s1tiling/libs/S1FileManager.py", line 350, in _download_and_extract_one_product
        ##     path = S1DownloadOutcome(dag.download(
        ##   File "site-packages/eodag/api/core.py", line 1487, in download
        ##     path = product.download(
        ##   File "site-packages/eodag/api/product/_product.py", line 288, in download
        ##     fs_path = self.downloader.download(
        ##   File "site-packages/eodag/plugins/download/http.py", line 269, in download
        ##     raise NotAvailableError(
        ## eodag.utils.exceptions.NotAvailableError: S1A_IW_GRDH_1SDV_20200401T044214_20200401T044239_031929_03AFBC_0C9E
        ##                                           is not available (OFFLINE) and could not be downloaded, timeout reached

        path = S1DownloadOutcome(e, product)

    return path


def download_and_extract_products(  # pylint: disable=too-many-arguments, too-many-locals
    *,
    dag:           EODataAccessGateway,
    raw_directory: str,
    products:      List[EOProduct],
    nb_procs:      int,
    context:       str,
    dl_wait:       int,
    dl_timeout:    int,
) -> List[S1DownloadOutcome]:
    """
    Takes care of downloading exactly all remote products and unzipping them,
    if required, in parallel.

    Returns :class:`S1DownloadOutcome` of :class:`EOProduct` or Exception.
    """
    nb_products = len(products)
    paths     : List[S1DownloadOutcome] = []
    log_queue : multiprocessing.Queue   = multiprocessing.Queue()
    log_queue_listener = logging.handlers.QueueListener(log_queue)
    dl_work = partial(_download_and_extract_one_product, dag, raw_directory, dl_wait, dl_timeout)
    with multiprocessing.Pool(nb_procs, mp_worker_config, [log_queue]) as pool:
        log_queue_listener.start()
        try:
            # In case timeout happens, we try again if and only if we have been able to download
            # other products after the timeout.
            # -> IOW, downloading instability justifies trying again.
            # /> On the contrary, on a complete network failure, we should not try again and again...
            while len(products) > 0:
                products_in_timeout : List[S1DownloadOutcome] = []
                nb_successes_since_timeout = 0
                for count, result in enumerate(pool.imap_unordered(dl_work, products), 1):
                    # logger.debug('DL -> %s', result)
                    if result:
                        logger.info("%s correctly downloaded", result.value())
                        logger.info(' --> Downloading products%s... %s%%', context, count * 100. / nb_products)
                        paths.append(result)
                        if len(products_in_timeout) > 0:
                            nb_successes_since_timeout += 1
                    else:
                        logger.warning("Cannot download %s: %s", result.related_product(), result.error())
                        # TODO: make it possible to detect missing products in the analysis
                        if isinstance(result.error(), ReadTimeout):
                            products_in_timeout.append(result)
                        else:
                            paths.append(result)
                products = []
                if nb_successes_since_timeout > nb_procs:
                    products = [r.related_product() for r in products_in_timeout]
                    logger.info("Attempting again to download %s products on timeout...", len(products))
                elif len(products_in_timeout) > 0:
                    paths.extend(products_in_timeout)
        finally:
            pool.close()
            pool.join()
            log_queue_listener.stop()  # no context manager for QueueListener unfortunately

    # paths returns the list of .SAFE directories
    return paths
