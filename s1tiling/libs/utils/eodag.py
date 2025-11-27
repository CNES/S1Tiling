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

from collections.abc import Callable
from functools import partial
import logging
import logging.handlers
import multiprocessing
import os
from types import ModuleType
from typing import List, Optional, Protocol, Tuple, cast

from requests.exceptions    import ReadTimeout, Timeout
from urllib3.exceptions     import ReadTimeoutError
from eodag.api.core         import EODataAccessGateway
from eodag.api.product      import EOProduct
from eodag.utils.exceptions import TimeOutError

from ..outcome      import ProductDownloadOutcome
from ..otbpipeline  import mp_worker_config


Sanatizer = Callable[[str, EOProduct, logging.Logger|ModuleType], Optional[Exception]]


# Default configuration value for people using S1Tiling API functions s1_process, and s1_process_lia.
EODAG_DEFAULT_DOWNLOAD_WAIT         = 2   #: If download fails, wait time in minutes between two download tries
EODAG_DEFAULT_DOWNLOAD_TIMEOUT      = 20  #: If download fails, maximum time in minutes before stop retrying to download
EODAG_DEFAULT_SEARCH_MAX_RETRIES    = 5   #: If search fails on timeout, number of retries attempted
EODAG_DEFAULT_SEARCH_ITEMS_PER_PAGE = 20  #: Number of items returns by each page search


logger = logging.getLogger('s1tiling.utils.eodag')


class EODAGConfiguration(Protocol):
    """
    Specialized protocol for configuration information related to function:`EODataAccessGateway
    factory<create>`.

    Can be seen an a ISP compliant concept for Configuration object regarding eodag object
    construction.
    """
    download     : bool
    eodag_config : str


def create(cfg: EODAGConfiguration) -> Optional[EODataAccessGateway]:
    """
    :class:`EODataAccessGateway` factory from S1Tiling configuration object.
    """
    if not cfg.download:
        return None

    logger.debug('Using %s EODAG configuration file', cfg.eodag_config or 'user default')
    dag = EODataAccessGateway(cfg.eodag_config)
    return dag


def _is_a_timeout(exception: BaseException) -> Optional[BaseException]:
    """
    Helper function that tries to detect the various kind of timeouts encountered with eodag
    """
    crt : Optional[BaseException] = exception
    prefix = '  '
    logger.debug('%s Analyse exception type', prefix)
    while crt:
        logger.debug("%s+- %s -> %s", prefix, type(crt), str(crt))
        if isinstance(crt,
                      (
                          ReadTimeout,       # requests.exceptions
                          Timeout,           # requests.exceptions
                          TimeOutError,      # eodag.utils.exceptions
                          ReadTimeoutError,  # urllib3.exceptions.ReadTimeoutError
                          TimeoutError,      # std python error
                      )
                      ):
            logger.debug("%s   => time out!!", prefix)
            return crt
        crt = crt.__cause__ or crt.__context__
        prefix += '  '
    logger.debug("%s   => Other error kind", prefix)
    return None


def _as_timeout(exception: Exception) -> TimeOutError:
    if isinstance(exception, TimeOutError):
        return exception
    return TimeOutError(exception)


def _download_and_extract_one_product(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dag:        EODataAccessGateway,
    output_dir: str,
    dl_wait:    int,
    dl_timeout: int,
    logger_:    logging.Logger|ModuleType,  # todo: pass the right global logger
    sanatize:   Optional[Sanatizer],
    product:    EOProduct,
) -> ProductDownloadOutcome[str, EOProduct]:
    """
    Takes care of downloading exactly one remote product and unzipping it, if required.

    Some products are already unzipped on the fly by eodag.
    """
    logger_.debug("  Starting download of %s... into %s", product, output_dir)
    ok_msg = f"  Successful download (and extraction) of {product}"  # because eodag'll clear product
    prod_id = product.as_dict()['id']
    zip_file = os.path.join(output_dir, f'{prod_id}.zip')
    path: ProductDownloadOutcome[str, EOProduct]
    try:
        path = ProductDownloadOutcome(
            dag.download(
                product,            # EODAG will clear this variable
                output_dir=output_dir,
                extract=True,       # Let's eodag do the job
                wait=dl_wait,       # Wait time in minutes between two download tries
                timeout=dl_timeout  # Maximum time in mins before stop retrying to download (default=20’)
            ),
            product)
        logger_.debug(ok_msg)
        if os.path.exists(zip_file) :
            try:
                logger_.debug('  Removing downloaded ZIP: %s', zip_file)
                os.remove(zip_file)
            except OSError:
                pass
        # eodag may say the product is correctly downloaded while it failed to do so
        # => let's do a quick sanity check
        if sanatize and (error := sanatize(output_dir, product, logger_)):
            path = ProductDownloadOutcome(error, product)

    except BaseException as e:  # pylint: disable=broad-except
        logger_.warning('  %s while attempting download of %s', e, prod_id)  # EODAG error message is good and precise enough, just use it!
        # logger_.error('Product is %s', product_property(product, 'storageStatus', 'online?'))
        logger_.debug('  Exception type is: %s', e.__class__.__name__)
        ## ERROR - Product is OFFLINE
        ## ERROR - Exception type is: NotAvailableError
        # logger_.error('======================')
        # logger_.exception(e)
        ## Traceback (most recent call last):
        ##   File "s1tiling/libs/S1FileManager.py", line 350, in _download_and_extract_one_product
        ##     path = ProductDownloadOutcome(dag.download(
        ##   File "site-packages/eodag/api/core.py", line 1487, in download
        ##     path = product.download(
        ##   File "site-packages/eodag/api/product/_product.py", line 288, in download
        ##     fs_path = self.downloader.download(
        ##   File "site-packages/eodag/plugins/download/http.py", line 269, in download
        ##     raise NotAvailableError(
        ## eodag.utils.exceptions.NotAvailableError: S1A_IW_GRDH_1SDV_20200401T044214_20200401T044239_031929_03AFBC_0C9E
        ##                                           is not available (OFFLINE) and could not be downloaded, timeout reached

        path = ProductDownloadOutcome(e, product)

    return path


def download_and_extract_products_parallel(  # pylint: disable=too-many-arguments, too-many-locals
    *,
    dag:        EODataAccessGateway,
    output_dir: str,
    products:   List[EOProduct],
    nb_procs:   int,
    context:    str,
    dl_wait:    int,
    dl_timeout: int,
    sanatize:   Optional[Sanatizer],
) -> List[ProductDownloadOutcome]:
    """
    Takes care of downloading exactly all remote products and unzipping them,
    if required, in parallel.

    Returns :class:`ProductDownloadOutcome` of :class:`EOProduct` or Exception.
    """
    nb_products = len(products)
    paths     : List[ProductDownloadOutcome] = []
    log_queue : multiprocessing.Queue   = multiprocessing.Queue()
    log_queue_listener = logging.handlers.QueueListener(log_queue)
    dl_work = partial(_download_and_extract_one_product, dag, output_dir, dl_wait, dl_timeout, logging, sanatize)
    with multiprocessing.Pool(nb_procs, mp_worker_config, [log_queue]) as pool:
        log_queue_listener.start()
        try:
            # In case timeout happens, we try again if and only if we have been able to download
            # other products after the timeout.
            # -> IOW, downloading instability justifies trying again.
            # /> On the contrary, on a complete network failure, we should not try again and again...
            while len(products) > 0:
                products_in_timeout : List[ProductDownloadOutcome] = []
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
                        if (timeout := _is_a_timeout(result.error())):
                            # Harmonize the exception type for all cases of download timeouts
                            # NB: Here we know that timeout is one of the possible timeout exception type
                            result.change_error(_as_timeout(cast(Exception, timeout)))
                            products_in_timeout.append(result)
                            assert isinstance(result.error(), TimeOutError)
                        else:
                            paths.append(result)
                products = []
                if nb_successes_since_timeout > nb_procs:
                    products = [r.related_product() for r in products_in_timeout]
                    logger.info("Attempting again to download %s products on timeout...", len(products))
                elif len(products_in_timeout) > 0:
                    logger.warning("No successful download since the first timeout observed => abort download")
                    for pit in products_in_timeout:
                        assert isinstance(pit.error(), TimeOutError)
                    paths.extend(products_in_timeout)
        finally:
            pool.close()
            pool.join()
            log_queue_listener.stop()  # no context manager for QueueListener unfortunately

    # paths returns the list of .SAFE directories
    return paths


def download_and_extract_products_sequential(  # pylint: disable=too-many-arguments, too-many-locals
    *,
    dag:        EODataAccessGateway,
    output_dir: str,
    products:   List[EOProduct],
    nb_procs:   int,
    context:    str,
    dl_wait:    int,
    dl_timeout: int,
    sanatize:   Optional[Sanatizer],
) -> List[ProductDownloadOutcome]:
    """
    Takes care of downloading exactly all remote products and unzipping them,
    if required, in parallel.

    Returns :class:`ProductDownloadOutcome` of :class:`EOProduct` or Exception.
    """
    nb_procs = 1  # Force nb of simultaneous DL to 1

    nb_products = len(products)
    paths     : List[ProductDownloadOutcome] = []
    dl_work = partial(_download_and_extract_one_product, dag, output_dir, dl_wait, dl_timeout, logger, sanatize)

    # In case timeout happens, we try again if and only if we have been able to download
    # other products after the timeout.
    # -> IOW, downloading instability justifies trying again.
    # /> On the contrary, on a complete network failure, we should not try again and again...

    indexed_products : List[Tuple[int, EOProduct]] = list(enumerate(products, 1))

    logger.info("Starting download of %d products...", nb_products)
    while len(indexed_products) > 0:
        products_in_timeout : List[ProductDownloadOutcome] = []
        nb_successes_since_timeout = 0
        for idx, product in indexed_products:
            # logger.info("Starting download of product #%d/%d: %s...", idx, nb_products, product)
            assert product
            result = dl_work(product)
            # logger.debug('DL -> %s', result)
            if result:
                logger.info("  Product #%d/%d %s correctly downloaded", idx, nb_products, result.value())
                logger.info('--> Downloading products%s... %s%%', context, (1+ len(paths)) * 100. / nb_products)
                paths.append(result)
                if len(products_in_timeout) > 0:
                    nb_successes_since_timeout += 1
            else:
                logger.warning("Cannot download %s: %s", result.related_product(), result.error())
                # TODO: make it possible to detect missing products in the analysis
                if (timeout := _is_a_timeout(result.error())):
                    # Harmonize the exception type for all cases of download timeouts
                    # NB: Here we know that timeout is one of the possible timeout exception type
                    result.change_error(_as_timeout(cast(Exception, timeout)))
                    products_in_timeout.append(result)
                    assert isinstance(result.error(), TimeOutError)
                else:
                    paths.append(result)
        indexed_products = []
        if nb_successes_since_timeout > nb_procs:
            indexed_products = [r.related_product() for r in products_in_timeout]
            logger.info("Attempting to download again %d products on timeout...", len(indexed_products))
        elif len(products_in_timeout) > 0:
            logger.warning("No successful download since the first timeout observed => abort download")
            for pit in products_in_timeout:
                assert isinstance(pit.error(), TimeOutError)
            paths.extend(products_in_timeout)

    # paths returns the list of .SAFE directories
    return paths


#: The default exported way of downloading
download_and_extract_products = download_and_extract_products_sequential
