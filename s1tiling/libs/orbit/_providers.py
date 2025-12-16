#!/usr/bin/env python3
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

"""This sub-module defines access clients to EOF Providers"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
import logging
import os
from typing import List, Protocol, TypeVar

from eodag import EOProduct
from eodag.api.core import EODataAccessGateway
from eodag.api.search_result import SearchResult
from eodag.plugins.authentication.base import Authentication
# from eodag.plugins.authentication.openid_connect import CodeAuthorizedAuth
from eodag.utils.exceptions import MisconfiguredError

from ..Utils           import rename
from ..exceptions      import ConfigurationError
from ..outcome         import ProductDownloadOutcome
from ..utils.algorithm import partition
from ..utils.eodag     import EODAG_DEFAULT_DOWNLOAD_TIMEOUT, EODAG_DEFAULT_DOWNLOAD_WAIT, download_and_extract_products
from ..utils.path      import AnyPath


logger = logging.getLogger("s1tiling.orbit")

Value   = TypeVar("Value")
File    = TypeVar('File')
Product = TypeVar('Product')
T       = TypeVar("T")


#: Outcome result for EOF file: (filename|error, related product)
EOFProductDownloadOutcome = ProductDownloadOutcome[Value, Product]


class Provider(Protocol):
    """
    Protocol that describes expected interface to search and download EOF files from data providers.
    """
    def search(
        self,
        first_date: datetime,
        last_date:  datetime,
        missions:   Collection[str] = (),
    ) -> SearchResult:
        """
        Search for precise orbit files in the specified time range in the actual provider.

        :param datetime first_date: Start of the search time range
        :param datetime last_date:  End of the search time range
        :param sequence missions:   Set of "S1A", "S1B", "S1C", "S1D" missions to retrict search.
                                    If empty, to filtering is done.
        :return: a list of EOF file specifications that match the search request parameters.
        """
        ...

    def download(
        self,
        eofs            : List[EOProduct],
        destination_dir : AnyPath,
    ) -> List[EOFProductDownloadOutcome]:
        """
        Download the precise orbit files specified.

        :param list eofs: List of orbit precise file specifications (returned by :meth:`search`)
        :param path destination_dir: Where the files will be downloaded.
        :return: List of filenames pointing to the downloaded files.
        """
        ...


class EodagProvider:
    """
    Actual provider for interacting with Copernicus Dataspace through EODAG.

    In case 2FA is activated with the Copernicus Dataspace account, you'll need to set:
    - :envvar:`$EODAG__COP_DATASPACE__AUTH__CREDENTIALS__TOTP`
    - at this moment, there is no way to generate an access token before running S1Tiling as
      described in https://documentation.dataspace.copernicus.eu/APIs/Token.html

      .. todo:: https://github.com/CS-SI/eodag/issues/1937 resolution would be required to support this feature

    The One Time Password approach has a short live span. That's why we will try to obtain an
    authentication token as soon as possible if
    :envvar:`$EODAG__COP_DATASPACE__AUTH__CREDENTIALS__TOTP` is set.

    We recommend using an external :envvar`$EODAG__COP_DATASPACE__AUTH__CACHED_TOKEN_RESPONSE`
    instead, once eodag supports this feature.
    """
    PROVIDER = 'cop_dataspace'

    def __init__(
        self,
        dag        : EODataAccessGateway,
        dl_wait    : int = EODAG_DEFAULT_DOWNLOAD_WAIT,
        dl_timeout : int = EODAG_DEFAULT_DOWNLOAD_TIMEOUT,
    ):
        self.__dag        = dag
        self.__dl_wait    = dl_wait
        self.__dl_timeout = dl_timeout

        ## <depends on https://github.com/CS-SI/eodag/issues/1937>
        ### * A long-lived token exist => use it
        ##if os.getenv("EODAG__COP_DATASPACE__AUTH__CACHED_TOKEN_RESPONSE", None):
        ##    logger.debug('An identification token to %s has been set, it will be used to fetch EOF products', self.PROVIDER)
        ##    return
        ## </depends>

        # * No short-lived TOTP => on-the-fly identification
        totp = os.getenv("EODAG__COP_DATASPACE__AUTH__CREDENTIALS__TOTP", None)
        if not totp:
            logger.debug('Identification to %s (for EOF product download) will be done on-the-fly', self.PROVIDER)
            return

        # * Short-lived TOTP => acquire now a long-lived token!
        try:
            search_plugins = dag._plugins_manager.get_search_plugins(provider=self.PROVIDER)
            if search_plugins:
                self._authentication_plugin = dag._plugins_manager.get_auth_plugin(next(search_plugins))
            # logger.debug("%s EODAG auth plugin: %s", self.provider_name, self._authentication_plugin)
        except KeyError:
            raise AssertionError(f"provider {self.PROVIDER!r} not supported by EODAG")  # pylint: disable=raise-missing-from
        if not self._authentication_plugin:
            # Dare we assert self._authentication_plugin to be never None???
            raise ConfigurationError(
                f"Cannot authenticate on {self.PROVIDER}",
                os.path.join(dag.conf_dir, "eodag.yml"),
            )
        assert isinstance(self._authentication_plugin, Authentication)
        try:
            logger.debug("Try to authenticate for %s", self.PROVIDER)
            self._authorization = self._authentication_plugin.authenticate()
            # logger.debug("authorization for %s: %s\n\t-> %s", self.PROVIDER, self._authorization, self._authorization.token)
        except MisconfiguredError as e:
            raise ConfigurationError(
                f"Cannot authenticate on {self.PROVIDER}",
                os.path.join(dag.conf_dir, "eodag.yml"),
            ) from e

    @classmethod
    def is_configured(cls, dag: EODataAccessGateway) -> bool:
        """
        Tells whether cop_dataspace credentials have been configured in eodag.yaml configuration file.
        """
        # logger.debug("COP.is_configured -> %s", getattr(dag.providers_config["cop_dataspace"].auth, 'credentials', 'NADA'))
        return "username" in getattr(dag.providers_config["cop_dataspace"].auth, 'credentials', {})

    def search(
            self,
            first_date: datetime,
            last_date:  datetime,
            missions:   Collection[str] = (),
    ) -> SearchResult:
        """
        Search for precise orbit files in the specified time range in the actual provider.

        :param datetime first_date: Start of the search time range
        :param datetime last_date:  End of the search time range
        :param sequence missions:   Set of "S1A", "S1B", "S1C", "S1D" missions to retrict search.
                                    If empty, to filtering is done.
        :return: a list of EOF file specifications that match the search request parameters.
        """
        assert isinstance(missions, (list, tuple)), f"{missions=!r} is a {missions.__class__.__name__}"
        dag_platform_list_param = missions[0] if len(missions) == 1 else None
        # logger.debug('mission filter: %s -> %s', list(missions), dag_platform_list_param)
        eofs = self.__dag.search_all(
            provider="cop_dataspace",
            collection="SENTINEL-1",
            productType="S1_AUX_POEORB",
            start=first_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            end=last_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            raise_errors=True,
            platformSerialIdentifier=dag_platform_list_param,
        )

        # Filter platform -- if it could not be done earlier in the search() request.
        if len(missions) > 1:
            filtered_products = SearchResult([])
            for platform in missions:
                filtered_products.extend(eofs.filter_property(platformSerialIdentifier=platform))
            eofs = filtered_products

        logger.debug("%s EOFs found:", len(eofs))
        for eof in eofs:
            logger.debug("- %s", eof)
        return eofs

    def download(
        self,
        eofs            : List[EOProduct],
        destination_dir : AnyPath,
    ) -> List[EOFProductDownloadOutcome]:
        """
        Download the precise orbit files specified.

        :param list eofs: List of orbit precise file specifications (returned by :meth:`search`)
        :param path destination_dir: Where the files will be downloaded.
        :return: List of filenames pointing to the downloaded files.
        """
        # Because EODAG 3.10.0 cannot download .EOF files directly in the destination_dir
        # (https://github.com/CS-SI/eodag/issues/1942)
        # and that as a consequence we nned to manually take care of moving them around,
        # => then we need to manually filter out the products already there.
        def eof_filename(eof: EOProduct) -> str:
            prod_name =  eof.as_dict()['id']
            return os.path.join(destination_dir, f"{prod_name}.EOF")

        # logger.debug("Requested EOFs: %s in %s", eofs, destination_dir)
        # for eof in eofs:
        #     logger.debug("- prod: %s ∃ %s -> %r ",
        #                  eof,
        #                  os.path.isfile(eof_filename(eof)),
        #                  eof_filename(eof))

        on_disk, to_download = partition(lambda eof : os.path.isfile(eof_filename(eof)), eofs)
        logger.debug("EOFs that will be downloaded: %s", to_download)

        # Do download
        products = download_and_extract_products(
            dag=self.__dag,
            output_dir=str(destination_dir),
            products=to_download,
            nb_procs=1,
            context=" POEORB",
            dl_wait=self.__dl_wait,
            dl_timeout=self.__dl_timeout,
            sanatize=None,
        )

        # Move back the EOF to their right place
        # (Eodag stores EOF products under ${dest}/{product}/{product}.EOF
        # https://github.com/CS-SI/eodag/issues/1942)
        for p in products:
            if p:
                old_dirname = p.value()
                basename    = os.path.basename(p.value())
                orig        = os.path.join(old_dirname, f"{basename}.EOF")
                p.inplace_transform(lambda name: f"{name}.EOF")
                rename(orig, p.value())
                os.rmdir(old_dirname)

        # And inject back the EOF files found on disk
        for eof in on_disk:
            products.append(EOFProductDownloadOutcome(eof_filename(eof), eof))

        return products
