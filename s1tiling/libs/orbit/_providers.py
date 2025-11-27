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

"""This sub-module defines access clients to EOF Providers"""

from abc import abstractmethod
from collections.abc import Collection
from datetime import datetime
import logging
from pathlib import Path
import os
from typing import Dict, List, Optional, TypeVar, Union

from eodag import EOProduct
from eodag.api.core import EODataAccessGateway
from eodag.api.search_result import SearchResult
from eodag.plugins.authentication.base import Authentication
from eodag.plugins.authentication.openid_connect import CodeAuthorizedAuth
from eodag.utils.exceptions import MisconfiguredError

from eof._auth import get_netrc_credentials
from eof.client import Client
from eof.download import ASFClient, DataspaceClient


from ..exceptions  import ConfigurationError
from ..outcome     import ProductDownloadOutcome
from ..utils.eodag import EODAG_DEFAULT_DOWNLOAD_TIMEOUT, EODAG_DEFAULT_DOWNLOAD_WAIT, download_and_extract_products
from ..utils.path  import AnyPath


logger = logging.getLogger("s1tiling.orbit")

Value   = TypeVar("Value")
File    = TypeVar('File')
Product = TypeVar('Product')
T       = TypeVar("T")


EOFDownloadOutcome = ProductDownloadOutcome[Value, Product]


class Provider:
    """
    Abstract class for wrapping all :class:`eof.client.Client` kinds.
    """

    def __init__(self, provider_name: str):
        self._provider_name = provider_name
        self._client = None

    @property
    def provider_name(self) -> str:
        """
        Actual provider name: "cop_dataspace" or "asf".
        """
        return self._provider_name

    def search(
            self,
            first_date: datetime,
            last_date:  datetime,
            missions:   Collection[str] = ()
    ) -> List:
        """
        Search for precise orbit files in the specified time range in the actual provider.

        :param datetime first_date: Start of the search time range
        :param datetime last_date:  End of the search time range
        :param sequence missions:   Set of "S1A", "S1B", "S1C" missions to retrict search.
                                    If empty, to filering is done.
        :return: a list of EOF file specifications that match the search request parameters.

        .. warning: This list can only be used with the :meth:`download_all` method of the provider of the same type.
        """
        assert isinstance(missions, (list, tuple))
        self._client = self._instantiate_client()
        assert self._client
        eofs = self._client.query_orbits_by_dt_range(first_date, last_date, missions)
        logger.debug("%s EOFs found:", len(eofs))
        for eof in eofs:
            logger.debug("- %s", eof)
        return eofs

    def download(self, eofs, destination_dir : AnyPath) -> Union[List[Path], List[AnyPath]]:
        """
        Download the precise orbit files specified.

        :param list eofs: List of orbit precise file specifications (returned by :meth:`search`
        :param path destination_dir: Where the files will be downloaded.
        :return: List of filenames pointing to the downloaded files.
        """
        auth_info = self._auth_info()
        # logger.debug("client.authenticate(%s)", auth_info)
        assert self._client
        session = self._client.authenticate(**auth_info)
        files = session.download_all(
                eofs,
                destination_dir,
                max_workers=1,  # Because VCR is not thread safe
        )
        logger.info("%s files downloaded:", len(files))
        for file in files:
            logger.info("- %s", file)
            assert os.path.exists(file), f"{file!r} doesn't exist..."
        return files

    @abstractmethod
    def _auth_info(self) -> Dict[str, str]:
        """
        Internal variation point that returns the dictionary of parameters to use with
        :meth:`eof.clent.Client.authenticate` method.
        """
        pass

    @abstractmethod
    def _instantiate_client(self) -> Client:
        """
        Internal variation point that'll instanciate a :class:`eof.client.Client` matching the actual provider.
        """
        pass


class EodagProvider:
    def __init__(
        self,
        dag         : EODataAccessGateway,
        dl_wait     : int = EODAG_DEFAULT_DOWNLOAD_WAIT,
        dl_timeout  : int = EODAG_DEFAULT_DOWNLOAD_TIMEOUT,
        access_token: Optional[str] = None
    ):
        self.__dag        = dag
        self.__dl_wait    = dl_wait
        self.__dl_timeout = dl_timeout

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
        :param sequence missions:   Set of "S1A", "S1B", "S1C" missions to retrict search.
                                    If empty, to filering is done.
        :return: a list of EOF file specifications that match the search request parameters.

        .. warning: This list can only be used with the :meth:`download_all` method of the provider of the same type.
        """
        assert isinstance(missions, (list, tuple)), f"{missions=!r} is a {missions.__class__.__name__}"
        dag_platform_list_param = missions[0] if len(missions) == 1 else None
        # logger.debug('mission filter: %s -> %s', list(missions), dag_platform_list_param)
        eofs = self.__dag.search_all(
            provider="cop_dataspace",
            collection="SENTINEL-1",
            productType="AUX_POEORB",
            start=first_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            end=last_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            raise_errors=True,
            platformSerialIdentifier=dag_platform_list_param,
        )

        # Filter platform -- if it could not be done earlier in the search() request.
        if len(missions) > 1:
            filtered_products = SearchResult([])
            for platform in missions:
                # There is a bug in eodag 3.10: #1930 -> Hence explicitly using the last char from platform
                filtered_products.extend(eofs.filter_property(platformSerialIdentifier=platform[-1]))
            eofs = filtered_products

        logger.debug("%s EOFs found:", len(eofs))
        for eof in eofs:
            logger.debug("- %s", eof)
        return eofs

    def download(
        self,
        eofs            : List[EOProduct],
        destination_dir : AnyPath,
        context         : str,
    ) -> List[EOFDownloadOutcome]:
        return download_and_extract_products(
            dag=self.__dag,
            raw_directory=str(destination_dir),
            products=eofs,
            nb_procs=1,
            context=context,
            dl_wait=self.__dl_wait,
            dl_timeout=self.__dl_timeout,
            sanatize=None,
        )


class DataspaceProvider(Provider):
    """
    Actual provider for interacting with Copernicus Dataspace through :class:`eof.DataspaceClient`.

    EODAG API is used to extract authentication information for "cop_dataspace" provider.

    In case 2FA is activated with the Copernicus Dataspace account, either set:
    - :envvar:`$EODAG__COP_DATASPACE__AUTH__CREDENTIALS__TOTP`
    - or generate the access token described https://documentation.dataspace.copernicus.eu/APIs/Token.html and
      store it into :envvar`$EODAG__COP_DATASPACE__AUTH__TOKEN`.
    """
    def __init__(self, dag: EODataAccessGateway, access_token: Optional[str] = None):
        super().__init__("cop_dataspace")
        self._token = os.getenv("EODAG__COP_DATASPACE__AUTH__TOKEN", access_token)
        if self._token:
            self._authentication_plugin  = None
            self._authorization          = None
        else:
            try:
                search_plugins = dag._plugins_manager.get_search_plugins(provider=self.provider_name)
                if search_plugins:
                    self._authentication_plugin = dag._plugins_manager.get_auth_plugin(next(search_plugins))
                # logger.debug("%s EODAG auth plugin: %s", self.provider_name, self._authentication_plugin)
            except KeyError:
                raise AssertionError(f"provider {self.provider_name!r} not supported by EODAG")  # pylint: disable=raise-missing-from
            if not self._authentication_plugin:
                # Dare we assert self._authentication_plugin to be never None???
                raise ConfigurationError(
                        f"Cannot authenticate on {self.provider_name}",
                        os.path.join(dag.conf_dir, "eodag.yml"),
                )
            assert isinstance(self._authentication_plugin, Authentication)
            try:
                logger.debug("Try to authenticate for %s", self.provider_name)
                self._authorization = self._authentication_plugin.authenticate()
                # logger.debug("authorization for %s: %s", self.provider_name, self._authorization)
            except MisconfiguredError as e:
                raise ConfigurationError(
                        f"Cannot authenticate on {self.provider_name}",
                        os.path.join(dag.conf_dir, "eodag.yml"),
                ) from e

    def _instantiate_client(self) -> DataspaceClient:
        """
        Specialization that instanciates a :class:`eof.DataspaceClient`.
        """
        return DataspaceClient()

    def _auth_info(self) -> Dict[str, str]:
        """
        Specialization that returns the already obtained "access_token" parameter for
        :meth:`eof.clent.DataspaceClient.authenticate` method.
        """
        return {"access_token": self.get_token()}

    def get_token(self) -> str:
        """
        Obtain the access token from
        """
        if self._token:
            return self._token
        assert isinstance(self._authorization, CodeAuthorizedAuth)
        # logger.debug("cop authorization: %s", self._authorization)
        # logger.debug("cop token: %s", self._authorization.token)
        return self._authorization.token

    @classmethod
    def is_configured(cls, dag: EODataAccessGateway) -> bool:
        """
        Tells whether cop_dataspace credentials have been configured in eodag.yaml configuration file.
        """
        # logger.debug("COP.is_configured -> %s", getattr(dag.providers_config["cop_dataspace"].auth, 'credentials', 'NADA'))
        return "username" in getattr(dag.providers_config["cop_dataspace"].auth, 'credentials', {})


class ASFProvider(Provider):
    """
    Actual provider for interacting with earthdata through :class:`eof.ASFClient`.

    Unfortunately earthdata providers are not supported yet by eodag -- see
    https://github.com/CS-SI/eodag/issues/755 and https://github.com/CS-SI/eodag/issues/8
    As a consequence, we need to fetch the ID elsewhere. Let's depend on :envvar:`$NETRC` for now...
    """

    def __init__(self, **kwargs):
        super().__init__("asf")
        self.__build_options = kwargs

    def _instantiate_client(self) -> ASFClient:
        """
        Specialization that instanciates a :class:`eof.DataspaceClient`.
        """
        return ASFClient(**self.__build_options)

    def _auth_info(self) -> Dict[str, str]:
        """
        Specialization that returns no extra parameter for :meth:`eof.clent.ASFClient.authenticate` method.
        """
        return {}

    @classmethod
    def is_configured(cls, dag: EODataAccessGateway) -> bool:  # pylint: disable=unused-argument
        """
        Tells whether EarthData credentials have been configured in $NETRC configuration file.
        """
        try:
            _, _ = get_netrc_credentials("urs.earthdata.nasa.gov")
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug("Cannot obtain ASF credentials in .netrc: %s", e)
            return False
