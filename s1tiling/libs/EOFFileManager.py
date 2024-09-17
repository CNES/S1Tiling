#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   All rights reserved.
#   Copyright 2017-2024 (c) CNES.
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

""" This module defines the EOFFileManager class"""

from abc import abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from dateutil.parser import parse
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

from eodag.api.core import EODataAccessGateway
from eodag.plugins.authentication.base import Authentication
from eodag.plugins.authentication.openid_connect import CodeAuthorizedAuth
from eodag.utils.exceptions import MisconfiguredError

from eof._auth import get_netrc_credentials
from eof.client import Client
from eof.download import ASFClient, DataspaceClient, Filename

from s1tiling.libs.configuration import Configuration
from s1tiling.libs.exceptions import ConfigurationError
from s1tiling.libs.outcome import DownloadOutcome
from enum import Enum


logger = logging.getLogger('s1tiling.filemanager')

class Provider:
    """
    Abstract class for wrapping all :class:`eof.client.Client` kinds.
    """
    def __init__(self, provider_name: str):
        self._provider_name = provider_name

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
            missions:   Sequence[str] = ()
    ) -> List:
        """
        Search for precise orbit files in the specified time range in the actual provider.

        :param datetime first_date: Start of the search time range
        :param datetime last_date:  End of the search time range
        :param sequence mission:    Set of "S1A", "S1B" missions to retrict search.
                                    If empty, to filering is done.
        :return: a list of EOF file specifications that match the search request parameters.

        .. warning: This list can only be used with the :meth:`download_all` method of the provider of the same type.
        """
        assert isinstance(missions, (list,tuple))
        self._client = self._instantiate_client()
        eofs = self._client.query_orbits_by_dt_range(first_date, last_date, missions)
        logger.debug("%s EOFs found:", len(eofs))
        for eof in eofs:
            logger.debug("- %s", eof)
        return eofs

    def download(self, eofs, destination_dir : Filename) -> Union[List[Path], List[Filename]]:
        """
        Download the precise orbit files specified.

        :param list eofs: List of orbit precise file specifications (returned by :meth:`search`
        :param path destination_dir: Where the files will be downloaded.
        :return: List of filenames pointing to the downloaded files.
        """
        auth_info = self._auth_info()
        # logger.debug("client.authenticate(%s)", auth_info)
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
    def _auth_info(self) -> Dict[str,str]:
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
        super().__init__('cop_dataspace')
        self._token = os.getenv('EODAG__COP_DATASPACE__AUTH__TOKEN', access_token)
        if (self._token):
            self._authentication_plugin  = None
            self._authorization          = None
        else:
            try:
                self._authentication_plugin = dag._plugins_manager.get_auth_plugin(self.provider_name)
            except KeyError as e:
                raise AssertionError(f"provider {self.provider_name!r} not supported by EODAG")
            if not self._authentication_plugin:
                # Dare we assert self._authentication_plugin to be never None???
                raise ConfigurationError(
                        f"Cannot authenticate on {self.provider_name}",
                        os.path.join(dag.conf_dir, "eodag.yml")
                )
            assert isinstance(self._authentication_plugin, Authentication)
            try:
                self._authorization = self._authentication_plugin.authenticate()
            except MisconfiguredError as e:
                raise ConfigurationError(
                        f"Cannot authenticate on {self.provider_name}",
                        os.path.join(dag.conf_dir, "eodag.yml")
                ) from e

    def _instantiate_client(self) -> DataspaceClient:
        """
        Specialization that instanciates a :class:`eof.DataspaceClient`.
        """
        return DataspaceClient()

    def _auth_info(self) -> Dict[str,str]:
        """
        Specialization that returns the already obtained "access_token" parameter for
        :meth:`eof.clent.DataspaceClient.authenticate` method.
        """
        return {'access_token': self.get_token()}

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
        return 'username' in getattr(dag.providers_config["cop_dataspace"].auth, 'credentials', {})


class ASFProvider(Provider):
    """
    Actual provider for interacting with earthdata through :class:`eof.ASFClient`.

    Unfortunately earthdata providers are not supported yet by eodag -- see
    https://github.com/CS-SI/eodag/issues/755 and https://github.com/CS-SI/eodag/issues/8
    As a consequence, we need to fetch the ID elsewhere. Let's depend on :envvar:`$NETRC` for now...
    """
    def __init__(self, **kwargs):
        super().__init__('asf')
        self.__build_options = kwargs

    def _instantiate_client(self) -> ASFClient:
        """
        Specialization that instanciates a :class:`eof.DataspaceClient`.
        """
        return ASFClient(**self.__build_options)

    def _auth_info(self) -> Dict[str,str]:
        """
        Specialization that returns no extra parameter for :meth:`eof.clent.ASFClient.authenticate` method.
        """
        return {}

    @classmethod
    def is_configured(cls, dag: EODataAccessGateway) -> bool:
        """
        Tells whether EarthData credentials have been configured in $NETRC configuration file.
        """
        try:
            _, _ = get_netrc_credentials('urs.earthdata.nasa.gov')
            return True
        except BaseException as e:
            logger.debug("Cannot obtain ASF credentials in .netrc: %s", e)
            return False


# =====[ The main public interface

class ProviderKind(Enum):
    COP_DATASPACE = 1
    EARTHDATA     = 2


class EOFFileManager:
    # TODO: Don't depend on Configuration
    def __init__(self, cfg: Configuration, dag: EODataAccessGateway):
        """
        constructor
        """
        assert(dag)
        self.__cfg           = cfg
        self.__dag           = dag
        self.__first_date    = parse(cfg.first_date)
        self.__last_date     = parse(cfg.last_date) + timedelta(days=1) - timedelta(seconds=1)
        self.__dest_dir      = cfg.eof_directory
        self.__missions      = cfg.platform_list
        self.__build_options : Dict[ProviderKind, Dict] = {
                ProviderKind.COP_DATASPACE : {
                    'class':   DataspaceProvider,
                    'options': {'dag': dag},
                },
                ProviderKind.EARTHDATA     : {
                    'class': ASFProvider, 
                    'options': {},
                },
        }

    def add_extra_build_option(self, provider: ProviderKind, **kwargs):
        """
        Permits to tune construction parameters passed to the :class:`Provider` instances.

        Typically, it can be used to set `cache_dir` when building :class:`ASFProvider`
        """
        self.__build_options[provider]['options'].update(**kwargs)

    def _instanciate_provider(self, provider: ProviderKind) -> Provider:
        """
        Internal method that do instantiate an EOF provider.
        """
        provider_data = self.__build_options[provider]
        return provider_data['class'](**provider_data['options'])

    def _ensure_workspaces_exist(self) -> None:
        """
        Makes sure the directories used for :
        - eof files
        all exist
        """
        for path in [self.__dest_dir]:
            if not os.path.isdir(path):
                os.makedirs(path, exist_ok=True)

    def download_eof(
            self,
            missions  : Sequence[str] = (),
            dryrun    : bool          = False,
    ) -> List[DownloadOutcome]:
        """
        Main entry point to search and download the EOF precise orbit files.

        The orbits files are searched in the specified time range (construction
        parameters), for the chosen missions (default is set during construction but can
        be overridden when calling :meth:`download_eof`.
        """
        if not self.__cfg.download:
            logger.info("Using EOF files already downloaded, as per configuration request")
            # TODO: Should do a glob/ls
            return []

        self._ensure_workspaces_exist()

        request = f"EOF files between {self.__first_date} and {self.__last_date}"
        errors : List[DownloadOutcome] = []

        provider_kinds = [p for p in ProviderKind if self.__build_options[p]['class'].is_configured(self.__dag)]
        if len(provider_kinds) == 0:
            logger.warning("No data provider has been configured for EOF files")
            return [DownloadOutcome(RuntimeError("No data provider has been configured for EOF files"), request)]
        logger.debug(
                "EOF files will be searched on %s between %s and %s",
                " and ".join((str(p) for p in provider_kinds)),
                self.__first_date,
                self.__last_date,
        )
        missions = missions or self.__missions
        for provider_kind in provider_kinds:
            try:
                provider = self._instanciate_provider(provider_kind)
                eofs = provider.search(self.__first_date, self.__last_date, missions)
                files = provider.download(eofs, self.__dest_dir)
                return [DownloadOutcome(f, f) for f in files]
            except BaseException as e:
                logger.warning(e, exc_info=False)
                logger.debug(e, exc_info=True)
                errors.append(DownloadOutcome(e, request))
        else:
            if len(errors) == 0:
                errors = [DownloadOutcome(RuntimeError("No data provider has been configured for EOF files"), request)]
            return errors

    def search_for(
            self,
            relative_orbit : int,
            missions       : Sequence[str] = (),
            dryrun         : bool          = False,
    ) -> List[DownloadOutcome]:
        # 1. scan dest_dir for EOF having relative_orbit
        #    priority to the files in the time range
        # 2. if not, download files in the time range
        #    analyse the new files
        #
        # @post: for each EOF file detected, build a dict of min-max abs- and/or rel- orbit numbers
        return []
