# =========================================================================
#
#   Copyright 2021 (c) CS Group France. All rights reserved.
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
# Authors: Aurélien BRICIER (CS Group France)
#          Mickaël SAVINAUD (CS Group France)
#          Luc HERMITTE     (CS Group France)
#
# =========================================================================
ARG OTB_VERSION=7.3.0
ARG REPOSITORY=registry.orfeo-toolbox.org/s1-tiling/s1tiling-dockers
FROM ${REPOSITORY}/s1tiling-base:ubuntu-otb${OTB_VERSION}
# This ARG needs to be after the FROM in order to not be cleared
ARG CI_PROJECT_DIR
ARG S1TILING_VERSION
LABEL maintainer="CS GROUP France"
# TODO: automatically propagate S1Tiling version into the LABEL
LABEL version="${S1TILING_VERSION}"
LABEL description="This docker allow to run S1Tiling processing chain."

WORKDIR /tmp
COPY "./" /tmp/clone
COPY ./docker/entrypoint.sh /opt

RUN \
        chmod +x /opt/entrypoint.sh \
        && . ${S1TILING_VENV}/bin/activate \
        && . "${OTB_INSTALL_DIRNAME}/otbenv.profile" \
        && python3 -m pip --no-cache-dir install -r /tmp/clone/docker/s1tiling-requirements.txt \
        && python3 -m pip --no-cache-dir install "/tmp/clone" --no-deps \
        && rm -rf /tmp/clone

        # Install S1Tiling requirements independently of S1Tiling
        # itself in two steps.
        # This permits to make sure all S1Tiling, but GDAL, dependencies
        # are installed recursivelly, and then only S1Tiling itself.
        # The problem being that S1Tiling wheel depends on a hard coded
        # version of GDAL while the docker may depend on another version
        # of GDAL through OTB.

ENTRYPOINT [ "/opt/entrypoint.sh" ]
CMD [ "-h" ]
