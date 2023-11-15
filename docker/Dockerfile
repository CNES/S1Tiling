# =========================================================================
#
#   Copyright 2022 (c) CS Group France. All rights reserved.
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
ARG REPOSITORY=registry.orfeo-toolbox.org/s1-tiling/s1tiling-dockers
ARG BASE_FULL_VERSION
FROM ${REPOSITORY}/s1tiling-base:${BASE_FULL_VERSION}

# This ARG needs to be after the FROM in order to not be cleared
ARG CI_COMMIT_SHA
ARG CI_PROJECT_DIR
ARG CREATION_DATE
ARG S1TILING_VERSION

LABEL org.opencontainers.image.authors="CS Group France, CNES"
LABEL org.opencontainers.image.base.name="${REPOSITORY}/s1tiling:${S1TILING_VERSION}"
LABEL org.opencontainers.image.created="${CREATION_DATE}"
LABEL org.opencontainers.image.description="This docker allow to run S1Tiling processing chain."
LABEL org.opencontainers.image.licences="Apache-2.0"
LABEL org.opencontainers.image.revision="$CI_COMMIT_SHA"
LABEL org.opencontainers.image.source="https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling"
LABEL org.opencontainers.image.title="S1Tiling public docker"
LABEL org.opencontainers.image.url="https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/blob/master/docker/Dockerfile"
LABEL org.opencontainers.image.vendor="CNES"
LABEL org.opencontainers.image.version="${S1TILING_VERSION}"

# TODO: automatically propagate S1Tiling version into the LABEL

WORKDIR /tmp
COPY "./" /tmp/clone
COPY ./docker/entrypoint.sh /opt

RUN \
        chmod +x /opt/entrypoint.sh \
        && . ${S1TILING_VENV}/bin/activate \
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
