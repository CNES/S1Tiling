#!/bin/sh

if [ -n "${CI_COMMIT_REF_NAME}" ] ; then
    echo "${CI_COMMIT_REF_NAME}"
else
    git describe --tags --exact-match 2> /dev/null || git symbolic-ref -q --short HEAD
fi

# test -n ${CI_COMMIT_REF_NAME}  \
    # && version="${CI_COMMIT_REF_NAME}" \
    # || version="$(git describe --tags --exact-match 2> /dev/null || git symbolic-ref -q --short HEAD)"
