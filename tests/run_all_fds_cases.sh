#!/bin/sh

#
# Copyright (c) 2020 by the FireDynamics group
#
# This file is part of the FDS particle spray postprocessor (fdspsp).
#
# fdspsp is free software; you can use it, redistribute it, and/or
# modify it under the terms of the MIT License. The full text of the
# license can be found in the file LICENSE.md at the top level
# directory of fdspsp.
#

# get path to FDS from command line
if [ $# -eq 0 ] ; then
  echo "Error: Provide path to a FDS executable as a command line parameter."
  exit 1
fi

# find all FDS cases in this folder recursively
basedir=$(dirname $(realpath $0))
casefiles=$(find ${basedir} -type f -name "*.fds")

# move into a dedicated folder in which all output files will be dumped
mkdir -p ${basedir}/fdsresults
cd ${basedir}/fdsresults

# run all cases
for case in ${casefiles} ; do
  echo "Running case: ${case}"
  $1 ${case}
done
