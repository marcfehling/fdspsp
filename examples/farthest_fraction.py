#!/usr/bin/env python

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

"""
Script to calculate the radius of the spray front and its velocity with
the farthest fraction method (FF).

Provide a path to FDS simulation data via the command line.
"""


import sys

import matplotlib.pyplot as plt
import numpy as np

from fdspsp import particle, read, reduce, select


assert len(sys.argv) > 1, \
    "Please provide a path to FDS simulation data via the command line!"
path_to_data = sys.argv[1]


absolute_threshold = 1
top_share = 0.1
reference_point = [0., 0., 0.]


# Read particle data
p_data = read.ParticleData(path_to_data)

# Determine spray front properties with the farthest fraction method (FF)
ff_distance = {c_label: np.empty(p_data.n_outputsteps)
               for c_label in p_data.classes}
ff_velocity = {c_label: np.empty(p_data.n_outputsteps)
               for c_label in p_data.classes}
for c_label in p_data.classes:
  for o_step in range(p_data.n_outputsteps):
    # Determine particle properties
    prt_distances = particle.distances_from_reference_point(
        p_data, c_label, o_step, reference_point)
    prt_velocities = particle.velocities(p_data, c_label, o_step)

    # Select only moving particles
    prt_indices_absvelocities = select.absolute_threshold(
        prt_velocities, absolute_threshold)

    # Further select top percentile
    prt_indices_ff = select.top_percentile(
        prt_distances, top_share, prt_indices_absvelocities)

    # Average over top percentile
    ff_distance[c_label][o_step] = reduce.mean(prt_distances, prt_indices_ff)
    ff_velocity[c_label][o_step] = reduce.mean(prt_velocities, prt_indices_ff)

# Plot distance and velocity over time
plt.figure()
plt.title("spray radius over time")
plt.xlabel("time [s]")
plt.ylabel("radius [m]")
for c_label in p_data.classes:
  plt.plot(p_data.times, ff_distance[c_label], label=c_label)

plt.figure()
plt.title("spray front velocity over time")
plt.xlabel("time [s]")
plt.ylabel("velocity [m/s]")
for c_label in p_data.classes:
  plt.plot(p_data.times, ff_velocity[c_label], label=c_label)

plt.show()
