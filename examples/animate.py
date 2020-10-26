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
Script to animate particle data with matplotlib.

Provide a path to FDS simulation data via command line.
"""


import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from fdspsp import read


assert len(sys.argv) > 1, \
    "Please provide a path to FDS simulation data via command line!"
path_to_data = sys.argv[1]


# Read particle data
p_data = read.ParticleData(path_to_data)

# Find global minimum and maximum for all particle positions
limits = [[sys.float_info.max, sys.float_info.min] for _ in range(3)]
for c_label in p_data.classes:
  for o_step in range(p_data.n_outputsteps):
    if p_data.n_particles[c_label][o_step] > 0:
      for dim in range(3):
        limits[dim][0] = min(limits[dim][0],
                             np.amin(p_data.positions[c_label][o_step][dim]))
        limits[dim][1] = max(limits[dim][1],
                             np.amax(p_data.positions[c_label][o_step][dim]))
size = [0] * 3
for dim in range(3):
  assert limits[dim][1] > limits[dim][0]
  size[dim] = limits[dim][1] - limits[dim][0]


# Animate all particle classes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
o_step = 0


def updatefig(*args):
  global o_step

  ax.clear()
  for c_label in p_data.classes:
    ax.scatter(p_data.positions[c_label][o_step][0],
               p_data.positions[c_label][o_step][1],
               p_data.positions[c_label][o_step][2],
               label=c_label)

  ax.set_xlim3d(limits[0])
  ax.set_ylim3d(limits[1])
  ax.set_zlim3d(limits[2])
  ax.set_box_aspect(size)

  ax.set_title("time [s]: {:.5f}".format(p_data.times[o_step]))
  ax.set_xlabel("x [m]")
  ax.set_ylabel("y [m]")
  ax.set_zlabel("z [m]")

  plt.legend()
  plt.draw()

  o_step += 1
  if o_step >= p_data.n_outputsteps:
    o_step = 0


ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=False)
plt.show()
