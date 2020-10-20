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
001: uniform particle distribution in a section of a cube

Read particle data and verify correctness.
"""


from os import path

from fdspsp import *

from tests import FDSRESULTS_DIR


pdata = read.ParticleData(path.join(FDSRESULTS_DIR,
                                    "001_distribution_cube_partial"))
class_id = 0


def test_n_particles():
  # verify total number of particles in each time step
  #   (6x5x3 cells) x 10 particles/cell
  for n_part in pdata.n_particles[class_id]:
    assert n_part == 900


def test_n_particles_per_cell():
  return 0
  # verify that each cell contains exactly 10 cells
  # cgrid = CartesianGrid()
  # for cell in cgrid:
  #   assert cell.particles == 10
