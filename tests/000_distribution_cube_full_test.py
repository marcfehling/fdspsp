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
000: uniform particle distribution in an entire cube

Read particle data and verify correctness
"""


from os import path

from fdspsp.read import ParticleData

from tests import FDSRESULTS_DIR


pdata = ParticleData(path.join(FDSRESULTS_DIR,
                               "000_distribution_cube_full"))
classid = 0


def test_nparticles():
  # verify total number of particles in each time step
  #   (6x6x6 cells) x 10 particles/cell
  for npart in pdata.nparts[classid]:
    assert npart == 2160


def test_nparticles_per_cell():
  return 0
  # verify that each cell contains exactly 10 cells
  # cgrid = CartesianGrid()
  # for cell in cgrid:
  #   assert cell.particles == 10
