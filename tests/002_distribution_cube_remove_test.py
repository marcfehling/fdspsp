#
# Copyright (c) 2020 by the FireDynamics authors
#
# This file is part of the FDS particle spray postprocessor (fdspsp).
#
# fdspsp is free software; you can use it, redistribute it, and/or
# modify it under the terms of the MIT License. The full text of the
# license can be found in the file LICENSE.md at the top level
# directory of fdspsp.
#

"""
002: uniform particle distribution in an entire cube from which
     particles will be removed

Read particle data and verify correctness.
"""


from os import path

from fdspsp.read import ParticleData

from tests import FDSRESULTS_DIR


def test_distribution_cube_remove():
  pdata = ParticleData(path.join(FDSRESULTS_DIR,
                                 "002_distribution_cube_remove"))
  classid = 0

  # verify total number of particles in each time step
  #   step 0: (6x6x6 cells) x 10 particles/cell
  #   step 0: (6x6x6 cells) x  0 particles/cell
  for out in range(pdata.info["noutputsteps"]):
    if out == 0:
      assert pdata.nparts[classid][out] == 2160
    elif out == 1:
      assert pdata.nparts[classid][out] == 0
    else:
      assert False

  # verify that each cell contains exactly 10 cells
  # cgrid = CartesianGrid()
  # for cell in cgrid:
  #   assert cell.particles == 10
