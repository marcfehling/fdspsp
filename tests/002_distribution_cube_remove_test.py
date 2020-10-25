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
002: uniform particle distribution in an entire cube from which
     particles will be removed

Read particle data and verify correctness.
"""


from os import path

from fdspsp import *

from tests import FDSRESULTS_DIR


path_to_data = path.join(FDSRESULTS_DIR, "002_distribution_cube_remove")


def test_particle_selection():
  """
  Check selection of particle classes
  """
  read.ParticleData(path_to_data, classes=["tracer"])
  read.ParticleData(path_to_data, classes=["tracer_noquantities"])


pdata = read.ParticleData(path_to_data)


def test_n_particles():
  """
  verify total number of particles in each time step for each class
    output step 0: (6x6x6 cells) x 10 particles/cell
    output step 1: (6x6x6 cells) x  0 particles/cell
  """
  for n_particles in pdata.n_particles.values():
    for o_step in range(pdata.n_outputsteps):
      if o_step == 0:
        assert n_particles[o_step] == 2160
      elif o_step == 1:
        assert n_particles[o_step] == 0
      else:
        assert False


def test_n_particles_per_cell():
  return 0
  # verify that each cell contains exactly 10 cells
  # cgrid = CartesianGrid()
  # for cell in cgrid:
  #   assert cell.particles == 10
