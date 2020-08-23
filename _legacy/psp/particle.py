#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def masses(pdata, ptype, iout):
  """
  Calculates the physical mass of each numerical particle.

  Parameters
  ----------
  pdata : ParticleData
    Container which stores all particle data read from file system.
  ptype: ParticleType
    Identifiers for particle quantities for a certain type of particles.
  iout : integer
    The output step under consideration.

  Returns
  -------
  masses : np.array(float)
    Physical mass of each numerical particle in [kg].
  """
  return pdata.qs[ptype.class_id][ptype.mass_id][iout] * pdata.qs[ptype.class_id][ptype.weighting_factor_id][iout]



def cross_sections(pdata, ptype, iout):
  """
  Calculates the physical cross section of each numerical particle.

  Parameters
  ----------
  pdata : ParticleData
    Container which stores all particle data read from file system.
  ptype: ParticleType
    Identifiers for particle quantities for a certain type of particles.
  iout : integer
    The output step under consideration.

  Returns
  -------
  cross_sections : np.array(float)
    Physical cross section of each numerical particle in [m^2].
  """
  return 0.25 * np.pi * pdata.qs[ptype.class_id][ptype.diameter_id][iout] ** 2 * 1e-12  # conversion from [µm^2] to [m^2]



def velocities(prm_global, pdata, ptype, iout):
  """
  Calculates the velocity of each particle considering only those velocity components specified.

  Parameters
  ----------
  prm_global : ParameterGlobal
    Container for all global parameters.
  pdata : ParticleData
    Container which stores all particle data read from file system.
  ptype: ParticleType
    Identifiers for particle quantities for a certain type of particles.
  iout : integer
    The output step under consideration.

  Returns
  -------
  velocities : np.array(float)
    Velocity of each particle in [m/s] determined from the specified components.
  """
  velocities = np.zeros(pdata.nparts[ptype.class_id][iout])
  for dim in range(3):
    if prm_global.direction[dim] > 0:
      velocities += (pdata.qs[ptype.class_id][ptype.velocity_ids[dim]][iout]) ** 2

  return np.sqrt(velocities)



def distances_from_reference_point(prm_global, pdata, ptype, iout):
  """
  Calculates distances of all particles from the reference point.

  Parameters
  ----------
  prm_global : ParameterGlobal
    Container for all global parameters.
  pdata : ParticleData
    Container which stores all particle data read from file system.
  ptype: ParticleType
    Identifiers for particle quantities for a certain type of particles.
  iout : integer
    The output step under consideration.

  Returns
  -------
  distances : np.array(float)
    Distance of each particle to the reference point in [m].
  """
  distances = np.zeros(pdata.nparts[ptype.class_id][iout])
  for dim in range(3):
    if prm_global.direction[dim] > 0:
      distances += (pdata.xp[ptype.class_id][iout][dim] - prm_global.reference_position[dim]) ** 2

  return np.sqrt(distances)





if __name__ == "__main__":
  """
  Provide path to 'prt5' stem as command line argument to start a test evaluation.
  """
  import sys
  import read, parameter

  print("Assign parameters.")
  ptype_water = parameter.ParticleType(class_id=0, diameter_id=0, mass_id=1, temperature_id=2, velocity_ids=[3,4,5], weighting_factor_id=6)
  prm_global = parameter.ParameterGlobal(reference_position=[4.0,3.4,1.0], particle_types=[ptype_water], direction=[1,0,1])

  prm_global.print_parameters()

  print("Read particle data.")
  pdata = read.ParticleData(sys.argv[1])

  print("Particle tests.")
  for ptype in prm_global.particle_types:
    for iout in range(pdata.info['n_outputsteps']):
      prt_masses = masses(pdata, ptype, iout)
      prt_crosssections = cross_sections(pdata, ptype, iout)
      prt_velocities = velocities(prm_global, pdata, ptype, iout)
      prt_distances = distances_from_reference_point(prm_global, pdata, ptype, iout)