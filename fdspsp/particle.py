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
PARTICLE module:

Calculate additional properties of each particle.
"""


import numpy as np


def masses(particle_data, class_label, output_step):
  """
  Calculates the physical mass of each numerical particle.

  Parameters
  ----------
  particle_data : fdspsp.read.ParticleData
    Container which stores all particle data read from file system.
  class_label: str
    Label of the particle class under consideration.
  output_step : int
    The output step under consideration.

  Returns
  -------
  masses : np.array(float)
    Physical mass of each numerical particle in [kg].
  """
  assert type(particle_data) == read.ParticleData
  assert class_label in particle_data.classes
  assert 0 <= output_step < particle_data.n_outputsteps

  return (particle_data.quantities[class_label]
          ["PARTICLE MASS"][output_step]
          *
          particle_data.quantities[class_label]
          ["PARTICLE WEIGHTING FACTOR"][output_step])


def cross_sections(particle_data, class_label, output_step):
  """
  Calculates the physical cross section of each numerical particle.

  Parameters
  ----------
  particle_data : fdspsp.read.ParticleData
    Container which stores all particle data read from file system.
  class_label: str
    Label of the particle class under consideration.
  output_step : int
    The output step under consideration.

  Returns
  -------
  cross_sections : np.array(float)
    Physical cross section of each numerical particle in [m^2].
  """
  assert type(particle_data) == read.ParticleData
  assert class_label in particle_data.classes
  assert 0 <= output_step < particle_data.n_outputsteps

  return (0.25 * np.pi
          *
          particle_data.quantities[class_label]
          ["PARTICLE DIAMETER"][output_step] ** 2
          *
          # conversion from [µm^2] to [m^2]
          1e-12)


velocity_labels = ["PARTICLE U", "PARTICLE V", "PARTICLE W"]


def velocities(particle_data, class_label, output_step,
               components=[True, True, True]):
  """
  Calculates the velocity of each numerical particle considering only
  those velocity components specified.

  Parameters
  ----------
  particle_data : fdspsp.read.ParticleData
    Container which stores all particle data read from file system.
  class_label: str
    Label of the particle class under consideration.
  output_step : int
    The output step under consideration.
  components : list -> bool
    Choose which Cartesian coordinate components are considered. For
    example, a value [True, True, False] denotes the x-y-plane. List has
    to be of length 3. By default, all components are taken into account.

  Returns
  -------
  velocities : np.array(float)
    Velocity of each particle in [m/s] determined from the specified
    components.
  """
  assert type(particle_data) == read.ParticleData
  assert class_label in particle_data.classes
  assert 0 <= output_step < particle_data.n_outputsteps
  assert len(components) == 3 and all(type(c) == bool for c in components)

  velocities = np.zeros(particle_data.n_particles[class_label][output_step])
  for dim in range(3):
    if components[dim] is True:
      velocities += (particle_data.quantities[class_label]
                     [velocity_labels[dim]][output_step]) ** 2

  return np.sqrt(velocities)


def distances_from_reference_point(particle_data, class_label, output_step,
                                   reference_point, components=[True, True, True]):
  """
  Calculates the distance of each numerical particle from a reference
  point considering only specified coordinate components.

  Parameters
  ----------
  particle_data : fdspsp.read.ParticleData
    Container which stores all particle data read from file system.
  class_label: str
    Label of the particle class under consideration.
  output_step : int
    The output step under consideration.
  reference_point : list -> float
    Coordinates of the reference point. List has to be of length 3.
  components : list -> bool
    Choose which Cartesian coordinate components are considered. For
    example, a value [True, True, False] denotes the x-y-plane. List has
    to be of length 3. By default, all components are taken into account.

  Returns
  -------
  distances : np.array(float)
    Distance of each particle to the reference point in [m].
  """
  assert type(particle_data) == read.ParticleData
  assert class_label in particle_data.classes
  assert 0 <= output_step < particle_data.n_outputsteps
  assert len(reference_point) == 3
  assert len(components) == 3 and all(type(c) == bool for c in components)

  distances = np.zeros(particle_data.n_particles[class_label][output_step])
  for dim in range(3):
    if components[dim] is True:
      distances += (particle_data.positions[class_label][output_step][dim] -
                    reference_point[dim]) ** 2

  return np.sqrt(distances)
