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
PARAMETER module: 

For convenience, everything in ParticleData class.

Lots of container formats for Parameters.
"""


from sys import stdout

import numpy as np


class ParticleType:
  """
  Identifiers for certain particle quantities from the simulation output
  file.

  class_id : int >= 0
    Identifier of the particle class from the simulation.
  diameter_id : int >= 0
    Quantity identifier for the particle diameter.
  mass_id : int >= 0
    Quantity identifier for the particle mass.
  temperature_id : int >= 0
    Quantity identifier for the particle temperature.
  velocity_ids: list[i1] -> int >= 0
    Quantity identifier for the velocity component i1.
  weighting_factor_id: int >= 0
    Quantity identifier for the particle weighting factor.
  """

  def __init__(self, class_id, diameter_id, mass_id, temperature_id, velocity_ids, weighting_factor_id):
    # Check validity of user input
    assert class_id >= 0
    assert diameter_id >= 0
    assert mass_id >= 0
    assert temperature_id >= 0
    assert len(velocity_ids) == 3 and all(v >= 0 for v in velocity_ids)
    assert weighting_factor_id >= 0

    # Assign class members from parameters
    self.class_id = class_id
    self.diameter_id = diameter_id
    self.mass_id = mass_id
    self.temperature_id = temperature_id
    self.velocity_ids = velocity_ids
    self.weighting_factor_id = weighting_factor_id


class ParameterGlobal:
  """
  Parameter container storing all necessary information about the
  evaluation setup.

  reference_position : list of length 3 -> float
    Position in [m] to which all distance calculations refer to.
  particle_types : list[i1] -> ParticleType
    Identifiers for certain quantities of particle type i1 from the
    simulation output file.
  direction : list of length 3 -> bool
    Describes which components of position and velocity are considered
    for evaluation. A value 'True' signals that this particular
    component will be considered, and a value 'False' ignores it.
    NOTE: Some evaluation methods require exactly TWO components that
          span the slicing plane.
  """

  def __init__(self, reference_position, particle_types, direction=None):
    # Avoid mutable default arguments warning
    if direction is None:
      direction = [True, True, True]

    # Check validity of user input
    assert len(reference_position) == 3
    assert all(isinstance(pt, ParticleType) for pt in particle_types)
    assert len(direction) == 3 and all(type(flag) == bool
                                       for flag in direction)

    # Assign class members from parameters
    self.reference_position = reference_position
    self.particle_types = particle_types
    self.direction = direction

  def print_parameters(self, file=stdout):
    print(" * ParameterGlobal:\n",
          "    - Direction: ", self.direction, "\n",
          "    - Reference position [m]: ", self.reference_position,
          sep="", file=file)
    for ilc, ptype in enumerate(self.particle_types):
      print("    - Particle type ", ilc, ":\n",
            "       - Class ID: ", ptype.class_id, "\n",
            "       - Diameter ID: ", ptype.diameter_id, "\n",
            "       - Mass ID: ", ptype.mass_id, "\n",
            "       - Temperature ID: ", ptype.temperature_id, "\n",
            "       - Velocity IDs: ", ptype.velocity_ids, "\n",
            "       - Weighting Factor ID: ", ptype.weighting_factor_id,
            sep="", file=file)


class ParameterCartesian:
  """
  Parameter container storing all information about the cartesian grid
  discretization.

  length_xi : float > 0
    Considered radius [m] around reference point.
  length_xj : float > 0
    Considered radius [m] around reference point.
  n_xi : int > 0
    Number of cells in direction of first component.
  n_xj : int > 0
    Number of cells in direction of second component.
  """

  def __init__(self, length_xi, length_xj, n_xi=100, n_xj=100):
    # Check validity of user input
    assert length_xi > 0
    assert length_xj > 0
    assert n_xi > 0
    assert n_xj > 0

    # Assign class members from parameters
    self.length_xi = length_xi
    self.length_xj = length_xj
    self.n_xi = int(n_xi)
    self.n_xj = int(n_xj)

    # Assign auxiliary variables
    self.d_xi = self.length_xi / self.n_xi
    self.d_xj = self.length_xj / self.n_xj
    # For a shared interface with the polar discretization,
    # we duplicate some variables for later
    self.n_i = self.n_xi
    self.n_j = self.n_xj

  def print_parameters(self, file=stdout):
    print(" * ParameterCartesian:\n",
          "    - Origin: Reference position\n",
          "    - Discretization coordinate 1 [m]:\n",
          "       - range: [", -0.5 *
          self.length_xi, ", ", 0.5*self.length_xi, "]\n",
          "       - stepsize: ", self.d_xi, "\n",
          "       - steps: ", self.n_xi, "\n",
          "    - Discretization coordinate 2 [m]:\n",
          "       - range: [", -0.5 *
          self.length_xj, ", ", 0.5*self.length_xj, "]\n",
          "       - stepsize: ", self.d_xj, "\n",
          "       - steps: ", self.n_xj,
          sep="", file=file)


class ParameterPolar:
  """
  Parameter container storing all information about the polar grid
  discretization.

  radius : float > 0
    Considered radius [m] around reference point.
  n_segments : int > 0
    Number of angle segments the grid will be divided into.
  n_shells : int > 0
    Number of shells the grid will be divided into.
  phi_offset : float
    Offset in RAD for the azimuthal discretization.
  """

  def __init__(self, radius, n_shells=100, n_sectors=72, phi_offset=0):
    # Check validity of user input
    assert radius > 0
    assert n_sectors > 0
    assert n_shells > 0

    # Assign class members from parameters
    self.radius = radius
    self.n_shells = int(n_shells)
    self.n_sectors = int(n_sectors)
    self.phi_offset = phi_offset

    # Assign auxiliary variables
    self.d_r = self.radius / self.n_shells
    self.d_phi = 2*np.pi / self.n_sectors
    # For a shared interface with the cartesian discretization,
    # we duplicate some variables for later
    self.n_i = self.n_shells
    self.n_j = self.n_sectors

  def print_parameters(self, file=stdout):
    print(" * ParameterPolar:\n",
          "    - Origin: Reference position\n",
          "    - Radial discretization [m]:\n",
          "       - range: [", 0., ", ", self.radius, "]\n",
          "       - stepsize: ", self.d_r, "\n",
          "       - steps: ", self.n_shells, "\n",
          "    - Azimuthal discretization [°]:\n",
          "       - range: [", np.degrees(self.phi_offset), ", ",
          np.degrees(2*np.pi + self.phi_offset), "]\n",
          "       - stepsize: ", np.degrees(self.d_phi), "\n",
          "       - steps: ", self.n_sectors,
          sep="", file=file)
