#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np


def cartesian_grid_properties(prm_cartesian):
  """
  Determine some properties of a Cartesian grid.

  For Cartesian grids, the first index i1 corresponds to the cell index in x-direction, while the second index i2
  corresponds to the cell index in y-direction. The zeroth index for both indices corresponds to the lower left corner
  of the grid.

  Parameters
  ----------
  prm_cartesian : ParameterCartesian
    Container for all parameters for the cartesian grid discretization.

  Returns
  -------
  distances : np.ndarray[i1,i2] -> float
    Distance of the center point of cell [i1,i2] to the center point of the whole grid in [m].
  areas : np.ndarray[i1,i2] -> float
    Area of cell [i1,i2] in [m^2].
  """
  # calculate each cell's distance from the reference point
  cellcenters_xi = np.arange(-.5*prm_cartesian.length_xi, .5*prm_cartesian.length_xi, prm_cartesian.d_xi) + .5*prm_cartesian.d_xi
  cellcenters_xj = np.arange(-.5*prm_cartesian.length_xj, .5*prm_cartesian.length_xj, prm_cartesian.d_xj) + .5*prm_cartesian.d_xj

  cellcenters_sqr_xi = cellcenters_xi ** 2
  cellcenters_sqr_xj = cellcenters_xj ** 2
  distances = np.fromfunction(lambda i, j: np.sqrt(cellcenters_sqr_xi[i] + cellcenters_sqr_xj[j]),
                              (prm_cartesian.n_xi, prm_cartesian.n_xj), dtype=int)

  # calculate each cell's area
  # rectangular cells have equal area on whole domain
  areas = np.full((prm_cartesian.n_xi, prm_cartesian.n_xj), prm_cartesian.d_xi * prm_cartesian.d_xj)
  # check if sum of areas of each cell matches domain
  assert np.round(np.sum(areas), 6) == np.round(prm_cartesian.length_xi * prm_cartesian.length_xj, 6)

  return distances, areas



def particles_on_cartesian_grid(prm_cartesian, prm_global, pdata, ptype, iout, indices=()):
  """
  Segment particles in cells of a cartesian grid.

  For Cartesian grids, the first index i1 corresponds to the cell index in x-direction, while the second index i2
  corresponds to the cell index in y-direction. The zeroth index for both indices corresponds to the lower left corner
  of the grid.

  Parameters
  ----------
  p_polar : ParameterCartesian
    Container for all parameters for the cartesian grid discretization.
  prm_global : ParameterGlobal
    Container for all global parameters, including the coordinates of the reference point.
  pdata : ParticleData
    Container which stores all particle data read from file system.
  ptype : ParticleType
    Identifiers for particle quantities for a certain type of particles.
  iout : int
    The output step under consideration.
  indices : tuple -> np.array(int)
    Specify a selection of particles which will be distributed on the grid. We require a one-component tuple that
    contains an array of particle indices. By default, all particles are considered.

  Returns
  -------
  distributed_particles : np.ndarray[i1,i2] -> np.array(int)
    Indices of particles in grid cell [i1,i2].
  """
  # make sure that we work on a plane spanned by two Cartesian unit vectors
  assert sum(prm_global.direction) == 2
  # extract the corresponding indices of both directions
  dir_i = prm_global.direction.index(1, 0)
  dir_j = prm_global.direction.index(1, dir_i+1)

  # get grid origin
  origin_xi = prm_global.reference_position[dir_i] - .5*prm_cartesian.length_xi
  origin_xj = prm_global.reference_position[dir_j] - .5*prm_cartesian.length_xj
  # get relative coordinates to grid origin
  pos_xi = pdata.xp[ptype.class_id][iout][dir_i][indices] - origin_xi
  pos_xj = pdata.xp[ptype.class_id][iout][dir_j][indices] - origin_xj
  # determine grid indices
  idx_xi = pos_xi // prm_cartesian.d_xi
  idx_xj = pos_xj // prm_cartesian.d_xj
  # mask particles
  mask_xi = [(idx_xi==i) for i in range(prm_cartesian.n_xi)]
  mask_xj = [(idx_xj==j) for j in range(prm_cartesian.n_xj)]

  # distribute particles on grid
  nparts = pdata.nparts[ptype.class_id][iout]
  particles = np.ravel_multi_index(indices, nparts) if indices else np.arange(nparts)
  distributed_particles = np.ndarray((prm_cartesian.n_xi, prm_cartesian.n_xj), dtype=object)
  for i,j in np.ndindex(distributed_particles.shape):
    distributed_particles[i,j] = particles[mask_xi[i] & mask_xj[j]]

  return distributed_particles



def polar_grid_properties(prm_polar):
  """
  Determine some properties of a polar grid.

  For polar grids, the first index i1 corresponds to the cell index in r-direction, starting from the center point of the
  grid. The second index i2 corresponds to the cell index in phi-direction, starting from the specified offset in the
  parameter file.

  Parameters
  ----------
  prm_polar : ParameterPolar
    Container for all parameters for the polar grid discretization.

  Returns
  -------
  distances : np.ndarray[i1,i2] -> float
    Distance of the center point of cell [i1,i2] to the center point of the whole grid in [m].
  areas : np.ndarray[i1,i2] -> float
    Area of cell [i1,i2] in [m^2].
  """
  # calculate each cell's distance from the reference point
  # calculate each annulus' center distance from the reference point
  annulicenters = np.arange(0, prm_polar.radius, prm_polar.d_r) + .5*prm_polar.d_r
  # distribute on all annuli
  distances = np.transpose(np.full((prm_polar.n_sectors, prm_polar.n_shells), annulicenters))

  # calculate each cell's area
  # area of each annulus: A = pi*(r+dr)**2 - pi*r**2 = pi*(2r+dr)*dr = pi*2*(r+0.5dr)*dr
  areas_annuli = np.pi * 2 * annulicenters * prm_polar.d_r
  # area on each annulus segment, valid for all sectors
  areas_segments = areas_annuli / prm_polar.n_sectors
  # distribute on all sectors
  areas = np.transpose(np.full((prm_polar.n_sectors, prm_polar.n_shells), areas_segments))
  # check if sum of areas of all annuli matches circular area
  assert np.round(np.sum(areas), 6) == np.round(np.pi * prm_polar.radius ** 2, 6)

  return distances, areas



def particles_on_polar_grid(prm_polar, prm_global, pdata, ptype, iout, indices=()):
  """
  Segment particles in cells of a polar grid.

  For polar grids, the first index i1 corresponds to the cell index in r-direction, starting from the center point of the
  grid. The second index i2 corresponds to the cell index in phi-direction, starting from the specified offset in the
  parameter file.

  Parameters
  ----------
  prm_polar : ParameterPolar
    Container for all parameters for the polar grid discretization.
  prm_global : ParameterGlobal
    Container for all global parameters, including the coordinates of the reference point.
  pdata : ParticleData
    Container which stores all particle data read from file system.
  ptype : ParticleType
    Identifiers for particle quantities for a certain type of particles.
  iout : int
    The output step under consideration.
  indices : tuple -> np.array(int)
    Specify a selection of particles which will be distributed on the grid. We require a one-component tuple that
    contains an array of particle indices. By default, all particles are considered.

  Returns
  -------
  distributed_particles : np.ndarray[i1,i2] -> np.array(int)
    Indices of particles in grid cell [i1,i2].
  """
  # make sure that we work on a plane spanned by two Cartesian unit vectors
  assert sum(prm_global.direction) == 2
  # extract the corresponding indices of both directions
  dir_i = prm_global.direction.index(1, 0)
  dir_j = prm_global.direction.index(1, dir_i+1)

  # get relative Cartesian coordinates
  pos_xi = pdata.xp[ptype.class_id][iout][dir_i][indices] - prm_global.reference_position[dir_i]
  pos_xj = pdata.xp[ptype.class_id][iout][dir_j][indices] - prm_global.reference_position[dir_j]
  # convert to polar coordinates
  polar_r = np.sqrt((pos_xi ** 2) + (pos_xj ** 2))
  polar_phi = np.arctan2(pos_xj, pos_xi) - prm_polar.phi_offset
  while any(polar_phi < 0):
    polar_phi[polar_phi<0] += 2*np.pi
  # determine grid indices
  idx_r = polar_r // prm_polar.d_r
  idx_phi = polar_phi // prm_polar.d_phi
  assert all(0 <= isector < prm_polar.n_sectors for isector in idx_phi)
  # mask particles
  mask_r = [(idx_r==ishell) for ishell in range(prm_polar.n_shells)]
  mask_phi = [(idx_phi==isector) for isector in range(prm_polar.n_sectors)]

  # distribute particles on grid
  nparts = pdata.nparts[ptype.class_id][iout]
  particles = np.ravel_multi_index(indices, nparts) if indices else np.arange(nparts)
  distributed_particles = np.ndarray((prm_polar.n_shells, prm_polar.n_sectors), dtype=object)
  for ishell,isector in np.ndindex(distributed_particles.shape):
    distributed_particles[ishell,isector] = particles[mask_r[ishell] & mask_phi[isector]]

  return distributed_particles



def polar_grid_indices_of_sector(prm_polar, desired_angle):
  """
  Find all cells on a polar grid that correspond to a given circular sector.

  The distribution of circular sectors is determined by the polar grid discretization. We pick the circular sector in
  which the specified angle is located, and return the corresponding grid indices. Here, complete circular sectors will
  be considered, which involves all indices in r-direction. If the specified angle is adjacent to two sectors, the
  indices of both adjoining sectors will be returned.

  Parameters
  ----------
  prm_polar : ParameterPolar
    Container for all parameters for the polar grid discretization.
  desired_angle : int
    The angle in [rad] that determines the circular sector(s).

  Returns
  -------
  indices : tuple -> np.array(int)
    Two-component tuple of grid indices. The first array corresponds to indices in r-direction, and the other one to
    indices in phi-direction.
  """
  # check user input
  assert 0 <= desired_angle < 2*np.pi

  relative_angle = desired_angle - prm_polar.phi_offset
  index_phi = relative_angle // prm_polar.d_phi

  indices_r = np.arange(prm_polar.n_shells)
  indices_phi = np.full_like(indices_r, index_phi)
  indices = (indices_r, indices_phi)

  # due to possible conversion errors to radians, we check with an uncertainty if we are close to neighboring segments
  if math.isclose(relative_angle / prm_polar.d_phi, index_phi):
    # we are close to the preceding segment
    indices = (np.append(indices_r, indices_r),
               np.append(np.full_like(indices_phi, index_phi-1), indices_phi))
  elif math.isclose(relative_angle / prm_polar.d_phi, index_phi+1):
    # we are close to the succeeding segment
    indices = (np.append(indices_r, indices_r),
               np.append(indices_phi, np.full_like(indices_phi, index_phi+1)))

  return indices



def number_of_particles(particle_indices_grid, pdata, ptype, iout):
  """
  Count the number of physical particles on each cell of any grid.

  Parameters
  ----------
  particle_indices_grid : np.ndarry[i1,i2] -> np.array(int)
    Indices of particles in grid cell [i1,i2].
  pdata : ParticleData
    Container which stores all particle data read from file system.
  ptype : ParticleType
    Identifiers for particle quantities for a certain type of particles.
  iout : int
    The output step under consideration.

  Returns
  -------
  numbers : np.ndarray[i1,i2] -> float
    The number of physical particles in grid cell [i1,i2].
  """
  # consider particle weighting factor to evaluate density on physical particles
  numbers = np.empty_like(particle_indices_grid, dtype=float)
  for i,j in np.ndindex(numbers.shape):
    numbers[i,j] = np.sum(
      pdata.qs[ptype.class_id][ptype.weighting_factor_id][iout][particle_indices_grid[i,j]]) \
      if len(particle_indices_grid[i,j]) > 0 else 0

  return numbers



def area_of_particles(particle_cross_sections, particle_indices_grid, pdata, ptype, iout):
  """
  Calculate the area that all physical particles occupy on each cell of any grid. No overlap will be considered, thus
  the individual cross sections will be cumulated.

  Parameters
  ----------
  particle_cross_sections : np.array(float)
    Physical cross section of each numerical particle in [m^2].
  particle_indices_grid : np.ndarray[i1,i2] -> np.array(int)
    Indices of particles in grid cell [i1,i2].
  pdata : ParticleData
    Container which stores all particle data read from file system.
  ptype : ParticleType
    Identifiers for particle quantities for a certain type of particles.
  iout : int
    The output step under consideration.

  Returns
  -------
  areas : np.ndarray[i1,i2] -> float
    The cumulated cross section in [m^2] of all physical particles in grid cell [i1,i2].
  """
  # consider particle weighting factor to evaluate area on physical particles
  areas = np.empty_like(particle_indices_grid, dtype=float)
  for i,j in np.ndindex(areas.shape):
    areas[i,j] = np.sum(
      particle_cross_sections[particle_indices_grid[i,j]] *
      pdata.qs[ptype.class_id][ptype.weighting_factor_id][iout][particle_indices_grid[i,j]]) \
      if len(particle_indices_grid[i,j]) > 0 else 0

  return areas



def particles_from_grid(particle_indices_grid, grid_indices=()):
  """
  Collect particles from a selection of cells on any grid.

  Note: There may be less particles on the full grid than there are in total. This may be the case whenever the grid is
        smaller than the simulation domain.

  Parameters
  ----------
  particle_indices_grid : np.ndarray[i1,i2] -> np.array(int)
    Indices of particles in grid cell [i1,i2].
  grid_indices : tuple -> np.array(int)
    Specify a selection of cells from which particles will be collected. We require a two-component tuple of grid
    indices, one array for each dimension. By default, all particles on the grid are considered.

  Returns
  -------
  particle_indices : tuple -> np.array(int)
    Particle indices that are associated with the selected grid cells. We return a one-component tuple that contains
    an array of particle indices.
  """
  return (np.concatenate(particle_indices_grid[grid_indices]),) \
         if particle_indices_grid[grid_indices].size else (np.array([], dtype=int),)





if __name__ == "__main__":
  """
  Provide path to 'prt5' stem as command line argument to start a test evaluation.
  """
  import sys
  import read, parameter, particle

  print("Assign parameters.")
  ptype_water = parameter.ParticleType(class_id=0, diameter_id=0, mass_id=1, temperature_id=2, velocity_ids=[3,4,5], weighting_factor_id=6)
  prm_global = parameter.ParameterGlobal(reference_position=[4.0,3.4,1.0], particle_types=[ptype_water], direction=[1,0,1])
  prm_cartesian = parameter.ParameterCartesian(length_xi=6.0, length_xj=2.0, n_xi=90, n_xj=30)
  prm_polar = parameter.ParameterPolar(radius=2.0, n_shells=75, n_sectors=36, phi_offset=np.radians(-5))

  prm_global.print_parameters()
  prm_cartesian.print_parameters()
  prm_polar.print_parameters()

  print("Get grid properties.")
  crt_distances, crt_areas = cartesian_grid_properties(prm_cartesian)
  plr_distances, plr_areas = polar_grid_properties(prm_polar)
  plr_cells5_plus = polar_grid_indices_of_sector(prm_polar, np.radians(5+1e-12))
  plr_cells5_minus = polar_grid_indices_of_sector(prm_polar, np.radians(5-1e-12))
  assert np.array_equal(plr_cells5_plus, plr_cells5_minus)

  print("Read particle data.")
  pdata = read.ParticleData(sys.argv[1])

  print("Grid tests.")
  for ptype in prm_global.particle_types:
    for iout in range(pdata.info['n_outputsteps']):
      # --- Particle ---
      prt_masses = particle.masses(pdata, ptype, iout)
      prt_crosssections = particle.cross_sections(pdata, ptype, iout)
      prt_velocities = particle.velocities(prm_global, pdata, ptype, iout)
      prt_distances = particle.distances_from_reference_point(prm_global, pdata, ptype, iout)

      # --- Grid ---
      # Cartesian grid
      crt_prtindices = particles_on_cartesian_grid(prm_cartesian, prm_global, pdata, ptype, iout)
      crt_prtnumbers = number_of_particles(crt_prtindices, pdata, ptype, iout)
      crt_prtareas = area_of_particles(prt_crosssections, crt_prtindices, pdata, ptype, iout)
      prt_indices_from_crt = particles_from_grid(crt_prtindices)

      # Polar grid
      plr_prtindices = particles_on_polar_grid(prm_polar, prm_global, pdata, ptype, iout)
      plr_prtnumbers = number_of_particles(plr_prtindices, pdata, ptype, iout)
      plr_prtareas = area_of_particles(prt_crosssections, plr_prtindices, pdata, ptype, iout)
      prt_indices_from_plr = particles_from_grid(plr_prtindices)
      prt_indices_from_plr5 = particles_from_grid(plr_prtindices, plr_cells5_plus)