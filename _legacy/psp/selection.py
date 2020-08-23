#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import heapq


def absolute_threshold(data, threshold, indices=()):
  """
  Find indices in a data set whose associated values are larger than a given absolute threshold.

  Parameters
  ----------
  data : array_like
    Any type of data, whether associated with particles or cells.
  threshold : float
    Absolute threshold with which the data set will be compared.
  indices : tuple -> np.array(int)
    Specify a selection of indices whose associated data this function will process only. The number of tuple components
    needs to be in accordance with the number of array dimensions. By default, the full data set is considered.

  Returns
  -------
  indices_reduced : tuple -> np.array(int)
    Selection of indices whose associated data fulfills the criterion. The number of tuple components corresponds to the
     number of array dimensions.
  """
  indices_reduced = tuple(np.array([], dtype=int) for _ in range(data.ndim))
  if data.size:
    # flatten array for convenience
    data_view = data.flatten()
    # only operate on predefined section of data (if specified)
    indices_view = np.ravel_multi_index(indices, data.shape) \
                   if indices else ()
    data_enumerate = enumerate(data_view[indices_view])
    # identify indices belonging to data fulfilling threshold criterion
    indices_threshold = np.array([i for i,e in data_enumerate if e >= threshold])
    # indexing multi-dimensional arrays in numpy
    if len(indices_threshold):
      if len(indices_view):
        indices_threshold = indices_view[indices_threshold]
      indices_reduced = np.unravel_index(indices_threshold, data.shape)

  return indices_reduced



def relative_threshold(data, threshold, indices=()):
  """
  Find indices in a data set whose associated values are larger than a given threshold relative to the maximum value.

  Parameters
  ----------
  data : array_like
    Any type of data, whether associated with particles or cells.
  threshold : float
    Fraction specifying the threshold relative to the maximum value of the data set.
  indices : tuple -> np.array(int)
    Specify a selection of indices whose associated data this function will process only. The number of tuple components
    needs to be in accordance with the number of array dimensions. By default, the full data set is considered.

  Returns
  -------
  indices_reduced : tuple -> np.array(int)
    Selection of indices whose associated data fulfills the criterion. The number of tuple components corresponds to the
    number of array dimensions.
  """
  # check user input
  assert 0 <= threshold <= 1

  indices_reduced = tuple(np.array([], dtype=int) for _ in range(data.ndim))
  if data.size:
    abs_threshold = threshold * np.max(data[indices])
    indices_reduced = absolute_threshold(data, abs_threshold, indices)

  return indices_reduced



def top_percentile(data, top_share, indices=()):
  """
  Find indices in a data set whose associated values correspond to a given top percentile.

  Parameters
  ----------
  data : array_like
    Any type of data, whether associated with particles or cells.
  top_share : float
    Fraction specifying the top percentile of the data set.
  indices : tuple -> np.array(int)
    Specify a selection of indices whose associated data this function will process only. The number of tuple components
    needs to be in accordance with the number of array dimensions. By default, the full data set is considered.

  Returns
  -------
  indices_reduced : tuple -> np.array(int)
    Selection of indices whose associated data fulfills the criterion. The number of tuple components corresponds to the
    number of array dimensions.
  """
  # check user input
  assert 0 <= top_share <= 1

  indices_reduced = tuple(np.array([], dtype=int) for _ in range(data.ndim))
  ntop = int(top_share * data[indices].size)
  if ntop:
    # flatten array for convenience
    data_view = data.flatten()
    # only operate on predefined section of data (if specified)
    indices_view = np.ravel_multi_index(indices, data.shape) \
                   if indices else ()
    data_enumerate = enumerate(data_view[indices_view])
    # sort top fraction of all data by value in descending order
    data_top = heapq.nlargest(ntop, data_enumerate, key=lambda x: x[1])
    # further add entries that have the same data
    data_top_min = data_top[-1][1]
    data_top += [e for e in data_enumerate if e[1] == data_top_min]
    # extract corresponding indices
    indices_top, _ = zip(*data_top)
    indices_top = np.array(indices_top)
    # indexing multi-dimensional arrays in numpy
    if len(indices_top):
      if len(indices_view):
        indices_top = indices_view[indices_top]
      indices_reduced = np.unravel_index(indices_top, data.shape)

  return indices_reduced





if __name__ == "__main__":
  """
  Provide path to 'prt5' stem as command line argument to start a test evaluation.
  """
  import sys
  import read, parameter, particle, grid

  print("Assign parameters.")
  ptype_water = parameter.ParticleType(class_id=0, diameter_id=0, mass_id=1, temperature_id=2, velocity_ids=[3,4,5], weighting_factor_id=6)
  prm_global = parameter.ParameterGlobal(reference_position=[4.0,3.4,1.0], particle_types=[ptype_water], direction=[1,0,1])
  prm_cartesian = parameter.ParameterCartesian(length_xi=6.0, length_xj=2.0, n_xi=90, n_xj=30)
  prm_polar = parameter.ParameterPolar(radius=2.0, n_shells=75, n_sectors=72, phi_offset=np.radians(-5))
  abs_threshold = 0.1
  rel_threshold = 0.1
  top_share = 0.1

  prm_global.print_parameters()
  prm_cartesian.print_parameters()
  prm_polar.print_parameters()
  print(" * Absolute threshold: ", abs_threshold)
  print(" * Relative threshold: ", rel_threshold)
  print(" * Top percentile: ", top_share)

  print("Get grid properties.")
  crt_distances, crt_areas = grid.cartesian_grid_properties(prm_cartesian)
  plr_distances, plr_areas = grid.polar_grid_properties(prm_polar)
  plr_cells45 = grid.polar_grid_indices_of_sector(prm_polar, np.radians(45))

  print("Read particle data.")
  pdata = read.ParticleData(sys.argv[1])

  print("Selection tests.")
  for ptype in prm_global.particle_types:
    for iout in range(pdata.info['n_outputsteps']):
      # --- Particle ---
      prt_distances = particle.distances_from_reference_point(prm_global, pdata, ptype, iout)
      prt_velocities = particle.velocities(prm_global, pdata, ptype, iout)

      # Select moving particles
      prt_indices_absvelocities = absolute_threshold(prt_velocities, abs_threshold)

      # --- Grid ---
      # Cartesian grid
      crt_prtindices = grid.particles_on_cartesian_grid(prm_cartesian, prm_global, pdata, ptype, iout, prt_indices_absvelocities)
      crt_prtnumbers = grid.number_of_particles(crt_prtindices, pdata, ptype, iout)

      # Polar grid
      plr_prtindices = grid.particles_on_polar_grid(prm_polar, prm_global, pdata, ptype, iout, prt_indices_absvelocities)
      plr_prtnumbers = grid.number_of_particles(plr_prtindices, pdata, ptype, iout)

      # --- Selection ---
      # Particle
      prt_indices_reldistance = relative_threshold(prt_distances, rel_threshold, prt_indices_absvelocities)
      prt_indices_topdistance = top_percentile(prt_distances, top_share, prt_indices_absvelocities)

      prt_indices45 = grid.particles_from_grid(plr_prtindices, plr_cells45)
      prt_indices45_reldistance = relative_threshold(prt_distances, rel_threshold, prt_indices45)
      prt_indices45_topdistance = top_percentile(prt_distances, top_share, prt_indices45)

      # Cartesian grid
      crt_cells_absprtnumber = absolute_threshold(crt_prtnumbers, abs_threshold)
      crt_cells_relprtnumber = relative_threshold(crt_prtnumbers, rel_threshold)
      crt_cells_topprtnumber = top_percentile(crt_prtnumbers, top_share)
      crt_cells_relprtnumber_topprtnumber = top_percentile(crt_prtnumbers, top_share, crt_cells_relprtnumber)

      # Polar grid
      plr_cells_absprtnumber = absolute_threshold(plr_prtnumbers, abs_threshold)
      plr_cells_relprtnumber = relative_threshold(plr_prtnumbers, rel_threshold)
      plr_cells_topprtnumber = top_percentile(plr_prtnumbers, top_share)
      plr_cells_relprtnumber_topprtnumber = top_percentile(plr_prtnumbers, top_share, plr_cells_relprtnumber)

      plr_cells45_relprtnumber = relative_threshold(plr_prtnumbers, rel_threshold, plr_cells45)