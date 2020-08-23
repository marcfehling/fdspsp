#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def mean(data, indices):
  """
  Average a certain selection of data from a data set.

  Parameters
  ----------
  data : array_like
    Any type of data, whether associated with particles or cells.
  indices : tuple -> np.array(int)
    Specify a selection of indices whose associated data this function will process only. The number of tuple components
    needs to be in accordance with the number of array dimensions.

  Returns
  -------
  mean : float
    Arithmetic mean of the provided data. Zero if there is no data to average.
  """
  # flatten array for convenience
  data_view = data.flatten()
  indices_view = np.ravel_multi_index(indices, data.shape)

  return np.mean(data_view[indices_view]) if indices_view.size > 0 else 0



def sum(data, indices):
  """
  Sum a certain selection of data from a data set.

  Parameters
  ----------
  data : array_like
    Any type of data, whether associated with particles or cells.
  indices : tuple -> np.array(int)
    Specify a selection of indices whose associated data this function will process only. The number of tuple components
    needs to be in accordance with the number of array dimensions.

  Returns
  -------
  sum : float
    Sum of the provided data. Zero if there is no data to sum.
  """
  # flatten array for convenience
  data_view = data.flatten()
  indices_view = np.ravel_multi_index(indices, data.shape)

  return np.sum(data_view[indices_view]) if indices_view.size > 0 else 0





if __name__ == "__main__":
  """
  Provide path to 'prt5' stem as command line argument to start a test evaluation.
  """
  import sys
  import read, parameter, particle, grid, selection

  print("Assign parameters.")
  ptype_water = parameter.ParticleType(class_id=0, diameter_id=0, mass_id=1, temperature_id=2, velocity_ids=[3,4,5], weighting_factor_id=6)
  prm_global = parameter.ParameterGlobal(reference_position=[4.0,3.4,1.0], particle_types=[ptype_water], direction=[1,0,1])
  prm_cartesian = parameter.ParameterCartesian(length_xi=6.0, length_xj=2.0, n_xi=90, n_xj=30)
  prm_polar = parameter.ParameterPolar(radius=2.0, n_shells=75, n_sectors=36, phi_offset=np.radians(-5))
  rel_threshold = 0.1
  top_share = 0.1

  prm_global.print_parameters()
  prm_cartesian.print_parameters()
  prm_polar.print_parameters()
  print(" * Relative threshold: ", rel_threshold)
  print(" * Top percentile: ", top_share)

  print("Get grid properties.")
  crt_distances, _ = grid.cartesian_grid_properties(prm_cartesian)
  plr_distances, _ = grid.polar_grid_properties(prm_polar)
  plr_cells45 = grid.polar_grid_indices_of_sector(prm_polar, np.radians(45))

  print("Read particle data.")
  pdata = read.ParticleData(sys.argv[1])

  print("Reduction tests.")
  for ptype in prm_global.particle_types:
    for iout in range(pdata.info['n_outputsteps']):
      # --- Particle ---
      # Particle
      prt_masses = particle.masses(pdata, ptype, iout)
      prt_distances = particle.distances_from_reference_point(prm_global, pdata, ptype, iout)

      # --- Grid ---
      # Cartesian grid
      crt_prtindices = grid.particles_on_cartesian_grid(prm_cartesian, prm_global, pdata, ptype, iout)
      crt_prtnumbers = grid.number_of_particles(crt_prtindices, pdata, ptype, iout)

      # Polar grid
      plr_prtindices = grid.particles_on_polar_grid(prm_polar, prm_global, pdata, ptype, iout)
      plr_prtnumbers = grid.number_of_particles(plr_prtindices, pdata, ptype, iout)

      # --- Selection ---
      # Particle
      prt_indices_reldistance = selection.relative_threshold(prt_distances, rel_threshold)
      prt_indices45 = grid.particles_from_grid(plr_prtindices, plr_cells45)
      prt_indices45_topdistance = selection.top_percentile(prt_distances, top_share, prt_indices45)

      # Cartesian grid
      crt_cells_relprtnumber = selection.relative_threshold(crt_prtnumbers, rel_threshold)

      # Polar grid
      plr_cells45_relprtnumber = selection.relative_threshold(plr_prtnumbers, rel_threshold, plr_cells45)

      # --- Reduction ---
      # Particle
      mean_prtdistances_reldistance = mean(prt_distances, prt_indices_reldistance)
      mean_prtdistances45_topdistance = mean(prt_distances, prt_indices45_topdistance)
      sum_prtmasses45 = sum(prt_masses, prt_indices45)
      # Cartesian grid
      mean_crtdistances_relprtnumber = mean(crt_distances, crt_cells_relprtnumber)
      # Polar grid
      mean_plrdistances45_relprtnumber = mean(plr_distances, plr_cells45_relprtnumber)