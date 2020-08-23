#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
from functools import partial
import numpy as np

import psp.read, psp.parameter, psp.particle, psp.grid, psp.selection, psp.reduction, psp.multiple

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


def parse_water(infilestem):
  """
  TODO: Doc
  """
  # ----- Parameters -----
  # Parameters from FDS file
  ptype_water = psp.parameter.ParticleType(class_id=0, diameter_id=0, mass_id=1, temperature_id=2, velocity_ids=[3,4,5], weighting_factor_id=6)
  prm_global = psp.parameter.ParameterGlobal(reference_position=[4.0,3.4,1.0], particle_types=[ptype_water], direction=[1,0,1])
  # Parameters for evaluation
  prm_polar = psp.parameter.ParameterPolar(radius=5, n_shells=100, n_sectors=72, phi_offset=np.radians(-5))
  absolute_threshold = 1
  relative_threshold_factor = 0.01
  top_share = 0.1
  # Supply additional analysis on different angular sectors
  assert prm_polar.phi_offset == np.radians(-5)
  assert prm_polar.d_phi == np.radians(5)
  angles = [90, 45, 0, 315]

  # ----- Read -----
  pdata = psp.read.ParticleData(infilestem)

  # ----- Prepare -----
  plr_distances, plr_areas = psp.grid.polar_grid_properties(prm_polar)
  # Provide indices corresponding to angles
  plr_cellsangle = [psp.grid.polar_grid_indices_of_sector(prm_polar, np.radians(angle)) for angle in angles]
  for cells in plr_cellsangle:
    assert (cells[0].size, cells[1].size) == (2*prm_polar.n_shells, 2*prm_polar.n_shells)

  # ----- Analyse -----
  dummy_container = np.empty(shape=(len(prm_global.particle_types), pdata.info['n_outputsteps']), dtype=object)
  dummy_container_angle = np.empty(shape=(len(angles), len(prm_global.particle_types), pdata.info['n_outputsteps']), dtype=object)

  # Farthest Fraction Method (FF)
  total_mass = np.empty_like(dummy_container)
  ff_distance = np.empty_like(dummy_container)
  ff_velocity = np.empty_like(dummy_container)
  ff_distance_angle = np.empty_like(dummy_container_angle)
  ff_velocity_angle = np.empty_like(dummy_container_angle)

  # Grid Density Method (GD)
  gd_distance = np.empty_like(dummy_container)
  gd_velocity = np.empty_like(dummy_container)
  gd_distance_angle = np.empty_like(dummy_container_angle)
  gd_velocity_angle = np.empty_like(dummy_container_angle)

  # Grid Coverage Method (GC)
  gc_distance = np.empty_like(dummy_container)
  gc_velocity = np.empty_like(dummy_container)
  gc_distance_angle = np.empty_like(dummy_container_angle)
  gc_velocity_angle = np.empty_like(dummy_container_angle)

  for ilc,ptype in enumerate(prm_global.particle_types):
    for iout in range(pdata.info['n_outputsteps']):
      # Particle
      prt_masses = psp.particle.masses(pdata, ptype, iout)
      prt_crosssections = psp.particle.cross_sections(pdata, ptype, iout)
      prt_velocities = psp.particle.velocities(prm_global, pdata, ptype, iout)
      prt_distances = psp.particle.distances_from_reference_point(prm_global, pdata, ptype, iout)

      # Select moving particles
      prt_indices_absvelocities = psp.selection.absolute_threshold(prt_velocities, absolute_threshold)

      # Grid
      plr_prtindices = psp.grid.particles_on_polar_grid(prm_polar, prm_global, pdata, ptype, iout, prt_indices_absvelocities)
      plr_prtnumbers = psp.grid.number_of_particles(plr_prtindices, pdata, ptype, iout)
      plr_density = plr_prtnumbers / plr_areas
      plr_prtareas = psp.grid.area_of_particles(prt_crosssections, plr_prtindices, pdata, ptype, iout)
      plr_coverage = plr_prtareas / plr_areas

      # Selection
      prt_indices_ff = psp.selection.top_percentile(prt_distances, top_share, prt_indices_absvelocities)

      plr_cells_reldensity = psp.selection.relative_threshold(plr_density, relative_threshold_factor)
      plr_cells_reldensity_topdistance = psp.selection.top_percentile(plr_distances, top_share, plr_cells_reldensity)
      plr_cells_relcoverage = psp.selection.relative_threshold(plr_coverage, relative_threshold_factor)
      plr_cells_relcoverage_topdistance = psp.selection.top_percentile(plr_distances, top_share, plr_cells_relcoverage)

      prt_indices_gd = psp.grid.particles_from_grid(plr_prtindices, plr_cells_reldensity_topdistance)
      prt_indices_gc = psp.grid.particles_from_grid(plr_prtindices, plr_cells_relcoverage_topdistance)

      # Reduction
      total_mass[ilc][iout] = np.sum(prt_masses)
      ff_distance[ilc][iout] = psp.reduction.mean(prt_distances, prt_indices_ff)
      ff_velocity[ilc][iout] = psp.reduction.mean(prt_velocities, prt_indices_ff)

      gd_distance[ilc][iout] = psp.reduction.mean(prt_distances, prt_indices_gd)
      gd_velocity[ilc][iout] = psp.reduction.mean(prt_velocities, prt_indices_gd)
      gc_distance[ilc][iout] = psp.reduction.mean(prt_distances, prt_indices_gc)
      gc_velocity[ilc][iout] = psp.reduction.mean(prt_velocities, prt_indices_gc)

      # For multiple sectors
      for iangle,_ in enumerate(angles):
        # Select
        # get all particle indices corresponding to this sector
        prt_indicesangle = psp.grid.particles_from_grid(plr_prtindices, plr_cellsangle[iangle])
        prt_indicesangle_ff = psp.selection.top_percentile(prt_distances, top_share, prt_indicesangle)

        plr_cellsangle_reldensity = psp.selection.relative_threshold(plr_density, relative_threshold_factor, plr_cellsangle[iangle])
        plr_cellsangle_gd = psp.selection.top_percentile(plr_distances, top_share, plr_cellsangle_reldensity)
        plr_cellsangle_relcoverage = psp.selection.relative_threshold(plr_coverage, relative_threshold_factor, plr_cellsangle[iangle])
        plr_cellsangle_gc = psp.selection.top_percentile(plr_distances, top_share, plr_cellsangle_relcoverage)

        prt_indicesangle_gd = psp.grid.particles_from_grid(plr_prtindices, plr_cellsangle_gd)
        prt_indicesangle_gc = psp.grid.particles_from_grid(plr_prtindices, plr_cellsangle_gc)

        # Reduce
        ff_distance_angle[iangle][ilc][iout] = psp.reduction.mean(prt_distances, prt_indicesangle_ff)
        ff_velocity_angle[iangle][ilc][iout] = psp.reduction.mean(prt_velocities, prt_indicesangle_ff)

        gd_distance_angle[iangle][ilc][iout] = psp.reduction.mean(prt_distances, prt_indicesangle_gd)
        gd_velocity_angle[iangle][ilc][iout] = psp.reduction.mean(prt_velocities, prt_indicesangle_gd)
        gc_distance_angle[iangle][ilc][iout] = psp.reduction.mean(prt_distances, prt_indicesangle_gc)
        gc_velocity_angle[iangle][ilc][iout] = psp.reduction.mean(prt_velocities, prt_indicesangle_gc)


  # ----- Lump -----
  # Prepare comment string
  comment = io.StringIO()
  comment.write("FDS Particle Spray Postprocessor (FDS-PSP)\n\n")
  prm_global.print_parameters(file=comment)
  prm_polar.print_parameters(file=comment)
  print(" * Absolute threshold: ", absolute_threshold, file=comment)
  print(" * Relative threshold factor: ", relative_threshold_factor, file=comment)
  print(" * Top share: ", top_share, file=comment)

  # Prepare data
  header = "time mass"
  # Farthest Fraction Method
  header += " ff_distance ff_velocity"
  for angle in angles:
    header += " ff_distance_{0} ff_velocity_{0}".format(str(angle))
  # Grid Density Method
  header += " gd_distance gd_velocity"
  for angle in angles:
    header += " gd_distance_{0} gd_velocity_{0}".format(str(angle))
  # Grid Coverage Method
  header += " gc_distance gc_velocity"
  for angle in angles:
    header += " gc_distance_{0} gc_velocity_{0}".format(str(angle))

  data = []
  for ilc,_ in enumerate(prm_global.particle_types):
    data_ilc = [pdata.times, total_mass[ilc]]
    # Farthest Fraction Method (FF)
    data_ilc.extend([ff_distance[ilc], ff_velocity[ilc]])
    for iangle, _ in enumerate(angles):
      data_ilc.extend([ff_distance_angle[iangle][ilc], ff_velocity_angle[iangle][ilc]])
    # Grid Density Method (GD)
    data_ilc.extend([gd_distance[ilc], gd_velocity[ilc]])
    for iangle, _ in enumerate(angles):
      data_ilc.extend([gd_distance_angle[iangle][ilc], gd_velocity_angle[iangle][ilc]])
    # Grid Coverage Method (GC)
    data_ilc.extend([gc_distance[ilc], gc_velocity[ilc]])
    for iangle,_ in enumerate(angles):
      data_ilc.extend([gc_distance_angle[iangle][ilc], gc_velocity_angle[iangle][ilc]])

    data.append(data_ilc)

  return data, header, comment.getvalue()



def plot_water(filepair):
  """
  TODO: Doc
  """
  my_plots = [("time", "distance"), ("time", "velocity")]
  my_labels = {"time": "Time [s]", "distance": "Distance from POI [m]", "velocity": "Velocity [m/s]"}
  my_legend = {"ff": "FF", "gd": "GD", "gc": "GC"}

  # ----- Read -----
  file_in = open(filepair[0], 'r')

  # Skip comments
  line = ""
  while True:
    line = file_in.readline()
    if not line.startswith('#'):
      break

  # 'line' now contains first uncommented line, which deals with particle type.
  # Extract particle type.
  particle_type = int(line.split("Particle Type ")[1])

  # 'file_in' now points to the second uncommented line, which contains column headers.
  # Load data and extract column headers.
  data = np.genfromtxt(file_in, names=True)
  columns = data.dtype.names

  # Prepare
  plots_data = []
  for xd,yd in my_plots:
    assert xd in columns
    plots_data.append([(xd,cn) for cn in columns if yd in cn])

  labels_data = {}
  for key,value in my_labels.items():
    labels_data.update({cn: value for cn in columns if key in cn})

  legend_data = {}
  for key,value in my_legend.items():
    legend_data.update({cn: value for cn in columns if key in cn})

  # Plot global
  dirname = os.path.basename(os.path.dirname(filepair[0]))
  for i,(x,y) in enumerate(my_plots):
    plt.figure(i)
    for (xd,yd) in plots_data[i]:
      # extract angle
      legend = legend_data[yd]
      angle = yd.rsplit('_', 1)
      if angle[1].isdigit():
        legend += "_" + angle[1]
      # perform plot
      plt.plot(data[xd], data[yd], label=dirname+"_"+legend)
      plt.xlabel(labels_data[xd])
      plt.ylabel(labels_data[yd])
    # title will be (ab)used as a file indicator later
    plt.title(x+"_"+y)

  # Plot local
  for i,(x,y) in enumerate(my_plots):
    plt.figure()
    for (xd,yd) in plots_data[i]:
      # extract angle
      legend = legend_data[yd]
      angle = yd.rsplit('_', 1)
      if angle[1].isdigit():
        legend += "_" + angle[1]
      # plot
      plt.plot(data[xd], data[yd], label=legend)
      plt.xlabel(labels_data[xd])
      plt.ylabel(labels_data[yd])
    plt.legend()
    file_out = filepair[1].split("_psp")[0] + "_" + x + "_" + y + ".pdf"
    plt.savefig(file_out)
    plt.close()





if __name__ == "__main__":
  """
  Provide paths to folder containing 'prt5' data and the output folder as command line arguments to start evaluation.
  """
  import sys, os

  # Store directories
  dir_in = os.path.abspath(sys.argv[1])
  dir_out = os.path.abspath(sys.argv[2])

  # Check if specified paths are valid
  assert os.path.isdir(dir_in), "Cannot find source directory!"
  if not os.path.isdir(dir_out):
    os.makedirs(dir_out)

  print("Simulate scenarios.")
  psp.multiple.simulate_multiple(psp.multiple.simulate_default, dir_in, skip_processed=True)

  print("Process scenarios.")
  my_processor = partial(psp.multiple.process_default, parser=parse_water)
  psp.multiple.process_multiple(my_processor, dir_in, dir_out, skip_processed=True)

  print("Plot scenarios.")
  psp.multiple.plot_multiple(plot_water, dir_out, dir_out)

  # Write out global plots to file
  for f in plt.get_fignums():
    plt.figure(f)
    plt.legend()
    ax = plt.gca()
    path_out = os.path.join(dir_out, ax.get_title() + ".pdf")
    plt.title("")
    plt.savefig(path_out)