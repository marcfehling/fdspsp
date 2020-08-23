#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
from functools import partial
import numpy as np

import psp.read, psp.parameter, psp.particle, psp.grid, psp.selection, psp.reduction, psp.multiple

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


def parse_fireball(infilestem):
  """
  TODO: Doc
  """
  # ----- Parameters -----
  # Parameters from FDS file
  ptype_heptane = psp.parameter.ParticleType(class_id=1, diameter_id=0, mass_id=1, temperature_id=2, velocity_ids=[3,4,5], weighting_factor_id=6)
  ptype_tracer = psp.parameter.ParticleType(class_id=2, diameter_id=0, mass_id=1, temperature_id=2, velocity_ids=[3,4,5], weighting_factor_id=6)
  prm_global = psp.parameter.ParameterGlobal(reference_position=[0.0,-0.1,15.0], particle_types=[ptype_heptane, ptype_tracer], direction=[1,0,1])
  # Parameters for evaluation
  prm_cartesian = psp.parameter.ParameterCartesian(length_xi=6.0, length_xj=2.0, n_xi=90, n_xj=30)
  absolute_threshold = 1
  relative_threshold_factor = 0.01
  top_share = 0.1

  # ----- Read -----
  pdata = psp.read.ParticleData(infilestem)

  # ----- Prepare -----
  crt_distances, crt_areas = psp.grid.cartesian_grid_properties(prm_cartesian)

  # ----- Analyse -----
  dummy_container = np.empty(shape=(len(prm_global.particle_types), pdata.info['n_outputsteps']), dtype=object)

  # Farthest Fraction Method (FF)
  total_mass = np.empty_like(dummy_container)
  ff_distance = np.empty_like(dummy_container)
  ff_velocity = np.empty_like(dummy_container)

  # Grid Density Method (GD)
  gd_distance = np.empty_like(dummy_container)
  gd_velocity = np.empty_like(dummy_container)

  # Grid Coverage Method (GC)
  gc_distance = np.empty_like(dummy_container)
  gc_velocity = np.empty_like(dummy_container)

  for ilc, ptype in enumerate(prm_global.particle_types):
    for iout in range(pdata.info['n_outputsteps']):
      # Particle
      prt_masses = psp.particle.masses(pdata, ptype, iout)
      prt_crosssections = psp.particle.cross_sections(pdata, ptype, iout)
      prt_velocities = psp.particle.velocities(prm_global, pdata, ptype, iout)
      prt_distances = psp.particle.distances_from_reference_point(prm_global, pdata, ptype, iout)

      # Select moving particles
      prt_indices_absvelocities = psp.selection.absolute_threshold(prt_velocities, absolute_threshold)

      # Grid
      crt_prtindices = psp.grid.particles_on_cartesian_grid(prm_cartesian, prm_global, pdata, ptype, iout, prt_indices_absvelocities)
      crt_prtnumbers = psp.grid.number_of_particles(crt_prtindices, pdata, ptype, iout)
      crt_density = crt_prtnumbers / crt_areas
      crt_prtareas = psp.grid.area_of_particles(prt_crosssections, crt_prtindices, pdata, ptype, iout)
      crt_coverage = crt_prtareas / crt_areas

      # Selection
      prt_indices_ff = psp.selection.top_percentile(prt_distances, top_share, prt_indices_absvelocities)

      crt_cells_reldensity = psp.selection.relative_threshold(crt_density, relative_threshold_factor)
      crt_cells_reldensity_topdistance = psp.selection.top_percentile(crt_distances, top_share, crt_cells_reldensity)
      crt_cells_relcoverage = psp.selection.relative_threshold(crt_coverage, relative_threshold_factor)
      crt_cells_relcoverage_topdistance = psp.selection.top_percentile(crt_distances, top_share, crt_cells_relcoverage)

      prt_indices_gd = psp.grid.particles_from_grid(crt_prtindices, crt_cells_reldensity_topdistance)
      prt_indices_gc = psp.grid.particles_from_grid(crt_prtindices, crt_cells_relcoverage_topdistance)

      # Reduction
      total_mass[ilc][iout] = np.sum(prt_masses)
      ff_distance[ilc][iout] = psp.reduction.mean(prt_distances, prt_indices_ff)
      ff_velocity[ilc][iout] = psp.reduction.mean(prt_velocities, prt_indices_ff)

      gd_distance[ilc][iout] = psp.reduction.mean(prt_distances, prt_indices_gd)
      gd_velocity[ilc][iout] = psp.reduction.mean(prt_velocities, prt_indices_gd)
      gc_distance[ilc][iout] = psp.reduction.mean(prt_distances, prt_indices_gc)
      gc_velocity[ilc][iout] = psp.reduction.mean(prt_velocities, prt_indices_gc)

  # ----- Lump -----
  # Prepare comment string
  comment = io.StringIO()
  comment.write("FDS Particle Spray Postprocessor (FDS-PSP)\n\n")
  prm_global.print_parameters(file=comment)
  prm_cartesian.print_parameters(file=comment)
  print(" * Absolute threshold: ", absolute_threshold, file=comment)
  print(" * Relative threshold factor: ", relative_threshold_factor, file=comment)
  print(" * Top share: ", top_share, file=comment)

  # Prepare data
  header = "time mass ff_distance ff_velocity gd_distance gd_velocity gc_distance gc_velocity"
  data = [[pdata.times, total_mass[ilc], ff_distance[ilc], ff_velocity[ilc], gd_distance[ilc], gd_velocity[ilc], gc_distance[ilc], gc_velocity[ilc]]
          for ilc,_ in enumerate(prm_global.particle_types)]

  return data, header, comment.getvalue()



def plot_fireball(filepair):
  """
  TODO: doc
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
  for xd, yd in my_plots:
    assert xd in columns
    plots_data.append([(xd, cn) for cn in columns if yd in cn])

  labels_data = {}
  for key, value in my_labels.items():
    labels_data.update({cn: value for cn in columns if key in cn})

  legend_data = {}
  for key, value in my_legend.items():
    legend_data.update({cn: value for cn in columns if key in cn})

  # Plot global
  dirname = os.path.basename(os.path.dirname(filepair[0]))
  for i,(x,y) in enumerate(my_plots):
    plt.figure(i)
    for (xd,yd) in plots_data[i]:
      plt.plot(data[xd], data[yd], label=dirname+"_"+legend_data[yd])
      plt.xlabel(labels_data[xd])
      plt.ylabel(labels_data[yd])
    # title will be (ab)used as a file indicator later
    plt.title(x+"_"+y)

  # Plot local
  for i,(x,y) in enumerate(my_plots):
    plt.figure()
    for (xd,yd) in plots_data[i]:
      plt.plot(data[xd], data[yd], label=legend_data[yd])
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
  my_processor = partial(psp.multiple.process_default, parser=parse_fireball)
  psp.multiple.process_multiple(my_processor, dir_in, dir_out, skip_processed=True)

  print("Plot scenarios.")
  psp.multiple.plot_multiple(plot_fireball, dir_out, dir_out)

  # Write out global plots to file
  for f in plt.get_fignums():
    plt.figure(f)
    plt.legend()
    ax = plt.gca()
    path_out = os.path.join(dir_out, ax.get_title() + ".pdf")
    plt.title("")
    plt.savefig(path_out)