#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io, os, subprocess
from functools import partial
import multiprocessing

import numpy as np

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


# #########################
# ## Handling filelists ###
# #########################

def only_one(files):
  """
  TODO: Doc.

  Parameters
  ----------
  files: list[i1] -> string
    Filelist

  Returns
  -------
  file : list[string] of size 0
    First entry in string
  """
  assert len(files) == 1
  return files


def filestem(files):
  """
  Get stem of files

  Parameters
  ----------
  files : list[i1] -> string
    FileList

  Returns
  -------
  filestem : string
    Stem of files.
  """

  # Find the correct filestem that all files share
  roots = [os.path.splitext(file)[0] for file in files]
  commonroot = os.path.commonprefix(roots)
  assert len(commonroot) > 0
  return [commonroot]



def get_filepairs(dir_in, dir_out, ext_in, ext_out, process_fnames=None):
  """
  TODO: Doc.

  Parameters
  ----------
  dir_in : string
    TODO: Doc.
  ext_in : string
    TODO: Doc.
  dir_out : string
    TODO: Doc.
  ext_out : string
    TODO: Doc '.csv'

  Returns
  -------
  filelist : list[i1] -> string
    TODO: Doc.
  """
  filelist = []
  for root, subdirs, files in os.walk(dir_in):
    # Prepare output directory
    relpath = os.path.relpath(root, start=dir_in)
    root_out = os.path.join(dir_out, relpath)

    # Specify on which files we have to work on
    files_in = [f for f in files if f.lower().endswith(ext_in)]
    if not files_in:
      continue

    # Determine which filepairs to process
    if process_fnames is not None:
      files_in = process_fnames(files_in)

    # Add pairs to list
    for file_in in files_in:
      path_in = os.path.join(root, file_in)

      # Prescribe output folder
      stem_out = os.path.splitext(file_in)[0]
      file_out = stem_out + ext_out
      path_out = os.path.join(root_out, file_out)

      # Add tuple to folder
      filelist.append((path_in, path_out))

  return filelist



# #######################
# ### Multiprocessing ###
# #######################

def process(processor_function, filelist, parallel=False, skip_processed=False):
  """
  TODO Doc

  Parameters
  ----------
  processor_function : function
    TODO Doc
  filelist : list[i1] : tuple(string, string)
    TODO Doc
  """

  # Ensure existence of output folders
  for pair in filelist:
    # Ensure output folder exists
    root_out = os.path.split(pair[1])[0]
    if not os.path.isdir(root_out):
      os.makedirs(root_out)

  # Specify files to work on
  mylist = []
  if skip_processed == False:
    mylist = filelist
  else:
    for pair in filelist:
      root_out = os.path.split(pair[1])[0]
      ext_out = os.path.splitext(pair[1])[1]
      files_out = [f for f in os.listdir(root_out) if f.lower().endswith(ext_out)]

      if len(files_out) == 0:
        mylist.append(pair)

  # Do the work
  if parallel == False:
    for pair in mylist:
      processor_function(pair)
  else:
    pool = multiprocessing.Pool(None)
    pool.map_async(processor_function, mylist)
    pool.close()
    pool.join()



def multiple(function, dir_in, dir_out="", ext_in="", ext_out="", process_fnames=None, parallel=False, skip_processed=False):
  """
  TODO: Doc
  """
  if not dir_out:
    dir_out = dir_in
  assert len(ext_in) > 0

  # Get filepair
  files = get_filepairs(dir_in, dir_out, ext_in, ext_out, process_fnames)
  process(function, files, parallel=parallel, skip_processed=skip_processed)


# Wrappers for different tasks.
simulate_multiple = partial(multiple, ext_in=".fds", ext_out=".prt5", process_fnames=only_one)
process_multiple = partial(multiple, ext_in=".prt5", ext_out=".csv", process_fnames=filestem)
plot_multiple = partial(multiple, ext_in=".csv", ext_out=".pdf", process_fnames=None)



# ##########################
# ### Default processors ###
# ##########################

def simulate_default(filepair):
  """
  TODO: Doc.

  Parameters
  ----------
  filepair : ?
    TODO: Doc.
  """
  call = "OMP_NUM_THREADS=1 fds " + filepair[0]
  subprocess.check_call(call, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)


def process_default(filepair, parser):
  """
  TODO: Doc.

  Parameters
  ----------
  filepair : ?
    TODO: Doc.
  parser : ?
    TODO: Doc.
  """
  data, header, comments = parser(filepair[0])
  # Check if data container and header are in accordance
  assert all(len(data[ilc]) == len(header.split()) for ilc in range(len(data)))

  # Add comment flag in front of each line of the comments string
  comments = ''.join(["# " + line for line in comments.splitlines(True)])

  for ilc in range(len(data)):
    # Extend filename
    splitext = os.path.splitext(filepair[1])
    filename = splitext[0] + "_ptype" + str(ilc) + "_psp" + splitext[1]
    # Write out data
    file_out = open(filename, 'w')
    file_out.write(comments)
    file_out.write("Particle Type " + str(ilc) + "\n")
    np.savetxt(file_out, np.column_stack(data[ilc]), fmt='%1.4e', header=header, comments='')
    file_out.close()


def plot_default(filepair):
  """
  TODO: Doc

  Parameters
  ----------
  filepair : ?
    TODO: Doc.
  """
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
  column_names = data.dtype.names

  # Plot global
  dirname = os.path.basename(os.path.dirname(filepair[0]))
  for i in range(1, len(column_names)):
    plt.figure(i - 1)
    plt.plot(data[column_names[0]], data[column_names[i]], label=dirname)
    plt.xlabel(column_names[0])
    plt.ylabel(column_names[i])

  # Plot local
  for i in range(1, len(column_names)):
    # Get a temporary figure nobody cares about
    plt.figure()
    plt.plot(data[column_names[0]], data[column_names[i]])
    plt.xlabel(column_names[0])
    plt.ylabel(column_names[i])
    file_out = filepair[1].split("_psp")[0] + "_" + column_names[i] + ".pdf"
    plt.savefig(file_out)
    plt.close()





if __name__ == "__main__":
  """
  Provide paths to folder containing 'prt5' data and the output folder as command line arguments to start a test evaluation.
  """
  import sys
  import read, parameter, particle, selection, reduction


  def parse_default(infilestem):
    """
    TODO: Doc

    Parameters
    ----------
    filestem : string
      Stem of files.
    """
    # ----- Parameters -----
    ptype_water = parameter.ParticleType(class_id=0, diameter_id=0, mass_id=1, temperature_id=2, velocity_ids=[3,4,5], weighting_factor_id=6)
    prm_global = parameter.ParameterGlobal(reference_position=[4.0,3.4,1.0], particle_types=[ptype_water], direction=[1,0,1])
    top_fraction = 0.1

    # ----- Read -----
    pdata = read.ParticleData(infilestem)

    # ----- Analyse -----
    dummy_container = np.empty(shape=(len(prm_global.particle_types), pdata.info['n_outputsteps']), dtype=object)

    sum_prtmasses = np.empty_like(dummy_container)
    mean_prtdistances_topdistance = np.empty_like(dummy_container)
    mean_prtvelocitiesw_topdistance = np.empty_like(dummy_container)
    mean_prtvelocities_topdistance = np.empty_like(dummy_container)
    for ilc,ptype in enumerate(prm_global.particle_types):
      for iout in range(pdata.info['n_outputsteps']):
        # Farthest Fraction Method
        # Particle data
        prt_masses = particle.masses(pdata, ptype, iout)
        prt_velocitiesw = pdata.qs[ptype.class_id][ptype.velocity_ids[2]][iout]
        prt_velocities = particle.velocities(prm_global, pdata, ptype, iout)
        prt_distances = particle.distances_from_reference_point(prm_global, pdata, ptype, iout)
        # Selection
        prt_indices_topdistance = selection.top_percentile(prt_distances, top_fraction)
        # Reduction
        sum_prtmasses[ilc][iout] = np.sum(prt_masses)
        mean_prtdistances_topdistance[ilc][iout] = reduction.mean(prt_distances, prt_indices_topdistance)
        mean_prtvelocitiesw_topdistance[ilc][iout] = reduction.mean(prt_velocitiesw, prt_indices_topdistance)
        mean_prtvelocities_topdistance[ilc][iout] = reduction.mean(prt_velocities, prt_indices_topdistance)

    # ----- Lump -----
    # Prepare comment string
    comment = io.StringIO()
    comment.write("FDS Particle Spray Postprocessor (FDS-PSP)\n\n")
    prm_global.print_parameters(file=comment)
    print(" * Top fraction: ", top_fraction, file=comment)

    # Prepare data
    header = "time mass ff_radius ff_velocity ff_velocity_z"
    data = [[pdata.times, sum_prtmasses[ilc],
         mean_prtdistances_topdistance[ilc], mean_prtvelocities_topdistance[ilc], mean_prtvelocitiesw_topdistance[ilc]]
        for ilc in range(len(prm_global.particle_types))]

    return data, header, comment.getvalue()



  # Store directories
  dir_in = os.path.abspath(sys.argv[1])
  dir_out = os.path.abspath(sys.argv[2])

  # Check if specified paths are valid
  assert os.path.isdir(dir_in), "Cannot find source directory!"
  if not os.path.isdir(dir_out):
    os.makedirs(dir_out)

  print("Simulate scenarios.")
  simulate_multiple(simulate_default, dir_in, skip_processed=True)

  print("Process scenarios.")
  my_processor = partial(process_default, parser=parse_default)
  process_multiple(my_processor, dir_in, dir_out, skip_processed=True)

  print("Plot scenarios.")
  plot_multiple(plot_default, dir_out, dir_out)

  # Write out global plots to file
  for f in plt.get_fignums():
    plt.figure(f)
    plt.legend()
    ax = plt.gca()
    path_out = os.path.join(dir_out, ax.get_ylabel() + ".pdf")
    plt.savefig(path_out)