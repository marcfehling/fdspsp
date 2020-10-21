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
READ module: Read

For convenience, everything in ParticleData class.
"""


from functools import partial
from glob import glob
from mmap import mmap, ACCESS_READ
from multiprocessing import Pool
from time import time

import numpy as np


def _get_fds_dtype(name, count=1, quantities=0):
  """
  Returns numpy.dtype on various FDS particle data types for reading
  from binary files.
  """

  # As the binary representation of raw data is compiler dependent,
  # this information must be provided by the user.
  # i4 -> 32 bit integer
  FDS_DTYPE_INT = "i4"
  # f4 -> 32 bit floating point
  FDS_DTYPE_FLOAT = "f4"
  # a  ->  8 bit character
  FDS_DTYPE_CHAR = "a"
  # sets whether blocks are ended with the size of the block
  FDS_FORTRAN_BACKWARD = True

  # Identifier LUPF
  fds_dtype = "{0}".format(FDS_DTYPE_INT)

  # Choose data type
  if name == "int":
    fds_dtype += ", ({0},){1}".format(count, FDS_DTYPE_INT)
  elif name == "float":
    fds_dtype += ", ({0},){1}".format(count, FDS_DTYPE_FLOAT)
  elif name == "char":
    fds_dtype += ", ({0}){1}".format(count, FDS_DTYPE_CHAR)
  elif name == "positions":
    fds_dtype += ", (3,{0}){1}".format(count, FDS_DTYPE_FLOAT)
  elif name == "tags":
    fds_dtype += ", ({0},){1}".format(count, FDS_DTYPE_INT)
  elif name == "quantities":
    fds_dtype += ", ({0},{1}){2}".format(quantities, count, FDS_DTYPE_FLOAT)
  else:
    assert False, "Unknown FDS particle data type. Aborting."

  # Size of block if available
  if FDS_FORTRAN_BACKWARD:
    fds_dtype += ", {0}".format(FDS_DTYPE_INT)

  return np.dtype(fds_dtype)


def _read_particle_file(filename, output_each=1, classes=None, n_timesteps=None, logs=True):
  """
  Function to parse and read in all particle data written by FDS from
  one single mesh into a single file.

  See below class documentation for details on parameters and returned
  values.

  The optional parameter ntimesteps indicates the number of timesteps of
  the specified FDS simulation. It will be used to identify whether
  there are no more particles in subsequent timesteps.
  """

  # Verify user input
  assert output_each > 0
  if classes is not None:
    assert all(classid >= 0 for classid in classes)
  if n_timesteps is not None:
    assert n_timesteps > 0

  #
  # INITIALIZATION: open file
  #

  if logs:
    time_start = time()
    print("Reading particle file '{}'.".format(filename))

  file_object = open(filename, 'rb')
  assert file_object, "Could not open file '{}'! Aborting.".format(filename)

  # Map file content to memory
  file_mm = mmap(file_object.fileno(), 0, access=ACCESS_READ)
  file_mm.flush()

  file_mm_pos = 0

  def _read_from_buffer(dtype, skip=False):
    """
    Reads exactly one instance of dtype from the current memory-mapped
    file object.

    Results will be returned as np.ndarray.
    """

    nonlocal file_mm, file_mm_pos

    # Proceed the input filestream by the corresponding amount of bytes
    if skip:
      file_mm_pos += dtype.itemsize
      return

    data_raw = np.frombuffer(file_mm, dtype, count=1, offset=file_mm_pos)
    file_mm_pos += dtype.itemsize

    # We are only interested in the actual data, ignore the remains of
    # the binary file
    return data_raw[0][1]

  #
  # READ: miscellaneous data
  #

  info = {"filename": filename}

  # Read endianess flag written by FDS which will be ignored
  _read_from_buffer(_get_fds_dtype("int"))

  # Read FDS version identifier
  info["fds_version"] = _read_from_buffer(_get_fds_dtype("int"))[0]

  # Read number of classes
  info["n_classes"] = _read_from_buffer(_get_fds_dtype("int"))[0]
  # If user did not provide any class identifiers, consider all by default
  if classes is None:
    classes = list(range(info["n_classes"]))

  # Read quantitiy properties
  # Create empty lists for each particle class
  info["n_quantities"] = [None for _ in range(info["n_classes"])]
  info["q_labels"] = [[] for _ in range(info["n_classes"])]
  info["q_units"] = [[] for _ in range(info["n_classes"])]

  # Loop over all classes and parse their individual quantity labels
  # and units
  for ic in range(info["n_classes"]):
    # Read number of quantities for current  class and skip a placeholder
    info["n_quantities"][ic] = _read_from_buffer(
        _get_fds_dtype("int", count=2))[0]

    # Read particle quantities as character lists, add as strings to the
    # info lists
    for _ in range(info["n_quantities"][ic]):
      qlabel = _read_from_buffer(_get_fds_dtype("char", count=30))
      info["q_labels"][ic].append(qlabel.decode())

      qunit = _read_from_buffer(_get_fds_dtype("char", count=30))
      info["q_units"][ic].append(qunit.decode())

  #
  # READ: particle data
  #

  # Particles in simulations may be removed, thus it quite often occurs
  # that there are no more particles in subsequent timesteps after some
  # point during the simulation. Meshes may not even contain any
  # particle at all. We are safe to stop reading the remaining file in
  # this case.
  #
  # To do so, we will determine how much space an empty timestep
  # occupies, calculate how many timesteps there would be if each of the
  # subsequent timesteps contains zero particles, and compare it to the
  # actual number of timesteps.

  # Size of zero particles for each particle class in a single timestep
  zero_particles_size = [
      _get_fds_dtype("positions", count=0).itemsize +
      _get_fds_dtype("tags", count=0).itemsize +
      (_get_fds_dtype("quantities", count=0, quantities=info["n_quantities"][ic]).itemsize
          if info["n_quantities"][ic] > 0 else 0)
      for ic in range(info["n_classes"])]

  # Size of a timestep without any particles in all classes
  # Current timestep
  empty_timestep_size = _get_fds_dtype("float").itemsize
  for ic in range(info["n_classes"]):
    empty_timestep_size += (
        # Number of particles is zero
        _get_fds_dtype("int").itemsize +
        # Size that zero particles occupy
        zero_particles_size[ic]
    )

  # Create empty lists for each particle class
  times = []
  n_particles = [[] for _ in range(info["n_classes"])]
  positions = [[] for _ in range(info["n_classes"])]
  tags = [[] for _ in range(info["n_classes"])]
  quantities = [[[] for _ in range(info["n_quantities"][ic])]
                for ic in range(info["n_classes"])]

  timestep = 0
  while file_mm_pos < file_mm.size():
    # If all remaining timesteps have no particles, we will skip this file
    if n_timesteps:
      n_timesteps_estimate_if_remaining_steps_empty = \
          timestep + (file_mm.size() - file_mm_pos) // empty_timestep_size
      if n_timesteps_estimate_if_remaining_steps_empty == n_timesteps:
        if logs:
          print("No more particles. Skip remaining file.")
        file_mm_pos = file_mm.size()
        break

    # Decide whether we want to process this timestep
    skip_timestep = (timestep % output_each) > 0

    # Read time of current output step
    time_at_timestep = _read_from_buffer(_get_fds_dtype("float"))[0]
    if not skip_timestep:
      times.append(time_at_timestep)

    # Read data for each particle class
    for ic in range(info["n_classes"]):
      # Decide whether we want to process this particle class
      skip_class = ic not in classes
      skip_current = skip_timestep or skip_class

      # Read number of particles
      n_particle = _read_from_buffer(_get_fds_dtype("int"))[0]

      # If no particles were found, skip this timestep
      if n_particle == 0:
        file_mm_pos += zero_particles_size[ic]

        # Store empty data if required
        if not skip_current:
          n_particles[ic].append(0)
          positions[ic].append([np.array([]),
                                np.array([]),
                                np.array([])])
          tags[ic].append(np.array([]))
          for iq in range(info["n_quantities"][ic]):
            quantities[ic][iq].append(np.array([]))

        continue

      # Read position lists
      raw_positions = _read_from_buffer(_get_fds_dtype("positions",
                                                       count=n_particle),
                                        skip=skip_current)
      # Read tags
      raw_tags = _read_from_buffer(_get_fds_dtype("tags",
                                                  count=n_particle),
                                   skip=skip_current)
      # Read each quantity data, if there is any
      raw_qs = None
      if info["n_quantities"][ic] > 0:
        raw_quantities = _read_from_buffer(_get_fds_dtype("quantities",
                                                          count=n_particle,
                                                          quantities=info["n_quantities"][ic]),
                                           skip=skip_current)

      # Store data if required
      if not skip_current:
        n_particles[ic].append(n_particle)
        positions[ic].append([np.copy(raw_positions[0]),
                              np.copy(raw_positions[1]),
                              np.copy(raw_positions[2])])
        tags[ic].append(np.copy(raw_tags))
        for iq in range(info["n_quantities"][ic]):
          quantities[ic][iq].append(np.copy(raw_quantities[iq]))

    # Continue with next timestep
    timestep += 1

  assert file_mm_pos == file_mm.size()

  # Add number of timesteps and outputsteps to dict
  info["n_timesteps"] = timestep
  info["n_outputsteps"] = len(times)

  if logs:
    data_size = file_mm.size()
    data_size_in_mb = data_size / (1024 ** 2)
    time_end = time()
    print("file size: {:.3f} MB, speed: {:.3f} MB/s".format(
          data_size_in_mb, data_size_in_mb / (time_end - time_start)))

  file_object.close()

  return info, times, n_particles, positions, tags, quantities


def _read_multiple_particle_files(filestem, fileextension, output_each=1, classes=None, logs=True, parallel=True):
  """
  Function to parse and read in all particle data written by FDS from
  multiple meshes into multiple files.

  See below class documentation for details on parameters and returned
  values.
  """

  # Verify user input
  filename_wildcard = filestem + "*." + fileextension
  filelist = sorted(glob(filename_wildcard))
  assert len(filelist) > 0, ("No files were found with the specified "
                             "credentials.")

  #
  # READ: all files
  #

  # Read very first file to know the total number of timesteps
  result_first = _read_particle_file(filelist.pop(0),
                                     output_each, classes, n_timesteps=None)
  # If there are no more files, we are done
  if not filelist:
    return result_first

  # Extract global information
  info = result_first[0]
  info["filename"] = filename_wildcard
  times = result_first[1]

  # Read remaining input files
  results = [result_first]
  if parallel:
    pool = Pool(None)
    worker = partial(_read_particle_file,
                     output_each=output_each, classes=classes, n_timesteps=info["n_timesteps"])
    results[1:] = pool.map(worker, filelist)
    pool.close()
    pool.join()
  else:
    for filename in filelist:
      results.append(
          _read_particle_file(filename,
                              output_each, classes, info["n_timesteps"]))

  if logs:
    print("Finished reading.")

  #
  # CONCATENATE: results of all files
  #

  # Calculate global number of particles for every particle class in
  # each timestep
  n_particles = [[0] * info["n_outputsteps"]] * info["n_classes"]
  for res in results:
    local_n_particles = res[2]
    for ic in range(info["n_classes"]):
      for iout, local_n_particle in enumerate(local_n_particles[ic]):
        n_particles[ic][iout] += local_n_particle

  # Prepare empty data containers one after another to avoid memory
  # fragmentation
  positions = [[[np.empty(n_particles[ic][iout]),
                 np.empty(n_particles[ic][iout]),
                 np.empty(n_particles[ic][iout])]
                for iout in range(info["n_outputsteps"])]
               for ic in range(info["n_classes"])]
  tags = [[np.empty(n_particles[ic][iout])
           for iout in range(info["n_outputsteps"])]
          for ic in range(info["n_classes"])]
  quantities = [[[np.empty(n_particles[ic][iout])
                  for iout in range(info["n_outputsteps"])]
                 for _ in range(info["n_quantities"][ic])]
                for ic in range(info["n_classes"])]

  # Offsets to build consecutive buffer
  offsets = [np.zeros(info["n_outputsteps"], dtype="int")
             for _ in range(info["n_classes"])]

  # Attach data
  for res in results:
    _, _, local_n_particles, local_positions, local_tags, local_quantities = res

    for ic in range(info["n_classes"]):
      for iout, n in enumerate(local_n_particles[ic]):
        o = offsets[ic][iout]
        positions[ic][iout][0][o:o+n] = np.copy(local_positions[ic][iout][0])
        positions[ic][iout][1][o:o+n] = np.copy(local_positions[ic][iout][1])
        positions[ic][iout][2][o:o+n] = np.copy(local_positions[ic][iout][2])
        tags[ic][iout][o:o+n] = np.copy(local_tags[ic][iout])
        for iq in range(info["n_quantities"][ic]):
          quantities[ic][iq][iout][o:o +
                                   n] = np.copy(local_quantities[ic][iq][iout])

        offsets[ic][iout] += n

  for ic in range(info["n_classes"]):
    for iout, n_particle in enumerate(n_particles[ic]):
      assert offsets[ic][iout] == n_particle

  return info, times, n_particles, positions, tags, quantities


class ParticleData:
  """
  Container to store particle data from particle simulations with FDS.

  The specified FDS output files will be read and all particle
  information will be stored in lists and numpy.arrays. These data
  structures can then be further processed with fdspsp in particular or
  python in general.

  Particle data will be read in the FORTRAN format as described in the
  FDS user guide (version 6.7.1, section 24.10).

  Parameters
  ----------
  filestem : str
  fileextension : str
    Those files will be read who start with the filestem parameter and end with the fileextenstion.
    By default, all files with the extension 'prt5' will be read.
  output_each : int
    Each n-th timestep will actually be read, those inbetween will be skipped.
    By default, every timestep will be considered.
  classes : list -> int > 0
    Specify identifiers of classes that will be read.
    By default, every particle class will be considered.
  logs : bool
    Choose if logs will be printed out.
  parallel : bool
    Decide if files from filesystem will be read in parallel.

  Class members
  -------------
  info : dict
    Contains meta information about the dataset, as:
      - filename : str
          The name of the read datafile
      - fds_version : int
          Data has been generated with this FDS version
      - n_classes : int
          Number of particle classes from the FDS simulation
      - n_timesteps : int
          Number of total timesteps from the FDS simulation
      - n_outputsteps : int
          Number of timesteps that have been read with fdspsp, i.e.,
          n_outputsteps = n_timesteps // output_each + 1
      - n_quantities : list[i1] -> int
          Number of quantities for particle class i1
      - q_labels : list[i1] -> str
          List of quantity labels for particle class i1
      - q_units : list[i1] -> str
          List of quantity units for particle class i1

  n_particles : list[i1][i2] -> int
    Number of particles in class i1 at output step i2

  times : list[i1] -> float
    Time of output step i1

  positions : list[i1][i2][i3] -> np.array(float)
    positions holds coordinate lists (as numpy-arrays) for particle
    class i1 at output step i2 for component (x,y,z) i3, where the
    length of the array is given by the number of particles of the
    selected class and step, i.e., n_particles[i1][i2]

  tags : list[i1][i2] -> np.array(int)
    numpy-array of particle tags for class i1 and output step i2

  quantities : list[i1][i2][i3] -> np.array(float)
    numpy-array of particle quantity i2 for class i1 and output step i3;
    the number of available quantities for the selected particle class
    is given in info["n_quantities"][i1]
  """

  def __init__(self, filestem, fileextension="prt5", output_each=1, classes=None, logs=True, parallel=True):
    self.info, self.times, self.n_particles, self.positions, self.tags, self.quantities = \
        _read_multiple_particle_files(
            filestem, fileextension, output_each, classes, logs, parallel)
