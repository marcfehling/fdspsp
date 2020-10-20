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
    elif name == "coordinates":
      fds_dtype += ", (3,{0}){1}".format(count, FDS_DTYPE_FLOAT)
    elif name == "tags":
      fds_dtype += ", ({0},){1}".format(count, FDS_DTYPE_INT)
    elif name == "quantities":
      fds_dtype += ", ({0},{1}){2}".format(quantities, count, FDS_DTYPE_FLOAT)
    else:
      assert false, "Unknown FDS particle data type. Aborting."

    # Size of block if available
    if FDS_FORTRAN_BACKWARD:
      fds_dtype += ", {0}".format(FDS_DTYPE_INT)

    return np.dtype(fds_dtype)



def _read_particle_file(filename, output_each=1, classes=None, ntimesteps=None, logs=True):
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
  if ntimesteps is not None:
    assert ntimesteps > 0


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

  info = {"filename" : filename}

  # Read endianess flag written by FDS which will be ignored
  _read_from_buffer(_get_fds_dtype("int"))

  # Read FDS version identifier
  info["fdsversion"] = _read_from_buffer(_get_fds_dtype("int"))[0]

  # Read number of classes
  info["nclasses"] = _read_from_buffer(_get_fds_dtype("int"))[0]
  # If user did not provide any class identifiers, consider all by default
  if classes is None:
    classes = list(range(info["nclasses"]))

  # Read quantitiy properties
  # Create empty lists for each particle class
  info["nquantities"] = [None for _ in range(info["nclasses"])]
  info["qlabels"] = [[] for _ in range(info["nclasses"])]
  info["qunits"] = [[] for _ in range(info["nclasses"])]

  # Loop over all classes and parse their individual quantity labels
  # and units
  for ic in range(info["nclasses"]):
    # Read number of quantities for current  class and skip a placeholder
    info["nquantities"][ic] = _read_from_buffer(_get_fds_dtype("int", count=2))[0]

    # Read particle quantities as character lists, add as strings to the
    # info lists
    for _ in range(info["nquantities"][ic]):
      qlabel = _read_from_buffer(_get_fds_dtype("char", count=30))
      info["qlabels"][ic].append(qlabel.decode())

      qunit = _read_from_buffer(_get_fds_dtype("char", count=30))
      info["qunits"][ic].append(qunit.decode())


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
    _get_fds_dtype("coordinates", count=0).itemsize +
    _get_fds_dtype("tags", count=0).itemsize +
    (_get_fds_dtype("quantities", count=0, quantities=info["nquantities"][ic]).itemsize
      if info["nquantities"][ic] > 0 else 0)
    for ic in range(info["nclasses"])]

  """
  zero_particles_size =                               \
    _get_fds_dtype("coordinates", count=0).itemsize + \
    _get_fds_dtype("tags", count=0).itemsize +        \
    [_get_fds_dtype("quantities", count=0, quantities=info["nquantities"][ic]).itemsize
     if info["nquantities"][ic] > 0 else 0 for ic in range(info["nclasses"])]
  """

  # Size of a timestep without any particles in all classes
  # Current timestep
  empty_timestep_size = _get_fds_dtype("float").itemsize
  for ic in range(info["nclasses"]):
    empty_timestep_size += (
      # Number of particles is zero
      _get_fds_dtype("int").itemsize +
      # Size that zero particles occupy
      zero_particles_size[ic]
    )


  # Create empty lists for each particle class
  times = []
  nparts = [[] for _ in range(info["nclasses"])]
  xps = [[] for _ in range(info["nclasses"])]
  tags = [[] for _ in range(info["nclasses"])]
  qs = [[[] for _ in range(info["nquantities"][ic])]
        for ic in range(info["nclasses"])]

  timestep = 0
  while file_mm_pos < file_mm.size():
    # If all remaining timesteps have no particles, we will skip this file
    if ntimesteps:
      ntimesteps_estimate_if_remaining_steps_empty = \
        timestep + 1 + (file_mm.size() - file_mm_pos) // empty_timestep_size
      if ntimesteps_estimate_if_remaining_steps_empty == ntimesteps:
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
    for ic in range(info["nclasses"]):
      # Decide whether we want to process this particle class
      skip_class = ic not in classes
      skip_current = skip_timestep or skip_class

      # Read number of particles
      npart = _read_from_buffer(_get_fds_dtype("int"))[0]

      # If no particles were found, skip this timestep
      if npart == 0:
        file_mm_pos += zero_particles_size[ic]

        # Store empty data if required
        if not skip_current:
          nparts[ic].append(0)
          xps[ic].append([np.array([]),
                          np.array([]),
                          np.array([])])
          tags[ic].append(np.array([]))
          for iq in range(info["nquantities"][ic]):
            qs[ic][iq].append(np.array([]))

        continue

      # Read coordinate lists
      raw_xyz = _read_from_buffer(_get_fds_dtype("coordinates",
                                  count=npart),
                                  skip=skip_current)
      # Read tags
      raw_tag = _read_from_buffer(_get_fds_dtype("tags",
                                  count=npart),
                                  skip=skip_current)
      # Read each quantity data, if there is any
      raw_qs = None
      if info["nquantities"][ic] > 0:
        raw_qs = _read_from_buffer(_get_fds_dtype("quantities",
                                   count=npart,
                                   quantities=info["nquantities"][ic]),
                                   skip=skip_current)

      # Store data if required
      if not skip_current:
        nparts[ic].append(npart)
        xps[ic].append([np.copy(raw_xyz[0]),
                        np.copy(raw_xyz[1]),
                        np.copy(raw_xyz[2])])
        tags[ic].append(np.copy(raw_tag))
        for iq in range(info["nquantities"][ic]):
          qs[ic][iq].append(np.copy(raw_qs[iq]))

    # Continue with next timestep
    timestep += 1

  assert file_mm_pos == file_mm.size()

  # Add number of timesteps and outputsteps to dict
  info["ntimesteps"] = timestep
  info["noutputsteps"] = len(times)

  if logs:
    data_size = file_mm.size()
    data_size_in_mb = data_size / (1024 ** 2)
    time_end = time()
    print("file size: {:.3f} MB, speed: {:.3f} MB/s".format(
          data_size_in_mb, data_size_in_mb / (time_end - time_start)))

  file_object.close()

  return info, times, nparts, xps, tags, qs



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
                                     output_each, classes, ntimesteps=None)
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
                     output_each=output_each, classes=classes, ntimesteps=info["ntimesteps"])
    results[1:] = pool.map(worker, filelist)
    pool.close()
    pool.join()
  else:
    for filename in filelist:
      results.append(
        _read_particle_file(filename,
                            output_each, classes, info["ntimesteps"]))

  if logs:
    print("Finished reading.")


  #
  # CONCATENATE: results of all files
  #

  # Calculate global number of particles for every particle class in
  # each timestep
  nparts = [[0] * info["noutputsteps"]] * info["nclasses"]
  for res in results:
    local_nparts = res[2]
    for ic in range(info["nclasses"]):
      local_noutputsteps = len(local_nparts[ic])
      for iout in range(local_noutputsteps):
        nparts[ic][iout] += local_nparts[ic][iout]

  # Prepare empty data containers one after another to avoid memory
  # fragmentation
  xps = [[[np.empty(nparts[ic][iout]),
           np.empty(nparts[ic][iout]),
           np.empty(nparts[ic][iout])]
           for iout in range(info["noutputsteps"])]
          for ic in range(info["nclasses"])]
  tags = [[np.empty(nparts[ic][iout])
           for iout in range(info["noutputsteps"])]
          for ic in range(info["nclasses"])]
  qs = [[[np.empty(nparts[ic][iout])
          for iout in range(info["noutputsteps"])]
         for _ in range(info["nquantities"][ic])]
        for ic in range(info["nclasses"])]

  # Offsets to build consecutive buffer
  offsets = [np.zeros(info["noutputsteps"], dtype="int")
             for _ in range(info["nclasses"])]

  # Attach data
  for res in results:
    _, _, local_nparts, local_xps, local_tags, local_qs = res

    for ic in range(info["nclasses"]):
      local_noutputsteps = len(local_nparts[ic])
      for iout in range(local_noutputsteps):
        o = offsets[ic][iout]
        n = local_nparts[ic][iout]
        xps[ic][iout][0][o:o+n] = local_xps[ic][iout][0]
        xps[ic][iout][1][o:o+n] = local_xps[ic][iout][1]
        xps[ic][iout][2][o:o+n] = local_xps[ic][iout][2]
        tags[ic][iout][o:o+n] = local_tags[ic][iout]
        for iq in range(info["nquantities"][ic]):
          qs[ic][iq][iout][o:o+n] = local_qs[ic][iq][iout]

        offsets[ic][iout] += n

  return info, times, nparts, xps, tags, qs



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
      - fdsversion : int
          Data has been generated with this FDS version
      - nclasses : int
          Number of particle classes from the FDS simulation
      - ntimesteps : int
          Number of total timesteps from the FDS simulation
      - noutputsteps : int
          Number of timesteps that have been read with fdspsp, i.e.,
          noutputsteps = ntimesteps // output_each + 1
      - nquantities : list[i1] -> int
          Number of quantities for particle class i1
      - qlabels : list[i1] -> str
          List of quantity labels for particle class i1
      - qunits : list[i1] -> str
          List of quantity units for particle class i1

  nparts : list[i1][i2] -> int
    Number of particles in class i1 at output step i2

  times : list[i1] -> float
    Time of output step i1

  xps : list[i1][i2][i3] -> np.array(float)
    xps holds coordinate lists (as numpy-arrays) for particle class i1
    at output step i2 for component (x,y,z) i3, where the length of the
    array is given by the number of particles of the selected class and
    step, i.e., nparts[i1][i2]

  tags : list[i1][i2] -> np.array(int)
    numpy-array of particle tags for class i1 and output step i2

  qs : list[i1][i2][i3] -> np.array(float)
    numpy-array of particle quantity i2 for class i1 and output step i3;
    the number of available quantities for the selected particle class
    is given in info["nquantities"][i1]
  """

  def __init__(self, filestem, fileextension="prt5", output_each=1, classes=None, logs=True, parallel=True):
    self.info, self.times, self.nparts, self.xps, self.tags, self.qs = \
      _read_multiple_particle_files(filestem, fileextension, output_each, classes, logs, parallel)
