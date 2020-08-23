#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, mmap, time
import multiprocessing
from functools import partial
import glob

import numpy as np


def readParticleData(filename, output_each=1, classes=None, ntimesteps=None):
  """
  Function to parse and read in all particle data written by FDS into a prt5-file.

  This function reads in data in the FORTRAN format described in the FDS user guide, section 21.10.

  Parameters
  ----------
  filename : str
    The name of the file containing the particle data.
  output_each : int > 0
    Each n-th time step will actually be read, those inbetween will be skipped.
    By default, every time step will be considered.
  classes : list -> int >= 0
    Specify identifiers of classes that will be read.
    By default, every particle class will be considered.

  Returns
  -------
  (info, nparts, times, xs, ys, zs, tags, qs) with:

  info : dict
    Contains meta information about the dataset, as:
      - filename : str
        the read filename
      - fds_version : int
        FDS version identifier
      - n_classes : int
        number of particle classes in simulation
      - n_timesteps : int
        number of timesteps from the simulation
      - n_outputsteps : int
        number of timesteps from data was processed
      - n_quantities : list[i1] -> int
        number of quantities for particle class i1
      - q_labels : list[i1] -> str
        list of quantity label for particle class i1
      - q_units : list[i1] -> str
        list of quantity units for particle class i1

  nparts : list[i1][i2] -> int
    number of particles in class i1 at output step i2

  times : list[i1] -> float
    time of output step i1

  xp : list[i1][i2][i3] -> np.array(float)
    xp holds coordinate lists (as numpy-arrays) for particle class i1 at output step i2 for component (x,y,z) i3,
    where the length of the array given by the number of particles of the selected class and step (i.e. nparts[i1][i2])

  tags : list[i1][i2] -> np.array(int)
    numpy-array of particle tags for class i1 and output step i2

  qs : list[i1][i2][i3] -> np.array(float)
    numpy-array of particle quantity i2 for class i1 and output step i3, the number of available quantitis for
    the selected particle class is given in info['n_quantities'][i1]
  """

  def getParticleType(name: str, n_size=1, n_qs=0):
    """
    Helper function to construct the data types needed for the numpy function fromfile.
    """

    # as the binary representation of raw data is compiler dependent, this information must be provided by the user
    fds_data_type_integer = "i4"  # i4 -> 32 bit integer (native endiannes, probably little-endian)
    fds_data_type_float = "f4"  # f4 -> 32 bit floating point (native endiannes, probably little-endian)
    fds_data_type_char = "a"  # a  ->  8 bit character
    fds_fortran_backward = True  # sets weather the blocks are ended with the size of the block

    if name == 'int':
      type_particle_int_str = "{0}, ({1}){2}".format(fds_data_type_integer, n_size, fds_data_type_integer)
      if fds_fortran_backward: type_particle_int_str += ", {0}".format(fds_data_type_integer)
      return np.dtype(type_particle_int_str)

    if name == 'char':
      type_particle_char_str = "{0}, ({1}){2}".format(fds_data_type_integer, n_size, fds_data_type_char)
      if fds_fortran_backward: type_particle_char_str += ", {0}".format(fds_data_type_integer)
      return np.dtype(type_particle_char_str)

    if name == 'time':
      type_slice_time_str = "{0}, {1}".format(fds_data_type_integer, fds_data_type_float)
      if fds_fortran_backward: type_slice_time_str += ", {0}".format(fds_data_type_integer)
      return np.dtype(type_slice_time_str)

    if name == 'coordinates':
      type_slice_data_str = "{0}, (3, {1}){2}".format(fds_data_type_integer, n_size, fds_data_type_float)
      if fds_fortran_backward: type_slice_data_str += ", {0}".format(fds_data_type_integer)
      return np.dtype(type_slice_data_str)

    if name == 'tags':
      type_slice_data_str = "{0}, ({1}){2}".format(fds_data_type_integer, n_size, fds_data_type_integer)
      if fds_fortran_backward: type_slice_data_str += ", {0}".format(fds_data_type_integer)
      return np.dtype(type_slice_data_str)

    if name == 'quantities':
      type_slice_data_str = "{0}, ({1},{2}){3}".format(fds_data_type_integer, n_qs, n_size, fds_data_type_float)
      if fds_fortran_backward: type_slice_data_str += ", {0}".format(fds_data_type_integer)
      return np.dtype(type_slice_data_str)

    return None


  def fromfileConditional(infile, dtype, skip=False):
    """
    Wrapper function to decide whether we want to read or skip data from the input filestream.
    """
    nonlocal infile_pos

    if skip:
      # proceed the input filestream by the corresponding amount of bytes
      infile_pos += dtype.itemsize
      return None
    else:
      # read and return data
      res = np.frombuffer(infile, dtype, count=1, offset=infile_pos)
      if np.isscalar(res[0][1]):
        res = [[res[0][0], np.array([res[0][1]]), res[0][2]]]
      infile_pos += dtype.itemsize
      return res

  t_start = time.time()
  print("Reading particle file '", filename, "'.", sep='')

  assert output_each > 0
  if classes is not None:
    assert all(classid >= 0 for classid in classes)

  # open prt5 file
  infile_raw = open(filename, 'rb')
  if infile_raw == None:
    print("could not open file: ", filename)
    sys.exit()

  # map file content to memory
  infile = mmap.mmap(infile_raw.fileno(), 0, access=mmap.ACCESS_READ)
  infile.flush()

  infile_pos = 0

  # create empty info dictionary
  info = {}
  info['filename'] = filename

  # read endianess flag written by FDS, this script uses the formatting prescribed in the helper function
  # getParticleType, i.e. the following data is ignored
  raw_end_flag = fromfileConditional(infile, dtype=getParticleType('int'))

  # read FDS version identifier
  raw_fds_version = fromfileConditional(infile, dtype=getParticleType('int'))
  info['fds_version'] = raw_fds_version[0][1]

  # read number of classes
  raw_n_classes = fromfileConditional(infile, dtype=getParticleType('int'))
  n_classes = raw_n_classes[0][1][0]
  info['n_classes'] = n_classes

  # if user did not provide any class ids, consider all classes by default
  if classes is None:
    classes = list(range(n_classes))

  # initial lists for number of quantities per class and their labels and units
  info['n_quantities'] = []
  info['q_labels'] = []
  info['q_units'] = []

  # loop over all classes and parse their individual quantities labels and units
  for ic in range(n_classes):
    # read number of quantities for current particle class
    raw_n_quantities = fromfileConditional(infile, dtype=getParticleType('int', n_size=2))
    n_quantities = raw_n_quantities[0][1][0]
    info['n_quantities'].append(n_quantities)

    # read particle quantities, add info to the lists
    for nq in range(n_quantities):
      raw_q_label = fromfileConditional(infile, dtype=getParticleType('char', n_size=30))
      raw_u_label = fromfileConditional(infile, dtype=getParticleType('char', n_size=30))
      info['q_labels'].append(raw_q_label[0][1])
      info['q_units'].append(raw_u_label[0][1])

  # top level lists
  times = []
  # create empty lists for each particle class
  nparts = [[] for _ in range(n_classes)]
  xp = [[] for _ in range(n_classes)]
  tags = [[] for _ in range(n_classes)]
  # create empty lists for each quantity of each class
  qs = [[[] for _ in range(info['n_quantities'][ic])] for ic in range(n_classes)]

  # compute skip sizes
  no_part_skip = getParticleType('coordinates', n_size=0).itemsize + \
           getParticleType('tags', n_size=0).itemsize + \
           getParticleType('quantities', n_size=0, n_qs=info['n_quantities'][0]).itemsize

  ts_est_skip = getParticleType('time').itemsize + n_classes * (getParticleType('int').itemsize + no_part_skip)

  # loops, until no data can be read
  ts = 0
  while infile_pos < infile.size():

    # read time of current output step
    raw_time = fromfileConditional(infile, dtype=getParticleType('time'))

    ts_final_est = ts + ((infile.size() - infile_pos) // ts_est_skip) + 1
    if ntimesteps and ts_final_est == ntimesteps:
      infile_pos = infile.size()
      print("no further particles, skip remaining file content")
      break

    # decide whether we want to process this timestep
    skip_timestep = (ts % output_each) > 0

    # read data for each particle class
    for ic in range(n_classes):
      # decide whether we want to process this particle class
      skip_class = ic not in classes
      skip_current = skip_timestep or skip_class

      # read number of particles
      raw_n_part = fromfileConditional(infile, dtype=getParticleType('int'))
      n_part = raw_n_part[0][1][0]

      if n_part == 0 or skip_current:
        infile_pos += no_part_skip

        if n_part == 0:
          nparts[ic].append(0)
          xp[ic].append([np.array([]), np.array([]), np.array([])])
          tags[ic].append(np.array([]))
          for iq in range(info['n_quantities'][ic]):
            qs[ic][iq].append(np.array([]))

        continue

      # read coordinate lists
      raw_xyz = fromfileConditional(infile, dtype=getParticleType('coordinates', n_size=n_part), skip=skip_current)

      # read tags
      raw_tag = fromfileConditional(infile, dtype=getParticleType('tags', n_size=n_part), skip=skip_current)

      # read each quantity data, if there is any
      raw_qs = None
      if info['n_quantities'][ic] > 0:
        raw_qs = fromfileConditional(infile, dtype=getParticleType('quantities', n_size=n_part, n_qs=info['n_quantities'][ic]), skip=skip_current)

      # store all read data in the constructed lists
      if not skip_current:
        nparts[ic].append(n_part)
        xp[ic].append([np.copy(raw_xyz[0][1][0]), np.copy(raw_xyz[0][1][1]), np.copy(raw_xyz[0][1][2])])
        tags[ic].append(np.copy(raw_tag[0][1]))
        for iq in range(info['n_quantities'][ic]):
          qs[ic][iq].append(np.copy(raw_qs[0][1][iq]))

    if not skip_timestep:
      times.append(raw_time[0][1][0])

    # continue with next timestep
    ts += 1

  assert infile_pos == infile.size()

  # add number of timesteps to dict
  info['n_timesteps'] = ts
  info['n_outputsteps'] = len(times)

  data_size = infile.size() / (1024 ** 2)

  infile_raw.close()

  t_end = time.time()

  print('file size: {:.3f} MB'.format(data_size), 'speed: {:.3f} MB/s'.format(data_size / (t_end - t_start)))

  # return the filled lists
  return info, nparts, times, xp, tags, qs


def readMultipleParticleData(filestem, fileextension, output_each=1, classes=None, parallel=True):

  filelist = sorted(glob.glob(filestem + "*." + fileextension))
  assert len(filelist) > 0

  # Read very first file to know the total number of timesteps
  result_first = readParticleData(filelist.pop(0), output_each, classes, ntimesteps=None)
  info = result_first[0]
  ntimesteps = info['n_timesteps']

  # Read remaining input files
  if parallel:
    pool = multiprocessing.Pool(None)
    worker = partial(worker_readpd, output_each=output_each, classes=classes, ntimesteps=ntimesteps)
    results = pool.map(worker, filelist)
    pool.close()
    pool.join()
  else:
    results = []
    for file in filelist:
      results.append(readParticleData(file, output_each, classes, ntimesteps))

  # Append results of first file to list
  results.insert(0, result_first)

  print('finished reading')

  nparts = [[0] * info['n_outputsteps']] * info['n_classes']
  for res in results:
    tmp_nparts = res[1]
    for ic in range(info['n_classes']):
      for iout in range(len(tmp_nparts[ic])):
        nparts[ic][iout] += tmp_nparts[ic][iout]

  times = result_first[2]

  # prepare empty data containers one after another to avoid memory fragmentation
  xp = [[[np.empty(nparts[ic][iout]),
          np.empty(nparts[ic][iout]),
          np.empty(nparts[ic][iout])]
          for iout in range(info['n_outputsteps'])]
         for ic in range(info['n_classes'])]
  tags = [[np.empty(nparts[ic][iout])
           for iout in range(info['n_outputsteps'])]
          for ic in range(info['n_classes'])]
  qs = [[[np.empty(nparts[ic][iout])
          for iout in range(info['n_outputsteps'])]
         for _ in range(info['n_quantities'][ic])]
        for ic in range(info['n_classes'])]

  # another container?
  npos = [np.zeros(info['n_outputsteps'], dtype='int')
          for _ in range(info['n_classes'])]

  # attach data
  for res in results:

    tmp_info, tmp_nparts, tmp_times, tmp_xp, tmp_tags, tmp_qs = res

    for ic in range(info['n_classes']):
      for iout in range(len(tmp_nparts[ic])):

        p = npos[ic][iout]
        n = tmp_nparts[ic][iout]
        xp[ic][iout][0][p:p+n] = tmp_xp[ic][iout][0]
        xp[ic][iout][1][p:p+n] = tmp_xp[ic][iout][1]
        xp[ic][iout][2][p:p+n] = tmp_xp[ic][iout][2]
        tags[ic][iout][p:p+n] = tmp_tags[ic][iout]

        for iq in range(info['n_quantities'][ic]):
          qs[ic][iq][iout][p:p+n] = tmp_qs[ic][iq][iout]

        npos[ic][iout] += n

  return info, nparts, times, xp, tags, qs


def worker_readpd(filename, output_each, classes, ntimesteps):
  res = readParticleData(filename, output_each, classes, ntimesteps)
  return res


def printParticleInfo(info):
  """
  Nice output of the particle info dictionary.
  """
  return None



class ParticleData:
  """
  Class wrapper that reads particles and stores them.

  Parameters
  ----------
  filestem : str
  fileextension : str
    Those files will be read who start with the filestem parameter and end with the fileextenstion.
  output_each : int
    Each n-th timestep will actually be read, those inbetween will be skipped.
  classes : list -> int > 0
    Specify identifiers of classes that will be read.
    By default, every particle class will be considered.
  """
  def __init__(self, filestem, fileextension="prt5", output_each=1, classes=None):
    # read particle data
    info, nparts, times, xp, tags, qs = readMultipleParticleData(filestem, fileextension, output_each, classes)

    # assign class members
    self.info = info
    self.nparts = nparts
    self.times = times
    self.xp = xp
    self.tags = tags
    self.qs = qs



if __name__ == "__main__":
  """
  Provide path to 'prt5' stem as command line argument to start a test evaluation.
  """
  import matplotlib.pyplot as plt
  import matplotlib.animation as animation
  from mpl_toolkits.mplot3d import Axes3D

  # Read particle data
  particle_data = ParticleData(sys.argv[1])
  particle_class = 0

  # Initialize scatter plot with data from last output step
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  s1 = ax.scatter(particle_data.xp[particle_class][-1][0],
          particle_data.xp[particle_class][-1][1],
          particle_data.xp[particle_class][-1][2])

  # Set boundaries of scatter plot
  xmax = -1.0e100
  ymax = -1.0e100
  zmax = -1.0e100
  xmin = +1.0e100
  ymin = +1.0e100
  zmin = +1.0e100
  for c in range(particle_data.info['n_classes']):
    for t in range(len(particle_data.times)):
      if particle_data.nparts[c][t] > 0:
        xmax = max(xmax, np.amax(particle_data.xp[c][t][0]))
        ymax = max(ymax, np.amax(particle_data.xp[c][t][1]))
        zmax = max(zmax, np.amax(particle_data.xp[c][t][2]))
        xmin = min(xmin, np.amin(particle_data.xp[c][t][0]))
        ymin = min(ymin, np.amin(particle_data.xp[c][t][1]))
        zmin = min(zmin, np.amin(particle_data.xp[c][t][2]))

  ax.set_xlim3d([xmin, xmax])
  ax.set_ylim3d([ymin, ymax])
  ax.set_zlim3d([zmin, zmax])

  # Animate read particle
  it = 0
  def updatefig(*args):
    global s1, it, xs, ys
    ax.clear()
    it += 1
    if it >= len(particle_data.times): it = 0
    ax.scatter(particle_data.xp[0][it][0],
           particle_data.xp[0][it][1],
           particle_data.xp[0][it][2])
    ax.set_xlim3d([xmin, xmax])
    ax.set_ylim3d([ymin, ymax])
    ax.set_zlim3d([zmin, zmax])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.draw()
    return s1,

  ani1 = animation.FuncAnimation(fig, updatefig, interval=50, blit=False)
  plt.show()