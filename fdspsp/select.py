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
SELECT module:

Select subsets of particles qualifying for certain criteria.
"""


import numpy as np
import heapq


def absolute_threshold(data, threshold, indices=()):
  """
  Find indices in a dataset whose associated values are larger than a
  given absolute threshold.

  Parameters
  ----------
  data : numpy.array
    Any type of data, whether associated with particles or cells.
  threshold : float
    Absolute threshold with which the data set will be compared.
  indices : tuple -> np.array(int)
    Specify a selection of indices whose associated data this function
    will process only. The number of tuple components needs to be in
    accordance with the number of dimensions of the data array. By
    default, the full data set is considered.

  Returns
  -------
  indices_reduced : tuple -> np.array(int)
    Selection of indices whose associated data fulfills the criterion.
    The number of tuple components corresponds to the number of
    dimensions of the data array.
  """
  indices_reduced = tuple(np.array([], dtype=int)
                          for _ in range(data.ndim))
  if data.size:
    # Flatten array for convenience
    data_view = data.flatten()
    # Only operate on predefined section of data (if specified)
    indices_view = np.ravel_multi_index(indices, data.shape) \
        if indices else ()
    data_enumerate = enumerate(data_view[indices_view])
    # Identify indices belonging to data fulfilling threshold criterion
    indices_threshold = np.array(
        [i for i, e in data_enumerate if e >= threshold])
    # Indexing multi-dimensional arrays in numpy
    if len(indices_threshold):
      if len(indices_view):
        indices_threshold = indices_view[indices_threshold]
      indices_reduced = np.unravel_index(indices_threshold, data.shape)

  return indices_reduced


def relative_threshold(data, threshold, indices=()):
  """
  Find indices in a dataset whose associated values are larger than a
  given threshold relative to the maximum value.

  Parameters
  ----------
  data : numpy.array
    Any type of data, whether associated with particles or cells.
  threshold : float
    Fraction specifying the threshold relative to the maximum value of
    the data set.
  indices : tuple -> np.array(int)
    Specify a selection of indices whose associated data this function
    will process only. The number of tuple components needs to be in
    accordance with the number of dimensions of the data array. By
    default, the full data set is considered.

  Returns
  -------
  indices_reduced : tuple -> np.array(int)
    Selection of indices whose associated data fulfills the criterion.
    The number of tuple components corresponds to the number of
    dimensions of the data array.
  """
  assert 0 <= threshold <= 1

  indices_reduced = tuple(np.array([], dtype=int)
                          for _ in range(data.ndim))
  if data.size:
    abs_threshold = threshold * np.max(data[indices])
    indices_reduced = absolute_threshold(data, abs_threshold, indices)

  return indices_reduced


def top_percentile(data, top_share, indices=()):
  """
  Find indices in a dataset whose associated values correspond to a
  given top percentile.

  Parameters
  ----------
  data : numpy.array
    Any type of data, whether associated with particles or cells.
  top_share : float
    Fraction specifying the top percentile of the data set.
  indices : tuple -> np.array(int)
    Specify a selection of indices whose associated data this function
    will process only. The number of tuple components needs to be in
    accordance with the number of dimensions of the data array. By
    default, the full data set is considered.

  Returns
  -------
  indices_reduced : tuple -> np.array(int)
    Selection of indices whose associated data fulfills the criterion.
    The number of tuple components corresponds to the number of
    dimensions of the data array.
  """
  assert 0 <= top_share <= 1

  indices_reduced = tuple(np.array([], dtype=int)
                          for _ in range(data.ndim))
  ntop = int(top_share * data[indices].size)
  if ntop:
    # Flatten array for convenience
    data_view = data.flatten()
    # Only operate on predefined section of data (if specified)
    indices_view = np.ravel_multi_index(indices, data.shape) \
        if indices else ()
    data_enumerate = enumerate(data_view[indices_view])
    # Sort top fraction of all data by value in descending order
    data_top = heapq.nlargest(ntop, data_enumerate, key=lambda x: x[1])
    # Further add entries that have the same data
    data_top_min = data_top[-1][1]
    data_top += [e for e in data_enumerate if e[1] == data_top_min]
    # Extract corresponding indices
    indices_top, _ = zip(*data_top)
    indices_top = np.array(indices_top)
    # Indexing multi-dimensional arrays in numpy
    if len(indices_top):
      if len(indices_view):
        indices_top = indices_view[indices_top]
      indices_reduced = np.unravel_index(indices_top, data.shape)

  return indices_reduced
