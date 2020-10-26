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
REDUCE module

Merge datasets to a single value.
"""


import numpy as np


def mean(data, indices):
  """
  Average a certain selection of data from a data set.

  Parameters
  ----------
  data : numpy.array
    Any type of data, whether associated with particles or cells.
  indices : tuple -> np.array(int)
    Specify a selection of indices whose associated data this function
    will process only. The number of tuple components needs to be in
    accordance with the number of array dimensions.

  Returns
  -------
  mean : float
    Arithmetic mean of the provided data. Zero if there is no data to
    average.
  """
  # Flatten array for convenience
  data_view = data.flatten()
  indices_view = np.ravel_multi_index(indices, data.shape)

  return np.mean(data_view[indices_view]) \
      if indices_view.size > 0 else None


def sum(data, indices):
  """
  Sum a certain selection of data from a data set.

  Parameters
  ----------
  data : numpy.array
    Any type of data, whether associated with particles or cells.
  indices : tuple -> np.array(int)
    Specify a selection of indices whose associated data this function
    will process only. The number of tuple components needs to be in
    accordance with the number of array dimensions.

  Returns
  -------
  sum : float
    Sum of the provided data. Zero if there is no data to sum.
  """
  # Flatten array for convenience
  data_view = data.flatten()
  indices_view = np.ravel_multi_index(indices, data.shape)

  return np.sum(data_view[indices_view]) \
      if indices_view.size > 0 else None
