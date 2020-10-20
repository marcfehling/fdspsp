#
# Copyright (c) 2020 by the FireDynamics authors
#
# This file is part of the FDS particle spray postprocessor (fdspsp).
#
# fdspsp is free software; you can use it, redistribute it, and/or
# modify it under the terms of the MIT License. The full text of the
# license can be found in the file LICENSE.md at the top level
# directory of fdspsp.
#


import setuptools


with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="fdspsp",
  version="1.0.0",
  author="FireDynamics",
  author_email="firesim@fz-juelich.de",
  description="Tools to postprocess FDS particle data",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/FireDynamics/fdspsp",
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  python_requires=">=3.8",
  install_requires=["numpy"]
)
