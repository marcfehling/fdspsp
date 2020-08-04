import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="fdspsp",
  version="1.0.0",
  author="marcfehling",
  author_email="mafehling@gmail.com",
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
  python_requires='>=3.8',
)
