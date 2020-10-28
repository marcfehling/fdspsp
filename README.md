# FDS particle spray postprocessor (fdspsp)

`fdspsp` offers tools to postprocess particle data generated with NIST's
[Fire Dynamics Simulator (FDS)](https://github.com/firemodels/fds).

- Read `prt5` binary files containing particle data in the FORTRAN
  format described in the [`FDS 6.7.1` user's guide, section 24.10
  particle data](https://github.com/firemodels/fds/releases/download/FDS6.7.1/FDS_User_Guide.pdf).
- Determine the spray front of particle clouds using algorithms
  described in the [`fdspsp` publication](https://www.google.com/).
  - *farthest fraction method (FF)*\
    The fraction of particles farthest from the reference point
    determine the spray front of the particle cloud. Their average
    distance from the reference describe the spray radius, and their
    average velocity define the spray front velocity.
  - *grid density method (GD)*\
    The domain is split into control volumes (cells), and the particle
    density within these volumes determine the extension of the particle
    cloud. With lower and upper thresholds, the particle cloud can be
    defined more precisely. The cells farthest from the reference point
    which were identified to be part of particle cloud determine the
    spray front. Their average distance from the reference describe the
    spray radius, and the average velocity of all particles within them
    define the spray front velocity.
  - *grid coverage method (GC)*\
    Similar to the grid density method, but instead of taking the
    particle density as a measure, the volume fraction that particles
    occupy within a cell are used. All particle volumes are considered
    and the overlap of particle volumes neglected.
- Visualize postprocessed particle data.

Currently, there is no intention to provide a wiki for this repository.
Please consult the documentation provided by `Docstrings` and study the
examples to get started.


## Idea

Catastrophe scenarios include the impact of fuel-loaded vessels into
critical structures, e.g., a plane crashing into a nuclear power plant.
The investigation of the thermal load from the ignited fuel is crucial
to assess the integrity of the building substance. In such scenarios,
the understanding of the dispersion behavior of fuel and its
simultaneous ignition forms the basis for this kind of assessment.

An attempt to investigate the dispersion of fluids after an impact has
been conducted in the *IMPACT2010* experiments at *VTT
Technical Research Centre of Finland Ltd* as part of the project
*SAFIR2010, The Finnish Research Programme on Nuclear Power Plant Safety
2007-2010*. Here, missiles with water tanks were fired at
reinforced concrete walls at high velocities. Typically, these missiles
contained ~25kg water, weighed ~50kg in total, and were
accelerated to ~100m/s. The dispersion of the water load after the
impact was captured with high-speed cameras. For details, consult
chapter 30 of the [final report](https://www.vttresearch.com/sites/default/files/pdf/tiedotteet/2011/T2571.pdf)
of the research project.

The idea of this project is to numerically investigate the thermal load
on structures resulting from an impact with the well-validated software
[Fire Dynamics Simulator (FDS)](https://github.com/firemodels/fds).
Impacts will be modeled with spray devices in FDS, and the ignition and
combustion of fuel with its fire models.

The dispersion of fluids and the characterization of particle clouds
will be validated using the results of the *IMPACT2010* experiments with
water particle sprays. The validated methods will be then applied on
fuel particle sprays to characterize the fuel cloud and determine the
thermal load on concrete walls.


## Installation

`fdspsp` requires [`python3`](https://www.python.org/) along with
[`numpy`](https://numpy.org/) for usage and [`setuptools`
](https://github.com/pypa/setuptools) during installation. We recommend
to work with particle data generated with a version of `FDS` more recent
than `>= 6.7.1`.

Install `fdspsp` by cloning this repository and installing it to your
`python3` distribution.
```
$ git clone https://github.com/FireDynamics/fdspsp
$ cd fdspsp
$ python setup.py install
```

You can then import it into your `python` scripts as a module
```
import fdspsp
```
or
```
from fdspsp import *
```


## Testing

Testing `fdspsp` requires particle data from `FDS` simulations. For this
purpose, each test script in `fdspsp` comes with a corresponding `FDS`
input file.

First, run all relevant `FDS` simulations by calling the following shell
script from the project directory like this:
```
$ ./tests/run_all_fds_cases.sh /path/to/fds_executable
```

Finally, run the testsuite by simply calling in the project directory:
```
$ pytest
```


## Contributing

As this is an open source project, we welcome any kind of contribution!
Before submitting a pull request, make sure you ran the testsuite
successfully and your code conforms to the [`PEP 8`
](https://www.python.org/dev/peps/pep-0008/) code style.

For code formatting, we use the automatic tool [`autopep8`
](https://github.com/hhatto/autopep8). Please prepare the files you want
to submit with the following configuration:
```
$ autopep8 --indent-size=2 --in-place /path/to/file
```


## Publications

This toolbox has been developed to characterize water sprays and fire
balls in the following literature. Please cite the former publication if
you use `fdspsp`.

```
@article{fdspsp2020,
  author  = {Marc Fehling and Michael Krampf and Lukas Arnold},
  title   = {? Characterizing the liftoff phase of impact-induced
             fireballs with particle spray simulations},
  journal = {? Fire Safety Journal},
  volume  = {?},
  issue   = {?},
  year    = {? 2021},
  pages   = {?},
  doi     = {?}
}

@mastersthesis{krampf2018,
  author  = {Michael Krampf},
  title   = {{C}harakterisierung der {L}iftoff {P}hase partikelbasierter
             {F}euerb{\"a}lle},
  school  = {Univ. Wuppertal},
  pages   = {69 p},
  year    = {2018},
  url     = {https://juser.fz-juelich.de/record/865224}
}
```
