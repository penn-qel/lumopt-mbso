## Introduction

This repository contains an expansion to the [Lumopt](https://lumopt.readthedocs.io/en/latest/) repository for adjoint optimizations using Lumerical. The main new features of this repository are free-space figures of merit and parameterizations for many-body shape optimizations on arrays of simple objects.

This repository is released under a MIT license. If you use this repository in published work, please cite: [publication pending]. As much of this work is based on Lumopt, consider citing [their work](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-21-18-21693) as well. 

## Dependencies

The following dependencies are needed to run this code. Version numbers are those used as of the latest commit. Newer versions of dependencies should generally be functional but are untested. Lumopt itself is also required, but it is assumed to be included with your install of Lumerical.

<ul>
  <li>Lumerical FDTD 2023 R1.1 </li>
  <li>Python 3.9.18</li>
  <li>Numpy 1.19.5</li>
  <li>Scipy 1.10.1</li>
  <li>Matplotlib 3.3.4</li>
  <li>NLOpt 2.7.1</li>
</ul>

## Installation

Clone this repository to a convenient spot. Install Python and then (pip) install numpy, scipy, matplotlib, and NLopt. If you use conda, I recommend creating an environment with the specified versions. Add this repository to your Python path along with the Lumopt repository, which should be included with your Lumerical distribution. On Linux, lumopt is located by default in `/opt/lumerical/v231/api/python/lumopt`, where "v231" is the current version as of writing and changes with each release.

Alternatively, you can also use the Python distribution bundled with Lumerical (which can be run within the Lumerical application). This contains numpy, scipy, and matplotlib pre-installed but does *not* include NLopt. If you are not using the NLopt wrapper in <code>lumopt_mbso.optimizer.nlopt_optimizer.py</code> then you can get by without this dependency. The versions of numpy, scipy, and matplotlib listed in the dependencies are otherwise chosen to match those shipped with the current Lumerical distribution.

## Usage

This package is built on top of the Lumopt package, which is now shipped with Lumerical distributions. If you are unfamiliar with it, you should start [here](https://lumopt.readthedocs.io/en/latest/index.html). Lumopt has an object-oriented approach. The core functionality is in an object called Optimization that takes in Geometry, Figure of Merit, and Optimizer objects. Our package contains alternative implementations of each of these latter three that can be used interchangeably with those included with Lumopt. More details on each of these are below, and further usage information is available in the Python docstrings for each class.

### Geometries

The geometry objects are responsible for interfacing the geometrical parameters with the FDTD simulation and also for calculating gradients given forward and adjoint fields. This repository implements the following:

<ul>
  <li>EllipseMBSO: Creates an array of elliptical objects with free parameters of axial lengths, position, and rotation. Additionally uses EllipseConstraints object in constraints.py to define constraints on element spacing</li>
  <li>EllipseMBSOSidewall: Same as above, but allows for a non-vertical sidewall angle to be set. Not set up for using in optimization currently but useful for characterization</li>
  <li>HexMBSO: Defines array of regular hexagons defined by position and radius. Uses HexConstraints to define constraints on element spacing</li>
</ul>

### Figures of Merit

FOM objects are reponsible for calculating the FOM and creating the corresponding adjoint source. This repository includes:

<ul>
  <li>CustomModeMatch: Field overlap with arbitrary user-input mode functions. custom_modes.py contains some standard options.</li>
  <li>TransmissionFom: Total Poynting vector integrated over a region in physical space. Allows use-defined boundary function to define the region, with built-in option for a spot at (0,0) with a given radius</li>
  <li>KTransmissionFom: Same as above, but additionally can define an arbitrary region in k-space or a target collection NA</li>
  <li>KTransmissonCustomAdj: A hybrid of KTransmissionFom and CustomModeMatch that calculates both internally. Returns to the optimizer the FOM from KTransmissionFom but uses CustomModeMatch result to create adjoint source. Often results in more well-defined adjoint sources</li>
</ul>

### Optimizers

The Optimizer object is responsible for directly wrapping onto the optimization algorithm. This repository includes:
<ul>
  <li>ConstrainedOptimizer: Extension of Lumopt's ScipyOptimizers that allows choice of algorithm in scipy.optimize but additionally allows input of constraint functions</li>
  <li>NLoptOptimizer (Recommended): Wrapper for NLOpt optimization algorithms</li>
</ul>

### Misc

Various helper functions and analysis functions and scripts can be found in the utils folder. myoptimization.py in the main folder can be used instead of Lumopt's optimization.py. It inherits almost entirely with some minor changes for specific use cases. Examples can be found in the examples folder.

Useful functions and scripts for analysis of finished optimizations can be found in <code>lumopt_mbso.utils.analysis.py</code> and in <code>lumopt_mbso/utils/analysis_scripts/</code>.
