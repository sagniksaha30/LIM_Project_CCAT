# lim

lim is a python application designed to analytically compute various statistics of line intensity maps using a wide variety of models.  It also contains functions to generate simulated intensity maps from peak-patch simulations provided by George Stein.  This code is a work in progress, so it may change significantly and there may be undetected bugs.

### Note:
This code was updated by Sagnik Saha so that it can be used
to make predictions for CCAT. The code required a major modification, particularly in the line_obs.py file, in order to account for the units in which the instrument specifications are
provided. The reference paper is :

arXiv:2208.10634 [astro-ph.IM]

### Prerequisites

lim requires several packages which should be familiar to astronomers working with python, including numpy, scipy, and astropy.  It also makes substantial use of Francisco Villaescusa-Navarro's [pylians](https://github.com/franciscovillaescusa/Pylians) package, which can be download from github and installed along with its dependencies with the commands

```
cd library
python setup.py build
```

Astropy units are used throughout this code to avoid unit conversion errors. To use the output of lim in any code which does not accept astropy units, simply replace output x with x.value.

Using the simulation functionality requires peak-patch catalogs.  One example catalog is included here, more can be obtained from George Stein (github.com/georgestein)

Finally, lim uses the python camb wrapper to compute all needed cosmological quantities.

### Quickstart

In the folder containing the lim functions, you can quickly get the default CO power spectrum by running in an interpreter

```
from lim import lim
m = lim()
m.Pk
```

Models are defined by dictionary objects in the params.py file.  The 'default_par' set of parameters is used by default, but any other set can be used by changing the 'model_params' input of lim().  For example, params.py contains another dict which defines parameters for the Li et al. (2015) CO emission model and the COMAP Phase I observation, which can be called with

```
m = lim(model_params='TonyLi_PhI')
```

You can also set parameters directly with a dictionary, for example

```
from params import TonyLi_PhI
m = lim(model_params = TonyLi_PhI)
```

All modules in lim use an update(), which allows parameter values to be changed after creating the model.  Most outputs are created as @cached_properties, which will update themselves using the new value after update() is called.  For example, to change the observing frequency of a survey you could run

```
m = lim()
m.update(nuObs=15*u.GHz)

```

The update() method is somewhat optimized, in that it will only rerun functions if required.  This speeds up update()'s which only change the line emission physics without altering the cosmology.

There is also a reset() method which will reset all input parameters back to the values they had when lim() was originally called.

### Examples

An ipython notebook fully commented is provided as an example. Following this notebook (LIM_PkFisher.ipynb) will get you familiar with the code (especially with the computation of the power spectrum multipoles and the corresponding covariance), and will allow you to reproduce the results appearing on the papers: arXiv:1907.10065 and arXiv:1907.10067. These papers (and the example), focus on the use of the multipoles of the LIM power spectrum to extract robust and optimal cosmological information, marginalizing over astrophysical uncertainties.

### Modules

The lim.lim() function reads a dict of parameters and creates an object which computes desired quantities from those parameters.  The object created can come from one of several modules, depending on the other inputs to lim().  The base class is the line_model.LineModel() class, which models a signal on the sky independent of survey design.  This object can output power spectra and VID's for a desired model.  If doObs=True in lim(), the line_obs.LineObs() class is used, which is a subclass of LineModel that adds in functionality related to instrumental characteristics, such as noise curves.  If doSim=True, the limlam.LimLam() class is used, which further adds the ability to generate simulated maps and compute statistics from them.

### Line Emission Models

Models for line emission physics are defined in one of two ways: either with a formula for the luminosity function dn/dL or by a mass/luminosity relation L(M).  Which is used is set by the 'model_type' input, which is 'LF' for the former and 'ML' for the latter.  Specific models are defined in the luminosity_functions.py and mass_luminosity.py files respectively.  The model_name input should be a string containing the name of a function in one of these two files, and the model_par should be a dict of that model's parameters.  Custom models can easily be added by adding additional functions to the relevant file.

## DocTests

To quickly check several of the parameters, lim includes doctests.  In a terminal, simply run

```
python lim.py
```

Note that the expected power spectra for the doctests were computed assuming the camb module is installed.  If you do not have camb installed, i.e. if

```
import camb
```
gives an error, hmf will use the EH transfer function and the doctests may fail.


## Usage

When used, please refer to the github page and cite arXiv:1907.10067


## Authors

* **Patrick C. Breysse**
* **José Luis Bernal**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Code based on matlab routines originally developed with Ely Kovetz
* LimLam simulation code adapted from limlam_mocker code written by George Stein and Dongwoo Chung
