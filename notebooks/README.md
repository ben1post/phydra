# Phydra plankton community models

Here the first models included within the library are presented in interactive Jupyter notebooks. 

The notebooks show all steps from creating the _model setup_ object to analyzing model output and provide a template for further exploration and experimentation with the provided plankton community models. More detailed descriptions of the methods are given the manuscript accompanying the release.

## Model applications

The following model applications are included in the Phydra library. All models are embedded in 0-dimensional physical settings.:

- A simple NP (Nutrient-Phytoplankton) model in a chemostat setting.
- A fully configured NPZD (Nutrient-Phytoplankton-Zooplankton-Detritus) model.
- A complex size-structured NPZ model in a simplified physical setting.


## The framework

Xarray-simlab-ODE extends [Xarray-simlab](https://xarray-simlab.readthedocs.io/en/latest/) to allow the modular construction, setup and execution of models based on ordinary differential equations, in an interactive workflow fully compatible with the Python scientific ecosystem through the [Xarray](https://docs.xarray.dev/en/stable/) data framework. You can read more about Xarray-simlab-ODE in the [XSO Documentation on ReadTheDocs](https://xarray-simlab-ode.readthedocs.io/en/latest/index.html).