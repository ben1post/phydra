import phydra
import numpy as np


@phydra.comp
class SlabSinking:
    """ """
    var = phydra.variable(foreign=True, flux='sinking', negative=True)
    mld = phydra.forcing(foreign=True)
    rate = phydra.parameter(description='sinking rate, units: m d^-1')

    @phydra.flux
    def sinking(self, var, rate, mld):
        return var * rate / mld


@phydra.comp
class SlabUpwelling:
    """ """
    n = phydra.variable(foreign=True, flux='mixing', description='nutrient mixed into system')
    n_0 = phydra.forcing(foreign=True, description='nutrient concentration below mixed layer depth')
    mld = phydra.forcing(foreign=True)
    mld_deriv = phydra.forcing(foreign=True)

    kappa = phydra.parameter(description='constant mixing coefficient')

    @phydra.flux
    def mixing(self, n, n_0, mld, mld_deriv, kappa):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return (n_0 - n) * (self.m.max(mld_deriv, 0) + kappa) / mld


@phydra.comp
class SlabMixing:
    """ """
    vars_sink = phydra.variable(foreign=True, negative=True, flux='mixing',
                                list_input=True, dims='sinking_vars', description='list of variables affected')

    mld = phydra.forcing(foreign=True)
    mld_deriv = phydra.forcing(foreign=True)

    kappa = phydra.parameter(description='constant mixing coefficient')

    @phydra.flux(dims='sinking_vars_full')
    def mixing(self, vars_sink, mld, mld_deriv, kappa):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return vars_sink * (self.m.max(mld_deriv, 0) + kappa) / mld
