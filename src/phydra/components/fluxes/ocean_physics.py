import xso


@xso.component
class SlabSinking:
    """ """
    var = xso.variable(foreign=True, flux='sinking', negative=True)
    mld = xso.forcing(foreign=True)
    rate = xso.parameter(description='sinking rate, units: m d^-1')

    @xso.flux
    def sinking(self, var, rate, mld):
        return var * rate / mld


@xso.component
class SlabUpwelling:
    """ """
    n = xso.variable(foreign=True, flux='mixing', description='nutrient mixed into system')
    n_0 = xso.forcing(foreign=True, description='nutrient concentration below mixed layer depth')
    mld = xso.forcing(foreign=True)
    mld_deriv = xso.forcing(foreign=True)

    kappa = xso.parameter(description='constant mixing coefficient')

    @xso.flux
    def mixing(self, n, n_0, mld, mld_deriv, kappa):
        """ componentute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return (n_0 - n) * (self.m.max(mld_deriv, 0) + kappa) / mld


@xso.component
class SlabMixing:
    """ """
    vars_sink = xso.variable(foreign=True, negative=True, flux='mixing',
                                list_input=True, dims='sinking_vars', description='list of variables affected')

    mld = xso.forcing(foreign=True)
    mld_deriv = xso.forcing(foreign=True)

    kappa = xso.parameter(description='constant mixing coefficient')

    @xso.flux(dims='sinking_vars_full')
    def mixing(self, vars_sink, mld, mld_deriv, kappa):
        """ componentute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return vars_sink * (self.m.max(mld_deriv, 0) + kappa) / mld


@xso.component
class Mixing_K:
    """ pre-componentutes K to be used in all mixing processes """
    mld = xso.forcing(foreign=True)
    mld_deriv = xso.forcing(foreign=True)

    kappa = xso.parameter(description='constant mixing coefficient')

    @xso.flux(group='mixing_K')
    def mixing(self, mld, mld_deriv, kappa):
        return (self.m.max(mld_deriv, 0) + kappa) / mld


@xso.component
class SlabUpwelling_KfromGroup:
    """ """
    n = xso.variable(foreign=True, flux='mixing', description='nutrient mixed into system')
    n_0 = xso.forcing(foreign=True, description='nutrient concentration below mixed layer depth')

    @xso.flux(group_to_arg='mixing_K')
    def mixing(self, n, n_0, mixing_K):
        """ componentute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return (n_0 - n) * mixing_K


@xso.component
class SlabMixing_KfromGroup:
    """ """
    vars_sink = xso.variable(foreign=True, negative=True, flux='mixing',
                                list_input=True, dims='sinking_vars', description='list of variables affected')

    @xso.flux(dims='sinking_vars_full', group_to_arg='mixing_K')
    def mixing(self, vars_sink, mixing_K):
        """ componentute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return vars_sink * mixing_K
