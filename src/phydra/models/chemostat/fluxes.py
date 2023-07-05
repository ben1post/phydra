import xso


@xso.component
class LinearForcingInput:
    var = xso.variable(foreign=True, flux='input', negative=False, description='variable affected by flux')
    forcing = xso.forcing(foreign=True, description='forcing affecting flux')
    rate = xso.parameter(description='linear rate of change')

    @xso.flux
    def input(self, var, forcing, rate):
        """ """
        return forcing * rate


@xso.component
class MonodGrowth:
    resource = xso.variable(foreign=True, flux='uptake', negative=True)
    consumer = xso.variable(foreign=True, flux='uptake', negative=False)  # dims='var',

    halfsat = xso.parameter(description='half-saturation constant')  # dims='var'

    @xso.flux
    def uptake(self, resource, consumer, halfsat):
        return resource / (resource + halfsat) * consumer


@xso.component
class LinearDecay_ListInput:
    """ """
    var_list = xso.variable(dims='decay_vars', list_input=True,
                           foreign=True, flux='decay', negative=True, description='list of variables affected by flux')
    rate = xso.parameter(description='linear rate of decay/mortality')

    @xso.flux(dims='decay_vars_full')
    def decay(self, var_list, rate):
        """ """
        return var_list * rate