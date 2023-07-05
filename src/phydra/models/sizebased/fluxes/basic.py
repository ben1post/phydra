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
class LinearDecay_Dims:
    """ """
    var = xso.variable(dims='var', foreign=True, flux='decay', negative=True, description='variable affected by flux')
    rate = xso.parameter(dims='var', description='linear rate of decay/mortality')

    @xso.flux(dims='var')
    def decay(self, var, rate):
        """ """
        return var * rate


@xso.component
class QuadraticDecay_Dim_Sum:
    """ """
    var = xso.variable(foreign=True, dims='var', flux='decay', negative=True, description='variable affected by flux')
    rate = xso.parameter(description='quadratic rate of change')

    @xso.flux(dims='var')
    def decay(self, var, rate):
        """ """
        return var * self.m.sum(var) * rate