import xso


@xso.component
class LinearInput_Dim:
    """ """
    var = xso.variable(foreign=True, dims='var', flux='input', negative=False, description='variable affected by flux')
    rate = xso.parameter(dims='var', description='linear rate of change')

    @xso.flux(dims='var')
    def input(self, var, rate):
        """ """
        return rate


@xso.component
class ExponentialGrowth_Dim:
    """ """
    var = xso.variable(foreign=True, dims='var', flux='growth', negative=False, description='variable affected by flux')
    rate = xso.parameter(description='linear rate of change')

    @xso.flux(dims='var')
    def growth(self, var, rate):
        """ """
        return var * rate


@xso.component
class LinearDecay_VarDim:
    """ """
    var = xso.variable(dims='var', foreign=True, flux='decay', negative=True, description='variable affected by flux')
    rate = xso.parameter(description='linear rate of decay/mortality')

    @xso.flux(dims='var')
    def decay(self, var, rate):
        """ """
        return var * rate


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
class LinearExchange_SourceDim:
    """ """
    source = xso.variable(foreign=True, dims='var', flux='decay', negative=True)
    sink = xso.variable(foreign=True, flux='decay', negative=False)
    rate = xso.parameter(description='decay/mortality rate')

    @xso.flux(dims='var')
    def decay(self, source, sink, rate):
        return source * rate


@xso.component
class QuadraticDecay_Dim:
    """ """
    var = xso.variable(foreign=True, dims='var', flux='decay', negative=True, description='variable affected by flux')
    rate = xso.parameter(description='quadratic rate of change')

    @xso.flux(dims='var')
    def decay(self, var, rate):
        """ """
        return var ** 2 * rate


@xso.component
class QuadraticDecay_Dim_Sum:
    """ """
    var = xso.variable(foreign=True, dims='var', flux='decay', negative=True, description='variable affected by flux')
    rate = xso.parameter(description='quadratic rate of change')

    @xso.flux(dims='var')
    def decay(self, var, rate):
        """ """
        return var * self.m.sum(var) * rate


@xso.component
class QuadraticExchange_SourceDim:
    """ """
    source = xso.variable(foreign=True, dims='var', flux='decay', negative=True)
    sink = xso.variable(foreign=True, flux='decay', negative=False)
    rate = xso.parameter(description='quadratic rate of change')

    @xso.flux(dims='var')
    def decay(self, source, sink, rate):
        """ """
        return source ** 2 * rate