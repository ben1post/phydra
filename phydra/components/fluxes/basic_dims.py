import phydra


@phydra.comp
class LinearInput_Dim:
    """ """
    var = phydra.variable(foreign=True, dims='var', flux='input', negative=False, description='variable affected by flux')
    rate = phydra.parameter(dims='var', description='linear rate of change')

    @phydra.flux(dims='var')
    def input(self, var, rate):
        """ """
        return rate


@phydra.comp
class ExponentialGrowth_Dim:
    """ """
    var = phydra.variable(foreign=True, dims='var', flux='growth', negative=False, description='variable affected by flux')
    rate = phydra.parameter(description='linear rate of change')

    @phydra.flux(dims='var')
    def growth(self, var, rate):
        """ """
        return var * rate


@phydra.comp
class LinearDecay_VarDim:
    """ """
    var = phydra.variable(dims='var', foreign=True, flux='decay', negative=True, description='variable affected by flux')
    rate = phydra.parameter(description='linear rate of decay/mortality')

    @phydra.flux(dims='var')
    def decay(self, var, rate):
        """ """
        return var * rate


@phydra.comp
class LinearDecay_Dims:
    """ """
    var = phydra.variable(dims='var', foreign=True, flux='decay', negative=True, description='variable affected by flux')
    rate = phydra.parameter(dims='var', description='linear rate of decay/mortality')

    @phydra.flux(dims='var')
    def decay(self, var, rate):
        """ """
        return var * rate


@phydra.comp
class LinearExchange_SourceDim:
    """ """
    source = phydra.variable(foreign=True, dims='var', flux='decay', negative=True)
    sink = phydra.variable(foreign=True, flux='decay', negative=False)
    rate = phydra.parameter(description='decay/mortality rate')

    @phydra.flux(dims='var')
    def decay(self, source, sink, rate):
        return source * rate


@phydra.comp
class QuadraticDecay_Dim:
    """ """
    var = phydra.variable(foreign=True, dims='var', flux='decay', negative=True, description='variable affected by flux')
    rate = phydra.parameter(description='quadratic rate of change')

    @phydra.flux(dims='var')
    def decay(self, var, rate):
        """ """
        return var ** 2 * rate


@phydra.comp
class QuadraticDecay_Dim_Sum:
    """ """
    var = phydra.variable(foreign=True, dims='var', flux='decay', negative=True, description='variable affected by flux')
    rate = phydra.parameter(description='quadratic rate of change')

    @phydra.flux(dims='var')
    def decay(self, var, rate):
        """ """
        return var * self.m.sum(var) * rate


@phydra.comp
class QuadraticExchange_SourceDim:
    """ """
    source = phydra.variable(foreign=True, dims='var', flux='decay', negative=True)
    sink = phydra.variable(foreign=True, flux='decay', negative=False)
    rate = phydra.parameter(description='quadratic rate of change')

    @phydra.flux(dims='var')
    def decay(self, source, sink, rate):
        """ """
        return source ** 2 * rate