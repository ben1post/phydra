import phydra


@phydra.comp
class LinearInput:
    """ """
    var = phydra.variable(foreign=True, flux='input', negative=False, description='variable affected by flux')
    rate = phydra.parameter(description='linear rate of change')

    @phydra.flux
    def input(self, var, rate):
        """ """
        return rate


@phydra.comp
class ExponentialGrowth:
    """ """
    var = phydra.variable(foreign=True, flux='growth', negative=False, description='variable affected by flux')
    rate = phydra.parameter(description='linear rate of change')

    @phydra.flux
    def growth(self, var, rate):
        """ """
        return var * rate


@phydra.comp
class LinearDecay:
    """ """
    var = phydra.variable(foreign=True, flux='decay', negative=True, description='variable affected by flux')
    rate = phydra.parameter(description='linear rate of decay/mortality')

    @phydra.flux
    def decay(self, var, rate):
        """ """
        return var * rate


@phydra.comp
class LinearExchange:
    """ """
    source = phydra.variable(foreign=True, flux='decay', negative=True)
    sink = phydra.variable(foreign=True, flux='decay', negative=False)
    rate = phydra.parameter(description='decay/mortality rate')

    @phydra.flux
    def decay(self, source, sink, rate):
        return source * rate


@phydra.comp
class QuadraticDecay:
    """ """
    var = phydra.variable(foreign=True, flux='decay', negative=True, description='variable affected by flux')
    rate = phydra.parameter(description='quadratic rate of change')

    @phydra.flux
    def decay(self, var, rate):
        """ """
        return var ** 2 * rate


@phydra.comp
class QuadraticExchange:
    """ """
    source = phydra.variable(foreign=True, flux='decay', negative=True)
    sink = phydra.variable(foreign=True, flux='decay', negative=False)
    rate = phydra.parameter(description='quadratic rate of change')

    @phydra.flux
    def decay(self, source, sink, rate):
        """ """
        return source ** 2 * rate