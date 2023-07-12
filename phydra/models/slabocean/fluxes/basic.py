import xso


@xso.component
class LinearExchange:
    """ """
    source = xso.variable(foreign=True, flux='decay', negative=True)
    sink = xso.variable(foreign=True, flux='decay', negative=False)
    rate = xso.parameter(description='decay/mortality rate')

    @xso.flux
    def decay(self, source, sink, rate):
        return source * rate


@xso.component
class QuadraticDecay:
    """ """
    var = xso.variable(foreign=True, flux='decay', negative=True, description='variable affected by flux')
    rate = xso.parameter(description='quadratic rate of change')

    @xso.flux
    def decay(self, var, rate):
        """ """
        return var ** 2 * rate


@xso.component
class QuadraticExchange:
    """ """
    source = xso.variable(foreign=True, flux='decay', negative=True)
    sink = xso.variable(foreign=True, flux='decay', negative=False)
    rate = xso.parameter(description='quadratic rate of change')

    @xso.flux
    def decay(self, source, sink, rate):
        """ """
        return source ** 2 * rate