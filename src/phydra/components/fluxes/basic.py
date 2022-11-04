import xso


@xso.component
class LinearInput:
    """ """
    var = xso.variable(foreign=True, flux='input', negative=False, description='variable affected by flux')
    rate = xso.parameter(description='linear rate of change')

    @xso.flux
    def input(self, var, rate):
        """ """
        return rate


@xso.component
class ExponentialGrowth:
    """ """
    var = xso.variable(foreign=True, flux='growth', negative=False, description='variable affected by flux')
    rate = xso.parameter(description='linear rate of change')

    @xso.flux
    def growth(self, var, rate):
        """ """
        return var * rate


@xso.component
class LinearDecay:
    """ """
    var = xso.variable(foreign=True, flux='decay', negative=True, description='variable affected by flux')
    rate = xso.parameter(description='linear rate of decay/mortality')

    @xso.flux
    def decay(self, var, rate):
        """ """
        return var * rate


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