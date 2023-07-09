import xso


@xso.component
class LinearInflow:
    """Component defining the linear inflow of one variable."""
    sink = xso.variable(foreign=True, flux='input', negative=False)
    source = xso.forcing(foreign=True)
    rate = xso.parameter(description='linear rate of inflow')

    @xso.flux
    def input(self, sink, source, rate):
        return source * rate


@xso.component
class MonodGrowth:
    """Component defining a growth process based on Monod-kinetics."""
    resource = xso.variable(foreign=True, flux='uptake', negative=True)
    consumer = xso.variable(foreign=True, flux='uptake', negative=False)

    halfsat = xso.parameter(description='half-saturation constant')
    mu_max = xso.parameter(description='maximum growth rate')

    @xso.flux
    def uptake(self, mu_max, resource, consumer, halfsat):
        return mu_max * resource / (resource + halfsat) * consumer


@xso.component
class LinearOutflow_ListInput:
    """Component defining the linear outflow of multiple variables."""
    var_list = xso.variable(dims='d', list_input=True,
                           foreign=True, flux='decay', negative=True, description='variables flowing out')
    rate = xso.parameter(description='linear rate of outflow')

    @xso.flux(dims='d')
    def decay(self, var_list, rate):
        # due to the list_input=True argument, var_list is an array of variables.
        # Thanks to vectorization we can just multiply the array with the rate.
        return var_list * rate
