import xso


@xso.component
class MonodGrowth:
    resource = xso.variable(foreign=True, flux='uptake', negative=True)
    consumer = xso.variable(foreign=True, flux='uptake', negative=False)  # dims='var',

    halfsat = xso.parameter(description='half-saturation constant')  # dims='var'

    @xso.flux
    def uptake(self, resource, consumer, halfsat):
        return resource / (resource + halfsat) * consumer


@xso.component
class MonodGrowth_ConsumerDim:
    resource = xso.variable(foreign=True, flux='uptake', negative=True)
    consumer = xso.variable(foreign=True, dims='var', flux='uptake', negative=False)  # dims='var',

    halfsat = xso.parameter(dims='var', description='half-saturation constant')  # dims='var'

    @xso.flux(dims='var')
    def uptake(self, resource, consumer, halfsat):
        return resource / (resource + halfsat) * consumer


@xso.component
class MonodGrowth_mu_ConsumerDim:
    resource = xso.variable(foreign=True, flux='uptake', negative=True)
    consumer = xso.variable(foreign=True, dims='var', flux='uptake', negative=False)  # dims='var',

    halfsat = xso.parameter(dims='var', description='half-saturation constant')  # dims='var'
    mu_max = xso.parameter(dims='var', description='maximum growth rate')

    @xso.flux(dims='var')
    def uptake(self, resource, consumer, halfsat, mu_max):
        return mu_max * resource / (resource + halfsat) * consumer
