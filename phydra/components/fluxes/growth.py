import phydra


@phydra.comp
class MonodGrowth:
    resource = phydra.variable(foreign=True, flux='uptake', negative=True)
    consumer = phydra.variable(foreign=True, flux='uptake', negative=False)  # dims='var',

    halfsat = phydra.parameter(description='half-saturation constant')  # dims='var'

    @phydra.flux
    def uptake(self, resource, consumer, halfsat):
        return resource / (resource + halfsat) * consumer


@phydra.comp
class MonodGrowth_ConsumerDim:
    resource = phydra.variable(foreign=True, flux='uptake', negative=True)
    consumer = phydra.variable(foreign=True, dims='var', flux='uptake', negative=False)  # dims='var',

    halfsat = phydra.parameter(dims='var', description='half-saturation constant')  # dims='var'

    @phydra.flux(dims='var')
    def uptake(self, resource, consumer, halfsat):
        return resource / (resource + halfsat) * consumer


@phydra.comp
class MonodGrowth_mu_ConsumerDim:
    resource = phydra.variable(foreign=True, flux='uptake', negative=True)
    consumer = phydra.variable(foreign=True, dims='var', flux='uptake', negative=False)  # dims='var',

    halfsat = phydra.parameter(dims='var', description='half-saturation constant')  # dims='var'
    mu_max = phydra.parameter(dims='var', description='maximum growth rate')

    @phydra.flux(dims='var')
    def uptake(self, resource, consumer, halfsat, mu_max):
        return mu_max * resource / (resource + halfsat) * consumer
