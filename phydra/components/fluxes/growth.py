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
