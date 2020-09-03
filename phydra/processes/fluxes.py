import phydra

# Standard Linear Fluxes:
@phydra.flux
class LinearOutputFlux:
    sv = phydra.sv(flow='output', description='state variable affected by flux')
    rate = phydra.param(description='flowing rate')

    def flux(sv, rate):
        return sv * rate


@phydra.flux
class LinearInputFlux:
    sv = phydra.sv(flow='input', description='state variable affected by flux')
    rate = phydra.param(description='flowing rate')

    def flux(sv, rate):
        return sv * rate


@phydra.flux
class ForcingLinearInputFlux:
    sv = phydra.sv(flow='input', description='state variable affected by forcing flux')
    fx = phydra.fx(description='forcing affecting rate')
    rate = phydra.param(description='flowing rate')

    def flux(sv, fx, rate):
        return fx * rate


# Michaelis Menten Flux
@phydra.flux
class MonodUptake:
    resource = phydra.sv(flow='output')
    consumer = phydra.sv(flow='input')
    halfsat = phydra.param(description='half saturation constant')

    def flux(resource, consumer, halfsat):
        return resource / (resource + halfsat) * consumer


# Multi limitation flux (wrap as phydra.multiflux)
#@phydra.multiflux(group='growth')
#class MonodUptake:
#    flux = xs.variable(intent='out', groups=group)

# Grazing Flux
@phydra.flux
class HollingTypeIIIGrazing:
    resource = phydra.sv(flow='output')
    consumer = phydra.sv(flow='input')
    feed_pref = phydra.param(description='feeding preferences') # add arg: units = ['micromolar'], add conversion to Latex
    Imax = phydra.param(description='maximum ingestion rate')
    kZ = phydra.param(description='feeding preferences')

    def flux(resource, consumer, feed_pref, Imax, kZ):
        return Imax * resource ** 2 \
               * feed_pref / (kZ ** 2 + sum([resource ** 2 * feed_pref])) * consumer

    # 2nd resource should be calculated for every resource
