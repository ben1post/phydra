import phydra


@phydra.comp(init_stage=3)
class LinearInput:
    var = phydra.variable(foreign=True, flux='input', negative=False, description='variable affected by flux')
    rate = phydra.parameter(description='linear rate of change')

    def input(self, var, rate):
        """ """
        return rate


import inspect
import numpy as np

@phydra.comp(init_stage=3)
class LinearForcingInput:
    var = phydra.variable(foreign=True, flux='input', negative=False, description='variable affected by flux')
    forcing = phydra.forcing(foreign=True, description='forcing affecting flux')
    rate = phydra.parameter(description='linear rate of change')

    def input(self, var, forcing, rate):
        """ """
        return forcing * rate


@phydra.comp(init_stage=3)
class ExponentialGrowth:
    var = phydra.variable(foreign=True, flux='input', negative=False, description='variable affected by flux')
    rate = phydra.parameter(description='linear rate of change')

    def input(self, var, rate):
        """ """
        return var * rate


@phydra.comp(init_stage=3)
class LinearMortality:
    var = phydra.variable(foreign=True, flux='death', negative=True, description='variable affected by flux')
    rate = phydra.parameter(description='linear rate of change')

    def death(self, var, rate):
        """ """
        return var * rate


@phydra.comp(init_stage=3)
class LinearMortalityExchange:
    source = phydra.variable(foreign=True, flux='death', negative=True)
    sink = phydra.variable(foreign=True, flux='death', negative=False)
    rate = phydra.parameter(description='mortality rate')

    def death(self, source, sink, rate):
        return source * rate


@phydra.comp(init_stage=3)
class QuadraticMortality:
    var = phydra.variable(foreign=True, flux='death', negative=True, description='variable affected by flux')
    rate = phydra.parameter(description='quadratic rate of change')

    def death(self, var, rate):
        """ """
        return var ** 2 * rate


@phydra.comp(init_stage=3)
class MonodGrowth:
    # TODO: add dimension to halfsat parameter!
    resource = phydra.variable(foreign=True, flux='uptake', negative=True)
    consumer = phydra.variable(foreign=True, dims='var', flux='uptake', negative=False)

    halfsat = phydra.parameter()

    def uptake(self, resource, consumer, halfsat):
        return resource / (resource + halfsat) * consumer


@phydra.comp(init_stage=3)
class HollingTypeIII:
    resource = phydra.variable(foreign=True, flux='grazing', negative=True)
    consumer = phydra.variable(foreign=True, flux='grazing', negative=False)
    feed_pref = phydra.parameter(description='feeding preferences')
    Imax = phydra.parameter(description='maximum ingestion rate')
    kZ = phydra.parameter(description='feeding preferences')

    def grazing(self, resource, consumer, feed_pref, Imax, kZ):
        return Imax * resource ** 2 \
               * feed_pref / (kZ ** 2 + sum([resource ** 2 * feed_pref])) * consumer

# NOTES:
#   so now I want to add the dimensionality feature
#   there is the other feature I thought about, and that is supplying a list of foreign vars
#   but these are not really compatible
#   1) I want to be able to run the model with vars that have dims
#       - some pars will need to share dims for this
#       -
#   2) some fluxes could have the additional functionality to supply a list of foreign vars
#       - but what about this, is there a better way?
#   / ocaya ktually i will not implement this, instead try to hard code fluxes
#   which is complex enough as it is

import numpy as np


@phydra.comp(init_stage=3)
class MultiFlux_Test:
    r_1 = phydra.variable(foreign=True, flux='grazing', negative=True,
                           description='resource 1')
    r_2 = phydra.variable(foreign=True, flux='grazing', negative=True,
                           description='resource 2')
    consumer = phydra.variable(foreign=True, flux='grazing', negative=False)
    fp_1 = phydra.parameter( description='feeding preference for resource 1')
    fp_2 = phydra.parameter( description='feeding preference for resource 2')
    Imax = phydra.parameter(description='maximum ingestion rate')
    kZ = phydra.parameter(description='feeding preferences')

    def grazing(self, resources, feed_prefs, consumer, Imax, kZ):

        total_grazing = sum(resources**2 * feed_prefs)

        out = Imax * resources ** 2 * feed_prefs / (kZ ** 2 + total_grazing) * consumer
        print(out)
        return out


# hm, so this needs to be simplified, at least from the interface
# vectorization is key here, the flux can take array and returns array
#


@phydra.comp(init_stage=3)
class Old_MultiFlux_Test:
    r_1 = phydra.variable(foreign=True, flux='r1_out', negative=True, description='resource 1')
    r_2 = phydra.variable(foreign=True, flux='r2_out', negative=True, description='resource 2')
    consumer = phydra.variable(foreign=True, flux='grazing', negative=False)
    fp_1 = phydra.parameter(description='feeding preference for resource 1')
    fp_2 = phydra.parameter(description='feeding preference for resource 2')
    Imax = phydra.parameter(description='maximum ingestion rate')
    kZ = phydra.parameter(description='feeding preferences')

    def flux(self, source, fp, r_1, r_2, consumer, fp_1, fp_2, Imax, kZ):

        resources = np.array([r_1, r_2])
        feed_prefs = np.array([fp_1, fp_2])

        total_grazing = sum(resources**2 * feed_prefs)

        return Imax * source**2 * fp / (kZ**2 + total_grazing) * consumer

    def grazing(self, **kwargs):
        r_1, r_2 = kwargs.get('r_1'), kwargs.get('r_2')
        fp_1, fp_2 = kwargs.get('fp_1'), kwargs.get('fp_2')
        return self.flux(source=r_1, fp=fp_1, **kwargs) + self.flux(source=r_2, fp=fp_2, **kwargs)

    def r1_out(self, **kwargs):
        r_1 = kwargs.get('r_1')
        fp_1 = kwargs.get('fp_1')
        return self.flux(source=r_1, fp=fp_1, **kwargs)

    def r2_out(self, **kwargs):
        r_2 = kwargs.get('r_2')
        fp_2 = kwargs.get('fp_2')
        return self.flux(source=r_2, fp=fp_2, **kwargs)


# yeah so the framework currently does not easily support having an array per flux..
# how can I change that?
# essentially I could have a multiflux routing type thing going on
# so that each single connection is it's own flux
# but i will first try and add an array of SVs now, maybe this will illuminate the way forward
