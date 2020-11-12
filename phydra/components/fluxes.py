import numpy as np
import phydra


@phydra.comp(init_stage=3)
class LinearInput:
    var = phydra.variable(foreign=True, flux='input', negative=False, description='variable affected by flux')
    rate = phydra.parameter(description='linear rate of change')

    @phydra.flux
    def input(self, var, rate):
        """ """
        return rate


@phydra.comp(init_stage=3)
class LinearForcingInput:
    var = phydra.variable(foreign=True, flux='input', negative=False, description='variable affected by flux')
    forcing = phydra.forcing(foreign=True, description='forcing affecting flux')
    rate = phydra.parameter(description='linear rate of change')

    @phydra.flux
    def input(self, var, forcing, rate):
        """ """
        return forcing * rate


@phydra.comp(init_stage=3)
class ExponentialGrowth:
    var = phydra.variable(foreign=True, flux='input', negative=False, description='variable affected by flux')
    rate = phydra.parameter(description='linear rate of change')

    @phydra.flux
    def input(self, var, rate):
        """ """
        return var * rate


@phydra.comp(init_stage=3)
class LinearMortality:
    var = phydra.variable(foreign=True, flux='death', negative=True, description='variable affected by flux')
    rate = phydra.parameter(description='linear rate of change')

    @phydra.flux
    def death(self, var, rate):
        """ """
        return var * rate


@phydra.comp(init_stage=3)
class LinearMortality_Array:
    var = phydra.variable(foreign=True, dims='var', flux='death', negative=True, description='variable affected by flux')
    rate = phydra.parameter(dims='var', description='linear rate of change')

    @phydra.flux
    def death(self, var, rate):
        """ """
        return var * rate


@phydra.comp(init_stage=3)
class LinearMortalityExchange:
    source = phydra.variable(foreign=True, flux='death', negative=True)
    sink = phydra.variable(foreign=True, flux='death', negative=False)
    rate = phydra.parameter(description='mortality rate')

    @phydra.flux
    def death(self, source, sink, rate):
        return source * rate


@phydra.comp(init_stage=3)
class QuadraticMortality:
    var = phydra.variable(foreign=True, flux='death', negative=True, description='variable affected by flux')
    rate = phydra.parameter(description='quadratic rate of change')

    @phydra.flux
    def death(self, var, rate):
        """ """
        return var ** 2 * rate


@phydra.comp(init_stage=3)
class MonodGrowth:
    resource = phydra.variable(foreign=True, flux='uptake', negative=True)
    consumer = phydra.variable(foreign=True, flux='uptake', negative=False)  # dims='var',

    halfsat = phydra.parameter(description='half-saturation constant')  # dims='var'

    @phydra.flux
    def uptake(self, resource, consumer, halfsat):
        return resource / (resource + halfsat) * consumer


@phydra.comp(init_stage=3)
class MonodGrowth_Array:
    resource = phydra.variable(foreign=True, flux='uptake', negative=True)
    consumer = phydra.variable(foreign=True, dims='var', flux='uptake', negative=False)  # dims='var',

    halfsat = phydra.parameter(dims='var', description='half-saturation constant')  # dims='var'

    @phydra.flux
    def uptake(self, resource, consumer, halfsat):
        return resource / (resource + halfsat) * consumer


@phydra.comp(init_stage=3)
class HollingTypeIII:
    resource = phydra.variable(foreign=True, flux='grazing', negative=True)
    consumer = phydra.variable(foreign=True, flux='grazing', negative=False)
    feed_pref = phydra.parameter(description='feeding preferences')
    Imax = phydra.parameter(description='maximum ingestion rate')
    kZ = phydra.parameter(description='feeding preferences')

    @phydra.flux
    def grazing(self, resource, consumer, feed_pref, Imax, kZ):
        return Imax * resource ** 2 \
               * feed_pref / (kZ ** 2 + sum([resource ** 2 * feed_pref])) * consumer


# NEW LIST INPUT FLUX:
@phydra.comp
class ListInputFlux:
    """ get variable input of multiple labels as list
        and do the routing etc.
    """
    resources = phydra.variable(foreign=True, negative=True, flux='growth', list_input=True, dims='resources')
    consumer = phydra.variable(foreign=True, flux='growth')
    halfsat = phydra.parameter()

    @phydra.flux(dims='resources')
    def growth(self, resources, consumer, halfsat):
        # print(resources, consumer, halfsat)
        return resources / (sum(resources) + halfsat) * consumer


@phydra.comp(init_stage=3)
class HollingTypeIII_2Resources:
    r_1 = phydra.variable(foreign=True, flux='r1_out', negative=True, description='resource 1')
    r_2 = phydra.variable(foreign=True, flux='r2_out', negative=True, description='resource 2')
    consumer = phydra.variable(foreign=True, flux='grazing', negative=False)
    fp_1 = phydra.parameter(description='feeding preference for resource 1')
    fp_2 = phydra.parameter(description='feeding preference for resource 2')
    Imax = phydra.parameter(description='maximum ingestion rate')
    kZ = phydra.parameter(description='feeding preferences')

    def flux(self, source, fp, r_1, r_2, consumer, fp_1, fp_2, Imax, kZ):
        resources = np.concatenate([r_1, r_2], axis=None)
        feed_prefs = np.concatenate([fp_1, fp_2], axis=None)

        total_grazing = sum(resources ** 2 * feed_prefs)

        return Imax * source ** 2 * fp / (kZ ** 2 + total_grazing) * consumer

    @phydra.flux
    def grazing(self, **kwargs):
        r_1, r_2 = kwargs.get('r_1'), kwargs.get('r_2')
        fp_1, fp_2 = kwargs.get('fp_1'), kwargs.get('fp_2')
        return self.flux(source=r_1, fp=fp_1, **kwargs) + self.flux(source=r_2, fp=fp_2, **kwargs)

    @phydra.flux
    def r1_out(self, **kwargs):
        r_1 = kwargs.get('r_1')
        fp_1 = kwargs.get('fp_1')
        return self.flux(source=r_1, fp=fp_1, **kwargs)

    @phydra.flux
    def r2_out(self, **kwargs):
        r_2 = kwargs.get('r_2')
        fp_2 = kwargs.get('fp_2')
        return self.flux(source=r_2, fp=fp_2, **kwargs)


# First Group Flux Prototype:
@phydra.comp(init_stage=4)
class GroupedFlux:
    """ Somehow i need to get other fluxes here as input args

            the normal xs.group way doesn't really work anymore
            i need to create a dict of labels, that the values of are retrieved from model STATE

            -
    """

    var = phydra.variable(foreign=True, flux='growth')

    @phydra.flux(group_to_arg='X_growth', description='HELLO')
    def growth(self, var, X_growth):
        # print(X_growth)
        return sum(X_growth)


@phydra.comp(init_stage=2)
class SubFlux:
    var = phydra.variable(foreign=True)
    rate = phydra.parameter()

    @phydra.flux(group='X_growth')
    def one_growth(self, var, rate):
        # print(var)
        return var * rate

    @phydra.flux(group='X_growth')
    def two_growth(self, var, rate):
        return - var * rate


# MULTI LIM TESTS:
# noinspection NonAsciiCharacters
@phydra.comp(init_stage=3)
class Growth_Monod_Eppley_Steele:
    resource = phydra.variable(foreign=True, flux='growth', negative=True)
    consumer = phydra.variable(foreign=True, flux='growth', negative=False)

    Temp = phydra.forcing(foreign=True, description='Temperature forcing')
    Light = phydra.forcing(foreign=True, description='Light forcing')
    MLD = phydra.forcing(foreign=True, description='Mixed Layer Depth forcing')

    halfsat = phydra.parameter(description='monod half-saturation constant')
    eppley = phydra.parameter(description='eppley exponent')

    i_opt = phydra.parameter(description='Optimal irradiance of consumer')
    µ_max = phydra.parameter(description='maximum growth rate')

    kw = phydra.parameter(description='light attenuation coef for water')
    kc = phydra.parameter(description='light attenuation coef for consumer')

    @phydra.flux
    def growth(self, resource, consumer, Temp, Light, MLD, i_opt, kw, kc, eppley, halfsat, µ_max):

        temp_lim = self.m.exp(eppley * Temp)
        monod_lim = resource / (resource + halfsat)
        kPAR = kw + kc * consumer
        light_lim = 1. / (kPAR * MLD) * (
                    -self.m.exp(1. - Light / i_opt) - (
                        -self.m.exp((1. - (Light * self.m.exp(-kPAR * MLD)) / i_opt))))

        return µ_max * temp_lim * monod_lim * light_lim * consumer
