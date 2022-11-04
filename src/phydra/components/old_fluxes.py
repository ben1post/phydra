import numpy as np
import phydra


# NEW LIST INPUT FLUX:
@phydra.comp
class ListInputFlux:
    """ get variable input of multiple labels as list
        and do the routing etc.
    """
    resources = phydra.variable(foreign=True, negative=True, flux='growth', list_input=True, dims='resources')
    consumer = phydra.variable(foreign=True, flux='growth')
    halfsats = phydra.parameter(dims='resources')

    @phydra.flux(dims='resources')
    def growth(self, resources, consumer, halfsats):
        print(resources, consumer, halfsats)
        print(sum(resources + halfsats))
        out = resources / sum(resources + halfsats) * consumer
        print("out:", out, np.shape(out))
        return out


# First Group Flux Prototype:
@phydra.comp(init_stage=4)
class GroupedFlux:
    """ XXX
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
