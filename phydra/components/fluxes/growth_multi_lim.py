import phydra


@phydra.comp(init_stage=4)
class Growth_ML:
    """ XXX
    """
    resource = phydra.variable(foreign=True, flux='growth', negative=True)
    consumer = phydra.variable(foreign=True, flux='growth', negative=False)

    mu_max = phydra.parameter(description='maximum growth rate')

    @phydra.flux(group_to_arg='growth_lims')
    def growth(self, resource, consumer, mu_max, growth_lims):
        # print("in growth flux func now", resource, consumer, mu_max, growth_lims)
        return mu_max * self.m.product(growth_lims) * consumer


@phydra.comp
class Monod_ML:
    """ """
    resource = phydra.variable(foreign=True)
    halfsat = phydra.parameter(description='monod half-saturation constant')

    @phydra.flux(group='growth_lims')
    def monod_lim(self, resource, halfsat):
        return resource / (resource + halfsat)


@phydra.comp
class Steele_ML:
    """ """
    pigment_biomass = phydra.variable(foreign=True)

    i_0 = phydra.forcing(foreign=True, description='Light forcing')
    mld = phydra.forcing(foreign=True, description='Mixed Layer Depth forcing')

    i_opt = phydra.parameter(description='Optimal irradiance of consumer')
    kw = phydra.parameter(description='light attenuation coef for water')
    kc = phydra.parameter(description='light attenuation coef for pigment biomass')

    @phydra.flux(group='growth_lims')
    def steele_light_lim(self, i_0, mld, pigment_biomass, i_opt, kw, kc):
        kPAR = kw + kc * pigment_biomass
        light_lim = 1. / (kPAR * mld) * (
                    - self.m.exp(1. - i_0 / i_opt) - (
                    - self.m.exp((1. - (i_0 * self.m.exp(-kPAR * mld)) / i_opt))))
        return light_lim


@phydra.comp
class Smith_ML:
    """ """
    pigment_biomass = phydra.variable(foreign=True)

    i_0 = phydra.forcing(foreign=True, description='Light forcing')
    mld = phydra.forcing(foreign=True, description='Mixed Layer Depth forcing')

    alpha = phydra.parameter(description='initial slop of PI curve')
    VpMax = phydra.parameter(description='Maximum photosynthetic rate')
    kw = phydra.parameter(description='light attenuation coef for water')
    kc = phydra.parameter(description='light attenuation coef for pigment biomass')

    @phydra.flux(group='growth_lims')
    def smith_light_lim(self, i_0, mld, pigment_biomass, alpha, VpMax, kw, kc):
        kPAR = kw + kc * pigment_biomass
        x_0 = alpha * i_0  # * self.m.exp(- kPAR * 0) # (== 1)
        x_H = alpha * i_0 * self.m.exp(- kPAR * mld)
        VpH = (VpMax / (kPAR * mld)) * \
              self.m.log(
                  (x_0 + self.m.sqrt(VpMax ** 2 + x_0 ** 2)) /
                  (x_H + self.m.sqrt(VpMax ** 2 + x_H ** 2))
              )
        return VpH


@phydra.comp
class Eppley_ML:
    """ """
    temp = phydra.forcing(foreign=True, description='Temperature forcing')

    eppley_exp = phydra.parameter(description='eppley exponent')

    @phydra.flux(group='growth_lims')
    def eppley_growth(self, temp, eppley_exp):
        return self.m.exp(eppley_exp * temp)


@phydra.comp(init_stage=4)
class Growth_ML_ConsumerDim:
    """ XXX
    """
    resource = phydra.variable(foreign=True, flux='growth', negative=True)
    consumer = phydra.variable(dims='vars', foreign=True, flux='growth', negative=False)

    mu_max = phydra.parameter(description='maximum growth rate')

    @phydra.flux(dims='vars', group_to_arg='growth_lims')
    def growth(self, resource, consumer, mu_max, growth_lims):
        # print("in growth flux func now", resource, consumer, mu_max, growth_lims)
        return consumer * mu_max * self.m.product(growth_lims, axis=0)


@phydra.comp
class Monod_ML_ConsumerDim:
    """ """
    resource = phydra.variable(foreign=True)
    halfsat = phydra.parameter(dims='vars', description='monod half-saturation constant')

    @phydra.flux(dims='vars', group='growth_lims')
    def monod_lim(self, resource, halfsat):
        #print("in monod lim", resource, halfsat)
        #print(resource.value.ndim)
        return resource / (resource + halfsat)


@phydra.comp
class Steele_ML_ConsumerDim:
    """ """
    pigment_biomass = phydra.variable(dims='vars', foreign=True)

    i_0 = phydra.forcing(foreign=True, description='Light forcing')
    mld = phydra.forcing(foreign=True, description='Mixed Layer Depth forcing')

    i_opt = phydra.parameter(description='Optimal irradiance of consumer')
    kw = phydra.parameter(description='light attenuation coef for water')
    kc = phydra.parameter(description='light attenuation coef for pigment biomass')

    @phydra.flux(group='growth_lims')
    def steele_light_lim(self, i_0, mld, pigment_biomass, i_opt, kw, kc):
        kPAR = kw + kc * self.m.sum(pigment_biomass)
        light_lim = 1. / (kPAR * mld) * (
                    - self.m.exp(1. - i_0 / i_opt) - (
                    - self.m.exp((1. - (i_0 * self.m.exp(-kPAR * mld)) / i_opt))))
        return light_lim


@phydra.comp
class Eppley_ML_ConsumerDim:
    """ """
    temp = phydra.forcing(foreign=True, description='Temperature forcing')

    eppley_exp = phydra.parameter(description='eppley exponent')

    @phydra.flux(group='growth_lims')
    def eppley_growth(self, temp, eppley_exp):
        return self.m.exp(eppley_exp * temp)