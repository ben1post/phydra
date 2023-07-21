import xso

import numpy as np


@xso.component
class EMPOWER_Growth_ML:
    """Growth flux component for the Phydra implementation of the EMPOWER model."""
    resource = xso.variable(foreign=True, flux='growth', negative=True)
    consumer = xso.variable(foreign=True, flux='growth', negative=False)

    mu_max = xso.parameter(description='maximum growth rate')

    @xso.flux(group_to_arg='growth_lims')
    def growth(self, resource, consumer, mu_max, growth_lims):
        """Flux function that receives all terms added to the group 'growth_lims' as an
         input argument and calculates resulting product to the growth flux."""
        return mu_max * self.m.product(growth_lims) * consumer


@xso.component
class EMPOWER_Monod_ML:
    """Component to calculate Monod-type nutrient limitation of growth
     for the Phydra implementation of the EMPOWER model.

     The flux value is added to the group 'growth_lims'."""
    resource = xso.variable(foreign=True)
    halfsat = xso.parameter(description='monod half-saturation constant')

    @xso.flux(group='growth_lims')
    def monod_lim(self, resource, halfsat):
        return resource / (resource + halfsat)


@xso.component
class EMPOWER_Eppley_ML:
    """Component to calculate temperature dependency of light-limited growth
     for the Phydra implementation of the EMPOWER model.

     The flux value is added to the group 'VpT'."""
    temp = xso.forcing(foreign=True, description='Temperature forcing')

    VpMax = xso.parameter(description='Maximum photosynthetic rate at 0 degrees celcius')

    @xso.flux(group='VpT')
    def temp_dependence(self, temp, VpMax):
        return VpMax * 1.066 ** temp


@xso.component
class EMPOWER_Smith_LambertBeer_ML:
    """Component to calculate light limitation of growth
     for the Phydra implementation of the EMPOWER model.

     The flux calculates light attenuation according to the simple Lamber-Beer law.

     The flux value is added to the group 'growth_lims'."""
    pigment_biomass = xso.variable(foreign=True)

    i_0 = xso.forcing(foreign=True, description='Light forcing')
    mld = xso.forcing(foreign=True, description='Mixed Layer Depth forcing')

    alpha = xso.parameter(description='initial slope of PI curve')
    CtoChl = xso.parameter(description='chlorophyll to carbon ratio')
    kw = xso.parameter(description='light attenuation coef for water')
    kc = xso.parameter(description='light attenuation coef for pigment biomass')

    @xso.flux(group_to_arg='VpT', group='growth_lims')
    def light_limitation(self, i_0, mld, pigment_biomass, alpha, VpT, kw, kc, CtoChl):
        kPAR = kw + kc * pigment_biomass
        i_0 = i_0 / 24  # from per day to per h
        x_0 = alpha * i_0
        x_H = alpha * i_0 * self.m.exp(- kPAR * mld)
        VpH = VpT / kPAR / mld * (
                self.m.log(x_0 + (VpT ** 2 + x_0 ** 2) ** 0.5) - self.m.log(x_H + (VpT ** 2 + x_H ** 2) ** 0.5))
        return VpH * 24 / CtoChl



@xso.component
class EMPOWER_Smith_Anderson3Layer_ML:
    """Component to calculate light limitation of growth
     for the Phydra implementation of the EMPOWER model.

     The flux calculates light attenuation according to a multi-layer model of the water column:
     > Anderson, T. R. (1993). A spectrally averaged model of light penetration and photosynthesis.
        Limnology and Oceanography, 38(7), 1403-1419.

     The flux value is added to the group 'growth_lims'."""
    pigment_biomass = xso.variable(foreign=True)

    i_0 = xso.forcing(foreign=True, description='Light forcing')
    mld = xso.forcing(foreign=True, description='Mixed Layer Depth forcing')

    alpha = xso.parameter(description='initial slop of PI curve')
    CtoChl = xso.parameter(description='chlorophyll to carbon ratio')
    kw = xso.parameter(description='light attenuation coef for water')
    kc = xso.parameter(description='light attenuation coef for pigment biomass')

    @xso.flux(group_to_arg='VpT', group='growth_lims')
    def light_limitation(self, i_0, mld, pigment_biomass, alpha, VpT, kw, kc, CtoChl):
        i_0 = i_0 / 24
        chl = pigment_biomass * 6.625 * 12.0 / CtoChl  # convert µM N to chlorophyll, mg m-3
        # (Redfield ratio of 6.625 mol C mol N-1 assumed for C:N of phytoplankton)

        ss = self.m.sqrt(chl)  # square root of chlorophyll

        # calculate layer specific attenuation coefficients:
        kPAR_1 = 0.13096 + 0.030969 * ss + 0.042644 * ss ** 2 - 0.013738 * ss ** 3 + 0.0024617 * ss ** 4 - 0.00018059 * ss ** 5
        kPAR_2 = 0.041025 + 0.036211 * ss + 0.062297 * ss ** 2 - 0.030098 * ss ** 3 + 0.0062597 * ss ** 4 - 0.00051944 * ss ** 5
        kPAR_3 = 0.021517 + 0.050150 * ss + 0.058900 * ss ** 2 - 0.040539 * ss ** 3 + 0.0087586 * ss ** 4 - 0.00049476 * ss ** 5

        kPAR = [kPAR_1, kPAR_2, kPAR_3]

        # calculate layer specific light intensities:
        if mld <= 5.0:
            jnlay = 1
            zbase = [0., mld]
            zdep = [0., mld]
            I_1 = i_0 * np.exp(-kPAR_1 * mld)
            Ibase = [i_0, I_1]
        elif mld > 5.0 and mld <= 23.0:
            jnlay = 2
            zbase = [0., 5.0, mld]
            zdep = [0., 5.0, mld - 5.0]
            mld_rest = mld - 5.0
            I_1 = i_0 * np.exp(-kPAR_1 * 5.0)
            I_2 = I_1 * np.exp(-kPAR_2 * mld - 5.0)
            Ibase = [i_0, I_1, I_2]
        elif mld > 23.0:
            jnlay = 3
            zbase = [0., 5.0, 23.0, mld]
            zdep = [0., 5.0, 23.0 - 5.0, mld - 23.0]
            I_1 = i_0 * np.exp(-kPAR_1 * 5.0)
            I_2 = I_1 * np.exp(-kPAR_2 * 23.0 - 5.0)
            I_3 = I_2 * np.exp(-kPAR_3 * mld - 23.0)
            Ibase = [i_0, I_1, I_2, I_3]

        # now add the light limitation of growth for each layer to the total light limitation:
        L_Isum = 0

        for ilay in range(1, jnlay + 1):
            # going through the layers, integrate light limitation:
            L_I = self.SmithFunc(zdep[ilay], Ibase[ilay - 1], Ibase[ilay], kPAR[ilay - 1], alpha, VpT)
            L_I = L_I * 24 / CtoChl  # convert units (gC gChl^-1 h^-1 to d-1)
            L_Isum = L_Isum + L_I * zdep[ilay]  # add to total weighted by layer depth

        L_I = L_Isum / mld  # divide by total depth to get average light limitation

        return L_I

    def SmithFunc(self, zdepth, Iin, Iout, kPARlay, alpha, Vp):
        """Helper function to calculate light limitation of growth according to the Smith function."""
        x0 = alpha * Iin
        xH = alpha * Iout
        VpH = Vp / kPARlay / zdepth * (
                np.log(x0 + (Vp ** 2 + x0 ** 2) ** 0.5) - np.log(xH + (Vp ** 2 + xH ** 2) ** 0.5))
        return VpH
