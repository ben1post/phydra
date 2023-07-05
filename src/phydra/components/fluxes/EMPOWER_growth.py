import xso

import numpy as np


@xso.component
class EMPOWER_Growth_ML:
    """ XXX
    """
    resource = xso.variable(foreign=True, flux='growth', negative=True)
    consumer = xso.variable(foreign=True, flux='growth', negative=False)

    mu_max = xso.parameter(description='maximum growth rate')

    @xso.flux(group_to_arg='growth_lims')
    def growth(self, resource, consumer, mu_max, growth_lims):
        # print("in growth flux func now", resource, consumer, mu_max, growth_lims)
        return mu_max * self.m.product(growth_lims) * consumer


@xso.component
class EMPOWER_Monod_ML:
    """ """
    resource = xso.variable(foreign=True)
    halfsat = xso.parameter(description='monod half-saturation constant')

    @xso.flux(group='growth_lims')
    def monod_lim(self, resource, halfsat):
        return resource / (resource + halfsat)


@xso.component
class EMPOWER_Eppley_ML:
    """ """
    temp = xso.forcing(foreign=True, description='Temperature forcing')

    VpMax = xso.parameter(description='Maximum photosynthetic rate at 0 degrees celcius')

    @xso.flux(group='VpT')
    def temp_dependence(self, temp, VpMax):
        return VpMax * 1.066 ** temp


# chl <- P*6.625*12.0/CtoChl

@xso.component
class EMPOWER_Smith_ML:
    """ """
    pigment_biomass = xso.variable(foreign=True)

    i_0 = xso.forcing(foreign=True, description='Light forcing')
    mld = xso.forcing(foreign=True, description='Mixed Layer Depth forcing')

    alpha = xso.parameter(description='initial slop of PI curve')
    CtoChl = xso.parameter(description='chlorophyll to carbon ratio')
    kw = xso.parameter(description='light attenuation coef for water')
    kc = xso.parameter(description='light attenuation coef for pigment biomass')

    @xso.flux(group_to_arg='VpT', group='growth_lims')
    def smith_light_lim(self, i_0, mld, pigment_biomass, alpha, VpT, kw, kc, CtoChl):
        kPAR = kw + kc * pigment_biomass
        i_0 = i_0 / 24  # from per day to per h
        x_0 = alpha * i_0  # * self.m.exp(- kPAR * 0) # (== 1)
        x_H = alpha * i_0 * self.m.exp(- kPAR * mld)
        VpH = VpT / kPAR / mld * (
                self.m.log(x_0 + (VpT ** 2 + x_0 ** 2) ** 0.5) - self.m.log(x_H + (VpT ** 2 + x_H ** 2) ** 0.5))
        return VpH * 24 / CtoChl


@xso.component
class EMPOWER_Anderson_Light_ML:
    """ """
    pigment_biomass = xso.variable(foreign=True)

    i_0 = xso.forcing(foreign=True, description='Light forcing')
    mld = xso.forcing(foreign=True, description='Mixed Layer Depth forcing')

    alpha = xso.parameter(description='initial slop of PI curve')
    CtoChl = xso.parameter(description='chlorophyll to carbon ratio')
    kw = xso.parameter(description='light attenuation coef for water')
    kc = xso.parameter(description='light attenuation coef for pigment biomass')

    @xso.flux(group_to_arg='VpT', group='growth_lims')
    def irradiance_out(self, i_0, mld, pigment_biomass, alpha, VpT, kw, kc, CtoChl):
        """ """
        i_0 = i_0 / 24
        chl = pigment_biomass * 6.625 * 12.0 / CtoChl  # chlorophyll, mg m-3 (Redfield ratio of 6.625 mol C mol N-1 assumed for C:N of phytoplankton)
        ss = chl ** 0.5  # square root of chlorophyll

        kPAR_1 = 0.13096 + 0.030969 * ss + 0.042644 * ss ** 2 - 0.013738 * ss ** 3 + 0.0024617 * ss ** 4 - 0.00018059 * ss ** 5
        kPAR_2 = 0.041025 + 0.036211 * ss + 0.062297 * ss ** 2 - 0.030098 * ss ** 3 + 0.0062597 * ss ** 4 - 0.00051944 * ss ** 5
        kPAR_3 = 0.021517 + 0.050150 * ss + 0.058900 * ss ** 2 - 0.040539 * ss ** 3 + 0.0087586 * ss ** 4 - 0.00049476 * ss ** 5

        kPAR = [kPAR_1, kPAR_2, kPAR_3]

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

        aphybase = [0, 0, 0, 0]
        aphybase[
            0] = 0.36796 + 0.17537 * ss - 0.065276 * ss ** 2 + 0.013528 * ss ** 3 - 0.0011108 * ss ** 4  # a (chl absorption) at ocean surface as function chl

        L_Isum = 0

        for ilay in range(1, jnlay + 1):
            L_I = self.SmithFunc(zdep[ilay], Ibase[ilay - 1], Ibase[ilay], kPAR[ilay - 1], alpha, VpT)
            L_I = L_I * 24 / CtoChl  # / VpT
            L_Isum = L_Isum + L_I * zdep[ilay]

            # for ilay in range(1, jnlay+1):
        #    ahash = self.FNaphy(ss,zbase[ilay-1],zbase[ilay],aphybase[ilay-1]) # change in a with depth
        #    aphybase[ilay] = aphybase[ilay-1]+ahash                       # a at base of layer
        #    aphyav = aphybase[ilay-1]+ahash*0.5                           # average a in layer (from which alpha is calculated: alpha = a*alphamax

        #    L_I = self.FNLIcalcA93(zdep[ilay],Ibase[ilay-1],Ibase[ilay],kPAR[ilay-1],alpha,VpT,aphyav)
        #    L_Isum = L_Isum + L_I*zdep[ilay]      # multiply by layer depth in order to set up weighted average for total mixed layer

        L_I = L_Isum / mld

        return L_I

    def SmithFunc(self, zdepth, Iin, Iout, kPARlay, alpha, Vp):
        x0 = alpha * Iin
        xH = alpha * Iout  # *np.exp(-kPARlay*zdepth)
        VpH = Vp / kPARlay / zdepth * (
                np.log(x0 + (Vp ** 2 + x0 ** 2) ** 0.5) - np.log(xH + (Vp ** 2 + xH ** 2) ** 0.5))
        return VpH

    def FNaphy(self, ss, ztop, zbottom, aphylast):
        """
        FNaphy calculates values of a (chlorophyll abosorption) for each layer;
        alpha[layer] is then calculated as a[layer]*alphamax
        """
        g = [0.048014, 0.00023779, -0.023074, 0.0031095, -0.0090545, 0.0027974, 0.00085217, -3.9804E-06, 0.0012398,
             -0.00061991]  # coeffs. for calculating a#

        x = zbottom + 1.0
        xlg = np.log(x)
        termf1a = x * xlg - x
        termf2a = x * xlg ** 2 - 2 * x * xlg + 2 * x
        termf3a = x * xlg ** 3 - 3 * x * xlg ** 2 + 6 * x * xlg - 6 * x
        x = ztop + 1.0
        xlg = np.log(x)
        termf1b = x * xlg - x
        termf2b = x * xlg ** 2 - 2 * x * xlg + 2 * x
        termf3b = x * xlg ** 3 - 3 * x * xlg ** 2 + 6 * x * xlg - 6 * x

        terma = g[0] + g[1] * ss + g[4] * ss ** 2 + g[6] * ss ** 3
        termb = g[2] + g[3] * ss + g[8] * ss ** 2
        termc = g[5] + g[9] * ss
        acalc = (zbottom + 1.0) * terma + termf1a * termb + \
                termf2a * termc + termf3a * g[7] - ((ztop + 1.0) * terma + \
                                                    termf1b * termb + termf2b * termc + termf3b * g[7])

        return acalc

    def FNLIcalcA93(self, zdepth, Iin, Iout, kPARlay, alpha, Vp, ahashnow):
        """
        # Photosynthesis calculated using polynomial approximation (Anderson, 1993)
        """

        omeg = [1.9004, -2.8333E-01, 2.8050E-02, -1.4729E-03,
                3.0841E-05]  # polynomial coefficients for calculating photosynthesis (Platt et al., 1990)

        # Calculate alphamax
        alphamax = alpha * 2.602  # alphamax is alpha at wavelength of maximum absorption cross section

        # Calculate daily photosynthesis in each layer, mg C m-2 d-1
        V0 = Vp / (np.pi * kPARlay)
        V1 = alphamax * ahashnow * Iin / Vp
        V2 = alphamax * ahashnow * Iout / Vp
        Qpsnow = V0 * (omeg[0] * (V1 - V2) + omeg[1] * (V1 ** 2 - V2 ** 2) + omeg[2] * (V1 ** 3 - V2 ** 3) + omeg[3] * (
                V1 ** 4 - V2 ** 4) + omeg[4] * (V1 ** 5 - V2 ** 5))

        # convert to dimensionless units to get 0 <= J <= 1
        Lim_I = Qpsnow / zdepth / Vp  # /24.

        return Lim_I
