import os
import numpy as np
import scipy.interpolate as intrp

import phydra
from phydra.utility.forcingdata import ClimatologyForcing


@phydra.comp(init_stage=2)
class ConstantForcing:
    forcing = phydra.forcing(foreign=False, setup_func='forcing_setup')
    value = phydra.parameter(description='constant value of forcing')

    def forcing_setup(self, value):
        cwd = os.getcwd()
        print("forcing function is in directory:", cwd)
        print("forcing_val:", value)

        @np.vectorize
        def forcing(time):
            return value

        return forcing


@phydra.comp(init_stage=2)
class SinusoidalForcing:
    forcing = phydra.forcing(foreign=False, setup_func='forcing_setup')
    period = phydra.parameter(description='period of sinusoidal forcing')

    def forcing_setup(self, period):
        cwd = os.getcwd()
        print("forcing function is in directory:", cwd)

        @np.vectorize
        def forcing(time):
            return np.cos(time / period * 2 * np.pi) + 1

        return forcing


@phydra.comp(init_stage=2)
class GlobalSlabClimatologyForcing:
    forcing = phydra.forcing(foreign=False, setup_func='forcing_setup')
    dataset = phydra.parameter(description="Options: 'n0x', 'mld', 'tmld', 'par'")
    lat = phydra.parameter(description='constant value of forcing')
    lon = phydra.parameter(description='constant value of forcing')
    rbb = phydra.parameter(description='constant value of forcing')
    smooth = phydra.parameter(description='smoothing conditions, larger values = stronger smoothing')
    k = phydra.parameter(description='The degree of the spline fit')
    deriv = phydra.parameter(description='order of derivative to store, for basic forcing pass 0')

    def forcing_setup(self, dataset, lat, lon, rbb, smooth, k, deriv):
        data = ClimatologyForcing(lat, lon, rbb, dataset).outForcing

        dayspermonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        dpm = dayspermonth
        dpm_cumsum = np.cumsum(dpm) - np.array(dpm) / 2

        time = np.concatenate([[0], dpm_cumsum, [365]], axis=None)

        boundary_int = [(data[0] + data[-1]) / 2]
        dat = np.concatenate([boundary_int, data, boundary_int], axis=None)

        spl = intrp.splrep(time, dat, per=True, k=k, s=smooth)

        def forcing(time):
            return intrp.splev(np.mod(time, 365), spl, der=deriv)

        return forcing
