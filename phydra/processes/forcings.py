import numpy as np
import xsimlab as xs
import scipy.interpolate as intrp

import matplotlib.pyplot as plt

from .main import GekkoContext, Time

from ..utility.forcingdata import ClimatologyForcing

def interpolate_monthly_climatology(data, show_plot=False):
    """ Function that returns periodic smoothed forcing from monthly climatology data

    returns interpolated spline object

    TODO in another function add return forcing for time.. etc.
    """
    dayspermonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    dpm = dayspermonth  # * 3
    dpm_cumsum = np.cumsum(dpm) - np.array(dpm) / 2

    time = np.concatenate([[0], dpm_cumsum, [365]], axis=None)

    boundary_int = [(data.outForcing[0] + data.outForcing[-1]) / 2]
    dat = np.concatenate([boundary_int, data.outForcing, boundary_int], axis=None)

    spl = intrp.splrep(time, dat, per=True, k=3, s=40)
    time_2int = np.linspace(0, 365, 1000)

    dat_int = intrp.splev(time_2int, spl)
    dat_int_deriv = intrp.splev(time_2int, spl, der=1)

    if show_plot is True:
        plt.plot(time, dat, 'o', time_2int, dat_int)
        plt.show()

    return dat_int



@xs.process
class Forcing(GekkoContext):
    label = xs.variable(intent='out')
    value = xs.variable(intent='out', dims='time', groups='forcing_value')
    deriv = xs.variable(intent='out', dims='time', groups='forcing_deriv')

    def initialize(self):
        raise ValueError('needs to be implemented in subclass')


@xs.process
class ConstantForcing(Forcing):

    initVal = xs.variable(intent='in')

    def initialize(self):
        self.label = self.__xsimlab_name__
        print(f"forcing {self.label} is initialized")

        self.m.phydra_forcings[self.label] = self.m.Param(self.initVal, name=self.label)
        self.value = self.m.phydra_forcings[self.label].value
        self.deriv = self.m.Param(0)


@xs.process
class InterpolatedForcing(Forcing):
    """"""
    time = xs.foreign(Time, 'time')

    def initialize(self):
        self.label = self.__xsimlab_name__
        print(f"forcing {self.label} is initialized")

        self.m.phydra_forcings[self.label] = self.m.Param(self.interpolated(self.time), name=self.label)
        self.m.phydra_forcings[self.label+'_deriv'] = self.m.Param(self.interpolated(self.time, deriv=True),
                                                                   name=self.label+'_deriv')

        self.value = self.m.phydra_forcings[self.label].value
        self.deriv = self.m.phydra_forcings[self.label+'_deriv'].value

    def repeat_forcing_yearly(self, time, deriv=False):
        if deriv == True:
            return self.interpolated_data_deriv(np.mod(time, 365.) + 365)  # add 365 here to return 2nd year
        return self.interpolated_data(np.mod(time, 365.) + 365)


@xs.process
class SinusoidalForcing(InterpolatedForcing):
    """provides a sinusoidal value forcing"""

    interpolated = xs.on_demand()
    getSinusoidal = xs.on_demand()

    @interpolated.compute
    def sinusoidal_interpolated(self):
        """ Function returns scipy.interpolate object"""

        self.data_time = np.arange(365)
        data = self.getSinusoidal

        self.interpolated_data = intrp.CubicSpline(self.data_time, data)
        self.interpolated_data_deriv = self.interpolated_data.derivative()

        return self.repeat_forcing_yearly

    @getSinusoidal.compute
    def sinusoidal(self):
        # N0 forcing params here
        return np.cos(self.data_time / 365 * 2 * np.pi) + 1


@xs.process
class GlobalSlabClimatologyForcing(InterpolatedForcing):
    """MLD Climatology from x
    WOA2018 Forcing for Nitrate, MLD, T_MLD
    MODIS aqua forcing PAR"""

    interpolated = xs.on_demand()
    getWOA2018 = xs.on_demand()

    dataset = xs.variable(intent='in', description="Options: 'n0x', 'mld', 'tmld', 'par'")

    lat = xs.variable(intent='in')
    lon = xs.variable(intent='in')
    rbb = xs.variable(intent='in')
    smooth = xs.variable(intent='in', description='smoothing factor used to choose number of knots')

    @getWOA2018.compute
    def getDatafromWOA2018(self):
        return ClimatologyForcing(self.lat, self.lon, self.rbb, self.dataset)

    @interpolated.compute
    def WOA2018_interpolate(self):
        """ Function returns scipy.interpolate object"""

        data = self.getWOA2018

        # to smooth out interpolated data, we append it by itself 3 times (over 3 years)
        # and take the interpolated values from the middle year
        dayspermonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        dpm = dayspermonth * 3
        dpm_cumsum = np.cumsum(dpm) - np.array(dpm) / 2

        x = np.r_[dayspermonth, dayspermonth[0]]
        y = np.r_[data.outForcing, data.outForcing[0]]

        tck, _ = intrp.splprep([x, y], s=0, per=True)

        xi, yi = intrp.splev(np.arange(30, 300), tck)

        print(xi)

        print(yi)

        # k=3 for cubic spline
        self.interpolated_data = intrp.UnivariateSpline(dpm_cumsum, data.outForcing * 3, k=3, s=self.smooth)
        self.interpolated_data_deriv = self.interpolated_data.derivative()

        return self.repeat_forcing_yearly

