import numpy as np
import xsimlab as xs
import scipy.interpolate as intrp

import matplotlib.pyplot as plt

from phydra.components.main import Context, Time

from phydra.utility.forcingdata import ClimatologyForcing


@xs.process
class Forcing(Context):
    label = xs.variable(intent='out')
    value = xs.variable(intent='out', dims='Time', groups='forcing_value')
    deriv = xs.variable(intent='out', dims='Time', groups='forcing_deriv')

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
    time = xs.foreign(Time, 'Time')

    def initialize(self):
        self.label = self.__xsimlab_name__
        print(f"forcing {self.label} is initialized")

        # (self, spline, Time, loop=365, derivative=0):
        self.m.phydra_forcings[self.label] = self.m.Param(self.interpolated(self.time), name=self.label)
        self.m.phydra_forcings[self.label+'_deriv'] = self.m.Param(self.interpolated(self.time, derivative=1),
                                                                   name=self.label+'_deriv')

        self.value = self.m.phydra_forcings[self.label].value
        self.deriv = self.m.phydra_forcings[self.label+'_deriv'].value


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

    def repeat_forcing_yearly(self, time, derivative=0):
        if derivative == 1:
            return self.interpolated_data.derivative(np.mod(time, 365.) + 365)  # add 365 here to return 2nd year
        return self.interpolated_data(np.mod(time, 365.) + 365)


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
    smooth = xs.variable(intent='in', description='smoothing conditions, larger values = stronger smoothing')
    k = xs.variable(intent='in', description='The degree of the spline fit')

    show_plot = xs.variable(default=False, description='show plot of interpolated data and interpolated forcing')

    @getWOA2018.compute
    def getDatafromWOA2018(self):
        return ClimatologyForcing(self.lat, self.lon, self.rbb, self.dataset)

    @interpolated.compute
    def WOA2018_interpolate(self):
        """ Function returns scipy.interpolate object"""

        data = self.getWOA2018

        # k=3 for cubic spline
        self.interpolated_data = self.interpolate_monthly_climatology(data.outForcing)

        return self.return_discretized_forcing


    def interpolate_monthly_climatology(self, data):
        """ Function that returns periodic smoothed forcing from monthly climatology data
        returns interpolated spline object
        """
        dayspermonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        dpm = dayspermonth
        dpm_cumsum = np.cumsum(dpm) - np.array(dpm) / 2

        time = np.concatenate([[0], dpm_cumsum, [365]], axis=None)

        boundary_int = [(data[0] + data[-1]) / 2]
        dat = np.concatenate([boundary_int, data, boundary_int], axis=None)
        print(dat, time)
        spl = intrp.splrep(time, dat, per=True, k=self.k, s=self.smooth)

        if self.show_plot is True:
            time_2int = np.arange(0, 365)
            dat_int = intrp.splev(time_2int, spl)

            plt.plot(time, dat, 'o', time_2int, dat_int)
            plt.ylim(bottom=0)
            plt.ylabel('Data')
            plt.xlabel('Time in days')
            plt.show()

        return spl


    def return_discretized_forcing(self, time, loop=365, derivative=0):
        """ Function returns discretized interpolated forcing, for use in the model """
        return intrp.splev(np.mod(time, loop), self.interpolated_data, der=derivative)