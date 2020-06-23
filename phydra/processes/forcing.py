import numpy as np
import xsimlab as xs
import scipy.interpolate as intrp

from .gekkocontext import InheritGekkoContext
from .components import Time
from ..utility.forcingdata import WOAForcing


@xs.process
class ForcingBase(InheritGekkoContext):
    """Base class for forcing,
    - interpolated and derivative need to be calculated in subclass
    - and passed as m.Param discretized in model timesteps"""
    fxindex = xs.variable(intent='out', groups='forcing_index')

    time = xs.foreign(Time, 'time')

    # subclass supplies : interpolated object that returns value for input of time
    interpolated = xs.on_demand()

    # interface to other processes:
    forcing = xs.variable(intent='out', groups='forcing_interpolated')
    derivative = xs.variable(intent='out', groups='forcing_interpolated_deriv')

    def initialize(self):
        """ basic initialisation of forcing """
        self.fxindex = getattr(self, '__xsimlab_name__')

        print(f"ForcingBase is initialized: {self.fxindex}")

        self.forcing = self.m.Param(self.interpolated(self.time))
        self.derivative = self.m.Param(self.interpolated(self.time, deriv=True))

    @interpolated.compute
    def interpolate(self):
        """ returns interpolated scipy object, unit : {d^-1} """
        raise ValueError('interpolate function needs to be initialized in subclass of forcing')

@xs.process
class ConstantValue(ForcingBase):
    """provides a constant value for N0 forcing"""

    value = xs.variable(intent='in', description='value of the constant forcing')

    interpolated = xs.on_demand()

    @interpolated.compute
    def constant_interpolated(self):
        """ Function returns scipy.interpolate object"""
        data_time = np.arange(365)
        data = np.full(365, self.value,  dtype='float64')
        interpolated_data = intrp.CubicSpline(data_time, data)
        interpolated_data_deriv = interpolated_data.derivative()

        def repeat_forcing_yearly(time, deriv=False):
            if deriv == True:
                return interpolated_data_deriv(np.mod(time, 365.))
            return interpolated_data(np.mod(time, 365.))

        return repeat_forcing_yearly

@xs.process
class SinusoidalValue(ForcingBase):
    """provides a sinusoidal value for N0 forcing"""
    interpolated = xs.on_demand()

    getSinusoidal = xs.on_demand()

    @getSinusoidal.compute
    def getSinusoidalData(self):
        raise Exception('Needs to be initialized in subclass')

    @interpolated.compute
    def sinusoidal_interpolated(self):
        """ Function returns scipy.interpolate object"""

        self.data_time = np.arange(365)
        data = self.getSinusoidal

        interpolated_data = intrp.CubicSpline(self.data_time, data)
        interpolated_data_deriv = interpolated_data.derivative()

        def repeat_forcing_yearly(time, deriv=False):
            if deriv == True:
                return interpolated_data_deriv(np.mod(time, 365.))
            return interpolated_data(np.mod(time, 365.))

        return repeat_forcing_yearly

@xs.process
class WOA2018(ForcingBase):
    """ Provides MLD forcing from WOA 2018 data based on chosen location in Env """
    interpolated = xs.on_demand()
    getWOA2018 = xs.on_demand()

    lat = xs.variable(intent='in')
    lon = xs.variable(intent='in')
    rbb = xs.variable(intent='in')
    smooth = xs.variable(intent='in', description='smoothing factor used to choose number of knots')

    @getWOA2018.compute
    def getDatafromWOA2018(self):
        raise Exception('Needs to be initialized in subclass')

    @interpolated.compute
    def WOA2018_interpolate(self):
        """ Function returns scipy.interpolate object"""

        data = self.getWOA2018

        # to smooth out interpolated data, we append it by itself 3 times (over 3 years)
        # and take the interpolated values from the middle year
        dayspermonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        dpm = dayspermonth * 3
        dpm_cumsum = np.cumsum(dpm) - np.array(dpm) / 2

        # k=3 for cubic spline
        interpolated_data = intrp.UnivariateSpline(dpm_cumsum, data.outForcing * 3, k=3, s=self.smooth)
        interpolated_data_deriv = interpolated_data.derivative()

        def repeat_forcing_yearly(time, deriv=False):
            if deriv == True:
                return interpolated_data_deriv(np.mod(time, 365.)+365)  # add 365 here to return 2nd year
            return interpolated_data(np.mod(time, 365.)+365)

        return repeat_forcing_yearly


# NUTIENT FORCING
@xs.process
class NutrientForcing(ForcingBase):
    """ This is a basic interface for N0 forcing to be called in Environment """
    interpolated = xs.on_demand()

    @interpolated.compute
    def interpolate(self):
        """ returns interpolated scipy object, unit : {d^-1} """
        raise ValueError('interpolate function needs to be initialized in subclass of NutrientForcing')

@xs.process
class ConstantN0(ConstantValue, NutrientForcing):
    """provides a constant value for N0 forcing"""
    value = xs.variable(intent='in', description='value of the constant forcing')

@xs.process
class SinusoidalN0(SinusoidalValue, NutrientForcing):
    """ provides a sinusoidal value for N0 forcing

    TODO: add specific arguments defining sinusoidal forcing range/values """

    getSinusoidal = xs.on_demand()

    @getSinusoidal.compute
    def sinusoidalN0(self):
        return np.cos(self.data_time / 365 * 2 * np.pi) + 1

@xs.process
class WOA2018_N0(WOA2018, NutrientForcing):
    """ Provides MLD forcing from WOA 2018 data based on chosen location in Env """
    getWOA2018 = xs.on_demand()

    @getWOA2018.compute
    def getN0fromWOA2018(self):
        return WOAForcing(self.lat, self.lon, self.rbb, 'n0x')


# MIXED LAYER DEPTH
@xs.process
class MLDForcing(ForcingBase):
    """ This is a basic interface for MLD forcing to be called in Environment """
    interpolated = xs.on_demand()

    @interpolated.compute
    def interpolate(self):
        """ returns interpolated scipy object, unit : {d^-1} """
        raise ValueError('interpolate function needs to be initialized in subclass of NutrientForcing')

@xs.process
class SinusoidalMLD(SinusoidalValue, MLDForcing):
    """ provides a sinusoidal value for N0 forcing
    """
    getSinusoidal = xs.on_demand()

    @getSinusoidal.compute
    def sinusoidalN0(self):
        return (np.cos(self.data_time / 365 * 2 * np.pi) + 1) * 100 + 20

@xs.process
class WOA2018_MLD(WOA2018, MLDForcing):
    """ Provides MLD forcing from WOA 2018 data based on chosen location in Env """
    getWOA2018 = xs.on_demand()

    @getWOA2018.compute
    def getN0fromWOA2018(self):
        return WOAForcing(self.lat, self.lon, self.rbb, 'mld')



# IRRADIANCE
@xs.process
class IrradianceForcing(ForcingBase):
    """ This is a basic interface for Irradiance forcing to be called in Environment """
    interpolated = xs.on_demand()

    @interpolated.compute
    def interpolate(self):
        """ returns interpolated scipy object, unit : {d^-1} """
        raise ValueError('interpolate function needs to be initialized in subclass of NutrientForcing')

@xs.process
class ConstantPAR(ConstantValue, IrradianceForcing):
    """provides a constant value for N0 forcing"""
    value = xs.variable(intent='in', description='value of the constant forcing')


@xs.process
class MODISaq_PAR(WOA2018, IrradianceForcing):
    """ Provides PAR forcing from MODIS aqua climatology based on chosen location in Env """
    getWOA2018 = xs.on_demand()

    @getWOA2018.compute
    def getPARfromWOA2018(self):
        return WOAForcing(self.lat, self.lon, self.rbb, 'par')

# TEMPERATURE
@xs.process
class TemperatureForcing(ForcingBase):
    """ This is a basic interface for N0 forcing to be called in Environment """
    interpolated = xs.on_demand()

    @interpolated.compute
    def interpolate(self):
        """ returns interpolated scipy object, unit : {d^-1} """
        raise ValueError('interpolate function needs to be initialized in subclass of NutrientForcing')

@xs.process
class ConstantTemp(ConstantValue, TemperatureForcing):
    """provides a constant value for N0 forcing"""
    value = xs.variable(intent='in', description='value of the constant forcing')

@xs.process
class WOA2018_Tmld(WOA2018, TemperatureForcing):
    """ Provides MLD forcing from WOA 2018 data based on chosen location in Env """
    getWOA2018 = xs.on_demand()

    @getWOA2018.compute
    def getTempfromWOA2018(self):
        return WOAForcing(self.lat, self.lon, self.rbb, 'tmld')