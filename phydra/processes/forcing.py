import numpy as np
import xsimlab as xs
import scipy.interpolate as intrp

from .gekkocontext import InheritGekkoContext
from .components import Time


@xs.process
class ForcingBase(InheritGekkoContext):
    """Base class for forcing,
    - interpolated and derivative need to be calculated in subclass
    - and passed as m.Param discretized in model timesteps"""

    time = xs.foreign(Time, 'time')

    # subclass supplies : interpolated object that returns value for input of time
    interpolated = xs.on_demand()

    # interface to other processes:
    forcing = xs.any_object()
    derivative = xs.any_object()

    def initialize(self):
        """ basic initialisation of forcing """
        print('initializing forcing base now')
        self.forcing = self.m.Param(self.interpolated(self.time))

        self.derivative = self.m.Param(self.interpolated(self.time, deriv=True))

    @interpolated.compute
    def interpolate(self):
        """ returns interpolated scipy object, unit : {d^-1} """
        raise ValueError('interpolate function needs to be initialized in subclass of forcing')


@xs.process
class WOA2018Forcing:
    """ This could be an interface to WOA2018, needs each subforcing
    initialized with different keywords in subclasses

    ToDo: perhaps it would make sense to put this in utility!!!!
    """
    pass


@xs.process
class NutrientForcing(ForcingBase):
    """ This is a basic interface for N0 forcing to be called in Environment """
    interpolated = xs.on_demand()

    @interpolated.compute
    def interpolate(self):
        """ returns interpolated scipy object, unit : {d^-1} """
        raise ValueError('interpolate function needs to be initialized in subclass of NutrientForcing')


@xs.process
class ConstantN0(NutrientForcing):
    """provides a constant value for N0 forcing"""
    constant_N0 = xs.variable(intent='in', description='value of the constant Nutrient forcing')

    interpolated = xs.on_demand()

    @interpolated.compute
    def constantN0(self):
        """ Function returns scipy.interpolate object"""
        data_time = np.arange(365)
        N0 = np.full(365, self.constant_N0,  dtype='float64')
        interpolted_N0 = intrp.CubicSpline(data_time, N0)
        interpolated_N0_deriv = interpolted_N0.derivative()

        def repeat_forcing_yearly(time, deriv=False):
            if deriv == True:
                return interpolated_N0_deriv(np.mod(time, 365.))
            return interpolted_N0(np.mod(time, 365.))

        return repeat_forcing_yearly


@xs.process
class SinuisoidalN0(NutrientForcing):
    """provides a sinusoidal value for N0 forcing"""
    interpolated = xs.on_demand()

    @interpolated.compute
    def sinusoidalN0(self):
        """ Function returns scipy.interpolate object"""

        data_time = np.arange(365)
        N0 = np.cos(data_time / 365 * 2 * np.pi) + 1
        interpolted_N0 = intrp.CubicSpline(data_time, N0)
        interpolated_N0_deriv = interpolted_N0.derivative()

        def repeat_forcing_yearly(time, deriv=False):
            if deriv == True:
                return interpolated_N0_deriv(np.mod(time, 365.))
            return interpolted_N0(np.mod(time, 365.))

        return repeat_forcing_yearly


@xs.process
class MLDForcing(ForcingBase):
    """ This is a basic interface for MLD forcing to be called in Environment """
    interpolated = xs.on_demand()

    @interpolated.compute
    def interpolate(self):
        """ returns interpolated scipy object, unit : {d^-1} """
        raise ValueError('interpolate function needs to be initialized in subclass of NutrientForcing')


@xs.process
class SinusoidalMLD(MLDForcing):
    """ Subclass of forcing of MLD forcing that defines the actual forcing supplied to the model"""
    interpolated = xs.on_demand()

    @interpolated.compute
    def SinusoidalMLD(self):
        """ Function returns scipy.interpolate object"""

        data_time = np.arange(365)
        MLD = (np.cos(data_time / 365 * 2 * np.pi) + 1) * 100 + 20
        interpolted_MLD = intrp.CubicSpline(data_time, MLD)
        interpolated_MLD_deriv = interpolted_MLD.derivative()

        def repeat_forcing_yearly(time, deriv=False):
            if deriv == True:
                return interpolated_MLD_deriv(np.mod(time, 365.))
            return interpolted_MLD(np.mod(time, 365.))

        return repeat_forcing_yearly