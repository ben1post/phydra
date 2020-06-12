import numpy as np
import xsimlab as xs
import scipy.interpolate as intrp

from .gekkocontext import InheritGekkoContext
from .components import Time
from .main import Grid0D

@xs.process
class Forcing(InheritGekkoContext):
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
        """self."""
        self.forcing = self.m.Param(self.interpolated(np.mod(self.time, 365.)))

        forcing_deriv = self.interpolated.derivative()
        self.derivative = self.m.Param(forcing_deriv(np.mod(self.time, 365.)))

    @interpolated.compute
    def interpolate(self):
        """ returns interpolated scipy object, unit : {d^-1} """
        raise ValueError('interpolate function needs to be initialized in subclass of forcing')


@xs.process
class NutrientForcing(Forcing):
    """ """
    interpolated = xs.on_demand()

    @interpolated.compute
    def SinusoidalN0(self):
        """ Function returns scipy.interpolate object"""

        data_time = np.arange(365)
        N0 = np.cos(data_time / 365 * 2 * np.pi) + 1

        return intrp.CubicSpline(data_time, N0)




@xs.process
class OldForcing:
    """Here we initialise the Nutrient Input Forcing (also spatially defined)"""

    mld = xs.variable(dims=('time'), intent='out', static=True)
    par = xs.variable(dims=('time'), intent='out', static=True)
    sst = xs.variable(dims=('time'), intent='out', static=True)

    time = xs.foreign(Time, 'days')

    def initialize(self):
        self.mld = np.cos(self.time / 365 * np.pi * 2) * 100 + 200
        self.par = np.sin(self.time / 365 * np.pi) * 50 + 0
        self.sst = np.sin(self.time / 365 * np.pi) * 10 + 10
        print(self.mld)