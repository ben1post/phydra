import xso
import numpy as np


@xso.component
class ConstantExternalNutrient:
    """Component that provides a constant external nutrient
     as a forcing value.
    """

    forcing = xso.forcing(foreign=False, setup_func='forcing_setup', description='external nutrient')
    value = xso.parameter(description='constant value')

    def forcing_setup(self, value):
        """Method that returns forcing function providing the
        forcing value as a function of time."""
        @np.vectorize
        def forcing(time):
            return value

        return forcing



#TODO: actually write sinusoidal forcing the way it is shown in the paper

@xso.component
class SinusoidalExternalNutrient:
    """Component that provides a sinusoidal forcing value.

    This component calculates a sinusoidal forcing value as a function of time, with a period
    defined by the `period` parameter. The forcing value is calculated as `cos(time / period * 2 * pi) + 1`.

    Attributes
    ----------
    forcing : xso.Forcing
        The XSO variable to define a forcing to be passed along to other components.
    period : xso.Parameter
        Defines the period of the sinusoidal forcing.

    Methods
    -------
    forcing_setup : function
        The function that defines the forcing value as a function of time.
    """
    forcing = xso.forcing(setup_func='forcing_setup', description='sinusoidal forcing')
    period = xso.parameter(description='period of sinusoidal forcing')
    mean = xso.parameter(description='mean of sinusoidal forcing')
    amplitude = xso.parameter(description='amplitude of sinusoidal forcing')

    def forcing_setup(self, period, mean, amplitude):
        """Method that returns forcing function providing the
        forcing value as a function of time."""
        @np.vectorize
        def forcing(time):
            return mean + amplitude * self.m.sin(time / period * 2 * self.m.pi)

        return forcing

