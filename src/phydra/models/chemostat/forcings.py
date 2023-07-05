import xso
import numpy as np


@xso.component
class ConstantForcing:
    """XSO component to define a constant forcing in the model."""

    forcing = xso.forcing(foreign=True, description='forcing affecting flux')
    rate = xso.parameter(description='constant rate of change')

    @xso.flux
    def input(self, forcing, rate):
        """Flux function for constant input flux

        Parameters
        ----------
        forcing : xso.forcing
            forcing affecting flux
        rate : xso.parameter
            constant rate of change

        Returns
        -------
        xso.flux
            constant input flux
        """
        return forcing * rate


#TODO: actually write sinusoidal forcing the way it is shown in the paper
@xso.component
class SinusoidalForcing:
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
    forcing = xso.forcing(setup_func='forcing_setup')
    period = xso.parameter(description='period of sinusoidal forcing')

    def forcing_setup(self, period):
        """Method that returns forcing function providing the
        forcing value as a function of time."""
        @np.vectorize
        def forcing(time):
            return np.cos(time / period * 2 * np.pi) + 1

        return forcing

# write function returning a sinusoidal curve
def sinusoidal(time, mean, amplitude, period):
    """Function that returns a sinusoidal curve as a function of time."""
    return mean + amplitude * np.sin(time / period * 2 * np.pi)
