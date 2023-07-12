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