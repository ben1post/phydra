from phydra.core.parts import Parameter
from .main import ThirdInit

import xsimlab as xs
import numpy as np

@xs.process
class Flux(ThirdInit):
    """represents a flux in the model"""

    k = xs.variable(intent='in', groups='pre_model_assembly')

    def initialize(self):
        super(Flux, self).initialize()  # handles initialization stages
        print(f"initializing state variable {self.label}")

        self.m.Parameters['k'] = Parameter(name='k', value=self.k)

        def linear_loss(state, parameters):
            # print("LINEAR", state, parameters)
            y = state['y']
            k = parameters['k']
            dydt = -k * y
            return dydt

        def linear_growth(state, parameters):
            # print("LINEAR", state, parameters)
            y = state['y']
            k = parameters['k']
            dydt = k * y
            return dydt

        self.m.Fluxes['y'].append(linear_loss)
        self.m.Fluxes['y'].append(linear_growth)