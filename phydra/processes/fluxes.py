from phydra.core.parts import Parameter
from .main import ThirdInit

import xsimlab as xs
import numpy as np

@xs.process
class LinearFlux(ThirdInit):
    """represents a flux in the model"""

    sv_label = xs.variable(intent='in')
    rate = xs.variable(intent='in')

    def initialize(self):
        super(LinearFlux, self).initialize()  # handles initialization stages
        print(f"initializing flux {self.label}")

        # setup parameter
        self.m.Parameters[self.label + '_rate'] = Parameter(name=self.label + '_rate', value=self.rate)
        # create flux
        self.m.Fluxes[self.sv_label].append(self.flux)


@xs.process
class LossLinearFlux(LinearFlux):
    """loss flux subclass of LinearFlux"""


    def flux(self, **kwargs):
        """linear loss flux of state variable"""
        state = kwargs.get('state')
        parameters = kwargs.get('parameters')

        sv = state[self.sv_label]
        rate = parameters[self.label + '_rate']
        delta = -rate * sv
        return delta


@xs.process
class ForcingInputLinearFlux(LinearFlux):
    """forcing flux subclass of LinearFlux"""

    fx_label = xs.variable(intent='in')

    def flux(self, **kwargs):
        """linear flux from forcing to state variable"""
        forcings = kwargs.get('forcings')
        parameters = kwargs.get('parameters')

        fx = forcings[self.fx_label]
        rate = parameters[self.label + '_rate']
        delta = fx * rate
        return delta
