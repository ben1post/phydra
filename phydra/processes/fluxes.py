from phydra.core.parts import Parameter, Forcing
from .main import ThirdInit

import xsimlab as xs
import numpy as np

@xs.process
class Flux(ThirdInit):
    """represents a flux in the model"""

    sv_label = xs.variable(intent='in')
    k = xs.variable(intent='in')

    def initialize(self):
        super(Flux, self).initialize()  # handles initialization stages
        print(f"initializing flux {self.label}")

        # setup parameter
        self.m.Parameters[self.label + 'k'] = Parameter(name=self.label + 'k', value=self.k)
        # create flux
        self.m.Fluxes[self.sv_label].append(self.linear_loss)

    def linear_loss(self, state, parameters, forcings):
        """FLUXXX"""
        y = state[self.sv_label]
        k = parameters[self.label + 'k']
        dydt = -k * y
        return dydt


@xs.process
class ForcingFlux(ThirdInit):
    """represents a flux in the model"""

    fx_label = xs.variable(intent='in')

    sv_label = xs.variable(intent='in')
    rate = xs.variable(intent='in')

    def initialize(self):
        super(ForcingFlux, self).initialize()  # handles initialization stages
        print(f"initializing flux {self.label}")

        # setup forcing
        self.m.Forcings[self.fx_label] = Forcing(name=self.fx_label, value=0.1)
        # setup parameter
        self.m.Parameters[self.label + '_rate'] = Parameter(name=self.label + '_rate', value=self.rate)
        # create flux
        self.m.Fluxes[self.sv_label].append(self.linear_loss)

    def linear_loss(self, state, parameters, forcings):
        """FLUXXX"""
        fx = forcings[self.fx_label]
        rate = parameters[self.label + '_rate']
        dsvdt = fx * rate
        return dsvdt