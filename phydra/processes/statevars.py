from phydra.core.parts import StateVariable
from .main import FirstInit, SecondInit

import xsimlab as xs
import numpy as np

@xs.process
class SV(SecondInit):
    """represents a state variable in the model"""

    init = xs.variable(intent='in')
    value = xs.variable(intent='out', dims='time')

    def initialize(self):
        super(SV, self).initialize()  # handles initialization stages
        print(f"initializing state variable {self.label}")

        self.value = self.m.setup_SV(self.label, StateVariable(name=self.label, initial_value=self.init))


@xs.process
class Time(FirstInit):
    """Time is represented as a state variable"""

    time = xs.variable(intent='in', dims='TIME', description='A sequence of Time points for which to solve for y.')
    value = xs.variable(intent='out', dims='time')

    def initialize(self):
        print('Initializing Model Time')
        self.m.Time = self.time

        self.value = self.m.setup_SV('time', StateVariable(name='time'))

        self.m.Fluxes['time'].append(self.timefunc)

    def timefunc(self, state, parameters, forcings):
        dydt = 1
        return dydt