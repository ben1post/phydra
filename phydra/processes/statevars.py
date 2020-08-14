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

    days = xs.variable(intent='in', dims='days', description='time in days')
    value = xs.variable(intent='out', dims='time')

    def initialize(self):
        print('Initializing Model Time')
        self.m.time = self.days

        self.value = self.m.setup_SV('time', StateVariable(name='time'))

        self.m.Fluxes['time'].append(self.timefunc)


    def timefunc(self, state, parameters):
        dydt = 1
        return dydt