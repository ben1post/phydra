from phydra.core.parts import StateVariable
from .main import FirstInit, SecondInit

import xsimlab as xs
import numpy as np

@xs.process
class SV(SecondInit):
    """represents a state variable in the model"""

    init = xs.variable(intent='in')
    value = xs.variable(intent='out', dims='time', groups='pre_model_assembly')

    def initialize(self):
        super(SV, self).initialize()  # handles initialization stages
        print(f"initializing state variable {self.label}")

        self.value = self.m.setup_SV(self.label, StateVariable(name=self.label, initial_value=self.init))


@xs.process
class Time(FirstInit):

    days = xs.variable(intent='in', dims='days', description='time in days')
    # for indexing xarray IO objects
    #time = xs.index(dims='time', description='time in days')
    #input = xs.variable(default=3)

    def initialize(self):
        print('Initializing Model Time')

        print(self.__xsimlab_state__)

        print('X', self.days)

        self.value = self.m.setup_SV('time', StateVariable(name='time', initial_value=0))

        self.value = self.days

        self.m.time = self.value

        def linear_growth(state, parameters):
            # print("LINEAR", state, parameters)
            t = state['time']
            dydt = 1 * t
            return dydt

        self.m.Fluxes['time'].append(linear_growth)

        # in order to
        #if self.m.core is not None:
        #    if self.m.solver == "gekko":
        #        self.m.core.gekko.Equation(self.SVs['time'].dt() == 1)