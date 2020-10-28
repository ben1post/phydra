import phydra
from .main import FirstInit, SecondInit

import xsimlab as xs


@phydra.comp(init_stage=2)
class SV:
    """represents a state variable in the model"""

    var = phydra.variable(description='basic state variable')

    #
    # init = xs.variable(intent='in', description='initial value for state variable')
    # value = xs.variable(intent='out', dims='time', description='value output of state variable')
    #
    # def initialize(self):
    #     super(SV, self).initialize()  # handles initialization stages
    #     print(f"initializing state variable {self.label}")
    #
    #     self.value = self.m.add_variable(self.label, initial_value=self.init)


@xs.process
class OldSV(SecondInit):
    """represents a state variable in the model"""

    init = xs.variable(intent='in', description='initial value for state variable')
    value = xs.variable(intent='out', dims='time', description='value output of state variable')

    def initialize(self):
        super(OldSV, self).initialize()  # handles initialization stages
        print(f"initializing state variable {self.label}")

        self.value = self.m.add_variable(self.label, initial_value=self.init)


@xs.process
class SV_Array(SecondInit):
    """represents a state variable in the model"""

    init = xs.variable(intent='in', dims='SV')
    dim = xs.variable(intent='in', dims='SV', static=True)
    value = xs.variable(intent='out', dims='time')

    def initialize(self):
        super(SV_Array, self).initialize()  # handles initialization stages
        print(f"initializing state variable {self.label}")

        self.value = self.m.add_variable(self.label, initial_value=self.init)


@xs.process
class Time(FirstInit):
    """Time is represented as a state variable"""

    time = xs.variable(intent='in', dims='input_time',
                       description='sequence of time points for which to solve the model')
    value = xs.variable(intent='out', dims='time')

    def initialize(self):
        print('Initializing Model Time')
        self.m.Model.time = self.time

        self.value = self.m.add_variable('time')

        self.m.Model.fluxes['time'].append(self.time_flux)

    def time_flux(self, state, parameters, forcings):
        dtdt = 1
        return dtdt
