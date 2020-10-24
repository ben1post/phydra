from .main import FirstInit, SecondInit

import xsimlab as xs


@xs.process
class StateVariable(SecondInit):
    """represents a state variable in the model"""

    init = xs.variable(intent='in')
    value = xs.variable(intent='out', dims='time')

    def initialize(self):
        super(StateVariable, self).initialize()  # handles initialization stages
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

        self.m.add_flux('time', self.time_flux)

    def time_flux(self, state, parameters, forcings):
        dtdt = 1
        return dtdt






#######################################################

@xs.process
class SV(SecondInit):
    """represents a state variable in the model"""

    init = xs.variable(intent='in')
    value = xs.variable(intent='out', dims='time')

    def initialize(self):
        super(SV, self).initialize()  # handles initialization stages
        print(f"initializing state variable {self.label}")

        self.value = self.m.setup_SV(self.label, Variable(name=self.label, initial_value=self.init))


@xs.process
class OLD_Time(FirstInit):
    """Time is represented as a state variable"""

    time = xs.variable(intent='in', dims='input_time',
                       description='A sequence of Time points for which to solve for y.')
    value = xs.variable(intent='out', dims='time')

    def initialize(self):
        print('Initializing Model Time')
        self.m.Time = self.time

        self.value = self.m.add_variable('time')

        self.m.Fluxes['time'].append(self.timefunc)

    def timefunc(self, state, parameters, forcings):
        dtdt = 1
        return dtdt