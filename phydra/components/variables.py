import xso
from xso.main import FirstInit, SecondInit

import xsimlab as xs

@xs.process
class Time(FirstInit):
    """Time is represented as a state variable"""

    time = xs.variable(intent='in', dims='input_time',
                       description='sequence of time points for which to solve the model')
    value = xs.variable(intent='out', dims='time')

    def initialize(self):
        print('Initializing Model Time')
        self.label = self.__xsimlab_name__
        self.m.Model.time = self.time

        self.value = self.m.add_variable('time')

        self.m.register_flux(self.label + '_' + self.time_flux.__name__, self.time_flux)
        self.m.add_flux(self.label, 'time', 'time_flux')

    def time_flux(self, **kwargs):
        dtdt = 1
        return dtdt


@xso.component(init_stage=2)
class SV:
    """represents a state variable in the model"""

    var = xso.variable(description='basic state variable')


@xso.component(init_stage=2)
class SVArray:
    """represents a state variable in the model"""

    var = xso.variable(dims='var', description='basic state variable')


@xso.component(init_stage=2)
class SVArraySize:
    """represents a state variable in the model"""

    var = xso.variable(dims='var', description='basic state variable')
    sizes = xso.parameter(dims='sizes', description='store of size array')
