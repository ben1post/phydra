import numpy as np
import xsimlab as xs

from ..processes.environments import BaseEnvironment

from ..processes.components import Nutrient, Phytoplankton

from ..processes.forcing import ChemostatForcing


@xs.process
class Mixing:
    """ This is a forcing flux """
    Model_dims = xs.foreign(BaseEnvironment, 'dims')

    N_0 = xs.foreign(ChemostatForcing, 'N_0')

    N = xs.foreign(Nutrient, 'state')
    N_input = xs.variable(dims=('x', 'y', 'Env', 'N'), intent='out', groups='N_flux')

    flowrate = xs.variable(intent='in')

    def initialize(self):
        self.N_input = np.zeros_like(self.N)

    def run_step(self):
        self.N_input = self.flowrate * self.N_0