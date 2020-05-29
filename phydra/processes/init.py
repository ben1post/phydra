import numpy as np
import xsimlab as xs

from ..processes.environments import BaseEnvironment

from ..processes.components import Nutrient, Phytoplankton


@xs.process
class ChemostatGridXYSetup:
    """
    This crucial process supplies the initial values to the components,
    more complicated parameter setup of MultiComps can be done here.
    """
    Model_dims = xs.foreign(BaseEnvironment, 'dims')

    # Input
    N_initval = xs.variable(intent='in', dims=[(), ('N')])
    P_initval = xs.variable(intent='in', dims=[(), ('P')])
    P_num = xs.variable(intent='in', dims=())

    # Initializes:
    P_halfsat = xs.foreign(Phytoplankton, 'halfsat', intent='out')
    P_mortality_rate = xs.foreign(Phytoplankton, 'mortality_rate', intent='out')

    N_state = xs.foreign(Nutrient, 'state', intent='out')
    P_state = xs.foreign(Phytoplankton, 'state', intent='out')
    P_dims = xs.foreign(Phytoplankton, 'dim', intent='out')

    def initialize(self):
        self.P_dims = self.P_num

        # initialize the state variables in the correct dimensions
        self.N_state = np.tile(np.array([self.N_initval], dtype='float64'), self.Model_dims)
        self.P_state = np.tile(np.array([self.P_initval / self.P_num for i in range(self.P_num)], dtype='float64'),
                               self.Model_dims)

        # initialize the model parameters
        self.P_halfsat = np.tile(np.array([1.5], dtype='float64'), self.Model_dims)
        self.P_mortality_rate = np.tile(np.array([0.1 for i in range(self.P_num)], dtype='float64'), self.Model_dims)
