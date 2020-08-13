from phydra.core.parts import StateVariable
from .main import ModelContext

import xsimlab as xs
import numpy as np

@xs.process
class SV(ModelContext):
    """represents a state variable in the model"""

    init = xs.variable(intent='in')
    value = xs.variable(intent='out', dims='time', groups='pre_model_assembly')

    def initialize(self):
        print('initializing state variable')
        print("HHHHHEEEE")
        print(self.m)
        self.value = self.m.setup_SV('y', StateVariable(name='y', initial_value=self.init))