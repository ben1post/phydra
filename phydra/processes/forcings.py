from phydra.core.parts import Forcing
from .main import SecondInit

import xsimlab as xs

@xs.process
class ConstantForcing(SecondInit):
    """represents a flux in the model"""

    value = xs.variable(intent='in')

    def initialize(self):
        super(ConstantForcing, self).initialize()  # handles initialization stages
        print(f"initializing forcing {self.label}")

        # setup forcing
        self.m.Forcings[self.label] = Forcing(name=self.label, value=self.value)