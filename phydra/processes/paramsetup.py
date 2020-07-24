import xsimlab as xs
import numpy as np

from .main import GekkoContext
#from .fluxes import GrowthMultiFlux

# TODO: \
# THIS IS ALL OLD AND BULLSHIT CODE

# instead move the parameter setup functions into utility!
# and pass size and all these other parameters externally into the framework!!


@xs.process
class GrowthParameterSetup(GekkoContext):
    label = xs.variable(intent='out')
    parameter = 1 # xs.foreign(GrowthMultiFlux, 'halfsat', intent='out')

    num = xs.variable(intent='in')

    minvalue = xs.variable(intent='in', description='minimum value of halfsat parameter')
    maxvalue = xs.variable(intent='in', description='minimum value of halfsat parameter')
    spacing = xs.variable(intent='in')

    def initialize(self):
        self.label = self.__xsimlab_name__

        if self.spacing == 'linear':
            parameter_range = np.linspace(self.minvalue, self.maxvalue, self.num)
        elif self.spacing == 'log':
            parameter_range = np.logspace(np.log10(self.minvalue), np.log10(self.maxvalue), self.num)

        self.parameter = np.array(
            [self.m.Param(name=f"{self.label}_{i}", value=parameter_range[i]) for i in range(self.num)])