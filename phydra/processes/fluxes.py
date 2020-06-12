import numpy as np
import xsimlab as xs
from itertools import product

from ..utility.modelcontext import GekkoMath
from .gekkocontext import InheritGekkoContext
from .forcing import NutrientForcing

@xs.process
class Flux(InheritGekkoContext):
    # THIS kind of flux needs to check if dims are the same
    # provide different kind of fluxes, that check dims and work accordingly!


    inputrate_c1 = xs.variable(intent='in')
    inputrate_c1_allo = xs.variable(intent='out')
    conversion2_c2 = xs.variable(intent='in')
    destruction_c2 = xs.variable(intent='in')

    def initialize(self):
        print('Initializing Flux')
        # get SVs
        C1 = self.gk_SVs[self.c1_label]
        C2 = self.gk_SVs[self.c2_label]

        # add some kind of allometric parameterisation (difference in parameter between component dims)
        self.inputrate_c1_allo = np.linspace(self.inputrate_c1, self.inputrate_c1 + 0.1,
                                             self.gk_context['comp_dims'][self.c1_label])

        # define flux for all dimensions of SV
        it1 = np.nditer(self.gk_SVshapes[self.c1_label], flags=['zerosize_ok', 'multi_index', 'refs_ok'])
        it2 = np.nditer(self.gk_SVshapes[self.c2_label], flags=['zerosize_ok', 'multi_index', 'refs_ok'])
        while not it1.finished:
            while not it2.finished:
                self.conversionrate = self.m.Intermediate(C1[it1.multi_index] * self.conversion2_c2)
                self.gk_Fluxes[self.c1_label][it1.multi_index] = self.m.Intermediate(
                    self.inputrate_c1_allo[it1.multi_index[1]] - self.conversionrate)
                self.gk_Fluxes[self.c2_label][it2.multi_index] = self.m.Intermediate(
                    self.conversionrate - self.destruction_c2 * C2[it2.multi_index])
                it1.iternext()
                it2.iternext()


@xs.process
class Mixing(InheritGekkoContext):
    N0_Forcing = xs.foreign(NutrientForcing, 'forcing')  # m.Param()
    N0_Forcing_deriv = xs.foreign(NutrientForcing, 'derivative')   # m.Param()

    C_label = xs.variable(intent='in', description='label of component that grows')

    mixingrate = xs.variable(intent='in')

    def initialize(self):
        self.C = self.gk_SVs[self.C_label]

        #it = np.nditer(self.gk_SVshapes[self.C_label], flags=['zerosize_ok', 'multi_index', 'refs_ok'])
        #while not it.finished:
        self.gk_Fluxes[self.C_label] = self.m.Intermediate((self.N0_Forcing - self.C) * self.m.Param(self.mixingrate))
        #it.iternext()


@xs.process
class LimitedGrowth(InheritGekkoContext):
    """This flux defines the growth rate of 1 phytoplankton component,
    based on
    - nutrient uptake, according to Droop
    - additional dependencies: Light, Temperature
    """

    mu = xs.variable(intent='in', description='Maximum growth rate of component')

    C_label = xs.variable(intent='in', description='label of component that grows')
    R_label = xs.variable(intent='in', description='label of ressource component that is consumed')

    halfsat = xs.variable(intent='in', description='half-saturation constant of nutrient uptake for component')

    # so this doesn't work with the current simulation stages of xsimlab
    # I will try to fix it by using the GekkoContext Fluxes Dict instead

    def initialize(self):
        # get SVs
        self.C = self.gk_SVs[self.C_label]
        self.R = self.gk_SVs[self.R_label]

        print('Initialize NutrientDependency')
        self.halfsat_Par = self.m.Param(self.halfsat)
        self.nutrient_limitation = self.m.Intermediate(
            self.R / (self.halfsat_Par + self.R))


        print('Growth Dependency_component:', self.C_label, self.gk_Fluxes[self.C_label])
        # first multiply by growth rate
        growth = self.m.Intermediate(self.mu * self.nutrient_limitation * self.C)

        self.gk_Fluxes[self.R_label] = self.m.Intermediate(- growth)
        self.gk_Fluxes[self.C_label] = self.m.Intermediate(growth)
        print('GK flux', self.gk_Fluxes)



@xs.process
class NutrientDependency(InheritGekkoContext):
    """ """
    #LG_GrowthDep = xs.foreign(LimitedGrowth, 'GrowthDependencies')
    #LG_C_label = xs.foreign(LimitedGrowth, 'C_label')

    halfsat = xs.variable(intent='in')


    def initialize(self):
        print('Initialize NutrientDependency')
        self.gk_SVs['nutrient'] = self.m.Var()

        NatT = self.m.Intermediate(self.m.cos(self.gk_SVs['time'] / 365 * np.pi * 1.5) * .8 + 2)

        self.m.Equation(self.gk_SVs['nutrient'].dt() == NatT)

        self.LG_GrowthDep[self.LG_C_label] = self.m.Intermediate(
            self.gk_SVs['nutrient'] / (self.halfsat + self.gk_SVs['nutrient']))
        # self.LG_GrowthDep[self.LG_C_label] = self.m.Intermediate(self.gk_SVs['nutrient'] / (self.halfsat +0.2 + self.gk_SVs['nutrient']))
