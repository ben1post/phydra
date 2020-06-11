import numpy as np
import xsimlab as xs
from xsimlab import runtime_hook

from ..utility.modelcontext import GekkoMath
from .gekkocontext import GekkoContext


@xs.process
class Flux:
    # THIS kind of flux needs to check if dims are the same
    # provide different kind of fluxes, that check dims and work accordingly!
    m = xs.foreign(GekkoContext, 'm')
    gk_context = xs.foreign(GekkoContext, 'context')

    gk_SVs = xs.foreign(GekkoContext, 'SVs')
    gk_SVshapes = xs.foreign(GekkoContext, 'SVshapes')
    gk_Fluxes = xs.foreign(GekkoContext, 'Fluxes')

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
class BaseFlux:
    """Base Flux already contains all references to model context
    that can be utilized in subclasses"""
    m = xs.foreign(GekkoContext, 'm')
    gk_context = xs.foreign(GekkoContext, 'context')

    gk_SVs = xs.foreign(GekkoContext, 'SVs')
    gk_SVshapes = xs.foreign(GekkoContext, 'SVshapes')
    gk_Fluxes = xs.foreign(GekkoContext, 'Fluxes')


@xs.process
class LimitedGrowth(BaseFlux):
    """This flux defines the growth rate of 1 phytoplankton component,
    based on
    - nutrient uptake, according to Droop
    - additional dependencies: Light, Temperature
    """

    mu = xs.variable(intent='in', description='Maximum growth rate of component')

    C_label = xs.variable(intent='in')
    GrowthDependencies = xs.any_object()

    def initialize(self):
        self.GrowthDependencies = GekkoMath()
        try:
            print(f"LimitedGrowth flux is initialized for {self.C_label}")
        except:
            raise ('LimitedGrowth needs to be initialized with label for affected component')

        # get SVs
        self.C = self.gk_SVs[self.C_label]

    def run_step(self):
        print('Initializing GrowthDependencies:', self.GrowthDependencies)
        self.growth_deps = self.GrowthDependencies[self.C_label]
        print(self.growth_deps)

        # define flux for all dimensions of SV
        it = np.nditer(self.gk_SVshapes[self.C_label], flags=['zerosize_ok', 'multi_index'])
        growth_deps_it = np.nditer(self.growth_deps, flags=['zerosize_ok', 'multi_index', 'refs_ok'])
        while not it.finished:
            # first multiply by growth rate
            self.gk_Fluxes[self.C_label][it.multi_index] = self.m.Intermediate(self.mu * self.C[it.multi_index])
            while not growth_deps_it.finished:
                # multiply all collected growth_deps individually
                self.gk_Fluxes[self.C_label][it.multi_index] = self.m.Intermediate(growth_deps_it.value *
                                                                                   self.C[it.multi_index])
                growth_deps_it.iternext()
            it.iternext()


@xs.process
class NutrientDependency(BaseFlux):
    """ """
    LG_GrowthDep = xs.foreign(LimitedGrowth, 'GrowthDependencies')
    LG_C_label = xs.foreign(LimitedGrowth, 'C_label')

    halfsat = xs.variable(intent='in')

    def nutrientForcing(self, t):
        return np.cos(t / 365 * np.pi * 1.5) * .8 + 2

    def initialize(self):
        print('Initialize NutrientDependency')
        self.gk_SVs['nutrient'] = self.m.Var()

        NatT = self.m.Intermediate(self.m.cos(self.gk_SVs['time'] / 365 * np.pi * 1.5) * .8 + 2)

        self.m.Equation(self.gk_SVs['nutrient'].dt() == NatT)

        self.LG_GrowthDep[self.LG_C_label] = self.m.Intermediate(
            self.gk_SVs['nutrient'] / (self.halfsat + self.gk_SVs['nutrient']))
        # self.LG_GrowthDep[self.LG_C_label] = self.m.Intermediate(self.gk_SVs['nutrient'] / (self.halfsat +0.2 + self.gk_SVs['nutrient']))
