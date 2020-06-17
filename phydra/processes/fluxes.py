import numpy as np
import xsimlab as xs
from itertools import product

from .gekkocontext import InheritGekkoContext
from .forcing import MLDForcing

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
class LimitedGrowth(InheritGekkoContext):
    """This flux defines the growth rate of 1 phytoplankton component,
    based on
    - nutrient uptake, according to Droop
    - additional dependencies: Light, Temperature
    """
    flux_label = xs.variable(intent='out', groups='fx_flux_label')
    fx_output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='fxflux_output')

    mu = xs.variable(intent='in', description='Maximum growth rate of component')

    C_label = xs.variable(intent='in', description='label of component that grows')
    R_label = xs.variable(intent='in', description='label of ressource component that is consumed')

    halfsat = xs.variable(intent='in', description='half-saturation constant of nutrient uptake for component')

    def initialize(self):
        self.flux_label = f"LimitedGrowth-{self.R_label}2{self.R_label}"
        print('Initializing flux:', self.flux_label)
        # get SVs
        self.C = self.gk_SVs[self.C_label]
        self.R = self.gk_SVs[self.R_label]

        self.halfsat_Par = self.m.Param(self.halfsat)
        self.nutrient_limitation = self.m.Intermediate(
            self.R / (self.halfsat_Par + self.R))


        # first multiply by growth rate
        growth = self.m.Intermediate(self.mu * self.nutrient_limitation * self.C)
        print('GROWTH', growth)
        rt = np.nditer(self.gk_SVshapes[self.R_label], flags=['multi_index'])
        it = np.nditer(self.gk_SVshapes[self.C_label], flags=['multi_index'])
        while not rt.finished:
            while not it.finished:
                self.gk_Fluxes.apply_exchange_flux(self.R_label, self.C_label, growth,
                                                   it.multi_index, rt.multi_index)
                it.iternext()
            rt.iternext()

    def finalize_step(self):
        """Store flux output to array here!"""
        print('storing Nutrient Limtiation from:', self.flux_label)
        self.out = []

        for flux in self.gk_Flux_Int[self.fxflux_label]:
            self.out.append([val for val in flux])

        self.fx_output = np.array(self.out, dtype='float64')