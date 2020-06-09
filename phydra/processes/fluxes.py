import numpy as np
import xsimlab as xs

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

    growthrate = xs.variable(intent='in')
    c1_label = xs.variable(intent='in')
    c2_label = xs.variable(intent='in')

    def initialize(self):
        print('Initializing Flux')
        # get SVs
        C1 = self.gk_SVs[self.c1_label]
        C2 = self.gk_SVs[self.c2_label]

        # define flux for all dimensions of SV
        it1 = np.nditer(self.gk_SVshapes[self.c1_label], flags=['multi_index', 'refs_ok'])
        it2 = np.nditer(self.gk_SVshapes[self.c2_label], flags=['multi_index', 'refs_ok'])
        while not it1.finished:
            while not it2.finished:
                self.gk_Fluxes[self.c1_label][it1.multi_index] = self.m.Intermediate(C1[it1.multi_index] * self.growthrate)
                self.gk_Fluxes[self.c2_label][it2.multi_index] = self.m.Intermediate(C2[it2.multi_index] * self.growthrate)
                it1.iternext()
                it2.iternext()



#TODO:
# rewrite everything below here, according to BaseFlux above!

def needstobemodified():
    @xs.process
    class Flux:
        var = xs.any_object("This contains the actual mathematical formulation described by this term")

        _in = xs.any_object("This handles input (and output) to var")

        def initialize(self):
            self.var = 0

        def run_step(self):
            self._delta = self.solve(self.var)


    #xs.Model({'Flux':Flux})


    @xs.process
    class NutrientUptake:
        """This is an example for a MultiComp interacting with a SingularComp"""
        Model_dims = xs.foreign(Flux, 'dims')

        N = xs.foreign(Flux, 'state')
        P = xs.foreign(Flux, 'state')

        N_uptake = xs.variable(dims=('x', 'y', 'Env', 'N'), intent='out', groups='N_flux')
        P_growth = xs.variable(dims=('x', 'y', 'Env', 'P'), intent='out', groups='P_flux')

        P_halfsat = xs.foreign(Flux, 'halfsat')

        NutLim = xs.variable(intent='out')

        @property
        def NutrientLimitation(self):
            lim = self.N / (self.P_halfsat + self.N)
            # print(lim.shape, np.zeros_like(self.N).shape)
            return lim

        def initialize(self):
            self.N_uptake = np.zeros_like(self.N)
            self.P_growth = np.zeros_like(self.P)

        def run_step(self):
            # calculate Nutrient limitation:
            self.NutLim = np.array(self.NutrientLimitation, dtype='float64')

            self.P_growth = self.NutLim * self.P

            # since there is only a single N, that dimension is summed up via "axis = -1"
            self.N_uptake = - np.sum(self.P_growth, axis=-1, keepdims=True)  # negative flux


    @xs.process
    class PhytoplanktonMortality:
        """Quadratic mortality """
        Model_dims = xs.foreign(Flux, 'dims')

        P = xs.foreign(Flux, 'state')

        P_mortality = xs.variable(dims=('x', 'y', 'Env', 'P'), intent='out', groups='P_flux')

        P_mortality_rate = xs.foreign(Flux, 'mortality_rate')

        def initialize(self):
            self.P_mortality = np.zeros_like(self.P)

        def run_step(self):
            self.P_mortality = - np.array(self.P_mortality_rate * self.P ** 2, dtype='float64')

