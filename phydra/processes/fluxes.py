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
        self.inputrate_c1_allo = np.linspace(self.inputrate_c1, self.inputrate_c1+0.1, self.gk_context['comp_dims'][self.c1_label])

        # define flux for all dimensions of SV
        it1 = np.nditer(self.gk_SVshapes[self.c1_label], flags=['multi_index', 'refs_ok'])
        it2 = np.nditer(self.gk_SVshapes[self.c2_label], flags=['multi_index', 'refs_ok'])
        while not it1.finished:
            while not it2.finished:
                self.conversionrate = self.m.Intermediate(C1[it1.multi_index] * self.conversion2_c2)
                self.gk_Fluxes[self.c1_label][it1.multi_index] = self.m.Intermediate(self.inputrate_c1_allo[it1.multi_index[1]] - self.conversionrate)
                self.gk_Fluxes[self.c2_label][it2.multi_index] = self.m.Intermediate(self.conversionrate - self.destruction_c2 * C2[it2.multi_index])
                it1.iternext()
                it2.iternext()