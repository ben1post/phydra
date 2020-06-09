import numpy as np
import xsimlab as xs

from .gekkocontext import GekkoContext

def createMultiComp(base_process, comp_label, comp_dim):
    """This function allows creating specific instance of component during model setup
    a new subclass with the appropriate labels and dimensions is created by a dynamically
    created xs.process AddIndexComplabel inheritng form the base_process
    """
    @xs.process
    class AddIndexCompLabel(base_process):
        label = xs.variable(intent='out')
        dim = xs.variable(intent='out')
        index = xs.index(dims=comp_label)

        output = xs.variable(intent='out', dims=(comp_label, 'time'))

        def initialize(self):
            self.label = comp_label
            self.dim = comp_dim
            self.index = [f"{comp_label}-{i}" for i in range(comp_dim)]
            super(AddIndexCompLabel, self).initialize()

    return AddIndexCompLabel


@xs.process
class Component:
    m = xs.foreign(GekkoContext, 'm')
    gk_context = xs.foreign(GekkoContext, 'context')

    gk_SVs = xs.foreign(GekkoContext, 'SVs')
    gk_SVshapes = xs.foreign(GekkoContext, 'SVshapes')
    gk_Fluxes = xs.foreign(GekkoContext, 'Fluxes')

    gridshape = xs.foreign(GekkoContext, 'shape')

    init = xs.variable(intent='in')

    def initialize(self):
        print('Initializing component ', self.label)
        # add label to gk_context - components list:
        self.gk_context['components'] = (self.label, self.dim)

        # define np.array of full dimensions for this component:
        self.FullDims = np.zeros((self.gridshape, self.dim))

        # add to SVDims dict:
        self.gk_SVshapes[self.label] = self.FullDims
        self.gk_Fluxes[self.label] = np.array(self.FullDims, dtype='object')


        # define m.SV array in full model dimensions, add to SV dict:
        self.gk_SVs[self.label] = self.m.Array(self.m.SV, (self.FullDims.shape))

        # initialize SV m.Array with self.init val through FullDims multi_index
        it = np.nditer(self.FullDims, flags=['multi_index'])
        while not it.finished:
            self.gk_SVs[self.label][it.multi_index].value = self.init
            it.iternext()

    def run_step(self):
        """Assemble component equations from initialized fluxes"""
        print('Assembling equation for component ',self.label)
        it1 = np.nditer(self.gk_SVshapes[self.label], flags=['multi_index', 'refs_ok'])
        while not it1.finished:
            self.m.Equation(
                self.gk_SVs[self.label][it1.multi_index].dt() == \
                self.gk_Fluxes[self.label][it1.multi_index])
            it1.iternext()

    def finalize_step(self):
        """Store component output to array here!"""
        print('Storing output component ', self.label)
        out = []
        _it = np.nditer(self.gk_SVshapes[self.label], flags=['multi_index', 'refs_ok'])
        while not _it.finished:
            out.append([val for val in self.gk_SVs[self.label][_it.multi_index].value])
            _it.iternext()

        self.output = np.array(out, dtype='float64')