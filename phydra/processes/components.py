import numpy as np
import xsimlab as xs

from .gekkocontext import GekkoContext

@xs.process
class AllComponents:
    """ Process that collects all component output into one single variable for easier plotting
    Note: only works for (ALL) one dimensional components for now! """

    components = xs.index(dims='components')
    outputs = xs.variable(intent='out', dims=('components', 'time'))

    comp_labels = xs.group('comp_label')
    comp_dims = xs.group('comp_dim')
    comp_indices = xs.group('comp_index')
    comp_outputs = xs.group('comp_output')

    def initialize(self):
        self.components = [index for indices in self.comp_indices for index in indices]

    def finalize_step(self):
        self.outputs = [output for outputs in self.comp_outputs for output in outputs]



@xs.process
class Component:
    """This is the basis for a state variable in the model,
    or an array of state variables that share the same mathematical equations

    ToDo:
    - Figure out how to best supply allometric parameterization for MultiComponents
        Options: with Flux, external xs.Process (intent='inout),
        or wrapper function for create_setup with dims passed at model creation
    """
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
        self.gk_context['comp_dims'] = (self.label,self.dim)

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