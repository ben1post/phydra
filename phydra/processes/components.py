import numpy as np
import xsimlab as xs

from .gekkocontext import InheritGekkoContext


@xs.process
class Time(InheritGekkoContext):
    days = xs.variable(dims='time', description='time in days')

    # for indexing xarray IO objects
    time = xs.index(dims='time', description='time in days')

    def initialize(self):
        print('Initializing Model Time')
        self.time = self.days

        # ASSIGN MODEL SOLVING TIME HERE:
        self.m.time = self.time

        self.gk_SVs['time'] = self.m.Var(0, lb=0)
        # add variable keeping track of time within model:
        self.m.Equation(self.gk_SVs['time'].dt() == 1)

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


def make_Component(cls_name, dim_name):
    """
    This functions creates a properly labeled xs.process from class Component.

    :args:
        cls_name (str): Name of returned process
        dim_name (str): Name of sub-dimension of returned process

    :returns:
        xs.process of class Component
    """
    new_cls = type(cls_name, Component.__bases__, dict(Component.__dict__))
    new_cls.index.metadata['dims'] = dim_name
    new_cls.output.metadata['dims'] = ((dim_name, 'time'),)

    return xs.process(new_cls)


class Component(InheritGekkoContext):
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.


    -------------------------------------------------------------------
    This is the basis for a state variable in the model,
    or an array of state variables that share the same mathematical equations

    has to be initialized through the make_Component function above!

    ToDo:
    - Figure out how to best supply allometric parameterization for MultiComponents
        Options: with Flux, external xs.Process (intent='inout),
        or wrapper function for create_setup with dims passed at model creation
    """

    index = xs.index(dims='not_initialized', groups='comp_index')

    output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='comp_output')

    # Necessary Input:
    init = xs.variable(intent='in')
    label = xs.variable(intent='in', groups='comp_label')
    dim = xs.variable(intent='in', groups='comp_dim')

    def initialize(self):
        if self.dim == 1:
            self.index = [f"{self.label}"]
        else:
            self.index = [f"{self.label}-{i}" for i in range(self.dim)]

        print('Initializing component ', self.label)
        print(self.index)
        # add label to gk_context - components list:
        self.gk_context['comp_dims'] = (self.label, self.dim)

        # define np.array of full dimensions for this component:
        self.FullDims = np.zeros((self.gridshape, self.dim))

        # add to SVDims dict:
        self.gk_SVshapes[self.label] = self.FullDims
        #self.gk_Fluxes[self.label] = np.array(self.FullDims, dtype='object')

        print('GKFLUXES', self.gk_Fluxes)
        # define m.SV array in full model dimensions, add to SV dict:
        #self.gk_SVs[self.label] = self.m.Array(self.m.SV, (self.FullDims.shape))
        self.gk_SVs[self.label] = self.m.SV()

        # initialize SV m.Array with self.init val through FullDims multi_index
        #it = np.nditer(self.FullDims, flags=['multi_index'])
        #while not it.finished:
        self.gk_SVs[self.label].value = self.init
        #    it.iternext()

    def run_step(self):
        """Assemble component equations from initialized fluxes"""
        print('Assembling equation for component ', self.label)
        #it1 = np.nditer(self.gk_SVshapes[self.label], flags=['multi_index', 'refs_ok'])
        #while not it1.finished:
        print('FLUXES:', self.gk_SVs[self.label], self.gk_Fluxes[self.label])
        print([flux for flux in self.gk_Fluxes[self.label]])
        self.m.Equation(
            self.gk_SVs[self.label].dt() == \
            sum([flux for flux in self.gk_Fluxes[self.label]]))
            #it1.iternext()

    def finalize_step(self):
        """Store component output to array here!"""
        print('Storing output component ', self.label)
        out = []
        #_it = np.nditer(self.gk_SVshapes[self.label], flags=['multi_index', 'refs_ok'])
        #while not _it.finished:
        out.append([val for val in self.gk_SVs[self.label]])
        #_it.iternext()

        self.output = np.array(out, dtype='float64')