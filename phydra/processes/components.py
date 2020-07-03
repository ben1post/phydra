import numpy as np
import xsimlab as xs

from .gekkocontext import InheritGekkoContext, GekkoContext
from time import process_time


@xs.process
class StateVariable(GekkoContext):
    """ TODO: allow modifying dims in environment, not here (?) """
    label = xs.variable(intent='out')
    value = xs.variable(intent='out', dims='time')
    SV = xs.variable(intent='out')

    def initialize(self):
        self.label = self.__xsimlab_name__
        print('initializing SV')
        self.SV = self.m.SV(name='Nutrient')
        self.value = self.SV.value



# So how about I pass dims to env, and this passes it to SVs, initializing at proper dims





###############################################################################################
def make_Component(cls_name, dim_name, comp_type):
    """
    This functions creates a properly labeled xs.process from class Component.

    :args:
        cls_name (str): Name of returned process
        dim_name (str): Name of sub-dimension of returned process

    :returns:
        new_cls (xs.process) : Component with specific names,
        an additional initialize_dim() function,
        and a new xs.index variable, as attribute 'dim_name'
    """
    new_dim = xs.index(dims=dim_name, groups='comp_index')
    base_dict = dict(comp_type.__dict__)
    base_dict[dim_name] = new_dim

    new_cls = type(cls_name, comp_type.__bases__, base_dict)
    new_cls.dim_labels.metadata['dims'] = dim_name

    def initialize_dim(self):
        dim = getattr(self, 'dim')
        cls_label = getattr(self, '__xsimlab_name__')
        print(f"dimensions of component {cls_label} are initialized at {dim}")
        setattr(self, 'comp_label', str(cls_label))
        if dim == 1:
            index_list = [f"{cls_label}"]
        else:
            index_list = [f"{cls_label}-{i}" for i in range(dim)]
        setattr(self, dim_name, index_list)
        setattr(self, 'dim_labels', index_list)

        cls_here = getattr(self, '__class__')
        super(cls_here, self).initialize_postdimsetup()

    setattr(new_cls, 'initialize', initialize_dim)
    new_cls.output.metadata['dims'] = ((dim_name, 'time'),)

    return xs.process(new_cls)


@xs.process
class BaseComponent(InheritGekkoContext):
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

    output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='comp_output')
    dim_labels = xs.variable(intent='out', groups='comp_label')

    # Necessary Input:
    init = xs.variable(intent='in')
    dim = xs.variable(intent='in', groups='comp_dim')

    def initialize_parametersetup(self):
        pass

    def initialize_postdimsetup(self):
        print('Initializing component ', self.comp_label, self.dim_labels)
        # add label to gk_context - components list:
        self.gk_context['comp_dims'] = (self.comp_label, self.dim)
        # define np.array of full dimensions for this component:
        self.FullDims = np.zeros((self.gridshape, self.dim))

        # add to SVDims dict:
        self.gk_SVshapes[self.comp_label] = self.FullDims

        # define m.SV array in full model dimensions, add to SV dict:
        self.gk_SVs[self.comp_label] = self.m.Array(self.m.SV, (self.FullDims.shape))

        # initialize SV m.Array with self.init val through FullDims multi_index
        it = np.nditer(self.FullDims, flags=['multi_index'])
        while not it.finished:
            self.gk_SVs[self.comp_label][it.multi_index].value = self.init
            # self.gk_SVs[self.comp_label][it.multi_index].lower = 0
            it.iternext()

        self.gk_Fluxes.setup_dims(self.comp_label, self.FullDims)

        super(getattr(self, '__class__'), self).initialize_parametersetup()

    def run_step(self):
        """Assemble component equations from initialized fluxes"""
        print('Assembling equation for component ', self.comp_label, self.gk_Fluxes[self.comp_label])

        # initialize SV m.Array with self.init val through FullDims multi_index
        it = np.nditer(self.FullDims, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            self.m.Equation(
                self.gk_SVs[self.comp_label][it.multi_index].dt() == \
                sum([flux for flux in self.gk_Fluxes[self.comp_label][it.multi_index]]))
            it.iternext()

    def finalize_step(self):
        """Store component output to array here!"""
        print('Storing output for component ', self.comp_label)
        out = []
        save_start = process_time()
        # initialize SV m.Array with self.init val through FullDims multi_index
        it = np.nditer(self.FullDims, flags=['multi_index'])
        while not it.finished:
            out.append([val for val in self.gk_SVs[self.comp_label][it.multi_index]])
            it.iternext()

        self.output = np.array(out, dtype='float64')

        save_end = process_time()

        print(f"storing component done in {round(save_end - save_start, 2)} seconds")


class Component(BaseComponent):
    """ Basic Component that can be initialized with make_Component """
    output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='comp_output')
    dim_labels = xs.variable(intent='out', groups='comp_label')

    # Necessary Input:
    init = xs.variable(intent='in')
    dim = xs.variable(intent='in', groups='comp_dim')



class SizeComponent(BaseComponent):
    """Component that stores an array of initialized sizes """
    output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='comp_output')
    dim_labels = xs.variable(intent='out', groups='comp_label')

    # Necessary Input:
    init = xs.variable(intent='in')
    dim = xs.variable(intent='in', groups='comp_dim')

    size_min = xs.variable(intent='in')
    size_max = xs.variable(intent='in')

    def initialize_parametersetup(self):
        self.gk_Parameters.setup_dims(self.comp_label, 'Size', self.gk_SVshapes[self.comp_label].shape)
        self.gk_Parameters.init_param_range(self.comp_label, 'Size', self.size_min,
                                            self.size_max, spacing='log')

        print(f"Size Range Initialized for {self.comp_label} with sizes {self.gk_Parameters[self.comp_label]['Size']}")
