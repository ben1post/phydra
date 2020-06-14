import numpy as np
import xsimlab as xs

from .gekkocontext import InheritGekkoContext
from .environments import BaseEnvironment, Slab


def make_FX_flux(cls_fx_flux, dim_name):
    """
    This functions creates a properly labeled xs.process from class Component.

    :args:
        cls_name (str): Name of returned process
        dim_name (str): Name of sub-dimension of returned process

    :returns:
        xs.process of class Component
    """
    print(cls_fx_flux, dim_name)
    new_cls_name = cls_fx_flux.__name__ + '_' + dim_name
    new_cls = type(new_cls_name, cls_fx_flux.__bases__, dict(cls_fx_flux.__dict__))
    print(new_cls)
    new_cls.C_labels.metadata['dims'] = dim_name
    new_cls.fx_index.metadata['dims'] = dim_name
    new_cls.fx_output.metadata['dims'] = ((dim_name, 'time'),)

    return xs.process(new_cls)


@xs.process
class ForcingFlux(InheritGekkoContext):
    """Base structure of a Forcing Flux

    The basics of a forcing flux, is that they affect a single COMP
    no interaction!
    but they can affect multiple COMPS in the sameway!
    """
    flux = xs.any_object()

    C_labels = xs.variable(intent='in',
                           dims='not_initialized',
                           description='label of component(s) that grows')

    flux_func = xs.on_demand()

    def initialize(self):
        """ basic initialisation of forcing """
        raise ValueError('initialization needs to be defined in ForcingFlux subclass')

        #for C_label in self.C_labels:
        #    self.gk_Fluxes[C_label] = self.m.Intermediate(self.C * self.mixing())

    @flux_func.compute
    def function_returning_flux(self):
        """ returns a function that is used in environment to calculate flux """
        raise ValueError('flux function needs to be defined in ForcingFlux subclass')

def notes():
    """
    properly got the idea now,
    simply every forcing flux can have multiple components as input!
    and then specific parameters can be initialized by passing that
    specific component through another instance.
    NICCE!
    this has much less complexity.

    only question remains, what to do of the environment?
    but anyways, will think of that later
    with the current gekko setup, I can create higher dimensions, after setting up equations, right?!
    let's quick try..
    NOPE, not possible, need to initialize
    perhaps I can wrap everything in an object though.. hm.. like input == output dims automatically

    BUT, I can pass multiple dims for any xs.process as an input var,
    then within ForcingFlux, simply iterate through all keys in input var

    """
    pass

@xs.process
class Mixing(InheritGekkoContext):
    """ Mixing:
    - this provides the mixing function to the environment that references it

    the mixing function takes
    - standard parameters, are passed along! as default args, right?
    - can I use the contextdict feature somehow, for these pars.. hm.
    - can so xs.group was the

    """
    fx_index = xs.index(dims='not_initialized', groups='flux_index')
    fx_output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='flux_output')


    MLD = xs.foreign(Slab, 'MLD')  # m.Param()
    MLD_deriv = xs.foreign(Slab, 'MLD_deriv')  # m.Param()

    N0_forcing = xs.foreign(Slab, 'N0_forcing')

    kappa = xs.variable(intent='in', groups='forcingpar_label', description='constant mixing coefficient')

    mixing = xs.on_demand(description='function to calculate mixing K')
    flux = xs.on_demand(description='function to calculate fluxes')

    C_labels = xs.variable(intent='in',
                           dims='not_initialized',
                           description='label of component(s) that grows')

    label = xs.variable(intent='out', groups='fx_flux_label')

    def initialize(self):
        self.label = self.__dict__['__xsimlab_name__']

        self.fx_index = [f"{C_label}-{self.label}" for C_label in self.C_labels]

        print('Flux label: ', self.label)

        for C_label in self.C_labels:
            self.C = self.gk_SVs[C_label]
            # this stores intermediate to be stored as output later on
            # needs a two-level dict, like "intermediate[comp][flux] = "
            self.gk_Flux_Int[self.label] = self.flux()
            # this supplies intermediate flux to specific component
            self.gk_Fluxes[C_label] = self.flux()

        print(self.gk_Flux_Int)


    def finalize_step(self):
        """Store flux output to array here!"""
        print('storing flux to output: ', self.label)
        self.out = []

        print('index', self.fx_index)
        #print('output', self.output)

        for flux in self.gk_Flux_Int[self.label]:
            self.out.append([val for val in flux])

        print(len(self.out), np.shape(self.out))

        self.fx_output = np.array(self.out, dtype='float64')

        print('hello there')

    @mixing.compute
    def mixing(self):
        h_pos = self.m.Intermediate(np.max(self.MLD_deriv, 0))
        H = self.MLD
        K = self.m.Intermediate((h_pos + self.kappa) / H)
        return K

    @flux.compute
    def flux(self):
        return self.m.Intermediate(- self.C * self.mixing())


class Sinking(Mixing):
    """negative mixing flux"""
    fx_index = xs.index(dims='not_initialized', groups='flux_index')
    fx_output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='flux_output')

    C_labels = xs.variable(intent='in',
                           dims='not_initialized',
                           description='label of component(s) that grows')

    flux = xs.on_demand(description='function to calculate fluxes')

    @flux.compute
    def flux(self):
        return self.m.Intermediate( - self.C * self.mixing())


class Upwelling(Mixing):
    """ Nutrient upwelling from a constant source below the mixed layer """
    fx_index = xs.index(dims='not_initialized', groups='flux_index')
    fx_output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='flux_output')

    C_labels = xs.variable(intent='in',
                           dims='not_initialized',
                           description='label of component(s) that grows')

    flux = xs.on_demand(description='function to calculate fluxes')

    @flux.compute
    def flux(self):
        return self.m.Intermediate((self.N0_forcing - self.C) * self.mixing())
