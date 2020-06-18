import numpy as np
import xsimlab as xs

from .gekkocontext import InheritGekkoContext
from .environments import BaseEnvironment, Slab


def make_FX_flux(fxflux_cls, fxflux_name):
    """
    This functions creates a properly labeled xs.process from class Component.

    :args:
        cls_name (cls): Class of forcing flux to be initialized
        dim_name (str): Name of sub-dimension of returned process, UPPERCASE!

    :returns:
        xs.process of class Component
    """
    new_dim = xs.index(dims=fxflux_name, groups='fxflux_index')
    base_dict = dict(fxflux_cls.__dict__)
    base_dict[fxflux_name] = new_dim

    new_cls_name = fxflux_cls.__name__ + '_' + fxflux_name
    print(new_cls_name)
    new_cls = type(new_cls_name, fxflux_cls.__bases__, base_dict)

    def initialize_dim(self):
        c_labels = getattr(self, 'C_labels')
        cls_label = getattr(self, '__xsimlab_name__')
        print(f"forcing flux {cls_label} is initialized at {c_labels}")
        setattr(self, 'fxflux_label', str(cls_label))
        fx_c_list = []
        for lab in c_labels:
            if self.gk_SVshapes[lab].size == 1:
                fx_c_list.append(f"{cls_label}-{lab}")
            else:
                for i in range(self.gk_SVshapes[lab].size):
                    fx_c_list.append(f"{cls_label}-{lab}-{i}")
        print(fx_c_list)
        setattr(self, fxflux_name, fx_c_list)

        cls_here = getattr(self, '__class__')
        super(cls_here, self).initialize_postdimsetup()

    setattr(new_cls, 'initialize', initialize_dim)

    print(fxflux_cls, fxflux_name)
    if fxflux_name.lower() == fxflux_name:
        raise ValueError(f"dimension label ({fxflux_name}) supplied to forcing flux {fxflux_cls} needs to be Upper Case")

    new_cls.C_labels.metadata['dims'] = fxflux_name.lower()
    new_cls.fx_output.metadata['dims'] = ((fxflux_name, 'time'),)
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

    @flux_func.compute
    def function_returning_flux(self):
        """ returns a function that is used in environment to calculate flux """
        raise ValueError('flux function needs to be defined in ForcingFlux subclass')


@xs.process
class Mixing(InheritGekkoContext):
    """ Mixing:
    - this provides the mixing function to the environment that references it

    the mixing function takes
    - standard parameters, are passed along! as default args, right?
    - can I use the contextdict feature somehow, for these pars.. hm.
    - can so xs.group was the

    """
    fxflux_label = xs.variable(intent='out', groups='fx_flux_label')
    fx_output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='fxflux_output')
    C_labels = xs.variable(intent='in',
                           dims='not_initialized',
                           description='label of component(s) that grows')
    flux = xs.on_demand(description='function to calculate fluxes')

    MLD = xs.foreign(Slab, 'MLD')  # m.Param()
    MLD_deriv = xs.foreign(Slab, 'MLD_deriv')  # m.Param()
    N0_forcing = xs.foreign(Slab, 'N0_forcing')

    kappa = xs.variable(intent='in', description='constant mixing coefficient')
    mixing = xs.on_demand(description='function to calculate mixing K')

    def initialize_postdimsetup(self):
        print(f"Initializing forcing flux {self.fxflux_label} for components {self.C_labels}")

        for C_label in self.C_labels:
            print(C_label, 'initFXflux', self.gk_Fluxes[C_label])

            # this supplies intermediate flux to specific component
            it = np.nditer(self.gk_SVshapes[C_label], flags=['multi_index'])
            while not it.finished:
                self.C = self.gk_SVs[C_label][it.multi_index]
                # this collects flux intermediates for output collection
                self.gk_Flux_Int[self.fxflux_label] = self.flux()
                # define actual flux
                self.gk_Fluxes.apply_across_dims(C_label, self.flux(), it.multi_index)
                it.iternext()

            print(C_label, 'initFXflux', self.gk_Fluxes[C_label])


    def finalize_step(self):
        """Store flux output to array here!"""
        print('storing flux to output: ', self.fxflux_label)
        self.out = []

        for flux in self.gk_Flux_Int[self.fxflux_label]:
            self.out.append([val for val in flux])

        self.fx_output = np.array(self.out, dtype='float64')

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
    fxflux_label = xs.variable(intent='out', groups='fx_flux_label')
    fx_output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='fxflux_output')
    C_labels = xs.variable(intent='in',
                           dims='not_initialized',
                           description='label of component(s) that grows',
                           )
    flux = xs.on_demand(description='function to calculate fluxes')

    @flux.compute
    def flux(self):
        return self.m.Intermediate( - self.C * self.mixing())


class Upwelling(Mixing):
    """ Nutrient upwelling from a constant source below the mixed layer """
    fxflux_label = xs.variable(intent='out', groups='fx_flux_label')
    fx_output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='fxflux_output')
    C_labels = xs.variable(intent='in',
                           dims='not_initialized',
                           description='label of component(s) that grows')
    flux = xs.on_demand(description='function to calculate fluxes')

    @flux.compute
    def flux(self):
        return self.m.Intermediate((self.N0_forcing - self.C) * self.mixing())
