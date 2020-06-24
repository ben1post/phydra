import numpy as np
import xsimlab as xs

from .gekkocontext import InheritGekkoContext
from .environments import Slab, Chemostat


def make_FX_flux(fxflux_cls, fxflux_name):
    """
    This functions creates a properly labeled xs.process from class Component.

    :args:
        cls_name (cls): Class of forcing flux to be initialized
        dim_name (str): Name of sub-dimension of returned process, UPPERCASE!

    :returns:
        xs.process of class Component
    """
    new_dim = xs.index(dims=(fxflux_name), groups='fxflux_index')
    base_dict = dict(fxflux_cls.__dict__)
    base_dict[fxflux_name] = new_dim

    new_cls_name = fxflux_cls.__name__ + '_' + fxflux_name
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
        setattr(self, fxflux_name, fx_c_list)

        cls_here = getattr(self, '__class__')
        super(cls_here, self).initialize_postdimsetup()

    setattr(new_cls, 'initialize', initialize_dim)

    if fxflux_name.lower() == fxflux_name:
        raise ValueError(f"dimension label ({fxflux_name}) supplied to forcing flux {fxflux_cls} needs to be Upper Case")
    # here the Component label affects all sub components, therefore C_labels dim != Forcingflux dims
    new_cls.C_labels.metadata['dims'] = (fxflux_name.lower())
    new_cls.fx_output.metadata['dims'] = ((fxflux_name, 'time'),)
    return xs.process(new_cls)


@xs.process
class BaseForcingFlux(InheritGekkoContext):
    """Base structure of a Forcing Flux

    The basics of a forcing flux, is that they affect a single COMP
    no interaction!
    but they can affect multiple COMPS in the sameway!
    """
    fxflux_label = xs.variable(intent='out', groups='fx_flux_label')
    fx_output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='fxflux_output')
    C_labels = xs.variable(intent='in',
                           dims='not_initialized',
                           description='label of component(s) that grows')
    fxflux = xs.on_demand(description='function to calculate fluxes')

    def initialize_postdimsetup(self):
        print(f"Initializing forcing flux {self.fxflux_label} for components {self.C_labels}")

        for C_label in self.C_labels:
            # this supplies intermediate flux to specific component
            it = np.nditer(self.gk_SVshapes[C_label], flags=['multi_index'])
            while not it.finished:
                self.C = self.gk_SVs[C_label][it.multi_index]
                # this collects flux intermediates for output collection
                self.gk_Flux_Int[self.fxflux_label] = self.fxflux
                # define actual flux
                self.gk_Fluxes.apply_flux(C_label, self.fxflux, it.multi_index)
                it.iternext()

    def finalize_step(self):
        """Store flux output to array here!"""
        print('storing flux to output: ', self.fxflux_label)
        self.out = []

        for flux in self.gk_Flux_Int[self.fxflux_label]:
            self.out.append([val for val in flux])

        self.fx_output = np.array(self.out, dtype='float64')

    @fxflux.compute
    def function_returning_flux(self):
        """ returns a function that is used in environment to calculate flux """
        raise ValueError('flux function needs to be defined in ForcingFlux subclass')


class LinearMortalityClosure(BaseForcingFlux):
    """negative mixing flux"""
    fxflux_label = xs.variable(intent='out', groups='fx_flux_label')
    fx_output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='fxflux_output')
    C_labels = xs.variable(intent='in',
                           dims='not_initialized',
                           description='label of component(s) that grows')
    fxflux = xs.on_demand(description='function to calculate fluxes')

    mortality_rate = xs.variable(intent='in', description='mortality rate of component')

    @fxflux.compute
    def linearmortality(self):
        return self.m.Intermediate(- self.mortality_rate * self.C)


@xs.process
class Flow(BaseForcingFlux):
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
    fxflux = xs.on_demand(description='function to calculate fluxes')

    Flow_forcing = xs.foreign(Chemostat, 'Flow')  # m.Param()

    flow = xs.on_demand(description='function to calculate mixing K')

    @fxflux.compute
    def fxflux_out(self):
        return self.m.Intermediate(self.C * self.flow)

    @flow.compute
    def influx(self):
        return self.m.Intermediate(self.Flow_forcing)

class Outflow(Flow):
    """negative mixing flux"""
    fxflux_label = xs.variable(intent='out', groups='fx_flux_label')
    fx_output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='fxflux_output')
    C_labels = xs.variable(intent='in',
                           dims='not_initialized',
                           description='label of component(s) that grows',
                           )
    fxflux = xs.on_demand(description='function to calculate fluxes')

    @fxflux.compute
    def influx(self):
        return self.m.Intermediate(- self.C * self.flow)

class N0_inflow(Flow):
    """negative mixing flux"""
    fxflux_label = xs.variable(intent='out', groups='fx_flux_label')
    fx_output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='fxflux_output')
    C_labels = xs.variable(intent='in',
                           dims='not_initialized',
                           description='label of component(s) that grows',
                           )
    fxflux = xs.on_demand(description='function to calculate fluxes')

    N0_forcing = xs.foreign(Chemostat, 'N0_forcing')

    @fxflux.compute
    def influx(self):
        return self.m.Intermediate(self.N0_forcing * self.flow)


@xs.process
class Mixing(BaseForcingFlux):
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
    fxflux = xs.on_demand(description='function to calculate fluxes')

    MLD = xs.foreign(Slab, 'MLD')  # m.Param()
    MLD_deriv = xs.foreign(Slab, 'MLD_deriv')  # m.Param()
    N0_forcing = xs.foreign(Slab, 'N0_forcing')  # m.Param()

    kappa = xs.variable(intent='in', description='constant mixing coefficient')
    mixing = xs.on_demand(description='function to calculate mixing K')

    @fxflux.compute
    def fxflux_out(self):
        return self.m.Intermediate(- self.C * self.mixing)

    @mixing.compute
    def mixing_out(self):
        h_pos = self.m.Intermediate(np.max(self.MLD_deriv, 0))
        H = self.MLD
        K = self.m.Intermediate((h_pos + self.kappa) / H)
        return K


class Sinking(Mixing):
    """negative mixing flux"""
    fxflux_label = xs.variable(intent='out', groups='fx_flux_label')
    fx_output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='fxflux_output')
    C_labels = xs.variable(intent='in',
                           dims='not_initialized',
                           description='label of component(s) that grows',
                           )
    fxflux = xs.on_demand(description='function to calculate fluxes')

    @fxflux.compute
    def sinking(self):
        return self.m.Intermediate( - self.C * self.mixing)


class Upwelling(Mixing):
    """ Nutrient upwelling from a constant source below the mixed layer """
    fxflux_label = xs.variable(intent='out', groups='fx_flux_label')
    fx_output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='fxflux_output')
    C_labels = xs.variable(intent='in',
                           dims='not_initialized',
                           description='label of component(s) that grows')
    fxflux = xs.on_demand(description='function to calculate fluxes')

    @fxflux.compute
    def upwelling(self):
        return self.m.Intermediate((self.N0_forcing - self.C) * self.mixing)
