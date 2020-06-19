import numpy as np
import xsimlab as xs

from .gekkocontext import InheritGekkoContext

def make_flux(flux_cls, flux_name):
    """
    This functions creates a properly labeled xs.process from class Component.

    :args:
        cls_name (cls): Class of forcing flux to be initialized
        dim_name (str): Name of sub-dimension of returned process, UPPERCASE!

    :returns:
        xs.process of class Component
    """
    new_dim = xs.index(dims=flux_name, groups='flux_index')
    base_dict = dict(flux_cls.__dict__)
    base_dict[flux_name] = new_dim

    new_cls_name = flux_cls.__name__ + '_' + flux_name
    new_cls = type(new_cls_name, flux_cls.__bases__, base_dict)

    def initialize_dim(self):
        r_label = getattr(self, 'R_label')
        c_label = getattr(self, 'C_label')

        cls_label = getattr(self, '__xsimlab_name__')
        setattr(self, 'cls_label', cls_label)
        print(f"flux {cls_label} is initialized for {r_label} --> {c_label}")

        setattr(self, 'flux_label', str(cls_label))
        fx_c_list = []
        for r_lab, c_lab in zip(r_label, c_label):
            if self.gk_SVshapes[r_lab].size == 1 and self.gk_SVshapes[c_lab].size == 1:
                fx_c_list.append(f"{cls_label}-{r_lab}-2-{c_lab}")
            elif self.gk_SVshapes[r_lab].size == 1:
                for c in range(self.gk_SVshapes[c_lab].size):
                    fx_c_list.append(f"{cls_label}-{r_lab}-2-{c_lab}{c}")
            else:
                for r in range(self.gk_SVshapes[r_lab].size):
                    for c in range(self.gk_SVshapes[c_lab].size):
                        fx_c_list.append(f"{cls_label}-{r_lab}{r}-2-{c_lab}{c}")

        setattr(self, flux_name, fx_c_list)

        cls_here = getattr(self, '__class__')
        super(cls_here, self).initialize_postdimsetup()

    setattr(new_cls, 'initialize', initialize_dim)

    if flux_name.lower() == flux_name:
        raise ValueError(f"dimension label ({flux_name}) supplied to forcing flux {flux_cls} needs to be Upper Case")

    new_cls.output.metadata['dims'] = ((flux_name, 'time'),)
    return xs.process(new_cls)


@xs.process
class BaseFlux(InheritGekkoContext):
    """ base class for a flux that defines an interaction between 2 or more components """
    flux_label = xs.variable(intent='out', groups='flux_label')
    output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='flux_output')

    R_label = xs.variable(intent='in', description='label of ressource that is consumed')
    C_label = xs.variable(intent='in', description='label of component that grows')

    flux = xs.on_demand()

    def initialize_postdimsetup(self):
        self.flux_label = f"{self.cls_label}-{self.R_label}2{self.C_label}"
        print('Initializing flux:', self.flux_label)

        # apply growth of all subdimensions of the consumer on all subdimensions of the ressource
        itR = np.nditer(self.gk_SVshapes[self.R_label], flags=['multi_index'])
        itC = np.nditer(self.gk_SVshapes[self.C_label], flags=['multi_index'])
        while not itR.finished:
            self.R = self.gk_SVs[self.R_label][itR.multi_index]
            while not itC.finished:
                self.C = self.gk_SVs[self.C_label][itC.multi_index]

                self.gk_Fluxes.apply_exchange_flux(self.R_label, self.C_label, self.flux,
                                                   itR.multi_index, itC.multi_index)
                self.gk_Flux_Int[self.flux_label] = self.flux
                itC.iternext()
            itR.iternext()

    @flux.compute
    def flux(self):
        """ basic initialisation of forcing,
        inherits SV for ressource self.R and for consumer self.C from context"""
        raise ValueError('flux needs to be defined in BaseFlux subclass')

    def finalize_step(self):
        """Store flux output to array here!"""
        print('storing flux:', self.flux_label)
        self.out = []

        for flux in self.gk_Flux_Int[self.flux_label]:
            self.out.append([val for val in flux])

        self.output = np.array(self.out, dtype='float64')


class LimitedGrowth(BaseFlux):
    """ Base Limited Growth, that inherits from BaseFlux and needs to be initialized from """
    flux_label = xs.variable(intent='out', groups='flux_label')
    output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='flux_output')

    R_label = xs.variable(intent='in', description='label of ressource component that is consumed')
    C_label = xs.variable(intent='in', description='label of component that grows')

    mu = xs.variable(intent='in', description='Maximum growth rate of component')
    halfsat = xs.variable(intent='in', description='half-saturation constant of nutrient uptake for component')

    flux = xs.on_demand()
    nutrient_limitation = xs.on_demand()

    @flux.compute
    def growth(self):
        return self.m.Intermediate(self.mu * self.nutrient_limitation() * self.C)

    @nutrient_limitation.compute
    def nutrient_limitation(self):
        self.halfsat_Par = self.m.Param(self.halfsat)
        return self.m.Intermediate(self.R / (self.halfsat_Par + self.R))


class LinearMortality(BaseFlux):
    """ Base Limited Growth, that inherits from BaseFlux and needs to be initialized from """
    flux_label = xs.variable(intent='out', groups='flux_label')
    output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='flux_output')

    R_label = xs.variable(intent='in', description='label of component that experiences mortality')
    C_label = xs.variable(intent='in', description='label of component that mortality feeds into')

    mortality_rate = xs.variable(intent='in', description='mortality rate of component')

    flux = xs.on_demand()

    @flux.compute
    def mortality(self):
        return self.m.Intermediate(self.mortality_rate * self.R)
