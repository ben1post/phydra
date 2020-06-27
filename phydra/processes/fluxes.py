import numpy as np
import xsimlab as xs

from collections import defaultdict

from .gekkocontext import InheritGekkoContext
from .environments import Slab


def make_flux(flux_cls, flux_name):
    """
    This functions creates a properly labeled xs.process from class Component.

    :args:
        cls_name (cls): Class of forcing flux to be initialized
        dim_name (str): Name of sub-dimension of returned process, UPPERCASE!

    :returns:
        xs.process of class Component
    """
    new_dim = xs.index(dims=(flux_name), groups='flux_index')
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
        super(cls_here, self).initialize_parametersetup()

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

        itC = np.nditer(self.gk_SVshapes[self.C_label], flags=['multi_index'])
        itR = np.nditer(self.gk_SVshapes[self.R_label], flags=['multi_index'])
        self.R_sum = np.sum([self.gk_SVs[self.R_label][multi_index] for multi_index in itR.multi_index])
        self.C_sum = np.sum([self.gk_SVs[self.C_label][multi_index] for multi_index in itC.multi_index])

        itC = np.nditer(self.gk_SVshapes[self.C_label], flags=['multi_index'])
        while not itC.finished:
            self.C_index = itC.multi_index
            self.C = self.gk_SVs[self.C_label][itC.multi_index]
            itR = np.nditer(self.gk_SVshapes[self.R_label], flags=['multi_index'])
            while not itR.finished:
                self.R_index = itR.multi_index
                self.R = self.gk_SVs[self.R_label][itR.multi_index]

                self.gk_Fluxes.apply_exchange_flux(self.R_label, self.C_label, self.flux,
                                                   itR.multi_index, itC.multi_index)
                self.gk_Flux_Int[self.flux_label] = self.flux
                itR.iternext()
            itC.iternext()

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


class LimitedGrowth_Monod(BaseFlux):
    """ Limited growth of C_label consuming R_label at Michealis-Menten/Monod kinetics """

    flux_label = xs.variable(intent='out', groups='flux_label')
    output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='flux_output')

    R_label = xs.variable(intent='in', description='label of ressource component that is consumed')
    C_label = xs.variable(intent='in', description='label of component that grows')

    mu_min = xs.variable(intent='in', description='Maximum growth rate of component')
    mu_max = xs.variable(intent='in', description='Maximum growth rate of component')
    halfsat_min = xs.variable(intent='in', description='half-saturation constant of nutrient uptake for component')
    halfsat_max = xs.variable(intent='in', description='half-saturation constant of nutrient uptake for component')

    flux = xs.on_demand()

    nutrient_limitation = xs.on_demand()

    def initialize_parametersetup(self):
        self.gk_Parameters.setup_dims(self.C_label, 'mu', self.gk_SVshapes[self.C_label].shape)
        self.gk_Parameters.init_param_range(self.C_label, 'mu', self.mu_min, self.mu_max, self.m)

        self.gk_Parameters.setup_dims(self.C_label, 'halfsat_Growth', self.gk_SVshapes[self.C_label].shape)
        self.gk_Parameters.init_param_range(self.C_label, 'halfsat_Growth', self.halfsat_min, self.halfsat_max, spacing='linear')
        super(getattr(self, '__class__'), self).initialize_postdimsetup()

    @flux.compute
    def growth(self):
        return self.m.Intermediate(
            self.gk_Parameters[self.C_label]['mu'][self.C_index] * self.nutrient_limitation() * self.C)

    @nutrient_limitation.compute
    def nutrient_limitation(self):
        return self.m.Intermediate(self.R / (self.gk_Parameters[self.C_label]['halfsat_Growth'][self.C_index] + self.R))


class SizeAllo_LimitedGrowth_Monod(BaseFlux):
    """ Limited growth of C_label consuming R_label at Michealis-Menten/Monod kinetics """

    flux_label = xs.variable(intent='out', groups='flux_label')
    output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='flux_output')

    R_label = xs.variable(intent='in', description='label of ressource component that is consumed')
    C_label = xs.variable(intent='in', description='label of component that grows')

    # Alloparams
    ks = xs.variable(intent='out', description='allometric half_saturation constant for uptake')
    mu0 = xs.variable(intent='out', description='allometric max growth rate')

    ks_allometry = xs.on_demand()
    mu0_allometry = xs.on_demand()

    flux = xs.on_demand()
    nutrient_limitation = xs.on_demand()

    def initialize_parametersetup(self):
        try:
            self.C_Size = self.gk_Parameters[self.C_label]['Size']
        except KeyError:
            raise BaseException(
                "the Size Grazing Kernel can not find 'Size' in gk_parameters, \n make sure to use SizeComponent")

        self.gk_Parameters.setup_dims(self.C_label, 'ks_Growth', self.gk_SVshapes[self.C_label].shape)

        self.gk_Parameters.setup_dims(self.C_label, 'mu0_Growth', self.gk_SVshapes[self.C_label].shape)

        itC = np.nditer(self.gk_SVshapes[self.C_label], flags=['f_index', 'multi_index'])
        while not itC.finished:
            self.C_index = itC.multi_index
            self.C = self.gk_SVs[self.C_label][self.C_index]
            self.C_findex = itC.index
            self.gk_Parameters.init_param_across_dims(self.C_label, 'ks_Growth',
                                                      self.ks_allometry, self.C_index)
            self.gk_Parameters.init_param_across_dims(self.C_label, 'mu0_Growth',
                                                      self.mu0_allometry, self.C_index)
            itC.iternext()

        self.ks = self.gk_Parameters[self.C_label]['ks_Growth']
        self.mu0 = self.gk_Parameters[self.C_label]['mu0_Growth']

        #print('ks, mu0', self.ks, self.mu0)

        super(getattr(self, '__class__'), self).initialize_postdimsetup()

    @flux.compute
    def growth(self):
        return self.m.Intermediate(self.mu0[self.C_index] * self.nutrient_limitation() * self.C)

    @nutrient_limitation.compute
    def nutrient_limitation(self):
        return self.m.Intermediate(self.R / (self.ks[self.C_index] + self.R))

    # ALLOMETRY FUNCTIONS:
    @ks_allometry.compute
    def return_ks(self):
        """ Iterates over self.C """
        return self.m.Param(2.6 * self.C_Size[self.C_index] ** -0.45)

    @mu0_allometry.compute
    def return_mu0(self):
        """ Iterates over self.C """
        return self.m.Param(0.1 * self.C_Size[self.C_index])



class LimitedGrowth_MonodTempLight(BaseFlux):
    """ Limited growth of C_label consuming R_label at Michealis-Menten/Monod kinetics """

    flux_label = xs.variable(intent='out', groups='flux_label')
    output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='flux_output')

    R_label = xs.variable(intent='in', description='label of ressource component that is consumed')
    C_label = xs.variable(intent='in', description='label of component that grows')

    # MONOD nutrient limitation
    mu = xs.variable(intent='in', description='Maximum growth rate of component')
    halfsat = xs.variable(intent='in', description='half-saturation constant of nutrient uptake for component')

    # STEELE's light limitation
    kw = xs.variable(intent='in', description='light attenuation coefficient of sea water')
    kc = xs.variable(intent='in', description='light attenuation coefficient of component biomass')
    OptI = xs.variable(intent='in', description='optimal integrated irradiance')

    flux = xs.on_demand()

    MLD = xs.foreign(Slab, 'MLD')
    Tmld = xs.foreign(Slab, 'Temp_forcing')
    I0 = xs.foreign(Slab, 'I0_forcing')

    nut_limitation = xs.on_demand()
    temp_dependence = xs.on_demand()
    light_limitation = xs.on_demand()

    def initialize_parametersetup(self):
        super(getattr(self, '__class__'), self).initialize_postdimsetup()

    @flux.compute
    def growth(self):
        return self.m.Intermediate(self.mu * self.nut_limitation *
                                   self.temp_dependence *
                                   self.light_limitation * self.C)

    @nut_limitation.compute
    def nutrient_limitation(self):
        self.halfsat_Par = self.m.Param(self.halfsat)
        return self.m.Intermediate(self.R / (self.halfsat_Par + self.R))

    @temp_dependence.compute
    def temperature_dependence(self):
        self.eppley = self.m.Param(0.063)
        return self.m.Intermediate(self.m.exp(self.eppley * self.Tmld))

    @light_limitation.compute
    def PAR_limitation(self):
        kPAR = self.m.Intermediate(self.kw + self.kc * self.C_sum)

        lighthrv = 1. / (kPAR * self.MLD) * \
                   (-self.m.exp(1. - self.I0 / self.OptI) - (
                       -self.m.exp((1. - (self.I0 * self.m.exp(-kPAR * self.MLD)) / self.OptI))))

        return self.m.Intermediate(lighthrv)


class LinearMortality(BaseFlux):
    """ Mortality of R_label feeding into C_label at a constant rate """
    flux_label = xs.variable(intent='out', groups='flux_label')
    output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='flux_output')

    R_label = xs.variable(intent='in', description='label of component that experiences mortality')
    C_label = xs.variable(intent='in', description='label of component that mortality feeds into')

    mortality_rate = xs.variable(intent='in', description='mortality rate of component')

    flux = xs.on_demand()

    def initialize_parametersetup(self):
        super(getattr(self, '__class__'), self).initialize_postdimsetup()

    @flux.compute
    def mortality(self):
        return self.m.Intermediate(self.mortality_rate * self.R)


class Remineralization(BaseFlux):
    """ Remineralisation of R_label to C_label at a constant rate """
    flux_label = xs.variable(intent='out', groups='flux_label')
    output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='flux_output')

    R_label = xs.variable(intent='in', description='label of component that experiences mortality')
    C_label = xs.variable(intent='in', description='label of component that mortality feeds into')

    remin_rate = xs.variable(intent='in', description='rate of remineralization')

    flux = xs.on_demand()

    def initialize_parametersetup(self):
        super(getattr(self, '__class__'), self).initialize_postdimsetup()

    @flux.compute
    def remineralisation(self):
        return self.m.Intermediate(self.remin_rate * self.R)


def make_multigrazing(flux_cls, flux_name):
    """
    This functions creates a properly labeled xs.process from class Component.

    :args:
        cls_name (cls): Class of forcing flux to be initialized
        dim_name (str): Name of sub-dimension of returned process, UPPERCASE!

    :returns:
        xs.process of class Component
    """
    new_dim = xs.index(dims=(flux_name), groups='flux_index')
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
        if self.gk_SVshapes[r_label].size == 1 and self.gk_SVshapes[c_label].size == 1:
            fx_c_list.append(f"{cls_label}-{r_label}-2-{c_label}")
        elif self.gk_SVshapes[r_label].size == 1:
            for c in range(self.gk_SVshapes[c_label].size):
                fx_c_list.append(f"{cls_label}-{r_label}-2-{c_label}{c}")
        else:
            for r in range(self.gk_SVshapes[r_label].size):
                for c in range(self.gk_SVshapes[c_label].size):
                    fx_c_list.append(f"{cls_label}-{r_label}{r}-2-{c_label}{c}")

        setattr(self, flux_name, fx_c_list)

        cls_here = getattr(self, '__class__')
        super(cls_here, self).initialize_parametersetup()

    setattr(new_cls, 'initialize', initialize_dim)

    if flux_name.lower() == flux_name:
        raise ValueError(f"dimension label ({flux_name}) supplied to multiflux {flux_cls} needs to be Upper Case")
    # here the Component label affects all sub components, therefore C_labels dim != Forcingflux dims
    # new_cls.R_label.metadata['dims'] = (flux_name + 'R')
    # new_cls.R_feed_prefs.metadata['dims'] = (flux_name + 'R')
    # new_cls.C_label.metadata['dims'] = (flux_name + 'C')

    new_cls.output.metadata['dims'] = ((flux_name, 'time'),)
    return xs.process(new_cls)


@xs.process
class BaseGrazingFlux(BaseFlux):
    """ Base Limited Growth, that inherits from BaseFlux and needs to be initialized from """
    flux_label = xs.variable(intent='out', groups='flux_label')
    output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='flux_output')

    R_label = xs.variable(intent='in', description='labels of components that is grazed upon')
    C_label = xs.variable(intent='in', description='label of component that grazes')

    # flux = xs.on_demand(description='returns total grazed flux of Ressource per Component')

    GrazePreferenceMatrix = xs.any_object(description='matrix that stores Ressources available to each Consumer')
    GrazePreference = xs.on_demand(description='returns Ressources available to Consumer')

    FoodAvailabilityMatrix = xs.any_object(description='matrix that stores Ressources available to each Consumer')
    FoodAvailability = xs.on_demand(description='returns Ressources available to Consumer')

    BiomassGrazedMatrix = xs.any_object(description='matrix that stores Ressources actually grazed by each Consumer')
    BiomassGrazed = xs.on_demand(description='returns Ressources actually grazed by Consumer')

    def initialize_postdimsetup(self):
        self.flux_label = f"{self.cls_label}-{self.R_label}2{self.C_label}"

        grazingmatrix = np.outer(self.gk_SVshapes[self.C_label], self.gk_SVshapes[self.R_label])
        self.gk_Parameters.setup_dims(self.flux_label, 'GrazePreference', grazingmatrix.shape)
        self.gk_Parameters.setup_dims(self.flux_label, 'FoodAvailability', grazingmatrix.shape)
        self.gk_Parameters.setup_dims(self.flux_label, 'BiomassGrazed', grazingmatrix.shape)

        itC1 = np.nditer(self.gk_SVshapes[self.C_label], flags=['f_index', 'multi_index'])
        while not itC1.finished:
            self.C_index = itC1.multi_index
            self.C = self.gk_SVs[self.C_label][self.C_index]
            self.C_findex = itC1.index

            itR1 = np.nditer(self.gk_SVshapes[self.R_label], flags=['f_index', 'multi_index'])
            while not itR1.finished:
                self.R_index = itR1.multi_index
                self.R = self.gk_SVs[self.R_label][self.R_index]

                self.R_findex = itR1.index

                self.gk_Parameters.init_param_across_dims(self.flux_label, 'GrazePreference',
                                                          self.GrazePreference, (self.C_findex, self.R_findex))
                itR1.iternext()
            itC1.iternext()

        self.GrazePreferenceMatrix = self.gk_Parameters[self.flux_label]['GrazePreference']

        itC2 = np.nditer(self.gk_SVshapes[self.C_label], flags=['f_index', 'multi_index'])
        while not itC2.finished:
            self.C_index = itC2.multi_index
            self.C = self.gk_SVs[self.C_label][self.C_index]
            self.C_findex = itC2.index

            itR2 = np.nditer(self.gk_SVshapes[self.R_label], flags=['f_index', 'multi_index'])
            while not itR2.finished:
                self.R_index = itR2.multi_index
                self.R = self.gk_SVs[self.R_label][self.R_index]
                self.R_findex = itR2.index

                self.gk_Parameters.init_param_across_dims(self.flux_label, 'FoodAvailability',
                                                          self.FoodAvailability, (self.C_findex, self.R_findex))
                itR2.iternext()
            itC2.iternext()

        self.FoodAvailabilityMatrix = self.gk_Parameters[self.flux_label]['FoodAvailability']

        itC3 = np.nditer(self.gk_SVshapes[self.C_label], flags=['f_index', 'multi_index'])
        while not itC3.finished:
            self.C_index = itC3.multi_index
            self.C = self.gk_SVs[self.C_label][self.C_index]
            self.C_findex = itC3.index

            itR3 = np.nditer(self.gk_SVshapes[self.R_label], flags=['f_index', 'multi_index'])
            while not itR3.finished:
                self.R_index = itR3.multi_index
                self.R = self.gk_SVs[self.R_label][self.R_index]
                self.R_findex = itR3.index

                self.gk_Parameters.init_param_across_dims(self.flux_label, 'BiomassGrazed',
                                                          self.BiomassGrazed, (self.C_findex, self.R_findex))
                itR3.iternext()
            itC3.iternext()

        self.BiomassGrazedMatrix = self.gk_Parameters[self.flux_label]['BiomassGrazed']

        itC = np.nditer(self.gk_SVshapes[self.C_label], flags=['f_index', 'multi_index'])
        while not itC.finished:
            self.C_index = itC.multi_index
            self.C = self.gk_SVs[self.C_label][self.C_index]
            self.C_findex = itC.index

            itR = np.nditer(self.gk_SVshapes[self.R_label], flags=['f_index', 'multi_index'])
            while not itR.finished:
                self.R_index = itR.multi_index
                self.R = self.gk_SVs[self.R_label][self.R_index]
                self.R_findex = itR.index

                grazed = self.BiomassGrazedMatrix[self.C_findex, self.R_findex]

                self.gk_Fluxes.apply_exchange_flux(self.R_label, self.C_label, grazed,
                                                   self.R_index, self.C_index)
                self.gk_Flux_Int[self.flux_label] = grazed

                itR.iternext()
            itC.iternext()

    @GrazePreference.compute
    def grazingprobability(self):
        raise ValueError('needs to be initialized in subclass')

    @FoodAvailability.compute
    def grazingprobability(self):
        raise ValueError('needs to be initialized in subclass')

    @BiomassGrazed.compute
    def FractionGrazed(self):
        raise ValueError('needs to be initialized in subclass')


class HollingTypeIII(BaseGrazingFlux):
    """ Base Limited Growth, that inherits from BaseFlux and needs to be initialized from """
    flux_label = xs.variable(intent='out', groups='flux_label')
    output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='flux_output')

    R_label = xs.variable(intent='in', description='labels of components that is grazed upon')

    C_label = xs.variable(intent='in', description='label of component that grazes')

    Imax = xs.variable(intent='in', description='maximum grazing rate for consumer')

    halfsat = xs.variable(intent='in', description='label of component that grazes')

    GrazePreferenceMatrix = xs.any_object(description='matrix that stores Ressources available to each Consumer')
    GrazePreference = xs.on_demand(description='returns grazingpreference of component to ressource')

    FoodAvailabilityMatrix = xs.any_object(description='matrix that stores Ressources available to each Consumer')
    FoodAvailability = xs.on_demand(description='returns Ressources available to Consumer')

    BiomassGrazedMatrix = xs.any_object(description='matrix that stores Ressources actually grazed by each Consumer')
    BiomassGrazed = xs.on_demand()

    def initialize_parametersetup(self):
        super(getattr(self, '__class__'), self).initialize_postdimsetup()

    @GrazePreference.compute
    def return_C2R_grazingpref(self):
        return self.m.Param(1)

    @FoodAvailability.compute
    def return_available_food(self):
        grazepref = self.GrazePreferenceMatrix[self.C_findex, self.R_findex]
        return self.m.Intermediate(grazepref * self.R ** 2)  # feed_pref *

    @BiomassGrazed.compute
    def return_grazed_biomass(self):
        grazedbiomass = self.FoodAvailabilityMatrix[self.C_findex, self.R_findex]
        grazedbiomass_total = self.m.sum(self.FoodAvailabilityMatrix[self.C_findex, :])

        grazing = self.Imax * (grazedbiomass / (self.halfsat ** 2 + grazedbiomass_total)) * self.C
        return self.m.Intermediate(grazing)


class SizeBasedKernelGrazing(BaseGrazingFlux):
    """ Grazing formulation adapted from Banas 2011 - the ASTroCAT model """
    flux_label = xs.variable(intent='out', groups='flux_label')
    output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='flux_output')

    R_label = xs.variable(intent='in', description='labels of components that is grazed upon')
    C_label = xs.variable(intent='in', description='label of component that grazes')

    # NEW PARAMS
    deltaxprey = xs.variable(default=0.25, description='log10 prey size tolerance')
    KsZ = xs.variable(default=3, description='razing half saturation constant')

    f_eg = xs.variable(default=.33)  # egested food
    epsilon = xs.variable(default=.33)  # assimilated food

    # Alloparams
    I0 = xs.variable(intent='out')
    xpreyopt = xs.variable(intent='out')
    phiP = xs.variable(intent='out')

    I0_allometry = xs.on_demand()
    xpreyopt_allometry = xs.on_demand()
    phiP_allometry = xs.on_demand()

    # GRAZING MATRICES
    GrazePreferenceMatrix = xs.any_object(description='matrix that stores Ressources available to each Consumer')
    GrazePreference = xs.on_demand(description='returns grazingpreference of component to ressource')

    FoodAvailabilityMatrix = xs.any_object(description='matrix that stores Ressources available to each Consumer')
    FoodAvailability = xs.on_demand(description='returns Ressources available to Consumer')

    BiomassGrazedMatrix = xs.any_object(description='matrix that stores Ressources actually grazed by each Consumer')
    BiomassGrazed = xs.on_demand()

    def initialize_parametersetup(self):
        try:
            self.C_Size = self.gk_Parameters[self.C_label]['Size']
            self.R_Size = self.gk_Parameters[self.R_label]['Size']
        except KeyError:
            raise BaseException(
                "the Size Grazing Kernel can not find 'Size' in gk_parameters, \n make sure to use SizeComponent")

        self.gk_Parameters.setup_dims(self.C_label, 'I0_Grazing', self.gk_SVshapes[self.C_label].shape)

        self.gk_Parameters.setup_dims(self.C_label, 'xpreyopt_Grazing', self.gk_SVshapes[self.C_label].shape)

        grazingmatrix = np.outer(self.gk_SVshapes[self.C_label], self.gk_SVshapes[self.R_label])
        self.gk_Parameters.setup_dims(self.flux_label, 'phiP_Grazing', grazingmatrix.shape)
        itC1 = np.nditer(self.gk_SVshapes[self.C_label], flags=['f_index', 'multi_index'])
        while not itC1.finished:
            self.C_index = itC1.multi_index
            self.C = self.gk_SVs[self.C_label][self.C_index]
            self.C_findex = itC1.index
            self.gk_Parameters.init_param_across_dims(self.C_label, 'I0_Grazing',
                                                      self.I0_allometry, self.C_index)
            self.gk_Parameters.init_param_across_dims(self.C_label, 'xpreyopt_Grazing',
                                                      self.xpreyopt_allometry, self.C_index)

            itC1.iternext()

        self.I0 = self.gk_Parameters[self.C_label]['I0_Grazing']
        self.xpreyopt = self.gk_Parameters[self.C_label]['xpreyopt_Grazing']

        itC = np.nditer(self.gk_SVshapes[self.C_label], flags=['f_index', 'multi_index'])
        while not itC.finished:
            self.C_index = itC.multi_index
            self.C = self.gk_SVs[self.C_label][self.C_index]
            self.C_findex = itC.index

            itR = np.nditer(self.gk_SVshapes[self.R_label], flags=['f_index', 'multi_index'])
            while not itR.finished:
                self.R_index = itR.multi_index
                self.R = self.gk_SVs[self.R_label][self.R_index]
                self.R_findex = itR.index

                self.gk_Parameters.init_param_across_dims(self.flux_label, 'phiP_Grazing',
                                                          self.phiP_allometry, (self.C_findex, self.R_findex))

                itR.iternext()
            itC.iternext()

        self.phiP = self.gk_Parameters[self.flux_label]['phiP_Grazing']

        #print('I0, xpreyopt, phiP', self.I0, self.xpreyopt, self.phiP)

        super(getattr(self, '__class__'), self).initialize_postdimsetup()

    @GrazePreference.compute
    def return_C2R_grazingpref(self):
        return self.phiP[self.C_findex, self.R_findex] * self.R

    @FoodAvailability.compute
    def return_available_food(self):
        # self.I0[j] * Z[j] * PscaledAsFood[i,j] / (1 + sum(PscaledAsFood[:,j]))
        # self.phiP[self.C_findex, self.R_findex] * self.R
        grazedbiomass = self.GrazePreferenceMatrix[self.C_findex, self.R_findex]
        grazedbiomass_total = self.m.sum(self.GrazePreferenceMatrix[self.C_findex, :])
        return self.I0[self.C_index] * grazedbiomass / (self.KsZ + grazedbiomass_total) * self.C

    @BiomassGrazed.compute
    def return_grazed_biomass(self):
        #grazedbiomass = self.FoodAvailabilityMatrix[self.C_findex, self.R_findex]
        return self.m.Intermediate(self.FoodAvailabilityMatrix[self.C_findex, self.R_findex])

    # ALLOMETRY FUNCTIONS:
    @I0_allometry.compute
    def return_I0(self):
        """ Iterates over self.C """
        return self.m.Param(26 * self.C_Size[self.C_index] ** -0.4)

    @xpreyopt_allometry.compute
    def return_xpreyopt(self):
        """ Iterates over self.C """
        #print(self.C_Size[self.C_index], self.R_Size[self.C_index])
        # 0.65 * self.C_Size[self.C_index] ** .56
        return self.m.Param(self.R_Size[self.C_index])

    @phiP_allometry.compute
    def return_phiP(self):
        """ Iterates over GRAZINGMATRIX, self.C, self.N """

        phiP_out = self.m.exp(-((self.m.log10(self.R_Size[self.R_index]) -
                             self.m.log10(self.xpreyopt[self.C_index])) / self.deltaxprey) ** 2)

        return self.m.Param(phiP_out)
