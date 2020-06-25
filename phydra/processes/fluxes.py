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
        self.gk_Parameters.init_param_range(self.C_label, 'halfsat_Growth', self.halfsat_min, self.halfsat_max, self.m)
        super(getattr(self, '__class__'), self).initialize_postdimsetup()

    @flux.compute
    def growth(self):
        return self.m.Intermediate(
            self.gk_Parameters[self.C_label]['mu'][self.C_index] * self.nutrient_limitation() * self.C)

    @nutrient_limitation.compute
    def nutrient_limitation(self):
        return self.m.Intermediate(self.R / (self.gk_Parameters[self.C_label]['halfsat_Growth'][self.C_index] + self.R))


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


class OldGrazingFlux(BaseFlux):
    """ Base Limited Growth, that inherits from BaseFlux and needs to be initialized from """
    flux_label = xs.variable(intent='out', groups='flux_label')
    output = xs.variable(intent='out', dims=('not_initialized', 'time'), groups='flux_output')

    R_label = xs.variable(intent='in', description='labels of components that is grazed upon')
    C_label = xs.variable(intent='in', description='label of component that grazes')

    grazing_rate = xs.variable(intent='in', description='maximum grazing rate')

    halfsat = xs.variable(intent='in', description='half-saturation constant of grazing response')

    flux = xs.on_demand()

    def initialize_parametersetup(self):
        super(getattr(self, '__class__'), self).initialize_postdimsetup()

    @flux.compute
    def grazing(self):
        return self.m.Intermediate(self.grazing_rate * (self.R / self.halfsat * self.R_sum) * self.C)


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

        itC = np.nditer(self.gk_SVshapes[self.C_label], flags=['f_index', 'multi_index'])
        while not itC.finished:
            self.C = self.gk_SVs[self.C_label][itC.multi_index]
            self.C_index = itC.multi_index
            self.C_findex = itC.index

            itR = np.nditer(self.gk_SVshapes[self.R_label], flags=['f_index', 'multi_index'])
            while not itR.finished:
                self.R = self.gk_SVs[self.R_label][itR.multi_index]
                self.R_index = itR.multi_index
                self.R_findex = itR.index

                self.gk_Parameters.init_param_across_dims(self.flux_label, 'GrazePreference',
                                                          self.GrazePreference, (self.C_findex, self.R_findex))
                itR.iternext()
            itC.iternext()

        self.GrazePreferenceMatrix = self.gk_Parameters[self.flux_label]['GrazePreference']

        itC = np.nditer(self.gk_SVshapes[self.C_label], flags=['f_index', 'multi_index'])
        while not itC.finished:
            self.C = self.gk_SVs[self.C_label][itC.multi_index]
            self.C_index = itC.multi_index
            self.C_findex = itC.index

            itR = np.nditer(self.gk_SVshapes[self.R_label], flags=['f_index', 'multi_index'])
            while not itR.finished:
                self.R = self.gk_SVs[self.R_label][itR.multi_index]
                self.R_index = itR.multi_index
                self.R_findex = itR.index

                self.gk_Parameters.init_param_across_dims(self.flux_label, 'FoodAvailability',
                                                          self.FoodAvailability, (self.C_findex, self.R_findex))
                itR.iternext()
            itC.iternext()

        print('Total Grazed full', self.gk_Parameters[self.flux_label]['FoodAvailability'])

        self.FoodAvailabilityMatrix = self.gk_Parameters[self.flux_label]['FoodAvailability']

        itC = np.nditer(self.gk_SVshapes[self.C_label], flags=['f_index', 'multi_index'])
        while not itC.finished:
            self.C = self.gk_SVs[self.C_label][itC.multi_index]
            self.C_index = itC.multi_index
            self.C_findex = itC.index

            itR = np.nditer(self.gk_SVshapes[self.R_label], flags=['f_index', 'multi_index'])
            while not itR.finished:
                self.R = self.gk_SVs[self.R_label][itR.multi_index]
                self.R_index = itR.multi_index
                self.R_findex = itR.index

                self.gk_Parameters.init_param_across_dims(self.flux_label, 'BiomassGrazed',
                                                          self.BiomassGrazed, (itC.index, itR.index))
                itR.iternext()
            itC.iternext()

        print('Total Grazed full', self.gk_Parameters[self.flux_label]['BiomassGrazed'])

        self.BiomassGrazedMatrix = self.gk_Parameters[self.flux_label]['BiomassGrazed']

        itC = np.nditer(self.gk_SVshapes[self.C_label], flags=['multi_index'])
        while not itC.finished:
            self.C_index = itC.multi_index
            self.C = self.gk_SVs[self.C_label][itC.multi_index]
            itR = np.nditer(self.gk_SVshapes[self.R_label], flags=['multi_index'])
            while not itR.finished:
                self.R_index = itR.multi_index
                self.R = self.gk_SVs[self.R_label][itR.multi_index]
                grazed = self.BiomassGrazedMatrix[self.C_findex, self.R_findex]
                self.gk_Fluxes.apply_exchange_flux(self.R_label, self.C_label, grazed,
                                                   itR.multi_index, itC.multi_index)
                self.gk_Flux_Int[self.flux_label] = grazed
                itR.iternext()
            itC.iternext()
        print('EXCHANGE FLUX CREATED!!')

        while True == False:
            itC = np.nditer(self.gk_SVshapes[self.C_label], flags=['f_index', 'multi_index'])
            while not itC.finished:
                self.C = self.gk_SVs[self.C_label][itC.multi_index]
                self.C_index = itC.multi_index
                self.C_findex = itC.index

                C_grazed_total = self.m.Intermediate(sum(self.BiomassGrazedMatrix[self.C_findex, :]))

                self.gk_Fluxes.apply_flux(self.C_label, C_grazed_total, itC.multi_index)
                # self.gk_Flux_Int[self.flux_label] = C_grazed_total
                print('hello')
                itC.iternext()

            itR = np.nditer(self.gk_SVshapes[self.R_label], flags=['f_index', 'multi_index'])
            while not itR.finished:
                self.R = self.gk_SVs[self.R_label][itR.multi_index]
                self.R_index = itR.multi_index
                self.R_findex = itR.index

                R_grazed_total = self.m.Intermediate(-sum(self.BiomassGrazedMatrix[:, self.R_findex]))

                self.gk_Fluxes.apply_flux(self.R_label, R_grazed_total, itR.multi_index)
                self.gk_Flux_Int[self.flux_label] = R_grazed_total
                itR.iternext()


    # @flux.compute
    # def grazing(self):
    #    raise ValueError('needs to be initialized in subclass')

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

    # flux = xs.on_demand(description='returns total grazed flux of Ressource per Consumer')

    Imax = xs.variable(intent='in', description='maximum grazing rate for consumer')

    halfsat = xs.variable(intent='in', description='label of component that grazes')

    GrazePreferenceMatrix = xs.any_object(description='matrix that stores Ressources available to each Consumer')
    GrazePreference = xs.on_demand(description='returns grazingpreference of component to ressource')

    FoodAvailabilityMatrix = xs.any_object(description='matrix that stores Ressources available to each Consumer')
    FoodAvailability = xs.on_demand(description='returns Ressources available to Consumer')

    BiomassGrazedMatrix = xs.any_object(description='matrix that stores Ressources actually grazed by each Consumer')
    BiomassGrazed = xs.on_demand()

    def initialize_parametersetup(self):
        # self.gk_Parameters.setup_dims(self.C_label, 'halfsat_Grazing', self.gk_SVshapes[self.C_label].shape)
        # self.gk_Parameters.init_param_across_dims(self.C_label, 'halfsat_Grazing', self.m.Param(self.halfsat))

        # self.gk_Parameters.setup_dims(self.C_label, 'Imax_Grazing', self.gk_SVshapes[self.C_label].shape)
        # self.gk_Parameters.init_param_across_dims(self.C_label, 'Imax_Grazing', self.m.Param(self.Imax))

        # print('HELLO', self.gk_Parameters)

        super(getattr(self, '__class__'), self).initialize_postdimsetup()

    # @flux.compute
    # def grazing(self):
    #    print('GRAZING_MATRIX', self.BiomassGrazedMatrix)
    #    print('SUm C', self.BiomassGrazedMatrix[self.C_findex, :])
    #
    #    return self.m.Intermediate(grazing)

    @GrazePreference.compute
    def returnC2Rgrazingpref(self):
        return self.m.Param(1)

    @FoodAvailability.compute
    def grazingprobability(self):
        print('MATRIX', self.GrazePreferenceMatrix)
        grazepref = self.GrazePreferenceMatrix[self.C_findex, self.R_findex]
        return self.m.Intermediate(grazepref * self.R ** 2)  # feed_pref *

    @BiomassGrazed.compute
    def FractionGrazed(self):
        # BiomassGrazed[i, j] = self.I0[j] * Z[j] * FoodAvailability[i, j] / (1 + sum(FoodAvailability[:, j]))
        print('MATRIX', self.FoodAvailabilityMatrix)
        print('SUm C', self.FoodAvailabilityMatrix[self.C_findex, :])

        # Imax = self.m.Param(self.Imax)
        # halfsat = self.m.Param(self.halfsat)

        grazedbiomass = self.FoodAvailabilityMatrix[self.C_findex, self.R_findex]
        grazedbiomass_total = sum(self.FoodAvailabilityMatrix[self.C_findex, :])

        grazing = self.Imax * (grazedbiomass / (self.halfsat ** 2 + grazedbiomass_total)) * self.C
        return self.m.Intermediate(grazing)


class putthisaway:
    try:
        def _grazingmatrix(self, P, Z):
            PscaledAsFood = np.zeros((self.NP, self.NP))
            for j in range(self.NP):
                for i in range(self.NP):
                    PscaledAsFood[i, j] = self.phiP[i, j] / self.KsZ * P[i]

            FgrazP = np.zeros((self.NP, self.NP))
            for j in range(self.NP):
                for i in range(self.NP):
                    FgrazP[i, j] = self.I0[j] * Z[j] * PscaledAsFood[i, j] / (1 + sum(PscaledAsFood[:, j]))

            return FgrazP

        def _ingestion(self, FgrazP):
            return [self.epsilon * sum(FgrazP[:, j]) for j in range(self.NP)]

        def _excretion(self, FgrazP):
            return [(1 - self.f_eg - self.epsilon) * sum(FgrazP[:, j]) for j in range(self.NP)]

        def _mortality(self, Z):
            return self.zeta * sum(Z)

        def calculate_sizes(self):
            zoosizes = 2.16 * self.phytosize ** 1.79
            return zoosizes

        def initialize_alloparams(self):
            # initializes allometric parameters as lists, based on sizes
            self.I0 = 26 * (self.size) ** -0.4
            self.xpreyopt = self.phytosize  # 0.65 * (self.size) ** .56

        def init_phiP(self):
            """creates array of feeding preferences [P...P10] for each [Z]"""
            phiP = np.array([[np.exp(-((np.log10(xpreyi) - np.log10(xpreyoptj)) / self.deltaxprey) ** 2)
                              for xpreyi in self.phytosize] for xpreyoptj in self.xpreyopt])
            return phiP

            itC = np.nditer(self.gk_SVshapes[self.C_label], flags=['f_index', 'multi_index'])
            while not itC.finished:
                self.C = self.gk_SVs[self.C_label][itC.multi_index]
                self.C_index = itC.multi_index
                self.C_findex = itC.index

                itR = np.nditer(self.gk_SVshapes[self.R_label], flags=['f_index', 'multi_index'])
                while not itR.finished:
                    self.R = self.gk_SVs[self.R_label][itR.multi_index]
                    self.R_index = itR.multi_index
                    self.R_findex = itR.index

                    FLUX = self.flux
                    # TODO: THIS MIGHT BE WHERE THE pROBLEM LIies, instead have apply_flux for R and C seperately!
                    self.gk_Fluxes.apply_exchange_flux(self.R_label, self.C_label, FLUX,
                                                       itR.multi_index, itC.multi_index)
                    self.gk_Flux_Int[self.flux_label] = FLUX

                    print('GRAZING-FLUX', FLUX)

                    itR.iternext()
                itC.iternext()

        # PHYTOPLANKTON
        def zoograzing(self, Gj, P, Z, D):
            """"""
            # take the general grazing term from each zooplankton, multiply by phyto fraction and sum
            Grazing = [Gj[j] * (self.grazepref[j] * P[self.num] ** 2) * Z[j] for j in range(self.zn)]
            GrazingPerZ = sum(Grazing)
            return GrazingPerZ

        # ZOOPLANKTON
        def hollingtypeIII(self):
            FrhoP = sum([self.feedpref[i] * P[i] ** 2 for i in range(self.pfn)])
            FrhoD = sum([self.detfeedpref[j] * D[j] ** 2 for j in range(self.dn)])
            Frho = FrhoP + FrhoD
            # print('Frho',Frho,'FrhoP',FrhoP,P,'FrhoD',FrhoD,D)
            GrazingProb = self.muZ / (self.ksat ** 2 + Frho)
            return GrazingProb

        def fullgrazing(self, Gj, P, Z, D):
            # phytouptake + zooplankton per zooplankton for each phyto
            IprobP = [Gj[self.num] * (self.feedpref[i] * P[i] ** 2) for i in range(self.pfn)]  # grazeprob per each PFT
            IprobD = [Gj[self.num] * (self.detfeedpref[j] * D[j] ** 2) for j in range(self.dn)]
            Iprob = IprobP + IprobD
            # perhaps here = multiply by Z before summing Iprobs!
            Itots = sum(Iprob)
            Itot = Itots * Z[self.num]
            # print('Itots', Itots,Z, 'IprobP', IprobP, P, 'IprobD', IprobD, D)
            return Itot

        def assimgrazing(self, ZooFeeding):
            # AssimGrazing = self.deltaZ * ZooFeeding[self.num]
            AssimGrazing = self.beta_feed * self.kN_feed * ZooFeeding[self.num]
            return AssimGrazing

        def unassimilatedgrazing(self, ZooFeeding, pool='N'):
            # UnAsGraze = (1. - self.deltaZ) * ZooFeeding[self.num]
            if pool == 'N':
                UnAsGraze = self.beta_feed * (1 - self.kN_feed) * ZooFeeding[self.num]
                return UnAsGraze
            elif pool == 'D':
                UnAsGraze = (1 - self.beta_feed) * ZooFeeding[self.num]
                return UnAsGraze

            # Grazing
            Gj = z.zoofeeding(P, Z, D, func='hollingtypeIII')  # feeding probability for all food
            ZooFeeding = z.fullgrazing(Gj, P, Z, D)

            # Phytoplankton
            PZooGrazed = p.zoograzing(Gj, P, Z, D)

            # Zooplankton Fluxes
            ZGains = z.assimgrazing(ZooFeeding)

    except AttributeError:
        pass
