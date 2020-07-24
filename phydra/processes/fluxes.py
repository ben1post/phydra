import numpy as np
import xsimlab as xs

from .main import GekkoContext

# FORCING FLUXES
@xs.process
class Flux(GekkoContext):
    """
    Base class for a flux that defines an interaction between 2 state variables

    Do not use this base class directly in a model! Use one of its
    subclasses instead.
    """
    label = xs.variable(intent='out', groups='flux_label')
    value = xs.variable(intent='out', dims='time')

    SV_label = xs.variable(intent='in')

    flux = xs.on_demand()

    def initialize(self):
        self.label = self.__xsimlab_name__
        print(f"flux {self.label} acting on {self.SV_label} is initialized")

        self.SV = self.m.phydra_SVs[self.SV_label]

        flx = self.flux
        self.m.phydra_fluxes[self.SV_label].append(flx)

        self.value = self.m.Intermediate(flx, name=self.label).value

    @flux.compute
    def flux(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        raise ValueError('flux needs to be defined in BaseFlux subclass')


@xs.process
class QuadraticLossFlux(Flux):
    """ Base Growth Flux, Monod function """

    flux = xs.on_demand()

    rate = xs.variable(intent='in', description='quadratic loss rate')

    @flux.compute
    def growth(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return - self.SV**2 * self.rate


@xs.process
class InFlux(GekkoContext):
    label = xs.variable(intent='out')
    value = xs.variable(intent='out', dims='time')

    forcing_label = xs.variable(intent='in')
    SV_label = xs.variable(intent='in')

    flowrate = xs.variable(intent='in', description='half saturation constant')

    def initialize(self):
        self.label = self.__xsimlab_name__
        print(f"flux {self.label} of {self.forcing_label} flowing to {self.SV_label} is initialized")

        self.forcing = self.m.phydra_forcings[self.forcing_label]
        self.flowrate_par = self.m.Param(self.flowrate, name='flowrate')

        flux = self.flowrate * self.forcing
        self.value = self.m.Intermediate(flux, name='influx').value

        self.m.phydra_fluxes[self.SV_label].append(flux)

#########################################################

# EXCHANGE FLUXES

@xs.process
class ExchangeFlux(GekkoContext):
    """
    Base class for a flux that defines an interaction between 2 state variables

    Do not use this base class directly in a model! Use one of its
    subclasses instead.
    """
    label = xs.variable(intent='out', groups='flux_label')
    value = xs.variable(intent='out', dims='time')

    resource_label = xs.variable(intent='in')
    consumer_label = xs.variable(intent='in')

    flux = xs.on_demand()

    def initialize(self):
        self.label = self.__xsimlab_name__
        print(f"flux {self.label} of {self.consumer_label} consuming {self.resource_label} is initialized")

        self.resource = self.m.phydra_SVs[self.resource_label]
        self.consumer = self.m.phydra_SVs[self.consumer_label]

        flx = self.flux
        self.m.phydra_fluxes[self.resource_label].append(-flx)
        self.m.phydra_fluxes[self.consumer_label].append(flx)

        self.value = self.m.Intermediate(flx, name=self.label).value

    @flux.compute
    def flux(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        raise ValueError('flux needs to be defined in BaseFlux subclass')

@xs.process
class LinearExchangeFlux(ExchangeFlux):
    """ Base Growth Flux, Monod function """

    flux = xs.on_demand()

    rate = xs.variable(intent='in', description='rate of exchange (e.g. linear transfer or mortality rate')

    @flux.compute
    def growth(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return self.resource * self.rate

@xs.process
class MonodUptake(ExchangeFlux):
    """ Base Growth Flux, Monod function """

    flux = xs.on_demand()

    halfsat = xs.variable(intent='in', description='half saturation constant for Monod growth')

    # def initialize(self):
    #    super(GrowthMonod, self).initialize()

    @flux.compute
    def growth(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return self.resource / (self.halfsat + self.resource) * self.consumer


##################
# GROWTH MULTI LIMITATION

class Growth_MultiLim(ExchangeFlux):
    """ Base Growth Flux, aggregates multiple growth limiting terms
    - nutrient Monod    { needs external nut conc + halfsat param
    - light lim         { needs light available + I opt
    - temp              { needs temp
    - + maximum growth rate param
    """
    fx_values = xs.group('forcing_value')  # hack to force process ordering (Forcing before Fluxes)

    value = xs.variable(intent='out', dims='time')

    flux = xs.on_demand()

    limiting_factors = xs.group('not_initialized')

    mumax = xs.variable(intent='in', description='maximum growth rate')

    @flux.compute
    def growth(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return self.mumax * np.product([lim_fact(self) for lim_fact in self.limiting_factors]) * self.consumer

    @classmethod
    def setup(cls, dim_label):
        """ create copy of process class with user specified name and dimension label """
        new_cls = type(cls.__name__ + dim_label, cls.__bases__, dict(cls.__dict__))
        # add new index with variable name of label (to avoid Zarr storage conflicts)
        new_group = xs.group(dim_label)
        setattr(new_cls, 'limiting_factors', new_group)
        # modify dimensions
        #new_cls.value.metadata['dims'] = ((dim_label, 'time'),)
        # return intialized xsimlab process
        return xs.process(new_cls)


class GML_BaseFlux(GekkoContext):
    """ Base Flux for multi-limitation Growth """

    limiting_factor = xs.variable(intent='out', groups='not_initialized')

    def initialize(self):
        self.limiting_factor = self.limiting_factor_func
        raise ValueError('initialize needs to be defined in GML_BaseFlux subclass')

    def limiting_factor_func(self, cls):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux

          to access state variables, use cls argument
          e.g. cls.resource / (self.halfsat + cls.resource)
          """
        raise ValueError('flux needs to be defined in GML_BaseFlux subclass')

    @classmethod
    def setup(cls, dim_label):
        """ create copy of process class with user specified name and dimension label """
        new_cls = type(cls.__name__ + dim_label, cls.__bases__, dict(cls.__dict__))
        # add new index with variable name of label (to avoid Zarr storage conflicts)
        new_flux = xs.variable(intent='out', groups=dim_label)
        setattr(new_cls, 'limiting_factor', new_flux)
        # modify dimensions
        # new_cls.value.metadata['dims'] = ((dim_label, 'time'),)
        # return intialized xsimlab process
        return xs.process(new_cls)


class GML_MonodUptake(GML_BaseFlux):
    """ Adding to Multi Growth Limitation Flux, Monod function """

    halfsat = xs.variable(intent='in', description='half saturation constant for Monod growth')

    def initialize(self):
        self.limiting_factor = self.monod

    def monod(self, cls):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return cls.resource / (self.halfsat + cls.resource)


class GML_EppleyTempLim(GML_BaseFlux):
    """ Adding to Multi Growth Limitation Flux, Monod function """
    exponent = xs.variable(intent='in', description='exponent for Eppley Temperature dependency of growth')
    FX_label = xs.variable(intent='in')

    def initialize(self):
        self.limiting_factor = self.eppley

    def eppley(self, cls):
        return self.m.exp(self.exponent * self.m.phydra_forcings[self.FX_label])


class GML_SteeleLightLim(GML_BaseFlux):
    """ Adding to Multi Growth Limitation Flux, Monod function """
    IOpt = xs.variable(intent='in', description='optimal integrated irradiance')

    kw = xs.variable(intent='in', description='light attenuation coefficient of sea water')
    kc = xs.variable(intent='in', description='light attenuation coefficient of component biomass')

    FX_label_I0 = xs.variable(intent='in', description='light forcing label')
    FX_label_MLD = xs.variable(intent='in', description='MLD forcing label')

    def initialize(self):
        self.limiting_factor = self.steele

    def steele(self, cls):
        kPAR = self.m.Intermediate(self.kw + self.kc * cls.consumer, name='kPAR')
        I0 = self.m.phydra_forcings[self.FX_label_I0]
        MLD = self.m.phydra_forcings[self.FX_label_MLD]

        lighthrv = 1. / (kPAR * MLD) * \
                   (-self.m.exp(1. - I0 / self.IOpt) - (
                       -self.m.exp((1. - (I0 * self.m.exp(-kPAR * MLD)) / self.IOpt))))

        return self.m.Intermediate(lighthrv, name='steele')



##################
# GRAZING

#########################################################

@xs.process
class GrazingFlux(GekkoContext):
    """
    Base class for a flux that defines an interaction between 1 state variables and multiple others
    (i.e. grazing to SV + fraction egested to another SV + fraction excreted to another SV)
    give the 3 fraction options, but can also simply pass None or 0.. beta, epsilon

    Do not use this base class directly in a model! Use one of its
    subclasses instead.
    """
    label = xs.variable(intent='out', groups='flux_label')
    value = xs.variable(intent='out', dims='time')

    resource_label = xs.variable(intent='in')
    consumer_label = xs.variable(intent='in')
    egested2_label = xs.variable(intent='in')
    excreted2_label = xs.variable(intent='in')

    beta = xs.variable(intent='in', description='absorption efficiency')
    epsilon = xs.variable(intent='in', description='net production efficiency')

    Imax = xs.variable(intent='in', description='maximum grazing rate')

    flux = xs.on_demand()

    def initialize(self):
        self.label = self.__xsimlab_name__
        print(f"flux {self.label} of {self.consumer_label} consuming {self.resource_label} is initialized \n",
              f"egesting to {self.egested2_label} and excreting to {self.excreted2_label}")

        self.resource = self.m.phydra_SVs[self.resource_label]
        self.consumer = self.m.phydra_SVs[self.consumer_label]
        self.egested2 = self.m.phydra_SVs[self.egested2_label]
        self.excreted2 = self.m.phydra_SVs[self.excreted2_label]

        flx = self.flux

        egestion = flx * (1 - self.beta)
        assimilation = flx * self.beta * self.epsilon
        excretion = flx * self.beta * (1 - self.epsilon)

        self.m.phydra_fluxes[self.consumer_label].append(assimilation)
        self.m.phydra_fluxes[self.egested2_label].append(egestion)
        self.m.phydra_fluxes[self.excreted2_label].append(excretion)

        # biomass grazed
        self.m.phydra_fluxes[self.resource_label].append(-flx)

        self.value = self.m.Intermediate(flx, name=self.label).value

    @flux.compute
    def growth(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return self.Imax * self.resource / (0.5 + self.resource) * self.consumer



@xs.process
class GrazingFlux_MultiRessource(GekkoContext):
    """
    Base class for a flux that defines an interaction between 1 state variables and multiple others
    (i.e. grazing to SV + fraction egested to another SV + fraction excreted to another SV)
    give the 3 fraction options, but can also simply pass None or 0.. beta, epsilon

    Do not use this base class directly in a model! Use one of its
    subclasses instead.
    """
    label = xs.variable(intent='out', groups='flux_label')
    values = xs.variable(intent='out', dims=('resource_index', 'time'))

    resource_index = xs.index(dims='resource_index')

    resource_labels = xs.variable(intent='in', dims='resource_index')
    consumer_label = xs.variable(intent='in')
    egested2_label = xs.variable(intent='in')
    excreted2_label = xs.variable(intent='in')

    beta = xs.variable(intent='in', description='absorption efficiency')
    epsilon = xs.variable(intent='in', description='net production efficiency')

    Imax = xs.variable(intent='in', description='maximum grazing rate')
    kZ = xs.variable(intent='in', description='half saturation constant of grazing')

    feed_prefs = xs.variable(intent='in', dims='resource_index',
                             description='preference of feeding, supply as list of same dims as resource_labels')

    flux = xs.on_demand()

    def initialize(self):
        self.label = self.__xsimlab_name__
        print(f"flux {self.label} of {self.consumer_label} consuming {self.resource_labels} is initialized \n",
              f"egesting to {self.egested2_label} and excreting to {self.excreted2_label}")

        self.consumer = self.m.phydra_SVs[self.consumer_label]
        self.egested2 = self.m.phydra_SVs[self.egested2_label]
        self.excreted2 = self.m.phydra_SVs[self.excreted2_label]

        self.resources = [self.m.phydra_SVs[label] for label in self.resource_labels]
        self.resource_index = [label for label in self.resource_labels]
        print(self.resource_index)

        fluxes = []
        i = 0
        for label in self.resource_labels:
            self.resource = self.m.phydra_SVs[label]

            self.feed_pref = self.feed_prefs[i]

            flx = self.flux
            self.m.phydra_fluxes[label].append(-flx)
            fluxes.append(flx)
            i += 1

        print(fluxes)

        egestion = sum(fluxes) * (1 - self.beta)
        assimilation = sum(fluxes) * self.beta * self.epsilon
        excretion = sum(fluxes) * self.beta * (1 - self.epsilon)

        self.m.phydra_fluxes[self.consumer_label].append(assimilation)
        self.m.phydra_fluxes[self.egested2_label].append(egestion)
        self.m.phydra_fluxes[self.excreted2_label].append(excretion)

        self.values = [self.m.Intermediate(flx, name=self.label+lab).value for flx, lab in zip(fluxes, self.resource_labels)]

    @flux.compute
    def growth(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        TotalGrazing = sum(np.array(self.resources)**2 * np.array(self.feed_prefs))
        return self.Imax * self.resource**2 * self.feed_pref / (self.kZ**2 + TotalGrazing) * self.consumer
