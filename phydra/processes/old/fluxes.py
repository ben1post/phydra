import numpy as np
import xsimlab as xs

from phydra.processes.main import ModelContext

@xs.process
class InFlux(ModelContext):
    label = xs.variable(intent='out')
    value = xs.variable(intent='out', dims='Time')

    FX_label = xs.variable(intent='in')
    SV_label = xs.variable(intent='in')

    rate = xs.variable(intent='in', description='half saturation constant')

    def initialize(self):
        self.label = self.__xsimlab_name__
        print(f"flux {self.label} of {self.FX_label} flowing to {self.SV_label} is initialized")

        self.forcing = self.m.phydra_forcings[self.FX_label]
        self.flowrate_par = self.m.Param(self.rate, name='rate')

        flux = self.rate * self.forcing
        self.value = self.m.Intermediate(flux, name='influx').value

        self.m.phydra_fluxes[self.SV_label].append(flux)


@xs.process
class EvansParslow_SlabPhysics(ModelContext):
    """"""
    pass


# FORCING FLUXES
@xs.process
class InputFlux(ModelContext):
    """
    Base class for a flux that defines an interaction between 2 state variables

    Do not use this base class directly in a model! Use one of its
    subclasses instead.
    """
    fx_values = xs.group('forcing_value')  # hack to force process ordering (Forcing before Fluxes)

    label = xs.variable(intent='out', groups='flux_label')
    value = xs.variable(intent='out', dims='Time')

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
    def flux_func(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        raise ValueError('flux needs to be defined in BaseFlux subclass')


@xs.process
class LinearInputFlux(InputFlux):
    """ Base Growth Flux, Monod function """

    flux = xs.on_demand()

    rate = xs.variable(intent='in', description='linear input rate')

    @flux.compute
    def growth(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return self.SV * self.rate


@xs.process
class Upwelling(InputFlux):
    """ Evan's and Parsley mixing LOSS flux (negative) """
    FX_label_MLD = xs.variable(intent='in', description='MLD forcing label')
    FX_label_N0 = xs.variable(intent='in', description='N0 forcing label')

    flux = xs.on_demand()

    kappa = xs.variable(intent='in', description='constant diffusive mixing rate')

    def initialize(self):
        self.N0 = self.m.phydra_forcings[self.FX_label_N0]

        self.MLD = self.m.phydra_forcings[self.FX_label_MLD]
        self.MLD_deriv = self.m.phydra_forcings[self.FX_label_MLD + '_deriv']

        self.h_pos = self.m.Param(np.maximum(self.MLD_deriv, 0), name='hpos2')
        super(Upwelling, self).initialize()

    @flux.compute
    def mixing(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        K = (self.h_pos + self.kappa) / self.MLD

        return (self.N0 - self.SV) * K


@xs.process
class LossFlux(ModelContext):
    """
    Base class for a flux that defines an interaction between 2 state variables

    Do not use this base class directly in a model! Use one of its
    subclasses instead.
    """
    label = xs.variable(intent='out', groups='flux_label')
    value = xs.variable(intent='out', dims='Time')

    SV_label = xs.variable(intent='in')

    flux = xs.on_demand()

    def initialize(self):
        self.label = self.__xsimlab_name__
        print(f"flux {self.label} acting on {self.SV_label} is initialized")

        self.SV = self.m.phydra_SVs[self.SV_label]

        flx = self.flux
        self.m.phydra_fluxes[self.SV_label].append(-flx)

        self.value = self.m.Intermediate(flx, name=self.label).value

    @flux.compute
    def flux_func(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        raise ValueError('flux needs to be defined in BaseFlux subclass')

@xs.process
class LinearLossFlux(LossFlux):
    """ Base Growth Flux, Monod function """

    flux = xs.on_demand()

    rate = xs.variable(intent='in', description='linear loss rate')

    @flux.compute
    def growth(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return self.SV * self.rate

@xs.process
class QuadraticLossFlux(LossFlux):
    """ Base Growth Flux, Monod function """

    flux = xs.on_demand()

    rate = xs.variable(intent='in', description='quadratic loss rate')

    @flux.compute
    def growth(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return self.SV**2 * self.rate


#############################################################################


@xs.process
class LossMultiFlux(ModelContext):
    """
    Base class for a flux that defines an interaction between 2 state variables

    Do not use this base class directly in a model! Use one of its
    subclasses instead.
    """

    label = xs.variable(intent='out', groups='flux_label')
    values = xs.variable(intent='out', dims=('loss_index', 'Time'))

    SV_labels = xs.variable(intent='in', dims='loss_index')

    flux = xs.on_demand()

    def initialize(self):
        self.label = self.__xsimlab_name__
        print(f"flux {self.label} acting on {self.SV_labels} is initialized")

        self.SV = [self.m.phydra_SVs[label] for label in self.SV_labels]
        self.SV_index = [label for label in self.SV_labels]
        print(self.SV_index)

        fluxes = []
        i = 0
        for label in self.SV_labels:
            self.SV = self.m.phydra_SVs[label]

            flx = self.flux

            self.m.phydra_fluxes[label].append(-flx)
            fluxes.append(flx)
            i += 1

        print(fluxes)
        self.values = [self.m.Intermediate(flx, name=self.label+lab).value for flx, lab in zip(fluxes, self.SV_labels)]

    @flux.compute
    def flux_func(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        raise ValueError('flux needs to be defined in BaseFlux subclass')

@xs.process
class Sinking(LossMultiFlux):
    """ Evan's and Parsley mixing LOSS flux (negative) """
    FX_label_MLD = xs.variable(intent='in', description='MLD forcing label')

    flux = xs.on_demand()

    sinking_rate = xs.variable(intent='in', description='constant diffusive mixing rate')

    values = xs.variable(intent='out', dims=('loss_index2', 'Time'))

    SV_labels = xs.variable(intent='in', dims='loss_index2')

    def initialize(self):
        self.MLD = self.m.phydra_forcings[self.FX_label_MLD]
        super(Sinking, self).initialize()

    @flux.compute
    def sinking(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """

        return self.SV * self.sinking_rate / self.MLD

@xs.process
class Mixing(LossMultiFlux):
    """ Evan's and Parsley mixing LOSS flux (negative) """
    FX_label_MLD = xs.variable(intent='in', description='MLD forcing label')

    flux = xs.on_demand()

    kappa = xs.variable(intent='in', description='constant diffusive mixing rate')

    def initialize(self):
        self.MLD = self.m.phydra_forcings[self.FX_label_MLD]
        self.MLD_deriv = self.m.phydra_forcings[self.FX_label_MLD + '_deriv']

        self.h_pos = self.m.Intermediate(np.max(self.MLD_deriv, 0), name='hpos')
        super(Mixing, self).initialize()

    @flux.compute
    def mixing(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """

        K = (self.h_pos + self.kappa) / self.MLD
        return self.SV * K

@xs.process
class InputMultiFlux(ModelContext):
    """
    Base class for a flux that defines an interaction between 2 state variables

    Do not use this base class directly in a model! Use one of its
    subclasses instead.
    """

    label = xs.variable(intent='out', groups='flux_label')
    values = xs.variable(intent='out', dims=('loss_index', 'Time'))

    SV_labels = xs.variable(intent='in', dims='loss_index')

    flux = xs.on_demand()

    def initialize(self):
        self.label = self.__xsimlab_name__
        print(f"flux {self.label} acting on {self.SV_labels} is initialized")

        self.SV = [self.m.phydra_SVs[label] for label in self.SV_labels]
        self.SV_index = [label for label in self.SV_labels]
        print(self.SV_index)

        fluxes = []
        i = 0
        for label in self.SV_labels:
            self.SV = self.m.phydra_SVs[label]

            flx = self.flux

            self.m.phydra_fluxes[label].append(flx)
            fluxes.append(flx)
            i += 1

        print(fluxes)
        self.values = [self.m.Intermediate(flx, name=self.label+lab).value for flx, lab in zip(fluxes, self.SV_labels)]

    @flux.compute
    def flux_func(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        raise ValueError('flux needs to be defined in BaseFlux subclass')


#########################################################

# EXCHANGE FLUXES

@xs.process
class ExchangeFlux(ModelContext):
    """
    Base class for a flux that defines an interaction between 2 state variables

    Do not use this base class directly in a model! Use one of its
    subclasses instead.
    """
    label = xs.variable(intent='out', groups='flux_label')
    value = xs.variable(intent='out', dims='Time')

    source_label = xs.variable(intent='in')
    sink_label = xs.variable(intent='in')

    flux = xs.on_demand()

    def initialize(self):
        self.label = self.__xsimlab_name__
        print(f"flux {self.label} of {self.sink_label} consuming {self.source_label} is initialized")

        self.source = self.m.phydra_SVs[self.source_label]
        self.sink = self.m.phydra_SVs[self.sink_label]

        flx = self.flux
        self.m.phydra_fluxes[self.source_label].append(-flx)
        self.m.phydra_fluxes[self.sink_label].append(flx)

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
        return self.source * self.rate

@xs.process
class QuadraticExchangeFlux(ExchangeFlux):
    """ Base Growth Flux, Monod function """

    flux = xs.on_demand()

    rate = xs.variable(intent='in', description='quadratic loss rate')

    @flux.compute
    def growth(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return self.source**2 * self.rate

@xs.process
class MonodUptake(ExchangeFlux):
    """ Base Growth Flux, Monod function """
    fx_values = xs.group('forcing_value')  # hack to force process ordering (Forcing before Fluxes)

    flux = xs.on_demand()

    halfsat = xs.variable(intent='in', description='half saturation constant for Monod growth')

    # def initialize(self):
    #    super(GrowthMonod, self).initialize()

    @flux.compute
    def growth(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return self.source / (self.halfsat + self.source) * self.sink


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

    value = xs.variable(intent='out', dims='Time')

    flux = xs.on_demand()

    limiting_factors = xs.group('not_initialized')

    mumax = xs.variable(intent='in', description='maximum growth rate')

    @flux.compute
    def growth(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return self.mumax * np.product([lim_fact(self) for lim_fact in self.limiting_factors]) * self.sink

    @classmethod
    def setup(cls, dim_label):
        """ create copy of process class with user specified name and dimension label """
        new_cls = type(cls.__name__ + dim_label, cls.__bases__, dict(cls.__dict__))
        # add new index with variable name of label (to avoid Zarr storage conflicts)
        new_group = xs.group(dim_label)
        setattr(new_cls, 'limiting_factors', new_group)
        # modify dimensions
        #new_cls.value.metadata['dims'] = ((dim_label, 'Time'),)
        # return intialized xsimlab process
        return xs.process(new_cls)


class GML_BaseFlux(ModelContext):
    """ Base Flux for multi-limitation Growth """

    limiting_factor = xs.variable(intent='out', groups='not_initialized')

    def initialize(self):
        self.limiting_factor = self.limiting_factor_func
        raise ValueError('initialize needs to be defined in GML_BaseFlux subclass')

    def limiting_factor_func(self, cls):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux

          to access state variables, use cls argument
          e.g. cls.source / (self.halfsat + cls.source)
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
        # new_cls.value.metadata['dims'] = ((dim_label, 'Time'),)
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
        return cls.source / (self.halfsat + cls.source)


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
        kPAR = self.m.Intermediate(self.kw + self.kc * cls.sink, name='kPAR')
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
class GrazingFlux(ModelContext):
    """
    Base class for a flux that defines an interaction between 1 state variables and multiple others
    (i.e. grazing to SV + fraction egested to another SV + fraction excreted to another SV)
    give the 3 fraction options, but can also simply pass None or 0.. beta, epsilon

    Do not use this base class directly in a model! Use one of its
    subclasses instead.
    """
    label = xs.variable(intent='out', groups='flux_label')
    value = xs.variable(intent='out', dims='Time')

    source_label = xs.variable(intent='in')
    sink_label = xs.variable(intent='in')
    egested2_label = xs.variable(intent='in')
    excreted2_label = xs.variable(intent='in')

    beta = xs.variable(intent='in', description='absorption efficiency')
    epsilon = xs.variable(intent='in', description='net production efficiency')

    Imax = xs.variable(intent='in', description='maximum grazing rate')

    flux = xs.on_demand()

    def initialize(self):
        self.label = self.__xsimlab_name__
        print(f"flux {self.label} of {self.sink_label} consuming {self.source_label} is initialized \n",
              f"egesting to {self.egested2_label} and excreting to {self.excreted2_label}")

        self.source = self.m.phydra_SVs[self.source_label]
        self.sink = self.m.phydra_SVs[self.sink_label]
        self.egested2 = self.m.phydra_SVs[self.egested2_label]
        self.excreted2 = self.m.phydra_SVs[self.excreted2_label]

        flx = self.flux

        egestion = flx * (1 - self.beta)
        assimilation = flx * self.beta * self.epsilon
        excretion = flx * self.beta * (1 - self.epsilon)

        self.m.phydra_fluxes[self.sink_label].append(assimilation)
        self.m.phydra_fluxes[self.egested2_label].append(egestion)
        self.m.phydra_fluxes[self.excreted2_label].append(excretion)

        # biomass grazed
        self.m.phydra_fluxes[self.source_label].append(-flx)

        self.value = self.m.Intermediate(flx, name=self.label).value

    @flux.compute
    def growth(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        return self.Imax * self.source / (0.5 + self.source) * self.sink



@xs.process
class GrazingFlux_MultiRessource(ModelContext):
    """
    Base class for a flux that defines an interaction between 1 state variables and multiple others
    (i.e. grazing to SV + fraction egested to another SV + fraction excreted to another SV)
    give the 3 fraction options, but can also simply pass None or 0.. beta, epsilon

    Do not use this base class directly in a model! Use one of its
    subclasses instead.
    """
    label = xs.variable(intent='out', groups='flux_label')
    values = xs.variable(intent='out', dims=('source_index', 'Time'))

    source_index = xs.index(dims='source_index')

    source_labels = xs.variable(intent='in', dims='source_index')
    sink_label = xs.variable(intent='in')
    egested2_label = xs.variable(intent='in')
    excreted2_label = xs.variable(intent='in')

    beta = xs.variable(intent='in', description='absorption efficiency')
    epsilon = xs.variable(intent='in', description='net production efficiency')

    Imax = xs.variable(intent='in', description='maximum grazing rate')
    kZ = xs.variable(intent='in', description='half saturation constant of grazing')

    feed_prefs = xs.variable(intent='in', dims='source_index',
                             description='preference of feeding, supply as list of same dims as source_labels')

    flux = xs.on_demand()

    def initialize(self):
        self.label = self.__xsimlab_name__
        print(f"flux {self.label} of {self.sink_label} consuming {self.source_labels} is initialized \n",
              f"egesting to {self.egested2_label} and excreting to {self.excreted2_label}")

        self.sink = self.m.phydra_SVs[self.sink_label]
        self.egested2 = self.m.phydra_SVs[self.egested2_label]
        self.excreted2 = self.m.phydra_SVs[self.excreted2_label]

        self.sources = [self.m.phydra_SVs[label] for label in self.source_labels]
        self.source_index = [label for label in self.source_labels]
        print(self.source_index)

        fluxes = []
        i = 0
        for label in self.source_labels:
            self.source = self.m.phydra_SVs[label]

            self.feed_pref = self.feed_prefs[i]

            flx = self.flux
            self.m.phydra_fluxes[label].append(-flx)
            fluxes.append(flx)
            i += 1

        print(fluxes)

        egestion = self.m.sum(fluxes) * (1 - self.beta)
        assimilation = self.m.sum(fluxes) * self.beta * self.epsilon
        excretion = self.m.sum(fluxes) * self.beta * (1 - self.epsilon)

        self.m.phydra_fluxes[self.sink_label].append(assimilation)
        self.m.phydra_fluxes[self.egested2_label].append(egestion)
        self.m.phydra_fluxes[self.excreted2_label].append(excretion)

        self.values = [self.m.Intermediate(flx, name=self.label+lab).value for flx, lab in zip(fluxes, self.source_labels)]

    @flux.compute
    def growth(self):
        """ compute function of on_demand xarray variable
         specific flux needs to be implemented in BaseFlux """
        TotalGrazing = self.m.sum(np.array(self.sources)**2 * np.array(self.feed_prefs))
        return self.Imax * self.source**2 * self.feed_pref / (self.kZ**2 + TotalGrazing) * self.sink
