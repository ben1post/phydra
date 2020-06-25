import numpy as np
import xsimlab as xs

from .gekkocontext import InheritGekkoContext
from .forcing import MLDForcing, FlowForcing, NutrientForcing, IrradianceForcing, TemperatureForcing

@xs.process
class BaseEnvironment(InheritGekkoContext):
    """ Process that collects all component output
    into one single variable for easier plotting
    Note: only works for (ALL) one dimensional components for now!

    Specific environments that inherit from the BaseEnvironment class
    provide an interface for external forcing"""

    components = xs.index(dims='components')
    fluxes = xs.index(dims='fluxes')
    forcingfluxes = xs.index(dims='forcingfluxes')
    forcings = xs.index(dims='forcings')


    comp_output = xs.variable(intent='out', dims=('components', 'time'))
    flux_output = xs.variable(intent='out', dims=('fluxes', 'time'))
    fxflux_output = xs.variable(intent='out', dims=('forcingfluxes', 'time'))
    forcing_output = xs.variable(intent='out', dims=('forcings', 'time'))

    comp_indices = xs.group('comp_index')
    comp_outputs = xs.group('comp_output')

    flux_indices = xs.group('flux_index')
    flux_outputs = xs.group('flux_output')

    fxflux_indices = xs.group('fxflux_index')
    fxflux_outputs = xs.group('fxflux_output')

    forcing_indices = xs.group('forcing_index')
    forcing_outputs = xs.group('forcing_interpolated')



    def initialize(self):
        self.components = [index for indices in self.comp_indices for index in indices]
        self.forcingfluxes = [index for indices in self.fxflux_indices for index in indices]
        self.fluxes = [index for indices in self.flux_indices for index in indices]

        self.forcings = [index for index in self.forcing_indices]

        print(f"\n")
        print(f"Initializing Environment: \n components:{self.components} \n fluxes:{self.fluxes} \
                \n fx-fluxes:{self.forcingfluxes} \n gekko context:{self.gk_context}")
        print(f"\n")

    def finalize_step(self):
        """ Step collects outputs generated in Components, Fluxes and ForcingFluxes"""
        self.comp_output = [output for outputs in self.comp_outputs for output in outputs]

        self.flux_output = [output for outputs in self.flux_outputs for output in outputs]
        self.fxflux_output = [output for outputs in self.fxflux_outputs for output in outputs]
        self.forcing_output = [forcing for forcing in self.forcing_outputs]


@xs.process
class Chemostat(BaseEnvironment):
    """ Physical Environment for Slab setting
    requires Forcing for:
        - Nutrient below the mixed layer (N0) - constant or variable
        (hm, how to switch and point to correct FX file?)
        -  Mixed Layer Depth (MLD)
        - Irradiance at surface (Is)

        optional (perhaps implement in subclass?):
        - Temperature

    also provides specific terms needed for Fluxes, i.e. K, integrated Light
    """
    Flow = xs.foreign(FlowForcing, 'forcing')  # m.Param()

    N0_forcing = xs.foreign(NutrientForcing, 'forcing')

    def initialize(self):
        super(Chemostat, self).initialize()


@xs.process
class Slab(BaseEnvironment):
    """ Physical Environment for Slab setting
    requires Forcing for:
        - Nutrient below the mixed layer (N0) - constant or variable
        (hm, how to switch and point to correct FX file?)
        -  Mixed Layer Depth (MLD)
        - Irradiance at surface (Is)

        optional (perhaps implement in subclass?):
        - Temperature

    also provides specific terms needed for Fluxes, i.e. K, integrated Light
    """
    MLD = xs.foreign(MLDForcing, 'forcing')  # m.Param()
    MLD_deriv = xs.foreign(MLDForcing, 'derivative')   # m.Param()

    N0_forcing = xs.foreign(NutrientForcing, 'forcing')

    I0_forcing = xs.foreign(IrradianceForcing, 'forcing')

    Temp_forcing = xs.foreign(TemperatureForcing, 'forcing')

    def initialize(self):
        super(Slab, self).initialize()