import numpy as np
import xsimlab as xs

from .gekkocontext import InheritGekkoContext
from .forcing import MLDForcing, NutrientForcing

@xs.process
class BaseEnvironment(InheritGekkoContext):
    """ Process that collects all component output
    into one single variable for easier plotting
    Note: only works for (ALL) one dimensional components for now!

    Specific environments that inherit from the BaseEnvironment class
    provide an interface for external forcing"""

    components = xs.index(dims='components')
    forcingfluxes = xs.index(dims='forcingfluxes')

    comp_output = xs.variable(intent='out', dims=('components', 'time'))
    fxflux_output = xs.variable(intent='out', dims=('forcingfluxes', 'time'))

    comp_indices = xs.group('comp_index')
    comp_outputs = xs.group('comp_output')

    fxflux_indices = xs.group('fxflux_index')
    fxflux_outputs = xs.group('fxflux_output')

    def initialize(self):
        print('hello there')
        self.components = [index for indices in self.comp_indices for index in indices]
        self.forcingfluxes = [index for indices in self.fxflux_indices for index in indices]
        print(f"\n")
        print(f"Initializing Environment: \n components:{self.components} \n fx-fluxes:{self.forcingfluxes}")
        print(f"\n")

    def finalize_step(self):
        if False==True:
            print(f"HERE OUT:,\n {list(self.comp_outputs)},\n {list(self.fxflux_outputs)}")
            print(f"\n")
            print([output for outputs in self.comp_outputs for output in outputs])
            print(f"\n")
            print([output for outputs in self.fxflux_outputs for output in outputs])
        self.comp_output = [output for outputs in self.comp_outputs for output in outputs]
        self.fxflux_output = [output for outputs in self.fxflux_outputs for output in outputs]


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

    def initialize(self):
        super(Slab, self).initialize()