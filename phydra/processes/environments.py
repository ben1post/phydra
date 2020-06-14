import numpy as np
import xsimlab as xs

from .gekkocontext import InheritGekkoContext
from .forcing import MLDForcing, NutrientForcing
#from .forcingfluxes import Mixing, Sinking, Upwelling

@xs.process
class BaseEnvironment(InheritGekkoContext):
    """ Process that collects all component output
    into one single variable for easier plotting
    Note: only works for (ALL) one dimensional components for now!

    Specific environments that inherit from the BaseEnvironment class
    provide an interface for external forcing"""

    components = xs.index(dims='components')
    outputs = xs.variable(intent='out', dims=('components', 'time'))

    comp_labels = xs.group('comp_label')
    comp_dims = xs.group('comp_dim')
    comp_indices = xs.group('comp_index')
    comp_outputs = xs.group('comp_output')

    def initialize(self):
        self.components = [index for indices in self.comp_indices for index in indices]

    def finalize_step(self):
        self.outputs = [output for outputs in self.comp_outputs for output in outputs]


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

    forcingpars_labels = xs.group('forcingpar_label')
    forcingpars_defaultvals = xs.group('forcingpar_defaultval')

    #K = xs.foreign(Mixing, 'flux_func')  # returns function to calculate mixing from input

    def initialize(self):
        super(Slab, self).initialize()

        #default_forcingpars = {label:value for label,value in zip(self.forcingpars_labels,self.forcingpars_defaultval)}
        #N_forcingpars = {'kappa':0.1, 'sinking':0.1, 'upwelling':'N0'} # perhaps upwelling should be modified in specific upwelling process
        #P_forcingpars = {'kappa':0.1, 'sinking':0.1, 'upwelling':None}
        #Z_forcingpars = {'kappa':0.1, 'sinking':0, 'upwelling':None}
        #D_forcingpars = {'kappa':0.1, 'sinking':0.1, 'upwelling':None}

        #for c_label in self.components:
        #    self.ForcingParameters[c_label] = self.components

