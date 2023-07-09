import xso


@xso.component
class SizebasedGrazingMatrix:
    """Size-based grazing function, adapted from Banas et al. (2011).

    The grazing function defines a complex pair-wise interaction between
    the size-spectra of phytoplankton and zooplankton. The grazing function
    scales with the size of the consumer and the feeding preference of the
    consumer for a given resource size. The grazing function is further
    scaled by the maximum ingestion rate of the consumer and the half-saturation
    constant of grazing.

    It is implemented in two parts. This component calculates the grazing matrix of each
    size class interaction. The second calculates receives the grazing matrix via the
    'group' argument to the flux function, and sums over the matrix to route the fluxes.
    """
    resource = xso.variable(foreign=True, dims='phyto')
    consumer = xso.variable(foreign=True, dims='zoo')
    phiP = xso.parameter(dims=('phyto', 'zoo'), description='feeding preferences')
    Imax = xso.parameter(dims='zoo', description='maximum ingestion rate')
    KsZ = xso.parameter(description='half saturation constant of grazing')

    @xso.flux(group='graze_matrix', dims=('phyto', 'zoo'))
    def grazing(self, resource, consumer, phiP, Imax, KsZ):
        """Here we are using a matrix calculation, to define the pair-wise interaction."""
        PscaledAsFood = phiP / KsZ * resource[:, None] # using np.newaxis to flip the array and create matrix
        FgrazP = Imax * consumer * PscaledAsFood / (1 + self.m.sum(PscaledAsFood, axis=0)) # sum over Zoo
        return FgrazP


@xso.component
class GrossGrowthEfficiency_MatrixGrazing:
    """ Coponent to calculate the grazing fluxes for each of the model variables, adapted from Banas et al. (2011).

    The grazing fluxes are calculated by multiplying the grazing matrix with the
    biomass of the resource. The grazing matrix is calculated by the SizebasedGrazingKernel_Dims

    to N: beta*(1-epsilon)
    to D: 1-beta
    to Z: beta*epsilon
    """
    grazed_resource = xso.variable(dims='phyto', foreign=True, flux='grazing', negative=True)
    assimilated_consumer = xso.variable(dims='zoo', foreign=True, flux='assimilation')
    egested_detritus = xso.variable(foreign=True, flux='egestion')

    f_eg = xso.parameter(description='fraction egested')
    epsilon = xso.parameter(description='net production efficiency')

    @xso.flux(dims='phyto', group_to_arg='graze_matrix')
    def grazing(self, assimilated_consumer, egested_detritus, grazed_resource, graze_matrix, f_eg, epsilon):
        """ """
        out = self.m.sum(graze_matrix, axis=1)
        return out

    @xso.flux(dims='zoo', group_to_arg='graze_matrix')
    def assimilation(self, assimilated_consumer, egested_detritus, grazed_resource, graze_matrix, f_eg, epsilon):
        """ """
        out = self.m.sum(graze_matrix, axis=0) * epsilon
        return out

    @xso.flux(group_to_arg='graze_matrix')
    def egestion(self, assimilated_consumer, egested_detritus, grazed_resource, graze_matrix, f_eg, epsilon):
        """ """
        out = self.m.sum(graze_matrix, axis=None) * (1 - f_eg - epsilon)
        return out
