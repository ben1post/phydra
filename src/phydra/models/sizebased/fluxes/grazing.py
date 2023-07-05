import xso


@xso.component
class SizebasedGrazingKernel_Dims:
    """ ASTroCAT Grazing Kernel """
    resource = xso.variable(foreign=True, dims='resource')
    consumer = xso.variable(foreign=True, dims='consumer')
    phiP = xso.parameter(dims=('resource', 'consumer'), description='feeding preferences')
    Imax = xso.parameter(dims='consumer', description='maximum ingestion rate')
    KsZ = xso.parameter(description='half sat of grazing')

    @xso.flux(group='graze_matrix', dims=('resource', 'consumer'))
    def grazing(self, resource, consumer, phiP, Imax, KsZ):
        """ """
        PscaledAsFood = phiP / KsZ * resource[:, None]
        FgrazP = Imax * consumer * PscaledAsFood / (1 + self.m.sum(PscaledAsFood, axis=0))
        return FgrazP


@xso.component
class GrossGrowthEfficiency_MatrixGrazing:
    """
    to N: beta*(1-epsilon)
    to D: 1-beta
    to Z: beta*epsilon
    """
    grazed_resource = xso.variable(dims='resource', foreign=True, flux='grazing', negative=True)
    assimilated_consumer = xso.variable(dims='consumer', foreign=True, flux='assimilation')
    egested_detritus = xso.variable(foreign=True, flux='egestion')

    f_eg = xso.parameter(description='fraction egested')
    epsilon = xso.parameter(description='net production efficiency')

    @xso.flux(dims='resource', group_to_arg='graze_matrix')
    def grazing(self, assimilated_consumer, egested_detritus, grazed_resource, graze_matrix, f_eg, epsilon):
        """ """
        out = self.m.sum(graze_matrix, axis=1)
        return out

    @xso.flux(dims='consumer', group_to_arg='graze_matrix')
    def assimilation(self, assimilated_consumer, egested_detritus, grazed_resource, graze_matrix, f_eg, epsilon):
        """ """
        out = self.m.sum(graze_matrix, axis=0) * epsilon
        return out

    @xso.flux(group_to_arg='graze_matrix')
    def egestion(self, assimilated_consumer, egested_detritus, grazed_resource, graze_matrix, f_eg, epsilon):
        """ """
        out = self.m.sum(graze_matrix, axis=None) * (1 - f_eg - epsilon)
        return out
