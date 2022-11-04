import xso


@xso.component
class HollingTypeIII:
    resource = xso.variable(foreign=True, flux='grazing', negative=True)
    consumer = xso.variable(foreign=True, flux='grazing', negative=False)
    feed_pref = xso.parameter(description='feeding preferences')
    Imax = xso.parameter(description='maximum ingestion rate')
    kZ = xso.parameter(description='feeding preferences')

    @xso.flux
    def grazing(self, resource, consumer, feed_pref, Imax, kZ):
        return Imax * resource ** 2 \
               * feed_pref / (kZ ** 2 + self.m.sum([resource ** 2 * feed_pref])) * consumer


@xso.component
class HollingTypeIII_ResourcesListInput:
    """ """
    resources = xso.variable(foreign=True, negative=True, flux='grazing', list_input=True, dims='resources')
    consumer = xso.variable(foreign=True, flux='grazing', negative=False)
    feed_prefs = xso.parameter(dims='resources', description='feeding preference for resources')
    Imax = xso.parameter(description='maximum ingestion rate')
    kZ = xso.parameter(description='feeding preferences')

    @xso.flux(dims='resources')
    def grazing(self, resources, consumer, feed_prefs, Imax, kZ):
        scaled_resources = resources ** 2 * feed_prefs

        return scaled_resources * Imax / (kZ ** 2 + self.m.sum(scaled_resources)) * consumer


@xso.component
class HollingTypeIII_ResourcesListInput_Consumption2Group:
    """
    """
    resources = xso.variable(foreign=True, negative=True, flux='grazing', list_input=True, dims='resources')
    consumer = xso.variable(foreign=True)
    feed_prefs = xso.parameter(dims='resources', description='feeding preference for resources')
    Imax = xso.parameter(description='maximum ingestion rate')
    kZ = xso.parameter(description='feeding preferences')

    @xso.flux(group='graze_out', dims='resources')
    def grazing(self, resources, consumer, feed_prefs, Imax, kZ):
        scaled_resources = resources ** 2 * feed_prefs
        return scaled_resources * Imax / (kZ ** 2 + self.m.sum(scaled_resources)) * consumer


@xso.component(init_stage=4)
class GrossGrowthEfficiency:
    """
    to N: beta*(1-epsilon)
    to D: 1-beta
    to Z: beta*epsilon
    """
    assimilated_consumer = xso.variable(foreign=True, flux='assimilation')
    egested_detritus = xso.variable(foreign=True, flux='egestion')
    excreted_nutrient = xso.variable(foreign=True, flux='excretion')

    beta = xso.parameter(description='absorption efficiency')
    epsilon = xso.parameter(description='net production efficiency')

    @xso.flux(group_to_arg='graze_out')
    def assimilation(self, assimilated_consumer, egested_detritus, excreted_nutrient, graze_out, beta, epsilon):
        return self.m.sum(graze_out) * beta * epsilon

    @xso.flux(group_to_arg='graze_out')
    def egestion(self, assimilated_consumer, egested_detritus, excreted_nutrient, graze_out, beta, epsilon):
        return self.m.sum(graze_out) * (1-beta)

    @xso.flux(group_to_arg='graze_out')
    def excretion(self, assimilated_consumer, egested_detritus, excreted_nutrient, graze_out, beta, epsilon):
        return self.m.sum(graze_out) * beta * (1-epsilon)


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


@xso.component(init_stage=4)
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


@xso.component
class SizebasedGrazingKernel_NoExtraFlux_Dims:
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