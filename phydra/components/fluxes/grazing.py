import phydra


@phydra.comp
class HollingTypeIII:
    resource = phydra.variable(foreign=True, flux='grazing', negative=True)
    consumer = phydra.variable(foreign=True, flux='grazing', negative=False)
    feed_pref = phydra.parameter(description='feeding preferences')
    Imax = phydra.parameter(description='maximum ingestion rate')
    kZ = phydra.parameter(description='feeding preferences')

    @phydra.flux
    def grazing(self, resource, consumer, feed_pref, Imax, kZ):
        return Imax * resource ** 2 \
               * feed_pref / (kZ ** 2 + self.m.sum([resource ** 2 * feed_pref])) * consumer


@phydra.comp
class HollingTypeIII_ResourcesListInput:
    """ """
    resources = phydra.variable(foreign=True, negative=True, flux='grazing', list_input=True, dims='resources')
    consumer = phydra.variable(foreign=True, flux='grazing', negative=False)
    feed_prefs = phydra.parameter(dims='resources', description='feeding preference for resources')
    Imax = phydra.parameter(description='maximum ingestion rate')
    kZ = phydra.parameter(description='feeding preferences')

    @phydra.flux(dims='resources')
    def grazing(self, resources, consumer, feed_prefs, Imax, kZ):
        scaled_resources = resources ** 2 * feed_prefs

        return scaled_resources * Imax / (kZ ** 2 + self.m.sum(scaled_resources)) * consumer


@phydra.comp
class HollingTypeIII_ResourcesListInput_Consumption2Group:
    """
    """
    resources = phydra.variable(foreign=True, negative=True, flux='grazing', list_input=True, dims='resources')
    consumer = phydra.variable(foreign=True)
    feed_prefs = phydra.parameter(dims='resources', description='feeding preference for resources')
    Imax = phydra.parameter(description='maximum ingestion rate')
    kZ = phydra.parameter(description='feeding preferences')

    @phydra.flux(group='graze_out', dims='resources')
    def grazing(self, resources, consumer, feed_prefs, Imax, kZ):
        scaled_resources = resources ** 2 * feed_prefs
        return scaled_resources * Imax / (kZ ** 2 + self.m.sum(scaled_resources)) * consumer


@phydra.comp(init_stage=4)
class GrossGrowthEfficiency:
    """
    to N: beta*(1-epsilon)
    to D: 1-beta
    to Z: beta*epsilon
    """
    assimilated_consumer = phydra.variable(foreign=True, flux='assimilation')
    egested_detritus = phydra.variable(foreign=True, flux='egestion')
    excreted_nutrient = phydra.variable(foreign=True, flux='excretion')

    beta = phydra.parameter(description='absorption efficiency')
    epsilon = phydra.parameter(description='net production efficiency')

    @phydra.flux(group_to_arg='graze_out')
    def assimilation(self, assimilated_consumer, egested_detritus, excreted_nutrient, graze_out, beta, epsilon):
        return self.m.sum(graze_out) * beta * epsilon

    @phydra.flux(group_to_arg='graze_out')
    def egestion(self, assimilated_consumer, egested_detritus, excreted_nutrient, graze_out, beta, epsilon):
        return self.m.sum(graze_out) * (1-beta)

    @phydra.flux(group_to_arg='graze_out')
    def excretion(self, assimilated_consumer, egested_detritus, excreted_nutrient, graze_out, beta, epsilon):
        return self.m.sum(graze_out) * beta * (1-epsilon)


@phydra.comp
class SizebasedGrazingKernel_Dims:
    """ ASTroCAT Grazing Kernel """
    resource = phydra.variable(foreign=True, dims='resource')
    consumer = phydra.variable(foreign=True, dims='consumer')
    phiP = phydra.parameter(dims=('resource', 'consumer'), description='feeding preferences')
    Imax = phydra.parameter(dims='consumer', description='maximum ingestion rate')
    KsZ = phydra.parameter(description='half sat of grazing')

    @phydra.flux(group='graze_matrix', dims=('resource', 'consumer'))
    def grazing(self, resource, consumer, phiP, Imax, KsZ):
        """ """
        PscaledAsFood = phiP / KsZ * resource[:, None]
        FgrazP = Imax * consumer * PscaledAsFood / (1 + self.m.sum(PscaledAsFood, axis=0))
        return FgrazP


@phydra.comp(init_stage=4)
class GrossGrowthEfficiency_MatrixGrazing:
    """
    to N: beta*(1-epsilon)
    to D: 1-beta
    to Z: beta*epsilon
    """
    grazed_resource = phydra.variable(dims='resource', foreign=True, flux='grazing', negative=True)
    assimilated_consumer = phydra.variable(dims='consumer', foreign=True, flux='assimilation')
    egested_detritus = phydra.variable(foreign=True, flux='egestion')

    f_eg = phydra.parameter(description='fraction egested')
    epsilon = phydra.parameter(description='net production efficiency')

    @phydra.flux(dims='resource', group_to_arg='graze_matrix')
    def grazing(self, assimilated_consumer, egested_detritus, grazed_resource, graze_matrix, f_eg, epsilon):
        """ """
        out = self.m.sum(graze_matrix, axis=1)
        return out

    @phydra.flux(dims='consumer', group_to_arg='graze_matrix')
    def assimilation(self, assimilated_consumer, egested_detritus, grazed_resource, graze_matrix, f_eg, epsilon):
        """ """
        out = self.m.sum(graze_matrix, axis=0) * epsilon
        return out

    @phydra.flux(group_to_arg='graze_matrix')
    def egestion(self, assimilated_consumer, egested_detritus, grazed_resource, graze_matrix, f_eg, epsilon):
        """ """
        out = self.m.sum(graze_matrix, axis=None) * (1 - f_eg - epsilon)
        return out


@phydra.comp
class SizebasedGrazingKernel_NoExtraFlux_Dims:
    """ ASTroCAT Grazing Kernel """
    resource = phydra.variable(foreign=True, dims='resource')
    consumer = phydra.variable(foreign=True, dims='consumer')
    phiP = phydra.parameter(dims=('resource', 'consumer'), description='feeding preferences')
    Imax = phydra.parameter(dims='consumer', description='maximum ingestion rate')
    KsZ = phydra.parameter(description='half sat of grazing')

    @phydra.flux(group='graze_matrix', dims=('resource', 'consumer'))
    def grazing(self, resource, consumer, phiP, Imax, KsZ):
        """ """
        PscaledAsFood = phiP / KsZ * resource[:, None]
        FgrazP = Imax * consumer * PscaledAsFood / (1 + self.m.sum(PscaledAsFood, axis=0))
        return FgrazP