import xso


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


@xso.component
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
