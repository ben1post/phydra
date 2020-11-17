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
class HollingTypeIII_ResourcesListInput_NoOutput:
    """
    to N: beta*(1-epsilon)
    to D: 1-beta
    to Z: beta*epsilon
    """
    resources = phydra.variable(foreign=True, negative=True, flux='grazing', list_input=True, dims='resources')
    consumer = phydra.variable(foreign=True)
    feed_prefs = phydra.parameter(dims='resources', description='feeding preference for resources')
    Imax = phydra.parameter(description='maximum ingestion rate')
    kZ = phydra.parameter(description='feeding preferences')

    @phydra.flux(group='graze_out', dims='resources')
    def grazing(self, resources, consumer, feed_prefs, Imax, kZ):
        # print(resources, consumer, feed_prefs, Imax, kZ)

        scaled_resources = resources ** 2 * feed_prefs

        return scaled_resources * Imax / (kZ ** 2 + self.m.sum(scaled_resources)) * consumer


@phydra.comp(init_stage=4)
class GGE_Routing:
    """ """
    nut = phydra.variable(foreign=True, flux='out3')
    det = phydra.variable(foreign=True, flux='out2')
    consumer = phydra.variable(foreign=True, flux='out1')

    beta = phydra.parameter()
    epsilon = phydra.parameter()

    @phydra.flux(group_to_arg='graze_out')
    def out1(self, nut, det, consumer, graze_out, beta, epsilon):
        return self.m.sum(graze_out) * beta * epsilon

    @phydra.flux(group_to_arg='graze_out')
    def out2(self, nut, det, consumer, graze_out, beta, epsilon):
        return self.m.sum(graze_out) * (1-beta)

    @phydra.flux(group_to_arg='graze_out')
    def out3(self, nut, det, consumer, graze_out, beta, epsilon):
        return self.m.sum(graze_out) * beta * (1-epsilon)

