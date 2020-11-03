import phydra


@phydra.comp(init_stage=3)
class LinearInput:
    var = phydra.variable(foreign=True, flux='input', negative=False, description='variable affected by flux')
    rate = phydra.parameter(description='linear rate of change')

    def input(var, rate):
        """ """
        return rate


@phydra.comp(init_stage=3)
class LinearForcingInput:
    var = phydra.variable(foreign=True, flux='input', negative=False, description='variable affected by flux')
    forcing = phydra.forcing(foreign=True, description='forcing affecting flux')
    rate = phydra.parameter(description='linear rate of change')

    def input(var, forcing, rate):
        """ """
        return forcing * rate


@phydra.comp(init_stage=3)
class ExponentialGrowth:
    var = phydra.variable(foreign=True, flux='input', negative=False, description='variable affected by flux')
    rate = phydra.parameter(description='linear rate of change')

    def input(var, rate):
        """ """
        return var * rate


@phydra.comp(init_stage=3)
class LinearMortality:
    var = phydra.variable(foreign=True, flux='death', negative=True, description='variable affected by flux')
    rate = phydra.parameter(description='linear rate of change')

    def death(var, rate):
        """ """
        return var * rate


@phydra.comp(init_stage=3)
class QuadraticMortality:
    var = phydra.variable(foreign=True, flux='death', negative=True, description='variable affected by flux')
    rate = phydra.parameter(description='quadratic rate of change')

    def death(var, rate):
        """ """
        return var ** 2 * rate


@phydra.comp(init_stage=3)
class MonodGrowth:
    resource = phydra.variable(foreign=True, flux='uptake', negative=True)
    consumer = phydra.variable(foreign=True, flux='uptake', negative=False)

    halfsat = phydra.parameter()

    def uptake(resource, consumer, halfsat):
        return resource / (resource + halfsat) * consumer


@phydra.comp(init_stage=3)
class HollingTypeIII:
    resource = phydra.variable(foreign=True, flux='grazing', negative=True)
    consumer = phydra.variable(foreign=True, flux='grazing', negative=False)
    feed_pref = phydra.parameter(description='feeding preferences')
    Imax = phydra.parameter(description='maximum ingestion rate')
    kZ = phydra.parameter(description='feeding preferences')

    def grazing(resource, consumer, feed_pref, Imax, kZ):
        return Imax * resource ** 2 \
               * feed_pref / (kZ ** 2 + sum([resource ** 2 * feed_pref])) * consumer
