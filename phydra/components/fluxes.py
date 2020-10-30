import phydra


@phydra.comp(init_stage=3)
class ExponentialGrowth:
    var = phydra.variable(foreign=True, flux='input', negative=False, description='variable affected by flux')
    rate = phydra.parameter(description='linear rate of change')

    def input(var, rate):
        """ """
        return var * rate





################################################################################################################

################################################################################################################

################OLD#CODE########################################################################################

################################################################################################################

################################################################################################################

################################################################################################################

# Standard Linear Fluxes:
@phydra.comp
class LinearOutputFlux:
    sv = phydra.variable(description='state variable affected by flux')
    rate = phydra.parameter(description='flowing rate')

    def flux(sv, rate):
        return sv * rate


@phydra.comp
class LinearInputFlux:
    sv = phydra.variable(description='state variable affected by flux')
    rate = phydra.parameter(description='flowing rate')

    def flux(sv, rate):
        return sv * rate


@phydra.comp
class ForcingLinearInputFlux:
    sv = phydra.variable(description='state variable affected by forcing flux')
    fx = phydra.forcing(description='forcing affecting rate')
    rate = phydra.parameter(description='flowing rate')

    def flux(sv, fx, rate):
        return fx * rate


# Michaelis Menten Flux
@phydra.comp
class MonodUptake:
    resource = phydra.variable()
    consumer = phydra.variable()
    halfsat = phydra.parameter(description='half saturation constant')

    def flux(resource, consumer, halfsat):
        return resource / (resource + halfsat) * consumer


# Multi limitation flux (wrap as phydra.multiflux)
#@phydra.multiflux(group='growth')
#class MonodUptake:
#    flux = xs.variable(intent='out', groups=group)


# Grazing Flux
@phydra.comp
class HollingTypeIIIGrazing:
    resource = phydra.variable()
    consumer = phydra.variable()
    feed_pref = phydra.parameter(description='feeding preferences')
    Imax = phydra.parameter(description='maximum ingestion rate')
    kZ = phydra.parameter(description='feeding preferences')

    def flux(resource, consumer, feed_pref, Imax, kZ):
        return Imax * resource ** 2 \
               * feed_pref / (kZ ** 2 + sum([resource ** 2 * feed_pref])) * consumer

# TODO: How to make vectorization explicit here,
#   and simplify the routing of output?


@phydra.comp
class MultiLossTest:
    sources = phydra.variable(dims='MultiLoss')
    sink = phydra.variable(partial_out='sink_out')
    rate = phydra.parameter(dims=[(), 'MultiLoss'])

    def flux(sources, sink, rate):
        return sources * rate

    def sink_out(flux):
        return sum(flux)





@phydra.comp
class MultiTest2:
    svs_input = phydra.variable(dims='Multi_input')
    svs_output = phydra.variable(dims='Multi_output')
    rate = phydra.parameter()

    def flux(svs_input, svs_output, rate):
        # TODO: SO I WANT THIS FUNCTION TO BE ONLY CALLED ONCE at each step
        #   and it should take inputs as arrays and return an array
        # but then somehow that array needs to be "re-routed" according to the arguments above
        return svs_input * rate

    def output(label, ):
        # here input dict that allows rerouting of output
        return out[label]



    # flux always needs to return a single value in the current framework
    #
    # this is the problem..
    #
    #


    def output(out):
        return()



#### different take:
# 1. calculate each individual sv output flux & apply to SV
# 2. sum all this up
# 3. define fractions that the sum is routed to



#### SO there are separate stages:

## 1. where I calculate the TOTAL portion of each flux, using an array of all variables, Params etc.

## 2. This Total is then run through a function
# that returns the output portion for each SV involved and assigns that a specific Flux

## 3. Then another calculation returns the input portions for the specified fluxes



# OKAY, so in order to get sv input either as FuncGroup or List, I need to add a dim to the xs.var
# the next step is how it retrieves the vars of the input labels within the flux function
# and how it assigns the fluxes to thoses state variabes

# FOR NOW I DON'T NEED TO CARE ABOUT FUNC GROUPS!

# so what I need is:
# 1. get a simple loss function to work, that takes multiple inputs, and returns multiple outputs
# the phydra.sv in this case is always dependent on a single item in the input list!

# So instead of fancy stuff, all I need to do is assign the flux to all items in list separately
# then why don't just create 3 fluxes, and pass as list? don't make sense

# actually, kinda does make sense.. but in a different way!

# I need to be able to supply a list of ressources, that each has their own output flux
# that feeds into a single input flux

# and then later, I can try and split the input flux up

# but these stages are always separated!









import xsimlab as xs

@phydra.comp
class PartialOutputLossTest:
    egested = phydra.variable()#, partial='egestion')
    rate = phydra.parameter()
    egested_frac = xs.variable(intent='in')

    # in flux, state comes in, flux needs to unpack full list to array
    # and be able to do vectorized calcs on it!
    def flux(sv, rate):
        return sv * rate

    # then, either array is returned, or a sum_total,
    # and in "partial_in/output", rest of the calculation specific to one SV can be made
    # HOWEVER: this needs to work with list of labels and func groups
    # ... how to do this in the backend?

    #
    def egestion(self):
        return self.flux * self.egested_frac

