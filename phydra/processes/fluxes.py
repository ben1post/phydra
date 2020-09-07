import phydra

# Standard Linear Fluxes:
@phydra.flux
class LinearOutputFlux:
    sv = phydra.sv(flow='output', description='state variable affected by flux')
    rate = phydra.param(description='flowing rate')

    def flux(sv, rate):
        return sv * rate


@phydra.flux
class LinearInputFlux:
    sv = phydra.sv(flow='input', description='state variable affected by flux')
    rate = phydra.param(description='flowing rate')

    def flux(sv, rate):
        return sv * rate


@phydra.flux
class ForcingLinearInputFlux:
    sv = phydra.sv(flow='input', description='state variable affected by forcing flux')
    fx = phydra.fx(description='forcing affecting rate')
    rate = phydra.param(description='flowing rate')

    def flux(sv, fx, rate):
        return fx * rate


# Michaelis Menten Flux
@phydra.flux
class MonodUptake:
    resource = phydra.sv(flow='output')
    consumer = phydra.sv(flow='input')
    halfsat = phydra.param(description='half saturation constant')

    def flux(resource, consumer, halfsat):
        return resource / (resource + halfsat) * consumer


# Multi limitation flux (wrap as phydra.multiflux)
#@phydra.multiflux(group='growth')
#class MonodUptake:
#    flux = xs.variable(intent='out', groups=group)


# Grazing Flux
@phydra.flux
class HollingTypeIIIGrazing:
    resource = phydra.sv(flow='output')
    consumer = phydra.sv(flow='input')
    feed_pref = phydra.param(description='feeding preferences') # add arg: units = ['micromolar'], add conversion to Latex
    Imax = phydra.param(description='maximum ingestion rate')
    kZ = phydra.param(description='feeding preferences')

    def flux(resource, consumer, feed_pref, Imax, kZ):
        return Imax * resource ** 2 \
               * feed_pref / (kZ ** 2 + sum([resource ** 2 * feed_pref])) * consumer

    # 2nd resource should be calculated for every resource


# so having a list input, is actually not what I need right now!
# hm actually I DO!
# so the main problem is, that a single flux needs to have an input of the

# either! define seperate flux functions for svs
# or! USE vectorization, so that a part of the input is ARRAYS..
# ... BUT, is that reasonable??
# this is necessary for functional groups,

# ?????????????????


# XxXxXxX

# 1. vectorize - both for list input and func group input
# 2. split output - define different outputs in func

# - so the flux,

import numpy as np

# TODO: How to make vectorization explicit here,
#   and simplify the routing of output?


@phydra.multiflux
class MultiLossTest:
    sources = phydra.sv(flow='output', dims='MultiLoss')

    rate = phydra.param()

    def flux(sources, rate):
        return sources * rate





@phydra.multiflux
class MultiTest2:
    svs_input = phydra.sv(flow='output', dims='Multi_input')
    svs_output = phydra.sv(flow='output', dims='Multi_output')
    rate = phydra.param()

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

## 1. where I calculate the TOTAL portion of each flux, using an array of all SVs, Params etc.

## 2. This Total is then run through a function
# that returns the output portion for each SV involved and assigns that a specific Flux

## 3. Then another calculation returns the input portions for the specified fluxes



# OKAY, so in order to get sv input either as FuncGroup or List, I need to add a dim to the xs.var
# the next step is how it retrieves the states of the input labels within the flux function
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

@phydra.flux
class PartialOutputLossTest:
    egested = phydra.sv(flow='output')#, partial='egestion')
    rate = phydra.param()
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

