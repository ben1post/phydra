import numpy as np
import xsimlab as xs

def phydra_setup(model, input_vars, output_vars):
    """ This function wraps create_setup and adds a dummy clock parameter
    necessary for model execution """
    return xs.create_setup(model=model,
                           # necessary for xsimlab
                           clocks={'clock': [0, 1]},
                           input_vars=input_vars,
                           output_vars=output_vars)
