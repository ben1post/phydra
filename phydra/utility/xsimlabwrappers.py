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


def createMultiComp(base_process, comp_label, comp_dim):
    """ This function allows addition of a specific label and dimension to create
    multiple interacting instances of a component"""
    @xs.process
    class AddIndexCompLabel(base_process):
        label = xs.variable(intent='out')
        dim = xs.variable(intent='out')
        index = xs.index(dims=comp_label)

        output = xs.variable(intent='out', dims=(comp_label, 'time'))

        def initialize(self):
            self.label = comp_label
            self.dim = comp_dim
            self.index = [f"{comp_label}-{i}" for i in range(comp_dim)]
            super(AddIndexCompLabel, self).initialize()

    return AddIndexCompLabel