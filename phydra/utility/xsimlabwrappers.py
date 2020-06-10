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


def createSingleComp(base_process, comp_label):
    """This function allows creating specific instance of component during model setup
       a new subclass with the appropriate labels and dimensions is created by a dynamically
       created xs.process AddIndexCompLabel inheritng from the base_process
       """
    @xs.process
    class AddIndexCompLabel(base_process):
        label = xs.variable(intent='out')
        dim = xs.variable(intent='out')
        index = xs.index(dims=comp_label)

        output = xs.variable(intent='out', dims=(comp_label, 'time'))

        def initialize(self):
            self.label = comp_label
            self.dim = 1
            self.index = [f"{comp_label}"]
            super(AddIndexCompLabel, self).initialize()

    return AddIndexCompLabel


def createMultiComp(base_process, comp_label, comp_dim):
    """This function allows creating specific instance of component during model setup
       a new subclass with the appropriate labels and dimensions is created by a dynamically
       created xs.process AddIndexCompLabel inheritng from the base_process
       """
    @xs.process
    class AddIndexCompDimsLabel(base_process):
        label = xs.variable(intent='out')
        dim = xs.variable(intent='out')
        index = xs.index(dims=comp_label)

        output = xs.variable(intent='out', dims=(comp_label, 'time'))

        def initialize(self):
            self.label = comp_label
            self.dim = comp_dim
            self.index = [f"{comp_label}-{i}" for i in range(comp_dim)]
            super(AddIndexCompDimsLabel, self).initialize()

    return AddIndexCompDimsLabel


def specifyComps4Flux(base_process, comp1_label, comp2_label):
    """This function allows creating specific instance of flux during model setup
       a new subclass with the appropriate labels and dimensions is created by a dynamically
       created xs.process AddCompLabels inheritng from the base_process

       ToDo:
       - This will need to also pass on specific component dimensions later on,
       in more complex implementations
       """
    @xs.process
    class AddCompLabels(base_process):
        c1_label = xs.variable(intent='out')
        c2_label = xs.variable(intent='out')

        def initialize(self):
            self.c1_label = comp1_label
            self.c2_label = comp2_label
            super(AddCompLabels, self).initialize()

    return AddCompLabels