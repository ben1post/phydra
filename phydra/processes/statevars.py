import numpy as np
import xsimlab as xs

from .main import GekkoContext


@xs.process
class StateVariable(GekkoContext):
    """ this process creates a single state variable with user specified label in our model """
    label = xs.variable(intent='out')

    value = xs.variable(intent='out', dims='time', description='stores the value of component state variable')
    SV = xs.any_object(description='stores the gekko variable')

    initVal = xs.variable(intent='in', description='initial value of component')

    def initialize(self):
        self.label = self.__xsimlab_name__
        print(f"state variable {self.label} is initialized")

        # store GEKKO SV object in Gekko context self.m
        self.m.phydra_SVs[self.label] = self.m.SV(self.initVal, name=self.label, lb=0)
        self.value = self.m.phydra_SVs[self.label].value

    def run_step(self):
        print(f"assembling equations for state variable {self.label}")
        print(self.m.phydra_fluxes)
        self.m.Equation(
            self.m.phydra_SVs[self.label].dt() == sum([flux for flux in self.m.phydra_fluxes[self.label]])
        )


class FunctionalGroup(GekkoContext):
    """ creates array of state variables """

    label = xs.variable(intent='out', description='the label supplied at model initialisation')
    value = xs.variable(intent='out', dims=('not_initalized', 'time'),
                        description='stores output in dimensions supplied to process_setup')

    num = xs.variable(intent='in', description='number of state variables within group')
    initVal = xs.variable(intent='inout', description='initial value of component')

    SV = xs.any_object(description='stores the gekko variable')

    # Simulation stages
    def initialize(self):
        self.label = self.__xsimlab_name__
        print(f"functional group {self.label} is initialized with size {self.num}")

        self.create_index()

        self.m.phydra_SVs[self.label] = np.array(
            [self.m.SV(name=f"{self.label}_{i}", value=self.initVal, lb=0) for i in range(self.num)])
        self.value = [sv.value for sv in self.m.phydra_SVs[self.label]]

    def run_step(self):
        print(f"assembling Equations for {self.label}")
        print(self.m.phydra_fluxes)
        gk_array = self.m.phydra_SVs[self.label]
        fluxes = self.m.phydra_fluxes[self.label]
        for i in range(self.num):
            self.m.Equation(gk_array[i].dt() == sum([flux[i] for flux in fluxes]))

    # Helper functions
    def create_index(self):
        # this creates numbered index with label:
        if self.num == 1:
            index_list = [f"{self.label}"]
        else:
            index_list = [f"{self.label}-{i}" for i in range(self.num)]
        setattr(self, self.label, index_list)

    @classmethod
    def setup(cls, dim_label):
        """ create copy of process class with user specified name and dimension label """
        new_cls = type(cls.__name__ + dim_label, cls.__bases__, dict(cls.__dict__))
        # add new index with variable name of label (to avoid Zarr storage conflicts)
        new_dim = xs.index(dims=dim_label, groups='comp_index')
        setattr(new_cls, dim_label, new_dim)
        # modify dimensions
        new_cls.value.metadata['dims'] = ((dim_label, 'time'),)
        # return intialized xsimlab process
        return xs.process(new_cls)