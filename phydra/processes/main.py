from phydra.core.backend import ModelBackend

import xsimlab as xs
import numpy as np

@xs.process
class ModelCore:
    """this object contains the backend GEKKO Solver and is modified or read by all other processes"""

    solver_type = xs.variable(intent='in')
    m = xs.any_object(description='model core instance is stored here')

    def initialize(self):
        print('initializing model core')
        self.m = ModelBackend(self.solver_type)

    def finalize(self):
        print('finalizing: cleanup')
        # self.m.open_folder()

        self.m.cleanup()  # for now only affects gekko solve

@xs.process
class ModelContext:
    """ Inherited by all other model processes to access GekkoCore"""
    m = xs.foreign(ModelCore, 'm')

    label = xs.variable(intent='out', groups='label')

    def initalize(self):
        print('calling model context')
        self.label = self.__xsimlab_name__  # assign given label to all subclasses

@xs.process
class FirstInit(ModelContext):
    """ Inherited by all other model processes to access GekkoCore"""
    group = xs.variable(intent='out', groups='FirstInit')

    def initialize(self):
        super(FirstInit, self).initalize()
        self.group = 1

@xs.process
class SecondInit(ModelContext):
    """ Inherited by all other model processes to access GekkoCore"""
    firstinit = xs.group('FirstInit')
    group = xs.variable(intent='out', groups='SecondInit')

    def initialize(self):
        super(SecondInit, self).initalize()
        self.group = 2


@xs.process
class ThirdInit(ModelContext):
    """ Inherited by all other model processes to access GekkoCore"""
    firstinit = xs.group('FirstInit')
    secondinit = xs.group('SecondInit')
    group = xs.variable(intent='out', groups='ThirdInit')

    def initialize(self):
        super(ThirdInit, self).initalize()
        self.group = 3


@xs.process
class Solver(ModelContext):

    firstinit = xs.group('FirstInit')
    secondinit = xs.group('SecondInit')
    thirdinit = xs.group('ThirdInit')

    def initialize(self):
        """TODO: assemble model + equations here"""
        print("assembling model")
        print("SOLVER :", self.m.Solver)
        self.m.assemble()

    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        # print("pre-solve:", self.m.Time, self.m)
        self.m.solve(dt)

