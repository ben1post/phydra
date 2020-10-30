import xsimlab as xs

from phydra.backend.core import PhydraCore


@xs.process
class Backend:
    """this object contains the backend model and is modified or read by all other components"""

    solver_type = xs.variable(intent='in')
    m = xs.any_object(description='model backend instance is stored here')

    def initialize(self):
        print('initializing model backend')
        self.m = PhydraCore(self.solver_type)

    def finalize(self):
        print('finalizing: cleanup')
        self.m.cleanup()  # for now only affects gekko solve


@xs.process
class Context:
    """ Inherited by all other model components to access backend"""
    m = xs.foreign(Backend, 'm')

    label = xs.variable(intent='out', groups='label')

    def initialize(self):
        self.label = self.__xsimlab_name__  # assign given label to all subclasses


@xs.process
class FirstInit(Context):
    """ Inherited by all other model components to access backend"""
    group = xs.variable(intent='out', groups='FirstInit')

    def initialize(self):
        super(FirstInit, self).initialize()
        self.group = 1


@xs.process
class SecondInit(Context):
    """ Inherited by all other model components to access backend"""
    firstinit = xs.group('FirstInit')
    group = xs.variable(intent='out', groups='SecondInit')

    def initialize(self):
        super(SecondInit, self).initialize()
        self.group = 2


@xs.process
class ThirdInit(Context):
    """ Inherited by all other model components to access backend"""
    firstinit = xs.group('FirstInit')
    secondinit = xs.group('SecondInit')
    group = xs.variable(intent='out', groups='ThirdInit')

    def initialize(self):
        super(ThirdInit, self).initialize()
        self.group = 3


@xs.process
class Solver(Context):

    firstinit = xs.group('FirstInit')
    secondinit = xs.group('SecondInit')
    thirdinit = xs.group('ThirdInit')

    def initialize(self):
        """"""
        print("assembling model")
        print("SOLVER :", self.m.Solver)
        self.m.assemble()

    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        self.m.solve(dt)

