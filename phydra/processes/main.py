from phydra.core.parts import StateVariable, Forcing, Flux, Parameter
from phydra.core.converters import OdeintConverter, GekkoConverter
from phydra.core.backend import ModelBackend

import xsimlab as xs
import numpy as np

@xs.process
class ModelCore:
    """this object contains the backend GEKKO solver and is modified or read by all other processes"""

    solver_type = xs.variable(intent='in')
    m = xs.any_object(description='model core instance is stored here')

    y_init = xs.variable(intent='in')
    y = xs.variable(intent='out', dims='time')

    def initialize(self):
        print('initializing model core')
        self.m = ModelBackend(self.solver_type)

        self.m.time = np.arange(1, 20, 0.1) #TODO: This currently needs to happen before SV setup!

        self.y = self.m.setup_SV('y', StateVariable(name='y', initial_value=self.y_init))

        self.m.Parameters['k'] = Parameter(name='k', value=0.5)

        def linear_loss(state, parameters):
            # print("LINEAR", state, parameters)
            y = state['y']
            k = parameters['k']
            dydt = -k * y
            return dydt

        def linear_growth(state, parameters):
            # print("LINEAR", state, parameters)
            y = state['y']
            k = parameters['k']
            dydt = k * y
            return dydt

        self.m.Fluxes['y'].append(linear_loss)


    def finalize(self):
        print('finalizing: cleanup')
        # self.m.open_folder()

        self.m.cleanup() #for now only affects gekko solve




@xs.process
class ModelContext:
    """ Inherited by all other model processes to access GekkoCore"""
    m = xs.foreign(ModelCore, 'm')


@xs.process
class Solver(ModelContext):

    def initialize(self):
        """TODO: assemble model + equations here"""
        print("assembling model")
        self.m.assemble()

    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        self.m.solve(dt)


@xs.process
class Time(ModelContext):
    days = xs.variable(dims='time', description='time in days')
    # for indexing xarray IO objects
    time = xs.index(dims='time', description='time in days')

    def initialize(self):
        print('Initializing Model Time')
        self.time = self.days

        # ASSIGN MODEL SOLVING TIME HERE:
        self.m.time = self.time

        # add state variable keeping track of time within model (for time-dependent functions):
        self.m.phydra_SVs['time'] = self.m.Var(0, lb=0, name='time')
        self.m.Equation(self.m.phydra_SVs['time'].dt() == 1)