from gekko import GEKKO
import xsimlab as xs
import numpy as np

# to create dynamic storage of fluxes, state variables and forcings
from collections import defaultdict

# to measure process time
import time as tm


@xs.process
class GekkoCore:
    """this object contains the backend GEKKO solver and is modified or read by all other processes"""

    m = xs.any_object(description='GEKKO model instance is stored here')

    def initialize(self):
        print('initializing model core')
        self.m = GEKKO(remote=False, name='phydra')

        # add defaultdict of list that dynamically stores fluxes by component label
        self.m.phydra_fluxes = defaultdict(list)
        # same for state variables
        self.m.phydra_SVs = defaultdict()
        # same for forcings
        self.m.phydra_forcings = defaultdict()
        # same for growth limiting terms (product of which is added to equation of state variable
        self.m.phydra_growthfluxes = defaultdict(list)

    def finalize(self):
        print('finalizing gekko core: cleanup')
        # self.m.open_folder()
        self.m.cleanup()


@xs.process
class GekkoContext:
    """ Inherited by all other model processes to access GekkoCore"""
    m = xs.foreign(GekkoCore, 'm')


@xs.process
class Solver(GekkoContext):
    solver_type = xs.variable(intent='out')


@xs.process
class GekkoSequentialSolve(Solver):
    """ time is supplied from the Time process """

    def initialize(self):
        self.solver_type = 'seq'

    def finalize_step(self):
        # print(self.m.__dict__)

        print([i.name for i in self.m._variables])
        print('Running solver now')

        # add solver options
        self.m.options.REDUCE = 3  # handles reduction of larger models, have not benchmarked it yet
        self.m.options.NODES = 3  # improves solution accuracy
        self.m.options.IMODE = 7  # sequential dynamic solver

        solve_start = tm.time()
        self.m.solve(disp=False)  # use option disp=True to print gekko output
        solve_end = tm.time()

        print(f"Model was solved in {round(solve_end - solve_start, 2)} seconds")
        # print(self.m.__dict__)


@xs.process
class Time(GekkoContext):
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