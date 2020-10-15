from abc import ABC, abstractmethod
from gekko import GEKKO
import numpy as np


class SolverABC(ABC):
    """Backend solver class (or should I use a function instead?)
    ## IMPORTANT: MAKE SURE THIS IS ADAPTABLE! allows using different solvers by user..

    - takes Model class as input ?
    hm, but how to pass data back and forth...

    TODO:
        - add all necessary abstract methods and add quick & dirty docs
    """

    #def __init__(self, time):
    #    self.time = time

    @abstractmethod
    def add_variable(self, label, time):
        pass

    @abstractmethod
    def assemble(self):
        pass

    @abstractmethod
    def solve(self, time_step):
        pass

    @abstractmethod
    def cleanup(self):
        pass


class ODEINTSolver(SolverABC):
    """ SolverABC can handle odeint solving of Model """

    def add_variable(self, label, time):
        """"""
        value = np.zeros(np.shape(time))
        return value

    def assemble(self):
        print("Hiho there, this is assembling using", self.__class__)
        pass

    def solve(self, time_step):
        print("Hullo there, this is solving using", self.__class__)
        pass

    def cleanup(self):
        print("Well hullo again, this is cleaning up using", self.__class__)
        pass



# TODO: FIX THESE BELOW; ONLY FOCUS ON ABOVE RIGHT NOW!

class StepwiseSolver(SolverABC):
    """ SolverABC can handle odeint solving of Model """

    def add_variable(self, label, time):
        """"""
        variable.value = np.zeros(np.shape(time))
        return variable

    def solve(self):
        print("Hullo there, this is solving using", self.__class__)
        pass

    def cleanup(self):
        print("Well hullo again, this is cleaning up using", self.__class__)
        pass



class GEKKOSolver(SolverABC):
    """ SolverABC can handle odeint solving of Model """

    def __init__(self):
        self.gekko = GEKKO(remote=False)

    def solve(self):
        print("Hullo there, this is solving using", self.__class__)
        pass

    def cleanup(self):
        print("Well hullo again, this is cleaning up using", self.__class__)
        pass