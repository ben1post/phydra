from abc import ABC, abstractmethod
import numpy as np

from scipy.integrate import odeint
from gekko import GEKKO


class SolverABC(ABC):
    """ abstract base class of backend solver class,
    use subclass to solve model within the Phydra framework

    TODO:
        - add all necessary abstract methods and add quick & dirty docs
    """

    @abstractmethod
    def add_variable(self, label, initial_value, time):
        pass

    @abstractmethod
    def add_parameter(self, label, value):
        pass

    @abstractmethod
    def add_flux(self, label, flux, time):
        pass

    @abstractmethod
    def assemble(self, model):
        pass

    @abstractmethod
    def solve(self, model, time_step):
        pass

    @abstractmethod
    def cleanup(self):
        pass


class ODEINTSolver(SolverABC):
    """ Solver that can handle odeint solving of Model """

    def __init__(self):
        self.y_init = []

    def add_variable(self, label, initial_value, time):
        """ this returns storage container """

        if time is None:
            raise Exception("To use ODEINT solver, model time needs to be supplied before adding variables")

        # store initial values of variables to pass to odeint function
        self.y_init.append(initial_value)

        return np.zeros(np.shape(time))

    def add_parameter(self, label, value):
        """ """
        return value

    def add_flux(self, label, flux, time):
        """ this returns storage container """
        if time is None:
            raise Exception("To use ODEINT solver, model time needs to be supplied before adding variables")

        return np.zeros(np.shape(time))

    def assemble(self, model):
        """ """
        pass

    def solve(self, model, time_step):
        """ """
        print("start solve now")

        state_out = odeint(model.model_function, self.y_init, model.time)

        state_rows = (row for row in state_out.T)
        state_dict = {y_label: values for y_label, values in zip(model.variables.keys(), state_rows)}

        for var, val in zip(model.variables.values(), state_dict.values()):
            var[:] = val

    def cleanup(self):
        pass


class StepwiseSolver(SolverABC):
    """ Solver that can handle stepwise calculation built into xarray-simlab framework """

    def add_variable(self, label, initial_value, time):
        """ """
        # return list to be appended to
        return [initial_value]

    def add_parameter(self, label, value):
        """ """
        return value

    def add_flux(self, label, flux, time):
        return [np.nan]

    def assemble(self, model):
        pass

    def solve(self, model, time_step):
        sv_state = [var[-1] for var in model.variables.values()]
        state_out = model.model_function(sv_state)
        state_dict = {sv_label: value for sv_label, value in zip(model.variables.keys(), state_out)}

        for var, val in zip(model.variables.values(), state_dict.values()):
            state = var[-1] + val * time_step  # model returns derivative, this calculates value
            var.append(state)

    def cleanup(self):
        pass


class GEKKOSolver(SolverABC):
    """ Solver that can handle solving the model with GEKKO """

    def __init__(self):
        self.gekko = GEKKO(remote=False)

    def add_variable(self, label, initial_value, time):
        """"""
        # return list of values to be appended to
        return self.gekko.SV(value=initial_value, name=label, lb=0)

    def add_parameter(self, label, value):
        # this needs to be sub-delegated to solver (hence it is)
        # TODO:
        #  particular use case: for GEKKO! add gekko.param() to backend
        return self.gekko.Param(value=value, name=label)

    def assemble(self, model):
        #model.parameters

        state_out = model.model_function(model.variables.values())
        state_dict = {label: eq for label, eq in zip(model.variables.keys(), state_out)}

        equations = []

        for label, var in model.variables.items():
            print(var, state_dict[label])
            try:
                state_dict[label]
            except KeyError:
                # if not, define derivative as 0
                equations.append(var.dt() == 0)
            else:
                equations.append(var.dt() == state_dict[label])

        # create Equations
        self.gekko.Equations(equations)

        self.gekko.time = model.time

        print([val.value for val in self.gekko.__dict__['_equations']])

    def solve(self, model, time_step):
        self.gekko.options.REDUCE = 3  # handles reduction of larger models, have not benchmarked it yet
        self.gekko.options.NODES = 3  # improves solution accuracy
        self.gekko.options.IMODE = 5  # 7  # sequential dynamic Solver

        self.gekko.solve(disp=False)  # use option disp=True to print gekko output

    def cleanup(self):
        self.gekko.cleanup()
