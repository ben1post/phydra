from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

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
    def add_forcing(self, label, flux, time):
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
        self.var_init = defaultdict()
        self.flux_init = defaultdict()

    def add_variable(self, label, initial_value, time):
        """ this returns storage container """

        if time is None:
            raise Exception("To use ODEINT solver, model time needs to be supplied before adding variables")

        # store initial values of variables to pass to odeint function
        self.var_init[label] = initial_value

        return np.zeros(np.shape(time))

    def add_parameter(self, label, value):
        """ """
        return value

    def add_flux(self, label, flux, model):
        """ this returns storage container """

        if model.time is None:
            raise Exception("To use ODEINT solver, model time needs to be supplied before adding fluxes")

        var_in_dict = defaultdict()
        for var, value in model.variables.items():
            var_in_dict[var] = self.var_init[var]

        forcing_now = defaultdict()
        for key, func in model.forcing_func.items():
            forcing_now[key] = func(0)

        self.flux_init[label] = flux(state=var_in_dict, parameters=model.parameters, forcings=forcing_now)

        return np.zeros(np.shape(model.time))

    def add_forcing(self, label, forcing_func, model):
        """ """
        return forcing_func(model.time)

    def assemble(self, model):
        """ """
        print(model)

    def solve(self, model, time_step):
        """ """
        print("start solve now")
        full_init = np.concatenate([[val for val in self.var_init.values()],
                                    [val for val in self.flux_init.values()]], axis=None)

        full_model_out = odeint(model.model_function, full_init, model.time)

        state_rows = (row for row in full_model_out.T)
        state_dict = {y_label: values for y_label, values in zip(model.full_model_state.keys(), state_rows)}

        for var, val in model.variables.items():
            state = state_dict[var]
            val[:] = state

        for var, val in model.flux_values.items():
            state = state_dict[var]
            val[:] = np.concatenate([state[0], np.diff(state) / time_step], axis=None)

    def cleanup(self):
        pass


class StepwiseSolver(SolverABC):
    """ Solver that can handle stepwise calculation built into xarray-simlab framework """

    def __init__(self):
        self.model_time = 0

    def add_variable(self, label, initial_value, time):
        """ """
        # return list to be appended to
        return [initial_value]

    def add_parameter(self, label, value):
        """ """
        return value

    def add_flux(self, label, flux, model):
        var_in_dict = defaultdict()
        for var, value in model.variables.items():
            var_in_dict[var] = value[-1]
        forc_in_dict = defaultdict()
        for forc, value in model.forcings.items():
            forc_in_dict[forc] = value[-1]
        return [flux(state=var_in_dict, parameters=model.parameters, forcings=forc_in_dict)]

    def add_forcing(self, label, forcing_func, model):
        """ """
        return [forcing_func(0)]

    def assemble(self, model):
        print(model)
        pass

    def solve(self, model, time_step):
        self.model_time += time_step

        for key, func in model.forcing_func.items():
            model.forcings[key].append(func(self.model_time))

        model_forcing = defaultdict()
        for key, val in model.forcings.items():
            model_forcing[key] = val[-1]

        model_state = [var[-1] for var in model.full_model_state.values()]
        state_out = model.model_function(model_state, forcing=model_forcing)
        state_dict = {label: value for label, value in zip(model.full_model_state.keys(), state_out)}

        for var, val in model.variables.items():
            state = val[-1] + state_dict[var] * time_step  # model returns derivative, this calculates value
            val.append(state)

        for var, val in model.flux_values.items():
            state = state_dict[var]
            val.append(state)

    def cleanup(self):
        pass


class GEKKOSolver(SolverABC):
    """ Solver that can handle solving the model with GEKKO """

    def __init__(self):
        self.gekko = GEKKO(remote=False)

    def add_variable(self, label, initial_value, model):
        """"""
        # return list of values to be appended to
        return self.gekko.SV(value=initial_value, name=label, lb=0)

    def add_parameter(self, label, value):
        return self.gekko.Param(value=value, name=label)

    def add_flux(self, label, flux, model):
        """ this returns storage container """
        return self.gekko.Intermediate(flux(state=model.variables,
                                            parameters=model.parameters,
                                            forcings=model.forcings), name=label)

    def add_forcing(self, label, forcing_func, model):
        """ """
        return self.gekko.Param(value=forcing_func(model.time), name=label)

    def assemble(self, model):
        """ """
        state_out = model.model_function(model.full_model_state.values())

        state_dict = {label: eq for label, eq in zip(model.full_model_state.keys(), state_out)}

        equations = []

        for label, var in model.variables.items():
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
        self.gekko.options.IMODE = 7  # 7  # sequential dynamic Solver

        self.gekko.solve(disp=False)  # use option disp=True to print gekko output

    def cleanup(self):
        self.gekko.cleanup()
