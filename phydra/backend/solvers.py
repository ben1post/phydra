from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

from scipy.integrate import odeint
from gekko import GEKKO


def return_dim_ndarray(value):
    """ helper function to always have at least 1d numpy array returned """
    if isinstance(value, list):
        return np.array(value)
    elif isinstance(value, np.ndarray):
        return value
    else:
        return np.array([value])


class SolverABC(ABC):
    """ abstract base class of backend solver class,
    use subclass to solve model within the Phydra framework

    TODO:
        - add all necessary abstract methods and add quick & dirty docs
    """

    @abstractmethod
    def add_variable(self, label, initial_value, model):
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

    def add_variable(self, label, initial_value, model):
        """ this returns storage container """

        if model.time is None:
            raise Exception("To use ODEINT solver, model time needs to be supplied before adding variables")

        # TODO:
        #      Problem is: if I keep extra dim here to loop over later, values are stored with extra dim in model!
        #   - try using np.nditer, or other solution that can loop over single values as well!
        #   - fix dis!

        # store initial values of variables to pass to odeint function
        self.var_init[label] = return_dim_ndarray(initial_value)

        print("adding variable here:")
        if np.size(initial_value) == 1:
            full_dims = (np.size(model.time),)
        else:
            full_dims = (np.size(initial_value), np.size(model.time))

        print("FULL DIMS", full_dims)

        return np.zeros(full_dims)

    def add_parameter(self, label, value):
        """ """
        return return_dim_ndarray(value)

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

        self.flux_init[label] = return_dim_ndarray(flux(state=var_in_dict,
                                                        parameters=model.parameters,
                                                        forcings=forcing_now))

        if np.size(self.flux_init[label]) == 1:
            full_dims = (np.size(model.time),)
        else:
            full_dims = (np.size(self.flux_init[label]), np.size(model.time))

        print("flux", label, full_dims)

        return np.zeros(full_dims)

    def add_forcing(self, label, forcing_func, model):
        """ """
        return forcing_func(model.time)

    def assemble(self, model):
        """ """
        # TODO: here I need to create full_model_dims so that minimal processing needs to happen within model
        #   - this includes: having a flat to nested mapping that can easily unpack full_model_init flat list

        for key, value in model.variables.items():
            print("variables", key, np.shape(value))
            dims = set(np.shape(value)) - set(np.shape(model.time))
            add_dims = dims.pop() if dims else None
            model.full_model_dims[key] = add_dims

        for key, value in model.flux_values.items():
            print("values", key, np.shape(value))
            dims = set(np.shape(value)) - set(np.shape(model.time))
            add_dims = dims.pop() if dims else None
            model.full_model_dims[key] = add_dims

        # finally print model repr for diagnostic purposes:
        print("Model is assembled:")
        print(model)

    def solve(self, model, time_step):
        """ """
        print("start solve now")
        full_init = np.concatenate([[v for val in self.var_init.values() for v in val.flatten()],
                                    [v for val in self.flux_init.values() for v in val.flatten()]], axis=None)

        full_model_out = odeint(model.model_function, full_init, model.time)

        state_rows = [row for row in full_model_out.T]

        state_dict = model.unpack_flat_state(state_rows)

        # assign solved model state to value storage in xsimlab framework:
        for key, val in model.variables.items():
            val[:] = state_dict[key]

        for var, val in model.flux_values.items():
            state = state_dict[var]
            # print(var, val, state)
            dims = model.full_model_dims[var]
            # rounding below to remove error from floating point arithmetics for nice plotting
            if dims:
                for v, row in zip(val, state):
                    v[:] = np.round(
                        np.concatenate([row[0], np.diff(row) / time_step], axis=None), decimals=7)
            else:
                val[:] = np.round(
                    np.concatenate([state[0], np.diff(state) / time_step], axis=None), decimals=7)

    def cleanup(self):
        pass


class StepwiseSolver(SolverABC):
    """ Solver that can handle stepwise calculation built into xarray-simlab framework """

    def __init__(self):
        self.model_time = 0
        self.time_index = 0

        self.full_model_values = defaultdict(list)

    def add_variable(self, label, initial_value, model):
        """ """
        # return list to be appended to
        print(label, initial_value)

        print("adding variable here:")
        if np.size(initial_value) == 1:
            full_dims = (np.size(model.time),)
            array_out = np.zeros(full_dims)
            array_out[0] = initial_value
        else:
            full_dims = (np.size(initial_value), np.size(model.time))
            array_out = np.zeros(full_dims)
            array_out[:, 0] = initial_value

        # print("FULL DIMS", full_dims)

        return array_out

    def add_parameter(self, label, value):
        """ """
        return return_dim_ndarray(value)

    def add_flux(self, label, flux, model):

        var_in_dict = defaultdict()
        for var, value in model.variables.items():
            var_in_dict[var] = value[0] if np.size(value[0]) < 2 else value[:, 0]

        forcing_now = defaultdict()
        for key, func in model.forcing_func.items():
            forcing_now[key] = func(0)

        flux_init = return_dim_ndarray(flux(state=var_in_dict,
                                            parameters=model.parameters,
                                            forcings=forcing_now))

        if np.size(flux_init) == 1:
            full_dims = (np.size(model.time),)
            array_out = np.zeros(full_dims)
            array_out[0] = flux_init
        else:
            full_dims = (np.size(flux_init), np.size(model.time))
            array_out = np.zeros(full_dims)
            array_out[:, 0] = flux_init

        # print("flux", label, full_dims)

        return array_out

    def add_forcing(self, label, forcing_func, model):
        """ """
        return forcing_func(model.time)

    def assemble(self, model):
        for key, value in model.variables.items():
            # print("variables", key, np.shape(value))
            self.full_model_values[key] = value
            dims = set(np.shape(value)) - set(np.shape(model.time))
            add_dims = dims.pop() if dims else None
            model.full_model_dims[key] = add_dims

        for key, value in model.flux_values.items():
            # print("values", key, np.shape(value))
            self.full_model_values[key] = value
            dims = set(np.shape(value)) - set(np.shape(model.time))
            add_dims = dims.pop() if dims else None
            model.full_model_dims[key] = add_dims

        # finally print model repr for diagnostic purposes:
        print("Model is assembled:")
        print(model)

    def solve(self, model, time_step):
        self.model_time += time_step

        model_forcing = defaultdict()
        for key, func in model.forcing_func.items():
            _forcing = func(self.model_time)
            model.forcings[key][self.time_index] = _forcing
            model_forcing[key] = _forcing

        model_state = []
        for key, val in self.full_model_values.items():
            # print(key, val)
            if model.full_model_dims[key]:
                model_state.append(val[:, self.time_index])
            else:
                model_state.append(val[self.time_index])

        model_state = np.concatenate(model_state, axis=None)

        state_out = model.model_function(model_state, forcing=model_forcing)
        state_dict = model.unpack_flat_state(state_out)
        # add time index, to assign new values to next slot in numpy arrays:
        self.time_index += 1

        for key, val in model.variables.items():
            if model.full_model_dims[key]:
                val[:, self.time_index] = val[:, self.time_index - 1] + state_dict[key] * time_step
            else:
                val[self.time_index] = val[self.time_index - 1] + state_dict[key] * time_step

        for key, val in model.flux_values.items():
            if model.full_model_dims[key]:
                val[:, self.time_index] = val[:, self.time_index - 1]
            else:
                val[self.time_index] = val[self.time_index - 1]

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
        state_out = model.model_function(model.full_model_dims.values())

        state_dict = {label: eq for label, eq in zip(model.full_model_dims.keys(), state_out)}

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
