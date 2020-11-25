from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

from scipy.integrate import odeint
from gekko import GEKKO


def to_ndarray(value):
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
    def register_flux(self, label, flux, model, dims):
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

    def return_dims_and_array(self, value, model_time):
        """ """
        if np.size(value) == 1:
            _dims = None
            full_dims = (np.size(model_time),)
        elif len(np.shape(value)) == 1:
            _dims = np.size(value)
            full_dims = (_dims, np.size(model_time))
        else:
            _dims = np.shape(value)
            full_dims = (*_dims, np.size(model_time))

        print("FULL DIMS", full_dims)
        array_out = np.zeros(full_dims)
        return array_out, _dims

    def add_variable(self, label, initial_value, model):
        """ this returns storage container """

        if model.time is None:
            raise Exception("To use ODEINT solver, model time needs to be supplied before adding variables")

        # store initial values of variables to pass to odeint function
        self.var_init[label] = to_ndarray(initial_value)

        array_out, dims = self.return_dims_and_array(initial_value, model.time)

        model.full_model_dims[label] = dims

        print("variable", label, np.shape(array_out))
        return array_out

    def add_parameter(self, label, value):
        """ """
        return to_ndarray(value)

    def register_flux(self, label, flux, model, dims):
        """ this returns storage container """

        if model.time is None:
            raise Exception("To use ODEINT solver, model time needs to be supplied before adding fluxes")

        var_in_dict = defaultdict()
        for var, value in model.variables.items():
            var_in_dict[var] = self.var_init[var]
        for var, value in self.flux_init.items():
            var_in_dict[var] = value
        # print("VAR IN DICT", var_in_dict)

        forcing_init = defaultdict()
        for key, func in model.forcing_func.items():
            forcing_init[key] = func(0)

        _flux_value = to_ndarray(flux(state=var_in_dict,
                                      parameters=model.parameters,
                                      forcings=forcing_init))
        self.flux_init[label] = _flux_value

        array_out, dims = self.return_dims_and_array(_flux_value, model.time)

        model.full_model_dims[label] = dims

        print("flux", label, np.shape(array_out))
        return array_out

    def add_forcing(self, label, forcing_func, model):
        """ """
        return forcing_func(model.time)

    def assemble(self, model):
        """ """
        # print model repr for diagnostic purposes:
        print("Model is assembled:")
        print(model)


    def solve(self, model, time_step):
        """ """
        print("start solve now")
        print("var init", self.var_init)
        print("flux_init", self.flux_init)
        full_init = np.concatenate([[v for val in self.var_init.values() for v in val.flatten()],
                                    [v for val in self.flux_init.values() for v in val.flatten()]], axis=None)

        print("full init", np.size(full_init), np.shape(full_init), full_init)
        full_model_out = odeint(model.model_function, full_init, model.time)

        print("FULL OUT", type(full_model_out), full_model_out)
        state_rows = [row for row in full_model_out.T]

        print(np.shape(state_rows))
        print(state_rows)

        state_dict = defaultdict()
        index = 0
        for key, dims in model.full_model_dims.items():
            # print(key, dims, index)
            if dims is None:
                state_dict[key] = state_rows[index]
                index += 1
            elif isinstance(dims, int):
                val_list = []
                for i in range(dims):
                    val_list.append(state_rows[index])
                    index += 1
                state_dict[key] = np.array(val_list)
            else:
                print(dims, model.time)
                full_dims = (*dims, np.size(model.time))
                print(full_dims)
                _length = int(np.prod(dims))
                print(_length)
                val_list = []
                for i in range(_length):
                    val_list.append(state_rows[index])
                    index += 1
                    print(np.shape(val_list))
                try:
                    state_dict[key] = np.array(val_list).reshape(full_dims)
                except:
                    raise Exception(f"{_length}-{index}-{val_list}")

        # assign solved model state to value storage in xsimlab framework:
        for var_key, val in model.variables.items():
            val[...] = state_dict[var_key]

        for flux_key, val in model.flux_values.items():
            # print(flux_key, val, state, np.diff(state))
            state = state_dict[flux_key]
            dims = model.full_model_dims[flux_key]
            val[...] = np.diff(state, prepend=0) / time_step

            # rounding below to remove error from floating point arithmetics for nice plotting
            #if dims:
            #    for v, row in zip(val, state):
            #        v[:] = np.round(
            #            np.concatenate([row[0], np.diff(row) / time_step], axis=None), decimals=7)
            #else:
            #    val[:] = np.round(
            #        np.concatenate([state[0], np.diff(state) / time_step], axis=None), decimals=7)

    def cleanup(self):
        pass


class StepwiseSolver(SolverABC):
    """ Solver that can handle stepwise calculation built into xarray-simlab framework """

    def __init__(self):
        self.model_time = 0
        self.time_index = 0

        self.full_model_values = defaultdict()

    def return_dims_and_array(self, value, model_time):
        """ """

        if np.size(value) == 1:
            _dims = None
            full_dims = (np.size(model_time),)
            array_out = np.zeros(full_dims)
            array_out[0] = value
        elif len(np.shape(value)) == 1:
            _dims = np.size(value)
            full_dims = (_dims, np.size(model_time))
            array_out = np.zeros(full_dims)
            array_out[:, 0] = value
        else:
            _dims = np.shape(value)
            full_dims = (*_dims, np.size(model_time))
            array_out = np.zeros(full_dims)
            array_out[..., 0] = value
        #print(value, array_out)
        return array_out, _dims

    def add_variable(self, label, initial_value, model):
        """ """
        array_out, _dims = self.return_dims_and_array(initial_value, model.time)

        model.var_dims[label] = _dims

        #self.full_model_values[label] = array_out

        #print("variable", label, _dims, initial_value)

        return array_out

    def add_parameter(self, label, value):
        """ """
        return to_ndarray(value)

    def register_flux(self, label, flux, model, dims):

        var_in_dict = defaultdict()
        for var, value in model.variables.items():
            _dims = model.var_dims[var]
            if _dims is None:
                var_in_dict[var] = value[0]
            elif isinstance(_dims, int):
                var_in_dict[var] = value[:, 0]
            else:
                var_in_dict[var] = value[..., 0]

        for var, value in model.flux_values.items():
            _dims = model.flux_dims[var]
            if _dims is None:
                var_in_dict[var] = value[0]
            elif isinstance(_dims, int):
                var_in_dict[var] = value[:, 0]
            else:
                var_in_dict[var] = value[..., 0]

        forcing_now = defaultdict()
        for key, func in model.forcing_func.items():
            forcing_now[key] = func(0)

        flux_init = to_ndarray(flux(state=var_in_dict,
                                    parameters=model.parameters,
                                    forcings=forcing_now))

        array_out, _dims = self.return_dims_and_array(flux_init, model.time)

        model.flux_dims[label] = _dims

        #self.full_model_values[label] = array_out

        #print("flux", label, _dims, flux_init)

        return array_out

    def add_forcing(self, label, forcing_func, model):
        """ """
        return forcing_func(model.time)

    def assemble(self, model):
        # assemble dimensions now to separate and order fluxes and values correctly for unpacking (computed separately)

        for var_key, value in model.variables.items():
            _dims = model.var_dims[var_key]
            model.full_model_dims[var_key] = _dims
            self.full_model_values[var_key] = value

        for var_key, value in model.flux_values.items():
            _dims = model.flux_dims[var_key]
            model.full_model_dims[var_key] = _dims
            self.full_model_values[var_key] = value

        # finally print model repr for diagnostic purposes:
        print("Model is assembled:")
        print(model)

    def solve(self, model, time_step):
        self.model_time += time_step
        self.time_index += 1

        model_forcing = defaultdict()
        for key, func in model.forcing_func.items():
            # retrieve pre-computed forcing:
            model_forcing[key] = model.forcings[key][self.time_index]

        model_state = []
        for key, val in self.full_model_values.items():
            # retrieve state computed in previous time step:
            # print(key, val)
            if model.full_model_dims[key]:
                model_state.append(val[..., self.time_index - 1])
            else:
                model_state.append(val[self.time_index - 1])

        # flatten list for model function:
        # print("model state pre flatten", model_state)
        flat_model_state = np.concatenate(model_state, axis=None)
        #print("flattened state", flat_model_state)

        state_out = model.model_function(flat_model_state, forcing=model_forcing)
        #print("state out", state_out)
        state_dict = model.unpack_flat_state(state_out)
        #print("state dict", state_dict)
        # add time index, to assign new values to next slot in numpy arrays:

        for key, val in model.variables.items():
            if model.full_model_dims[key]:
                #print(key, val[..., self.time_index], val[..., self.time_index - 1], state_dict[key])
                val[..., self.time_index] = val[..., self.time_index - 1] + state_dict[key] * time_step
            else:
                val[self.time_index] = val[self.time_index - 1] + state_dict[key] * time_step

            #print("Post assign", key, val[..., self.time_index])

        for key, val in model.flux_values.items():
            if model.full_model_dims[key]:
                val[..., self.time_index] = state_dict[key]
            else:
                val[self.time_index] = state_dict[key]

    def cleanup(self):
        pass


class GEKKOSolver(SolverABC):
    """ Solver that can handle solving the model with GEKKO """

    def __init__(self):
        self.gekko = GEKKO(remote=False)

        self.full_model_values = defaultdict()

        self.reserved_labels = ['abs', 'exp', 'log10', 'log',
                                'sqrt', 'sinh', 'cosh', 'tanh',
                                'sin', 'cos', 'tan', 'asin',
                                'acos', 'atan', 'erf', 'erfc']

    def check_label(self, label):
        """ check if string label coincides with reserved gekko mathematical function and change accordingly """
        if label.lower()[:3] in self.reserved_labels:
            return 'x' + label
        else:
            return label

    def add_variable(self, label, initial_value, model):
        """ """
        label = self.check_label(label)

        if isinstance(initial_value, list) or isinstance(initial_value, np.ndarray):
            var_out = [self.gekko.SV(value=initial_value[i], name=label + str(i), lb=0)
                       for i in range(len(initial_value))]
        else:
            var_out = self.gekko.SV(value=initial_value, name=label, lb=0)

        return var_out

    def add_parameter(self, label, value):
        """ """
        label = self.check_label(label)

        # print("adding parameter", label, value)
        if isinstance(value, str):
            return value

        if isinstance(value, list) or isinstance(value, np.ndarray):
            var_out = [self.gekko.Param(value=value[i], name=label + str(i)) for i in range(len(value))]
        else:
            var_out = self.gekko.Param(value=value, name=label)

        return var_out

    def register_flux(self, label, flux, model, dims):
        """ this returns storage container """
        label = self.check_label(label)

        var_in_dict = {**model.variables, **model.flux_values}
        # need to force vectorization here, otherwise lists/arrays of gekko object are not iterated over:
        # print("PARAMETERS", model.parameters)
        # print("CALCULATING FLUX", dims)
        _flux = flux(state=var_in_dict,
                     parameters=model.parameters,
                     forcings=model.forcings, vectorized=True, dims=dims)

        # print(label, _flux, type(_flux), dims)

        if np.size(_flux) > 1:
            # print([_flux[i] for i in range(len(_flux))])
            flux_out = [self.gekko.Intermediate(_flux[i], name=label + str(i)) for i in range(len(_flux))]
        else:
            flux_out = self.gekko.Intermediate(_flux, name=label)

        # print("Flux_OUT", flux_out)

        return flux_out

    def add_forcing(self, label, forcing_func, model):
        """ """
        label = self.check_label(label)

        return self.gekko.Param(value=forcing_func(model.time), name=label)

    def assemble(self, model):
        """ """
        for key, value in model.variables.items():
            # print("variables", key, np.shape(value))
            self.full_model_values[key] = value
            if isinstance(value, list) or isinstance(value, np.ndarray):
                model.full_model_dims[key] = np.size(value)
            else:
                model.full_model_dims[key] = None

        for key, value in model.flux_values.items():
            # print("fluxes", key, np.shape(value))
            self.full_model_values[key] = value
            if isinstance(value, list) or isinstance(value, np.ndarray):
                if np.size(value) == np.size(model.time):
                    model.full_model_dims[key] = None
                else:
                    model.full_model_dims[key] = np.size(value)
            else:
                model.full_model_dims[key] = None

        # print("FULL MODEL VALS", self.full_model_values)

        # finally print model repr for diagnostic purposes:
        print("Model dicts are assembled:")
        print(model)

        print("Now assembling gekko model:")
        # Assign fluxes to variables:
        equations = []

        # Route list input fluxes:
        list_input_fluxes = defaultdict(list)
        for flux_var_dict in model.fluxes_per_var["list_input"]:
            flux_label, negative, list_input = flux_var_dict.values()
            # print(flux_label, negative, model.flux_values[flux_label], list_input)

            flux_val = model.flux_values[flux_label]
            flux_dims = model.full_model_dims[flux_label]

            list_var_dims = []
            for var in list_input:
                _dim = model.full_model_dims[var]
                list_var_dims.append(_dim or 1)
            # print(len(list_input), flux_dims)

            if len(list_input) == flux_dims:
                for var, flux in zip(list_input, flux_val):
                    # var_dims = self.full_model_dims[var]
                    # print(var, var_dims, flux, flux_dims)
                    if negative:
                        list_input_fluxes[var].append(-flux)
                    else:
                        list_input_fluxes[var].append(flux)
            elif sum(list_var_dims) == flux_dims:
                _dim_counter = 0
                for var, dims in zip(list_input, list_var_dims):
                    flux = np.array(flux_val[_dim_counter:_dim_counter + dims])
                    _dim_counter += dims
                    # print(var, dims, "flux", flux, _dim_counter)
                    if negative:
                        list_input_fluxes[var].append(-flux)
                    else:
                        list_input_fluxes[var].append(flux)
            else:
                raise Exception(f"ERROR: list input vars dims {list_var_dims} and "
                                f"flux output dims {flux_dims} do not match")

        for var_label, value in model.variables.items():
            flux_applied = False
            var_fluxes = []
            dims = model.full_model_dims[var_label]
            if var_label in model.fluxes_per_var:
                flux_applied = True
                for flux_var_dict in model.fluxes_per_var[var_label]:
                    flux_label, negative, list_input = flux_var_dict.values()
                    _flux = model.flux_values[flux_label]
                    # print(var_label, dims, flux_label, _flux, type(_flux))
                    flux_dims = np.size(_flux)

                    if negative:
                        if flux_dims > 1 or isinstance(_flux, list) or isinstance(_flux, np.ndarray):
                            var_fluxes.append([-_flux[i] for i in range(flux_dims)])
                        else:
                            var_fluxes.append(-_flux)
                    else:
                        if flux_dims > 1 or isinstance(_flux, list) or isinstance(_flux, np.ndarray):
                            var_fluxes.append([_flux[i] for i in range(flux_dims)])
                        else:
                            var_fluxes.append(_flux)

            if var_label in list_input_fluxes:
                flux_applied = True
                # print(list_input_fluxes[var_label])
                for flux in list_input_fluxes[var_label]:
                    if dims:
                        _flux = flux
                    else:
                        _flux = np.sum(flux)
                    # print(_flux)
                    var_fluxes.append(_flux)
                # print(_flux)
                # var_fluxes.append(_flux)

            if not flux_applied:
                # print(var_label, "appending 0")
                if dims:
                    var_fluxes.append([0 for i in range(dims)])
                else:
                    var_fluxes.append(0)

            # print("VAR FLUXES", var_fluxes)
            if dims:
                for i in range(dims):
                    equations.append(value[i].dt() == sum([var_flx[i] for var_flx in var_fluxes]))
            else:
                _var_fluxes = []
                for flx in var_fluxes:
                    if np.size(flx) > 1:
                        _var_fluxes.append(sum(flx))
                    else:
                        _var_fluxes.append(flx)
                equations.append(value.dt() == sum(_var_fluxes))

        # create Equations
        self.gekko.Equations(equations)

        self.gekko.time = model.time

        print("Model equations:")
        for val in self.gekko.__dict__['_equations']:
            print(val.value)

    def solve(self, model, time_step):
        self.gekko.options.REDUCE = 3  # handles reduction of larger models, have not benchmarked it yet
        self.gekko.options.NODES = 3  # improves solution accuracy
        self.gekko.options.IMODE = 7  # sequential dynamic Solver

        self.gekko.solve(disp=False)  # use option disp=True to print gekko output
        # print(self.gekko.__dict__)

    def cleanup(self):
        self.gekko.cleanup()
