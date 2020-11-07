from collections import defaultdict
import numpy as np


def return_dim_ndarray(value):
    """ helper function to always have at least 1d numpy array returned """
    if isinstance(value, list):
        return np.array(value)
    elif isinstance(value, np.ndarray):
        return value
    else:
        return np.array([value])


class PhydraModel:
    """Backend model class
    - collects all things relevant to the model instance (i.e. variables, parameters, ...)
    - can be solved by passing it to the SolverABC class (that's where conversion (if necessary) happens)
    """

    def __init__(self):

        self.time = None

        self.variables = defaultdict()
        self.parameters = defaultdict()

        self.forcing_func = defaultdict()
        self.forcings = defaultdict()

        self.fluxes = defaultdict()
        self.flux_values = defaultdict()
        self.fluxes_per_var = defaultdict(list)

        self.full_model_dims = defaultdict()

    def __repr__(self):
        return (f"Model contains: \n"
                f"Variables:{[var for var in self.variables]} \n"
                f"Parameters:{[par for par in self.parameters]} \n"
                f"Forcings:{[forc for forc in self.forcings]} \n"
                f"Fluxes:{[flx for flx in self.fluxes]} \n"
                f"Full Model Dimensions:{[(state,dim) for state,dim in self.full_model_dims.items()]} \n")

    def unpack_flat_state(self, flat_state):
        """ """
        #print("FLAT STATE:", flat_state)
        #print(self.full_model_dims)

        state_dict = defaultdict()
        index = 0
        for key, dims in self.full_model_dims.items():
            if dims:
                val_list = []
                for i in range(dims):
                    val_list.append(flat_state[index])
                    index += 1
                state_dict[key] = np.array(val_list)
            else:
                state_dict[key] = flat_state[index]
                index += 1
            #print(key, dims, state_dict, index)

        return state_dict

    def model_function(self, current_state, time=None, forcing=None):
        """ general model function that matches fluxes to state variables

        :param current_state:
        :param time: argument is necessary for odeint solve
        :param forcing:
        :return:
        """
        # print("CURRENT STATE")
        # print(current_state)

        state = self.unpack_flat_state(current_state)

        # Return forcings for time point:
        if time is not None:
            forcing_now = defaultdict()
            for key, func in self.forcing_func.items():
                forcing_now[key] = func(time)
            forcing = forcing_now
        elif forcing is None:
            forcing = self.forcings

        # Compute fluxes:
        flux_values = defaultdict()
        fluxes_out = []
        for flx_label, flux in self.fluxes.items():
            _value = return_dim_ndarray(flux(state=state, parameters=self.parameters, forcings=forcing))
            flux_values[flx_label] = _value
            fluxes_out.append(_value)

        # print("fluxes_out", fluxes_out)

        # Assign fluxes to variables:
        state_out = []
        for var_label, value in self.variables.items():
            var_fluxes = []
            if var_label in self.fluxes_per_var:
                dims = self.full_model_dims[var_label]
                for flux_var_dict in self.fluxes_per_var[var_label]:
                    flux_label, negative = flux_var_dict.values()
                    # print(flux_label, flux_values[flux_label])

                    # TODO: add checking dims here, or safer handling!

                    if dims:
                        _flux = flux_values[flux_label]
                    else:
                        _flux = np.sum(flux_values[flux_label])

                    if negative:
                        var_fluxes.append(-_flux)
                    else:
                        var_fluxes.append(_flux)
            else:
                # print("here appending 0")
                dims = self.full_model_dims[var_label]
                if dims:
                    var_fluxes.append(np.array([0 for i in range(dims)]))
                else:
                    var_fluxes.append(np.array([0]))

            # print("var_fluxes", var_fluxes)
            state_out.append(np.sum(var_fluxes, axis=0))

        # print("state_out", state_out)

        full_output = np.concatenate([[v for val in state_out for v in val.flatten()],
                                      [v for val in fluxes_out for v in val.flatten()]], axis=None)

        # print(full_output)

        return full_output
