from collections import defaultdict
import numpy as np


class PhydraModel:
    """Backend model class
    - collects all things relevant to the model instance (i.e. variables, parameters, ...)
    - can be solved by passing it to the SolverABC class (that's where conversion (if necessary) happens)
    """

    def __init__(self):

        self.time = None

        self.variables = defaultdict()
        self.parameters = defaultdict()
        self.forcings = defaultdict()

        self.fluxes = defaultdict()
        self.flux_values = defaultdict()
        self.fluxes_per_var = defaultdict(list)

        self.full_model_state = defaultdict()

    def __repr__(self):
        return (f"Model contains: \n"
                f"Variables:{[var for var in self.variables]} \n"
                f"Parameters:{[par for par in self.parameters]} \n"
                f"Forcings:{[forc for forc in self.forcings]} \n"
                f"Fluxes:{[flx for flx in self.fluxes]} \n")

    def model_function(self, current_state, time=0):
        """ general model function that matches fluxes to state variables

        :param current_state:
        :param time: argument is necessary for odeint solve
        :return:
        """
        state = {label: val for label, val in zip(self.full_model_state.keys(), current_state)}

        # Compute fluxes:
        flux_values = defaultdict()
        fluxes_out = []
        for flx_label, flux in self.fluxes.items():
            _value = flux(state=state, parameters=self.parameters, forcings=self.forcings)
            flux_values[flx_label] = _value
            fluxes_out.append(_value)

        # Assign fluxes to variables:
        state_out = []
        for var_label in self.variables.keys():
            var_fluxes = []
            for flux_label in self.fluxes_per_var[var_label]:
                var_fluxes.append(flux_values[flux_label])
            state_out.append(sum(var_fluxes))

        full_output = np.concatenate([state_out, fluxes_out], axis=None)

        return full_output
