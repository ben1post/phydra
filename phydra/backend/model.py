from collections import defaultdict


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
        self.fluxes_per_var = defaultdict(list)
        self.multi_fluxes = defaultdict()

    def __repr__(self):
        return (f"Model contains: \n"
                f"Variables:{[var for var in self.variables]} \n"
                f"Parameters:{[par for par in self.parameters]} \n"
                f"Forcings:{[forc for forc in self.forcings]} \n"
                f"Fluxes:{[flx for flx in self.fluxes]} \n"
                f"Multi-Fluxes:{[multflx for multflx in self.multi_fluxes]}")

    def model_function(self, current_state, time=0):
        """ general model function that matches fluxes to state variables

        :param current_state:
        :param time: argument is necessary for odeint solve
        :return:

        # TODO:
        #   simplify multi_flux calc, by including subfunctions (i.e. self.multi_fluxes() to call here)
        """
        state = {label: val for label, val in zip(self.variables.keys(), current_state)}

        # NOTE:
        # currently I compute the model centered on state vars,
        # the value I am looking for is compute with every call to the flux function below
        # all I need to do is match fluxes_per_var with the new fluxes defaultdict..
        # so only compute once, use/store twice!

        fluxes = []
        for flux in self.fluxes.keys():
            print(flux)

        fluxes_out = []
        for label in self.variables.keys():

            sv_fluxes = []
            for flux in self.fluxes_per_var[label]:
                sv_fluxes.append(flux(state=state, parameters=self.parameters, forcings=self.forcings))

            fluxes_out.append(sum(sv_fluxes))
        print(fluxes_out)
        return fluxes_out
