import numpy as np
import xsimlab as xs

def create(*args):
    return xs.Model(*args)


def setup(solver, model, input_vars, output_vars, time=None):
    """ This function wraps create_setup and adds a dummy clock parameter
    necessary for model execution """

    input_vars.update({'core__solver_type': solver})

    if solver == "odeint" or solver == "gekko":
        input_vars.update({'time__days': time})
        return xs.create_setup(model=model,
                            # supply a single time step to xsimlab model setup
                            clocks={'clock': [0, 1]},
                            input_vars=input_vars,
                            output_vars=output_vars)
    elif solver == "stepwise":
        if time is None:
            raise Exception("Please supply (numpy) array of explicit timesteps to time keyword argument")
        input_vars.update({'time__days': [0]})
        return xs.create_setup(model=model,
                            clocks={'time': time},
                            input_vars=input_vars,
                            output_vars=output_vars)
    else:
        raise Exception("Please supply one of the available solvers: 'odeint', 'gekko' or 'stepwise'")
