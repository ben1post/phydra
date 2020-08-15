import numpy as np
import xsimlab as xs

from ..processes.main import ModelCore, Solver
from ..processes.statevars import Time


def create(model_dict):
    model_dict.update({'Core': ModelCore, 'Solver': Solver, 'Time': Time})
    return xs.Model(model_dict)


def setup(solver, model, input_vars, output_vars, time=None):
    """ This function wraps create_setup and adds a dummy clock parameter
    necessary for model execution """

    input_vars.update({'Core__solver_type': solver})

    if solver == "odeint" or solver == "gekko":
        input_vars.update({'Time__time': time})
        return xs.create_setup(model=model,
                               # supply a single Time step to xsimlab model setup
                               clocks={'clock': [0, 1]},
                               input_vars=input_vars,
                               output_vars=output_vars)
    elif solver == "stepwise":
        if time is None:
            raise Exception("Please supply (numpy) array of explicit timesteps to Time keyword argument")
        input_vars.update({'Time__time': [0]})
        return xs.create_setup(model=model,
                               clocks={'Time': time},
                               input_vars=input_vars,
                               output_vars=output_vars)
    else:
        raise Exception("Please supply one of the available solvers: 'odeint', 'gekko' or 'stepwise'")
