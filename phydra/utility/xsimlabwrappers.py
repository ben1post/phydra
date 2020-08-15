import xsimlab as xs

from ..processes.main import ModelCore, Solver
from ..processes.statevars import Time


def create(model_dict):
    """Function creates xsimlab Model instance,
    automatically adding the necessary model core, solver and time processes"""
    model_dict.update({'Core': ModelCore, 'Solver': Solver, 'Time': Time})
    return xs.Model(model_dict)


def setup(solver, model, input_vars, output_vars, time=None):
    """ This function wraps create_setup and adds a dummy clock parameter
    necessary for model execution """
    if time is None:
        raise Exception("Please supply (numpy) array of explicit timesteps to Time keyword argument")

    input_vars.update({'Core__solver_type': solver,
                       'Time__time': time})

    if solver == "odeint" or solver == "gekko":
        return xs.create_setup(model=model,
                               # supply a single Time step to xsimlab model setup
                               clocks={'clock': [0, 1]},
                               input_vars=input_vars,
                               output_vars=output_vars)
    elif solver == "stepwise":
        return xs.create_setup(model=model,
                               clocks={'clock': time},
                               input_vars=input_vars,
                               output_vars=output_vars)
    else:
        raise Exception("Please supply one of the available solvers: 'odeint', 'gekko' or 'stepwise'")


def update_setup(model, old_setup, new_solver, new_time=None):
    """Change instantiated model setup to another solver type,
    with the possibility to update solver time"""

    if new_time is None:
        time = old_setup.Time__time.values
    else:
        time = new_time

    if new_solver == "odeint" or new_solver == "gekko":
        with model:
            setup1 = old_setup.xsimlab.update_vars(input_vars={'Core__solver_type': new_solver,
                                                           'Time__time': time})
            setup2 = setup1.xsimlab.update_clocks(clocks={'clock': [0, 1]},
                                                  master_clock='clock')

    elif new_solver == "stepwise":
        with model:
            setup1 = old_setup.xsimlab.update_vars(input_vars={'Core__solver_type': new_solver})#,
            setup2 = setup1.xsimlab.update_clocks(clocks={'clock': time},
                                                  master_clock='clock')

    else:
        raise Exception("Please supply one of the available solvers: 'odeint', 'gekko' or 'stepwise'")

    return setup2
