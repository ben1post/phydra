import time as tm

from attr._make import _CountingAttr


from .model import PhydraModel
from .solvers import SolverABC, ODEINTSolver, GEKKOSolver, StepwiseSolver


_built_in_solvers = {'odeint': ODEINTSolver, 'gekko': GEKKOSolver, 'stepwise': StepwiseSolver}


class PhydraCore:
    """"""
    def __init__(self, solver):

        self.Model = PhydraModel()

        self.solve_start = None
        self.solve_end = None

        if isinstance(solver, str):
            self.Solver = _built_in_solvers[solver]()
        elif isinstance(solver, SolverABC):
            self.Solver = solver
        else:
            raise Exception("Solver argument passed to model is not built-in or subclass of SolverABC")

    def add_variable(self, label, initial_value=0):
        """
        function that takes the state variable setup as input
        and returns the storage values
        """
        # the following step registers the variable within the framework
        self.Model.variables[label] = self.Solver.add_variable(label, initial_value, self.Model.time)

        # return actual value store of variable to xsimlab framework
        return self.Model.variables[label]

    def add_flux(self, variable, flux_dict):
        print("ADDING FLUX NOW IN BACKEND")
        print(variable, flux_dict)

        if flux_dict['negative'] is True:
            self.Model.fluxes[variable].append(flux.negative_flux)
        elif flux_dict['negative'] is False:
            self.Model.fluxes[variable].append(flux.comp)

        for var_name, var in flux.flux_dict.items():
            print(var_name, var)

            if isinstance(var, _CountingAttr):
                print('HELAU')
                var_value = getattr(flux, var_name)
                var_type = var.metadata.get('var_type')
                var_flow = var.metadata.get('flow')

                print('FLOW', var_flow, '\n \n TYPE', var_type, '\n \n NAME', var_name, '\n \n value ',var_value)

                # parameters var_name is a float, statevariable var_name is string!
                if var_type is FluxVarType.PARAMETER:
                    self.Model.parameters[label + '_' + var_name] = var_value
                elif var_type is FluxVarType.VARIABLE:
                    if var_flow is FluxVarFlow.OUTPUT:
                        self.Model.fluxes[var_value].append(flux.negative_flux)
                    elif var_flow is FluxVarFlow.INPUT:
                        self.Model.fluxes[var_value].append(flux.comp)

    def assemble(self):
        self.Solver.assemble(self.Model)

        self.solve_start = tm.time()

    def solve(self, time_step):

        print(self.Model)

        if self.Model.time is None:
            raise Exception('Time needs to be supplied to Model before solve')

        self.Solver.solve(self.Model, time_step)

    def cleanup(self):
        self.solve_end = tm.time()
        print(f"Model was solved in {round(self.solve_end - self.solve_start, 5)} seconds")

        print("Cleanup method is called within Model class")
        self.Solver.cleanup()
