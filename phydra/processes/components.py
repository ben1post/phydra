import numpy as np
import xsimlab as xs


@xs.process
class Component:
    """
    Basis for all components, defines the calculation of fluxes and state.
    specific fluxes, variables, and parameters need to be defined in subclass.
    """

    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        self.delta = sum((v for v in self.fluxes)) * dt  # multiply by time step

    def finalize_step(self):
        self.state += self.delta


@xs.process
class SingularComp(Component):
    dim = xs.variable(intent='out')

    # e.g. N

    def initialize(self):
        self.dim = 1


@xs.process
class Nutrient(SingularComp):
    # create the own N dimension
    dim_label = xs.variable(default='N')
    N = xs.index(dims='N')

    state = xs.variable(intent='inout', dims=[('N'), ('Env', 'N'), ('x', 'y', 'Env', 'N')])
    fluxes = xs.group('N_flux')

    def initialize(self):
        super(Nutrient, self).initialize()

        self.N = np.arange(self.dim)


@xs.process
class MultiComp(Component):
    # e.g. P, Z
    dim = xs.variable(intent='inout')


@xs.process
class Phytoplankton(MultiComp):
    dim_label = xs.variable(default='P')
    P = xs.index(dims='P')

    state = xs.variable(intent='inout', dims=[('P'), ('Env', 'P'), ('x', 'y', 'Env', 'P')])
    fluxes = xs.group('P_flux')

    halfsat = xs.variable(intent='inout', dims=[('P'), ('Env', 'P'), ('x', 'y', 'Env', 'P')])
    mortality_rate = xs.variable(intent='inout', dims=[('P'), ('Env', 'P'), ('x', 'y', 'Env', 'P')])

    def initialize(self):
        self.P = np.arange(self.dim)