import numpy as np
import xsimlab as xs

def phydra_setup(model, input_vars, output_vars):
    """ This function wraps create_setup and adds a dummy clock parameter
    necessary for model execution """
    return xs.create_setup(model=model,
                           # necessary for xsimlab
                           clocks={'clock': [0, 1]},
                           input_vars=input_vars,
                           output_vars=output_vars)

def add_fulldims_EQ(m, Component, EQ2add):
    a = np.nditer(Component, op_flags=['readwrite'], flags=['zerosize_ok', 'refs_ok', 'multi_index'])
    b = np.nditer(EQ2add, op_flags=['readwrite'], flags=['zerosize_ok', 'refs_ok', 'multi_index'])
    while not b.finished:
        while not a.finished:
            print(a.value, b.value)
            m.Equation(Component[a.multi_index].dt() == EQ2add[b.multi_index])
            a.iternext()
        b.iternext()

def blowup_Dims(IN, FullDims, type):
    return np.tile(type(IN), FullDims)



def add_fulldims_Param(Param, Part, Part_dims='component'):
    if Part_dims == 'component':
        a = np.nditer(Param, op_flags=['readwrite'], flags=['zerosize_ok', 'refs_ok', 'multi_index'])
        b = np.nditer(Part, op_flags=['readwrite'], flags=['zerosize_ok', 'refs_ok', 'multi_index'])
        while not b.finished:
            while not a.finished:
                print(a.value, type(a.value), b.value, type(b.value))
                Param[a.multi_index] = Part[b.multi_index]
                a.iternext()
            b.iternext()
        return Param
    elif Part_dims == 'environment':
        pass

def add_fulldims_SVs(m, Component, FullShape):
    a = np.nditer(Component, op_flags=['readwrite'], flags=['zerosize_ok', 'refs_ok', 'multi_index'])
    b = np.nditer(FullShape, op_flags=['readwrite'], flags=['zerosize_ok', 'refs_ok', 'multi_index'])
    while not b.finished:
        while not a.finished:
            print(a.value, type(a.value), b.value, type(b.value))
            Component[a.multi_index].value = FullShape[b.multi_index]
            a.iternext()
        b.iternext()
    return Param