import numpy as np

import matplotlib.pyplot as plt


PZ_num = 2

phyto_init = np.tile(.5/PZ_num, (PZ_num))

zoo_init = np.tile(.1/PZ_num, (PZ_num))


def calculate_sizes(size_min, size_max, num):
    """initializes log spaced array of sizes from ESD size range"""
    numbers = np.array([i for i in range(num)])
    sizes = (np.log(size_max) - np.log(size_min))* numbers / (num-1) + np.log(size_min)
    return np.exp(sizes)


phyto_sizes = calculate_sizes(1,20,PZ_num)

def calculate_zoo_sizes(phytosizes):
    return 2.16 * phytosizes ** 1.79

zoo_sizes = calculate_zoo_sizes(phyto_sizes)

def calculate_zoo_I0(sizes):
    """initializes allometric parameters based on array of sizes (ESD)"""
    return 26 * sizes ** -0.4 #* .5

zoo_I0 = calculate_zoo_I0(zoo_sizes)

def calculate_phyto_mu0(sizes):
    """initializes allometric parameters based on array of sizes (ESD)
    allometric relationships are taken from meta-analyses of lab data"""
    return 2.6 * sizes ** -0.45
    
phyto_mu0 = calculate_phyto_mu0(phyto_sizes)


def calculate_phyto_ks(sizes):
    return sizes * .1

phyto_ks = calculate_phyto_ks(phyto_sizes)

def init_phiP(phytosize, preyoptsize):
    """creates array of feeding preferences [P...P10] for each [Z]"""
    phiP = np.array([[np.exp(-((np.log10(xpreyi) - np.log10(xpreyoptj)) / 0.25) ** 2)
                      for xpreyi in phytosize] for xpreyoptj in preyoptsize])
    return phiP

phiP = init_phiP(phyto_sizes, phyto_sizes)



import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import phydra



from phydra.components.variables import SV, SVArraySize

from phydra.components.fluxes.basic import LinearDecay, LinearExchange, LinearDecay_ListInput
from phydra.components.fluxes.basic_dims import LinearDecay_Dims, QuadraticDecay_Dim_Sum
from phydra.components.fluxes.basic_forcing import LinearForcingInput
from phydra.components.fluxes.growth import MonodGrowth_mu_ConsumerDim

from phydra.components.forcings import ConstantForcing, SinusoidalForcing

from phydra.components.fluxes.grazing import SizebasedGrazingKernel_Dims, GrossGrowthEfficiency_MatrixGrazing


ASTroCAT = phydra.create({
    # State variables
    'Nutrient':SV,
    'Phytoplankton':SVArraySize,
    'Zooplankton':SVArraySize,
    #'Detritus':SV,
    
    # Flows:
    'Inflow':LinearForcingInput,

    # Growth
    'Growth':MonodGrowth_mu_ConsumerDim,
    
    # Grazing
    'Grazing':SizebasedGrazingKernel_Dims,
    'GGE':GrossGrowthEfficiency_MatrixGrazing,

    # Mortality
    'PhytoMortality':LinearDecay_Dims,
    'ZooMortality':QuadraticDecay_Dim_Sum,
    
    # Forcings
    'N0':ConstantForcing,
                     })


model_setup = phydra.setup(solver='odeint', model=ASTroCAT,
            time= np.arange(0,365*10),  # np.concatenate([np.arange(0,50,0.005),np.arange(50,365*10,.1)],axis=None),  # *365
            input_vars={
                    # State variables
                    'Nutrient':{'var_label':'N','var_init':5.},
                    'Phytoplankton':{'var_label':'P','var_init':phyto_init, 'sizes':phyto_sizes},
                    'Zooplankton':{'var_label':'Z','var_init':zoo_init, 'sizes':zoo_sizes},
                
                    # Flows:
                    'Inflow':{'forcing':'N0', 'rate':1., 'var':'N'},
                
                    # Growth
                    'Growth':{'resource':'N', 'consumer':'P', 'halfsat':phyto_ks, 'mu_max':phyto_mu0},

                    # Grazing
                    'Grazing':{'resource':'P', 'consumer':'Z',
                               'Imax':zoo_I0, 'KsZ':3, 'phiP':phiP},
                    'GGE':{'grazed_resource':'P', 'assimilated_consumer':'Z', 'egested_detritus':'N', 
                           'epsilon':0.33, 'f_eg':0.33},
                
                    # Mortality
                    'PhytoMortality':{'var':'P', 'rate':0.1*phyto_mu0},
                    'ZooMortality':{'var':'Z', 'rate':1.},

                    # Forcings
                    'N0':{'forcing_label':'N0', 'value':1.},
            })


from xsimlab.monitoring import ProgressBar

with ProgressBar():
    model_out = model_setup.xsimlab.run(model=ASTroCAT)

#model_out.to_netcdf('5PZ_ASTroCAT_odeint_out_10years_1.nc')