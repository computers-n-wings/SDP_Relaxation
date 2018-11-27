import os
import numpy as np
from shutil import copy,rmtree
import collections
from pyDOE import lhs
import matplotlib.pyplot as plt
from equadratures import *
from matplotlib.mlab import griddata

import sys
# Append the path to CDO.py
sys.path.append('C:\Users\james\Documents\BladeGen')
# Append the path to SDPR.py
sys.path.append('C:\Users\james\Documents\PolyOpt')
# Import CDO and sdprelaxation classes
from CDO import CDO
from SDPR import sdprelaxation

# Generate the designs for DOE from a LHS
def GenerateDesigns():
    num_designs = 600
    xp = [0,3,7,14,20,24]
    fp = [.003,.007,.004,.0035,.003,.0025]
    total_num_vars = 30
    
    # Interpolates xp and fp with 25 points
    bounds = np.interp(np.arange(15),xp,fp)
    # Distance along chord
    locations = np.linspace(0.05,0.9,15)
    
    x_design = lhs(total_num_vars, num_designs)
    x_design = 2.0*x_design - 1.0 # so that they are now btn [-1,1]
    bounds_double = np.repeat(bounds,2)
    x_design *= bounds_double
    
    np.savetxt('design_params.txt', x_design, delimiter=',')
    np.savetxt('locations.txt', locations, delimiter=',')
    return None

# Create grid to be used in 2D active variables plot
def grid(x, y, z, resX=40, resY=40):
    "Convert 3 column data to matplotlib grid"
    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi, interp='linear')
    X, Y = np.meshgrid(xi, yi)
    return X, Y, Z

# Extract the Cd and Cl from forces_breakdown.dat created by SU2 run
def ExtractResultsFromDAT(FileName):
    with open(FileName,'r') as f:
        filedata = f.readlines()
    Cl = float(filedata[91].split()[2])
    Cd = float(filedata[92].split()[2])
    return Cd, Cl

# Run DOE study using inputs from design_params.txt (generated using a LHS) 
# with locations along aerofoil determined from locations.txt
def DOE():
#   Load the design parameters and the locations from the txt files
    all_designs = np.loadtxt('design_params.txt',delimiter=',')
    locations = np.loadtxt('locations.txt', delimiter=',')
    num_exps = all_designs.shape[0]
    for n in range(num_exps):
#       Specify current design
        current_design = all_designs[n,:]
    
        # Create cfg file for SU2_DEF and make the necessary modifications
        with open('baseline_config.cfg','r') as cfg_file:
            configs = cfg_file.readlines()
    
        line_185 = "DV_KIND= "
        line_210 = "DV_PARAM= "
        line_213 = "DV_VALUE= "
        line_263 = "MESH_FILENAME= baseline_mesh.su2"
        line_263_cfd = "MESH_FILENAME= mesh_out.su2"
    
        for i in range(len(locations)):
            line_185 += "HICKS_HENNE, HICKS_HENNE, "
            line_210 += "(0," + str(locations[i]) + "); (1," + str(locations[i]) + "); "
            line_213 += str(current_design[i*2]) + ", " + str(current_design[i*2 + 1]) + ", "
    
        line_185 = line_185[:-2] + '\n'
        line_210 = line_210[:-2] + '\n'
        line_213 = line_213[:-2] + '\n'
    
        configs[185] = line_185
        configs[210] = line_210
        configs[213] = line_213
        configs[263] = line_263
#       Name the directory, create it, copy the necessary files to it, and 
#       write the modified files
        dir_name = 'design_%03d' % (n)
        try:
            os.mkdir(dir_name)
        except:
            # Directory already exists?
            pass
        copy('baseline_mesh.su2',dir_name)
        os.chdir(dir_name)
        
        with open('new_config_def.cfg','w') as new_cfg_file:
            new_cfg_file.writelines(configs)
    
        configs[263] = line_263_cfd
    
        with open('new_config_cfd.cfg','w') as new_cfg_file_cfd:
            new_cfg_file_cfd.writelines(configs)
    
        # Run SU2_DEF on this
        os.system('SU2_DEF new_config_def.cfg')
    
        # Run SU2_CFD on the mesh
        os.system('SU2_CFD new_config_cfd.cfg')
#       Extract the Cd and Cl from forces_breakdown.dat
        Results = ExtractResultsFromDAT('forces_breakdown.dat')
#       Create an ordered dict and write inputs and outputs to dict
        Dict = collections.OrderedDict()
        for num,param in enumerate(current_design):
            key = 'HH'+str(num)
            Dict[key] = param
        Dict['Cd'] = Results[0]
        Dict['Cl'] = Results[1]
        # Return to parent dir
        os.chdir('../')
#       Append ordered dict values to Results.csv
        CDO.OrderedDict2CSV('Results.csv',Dict)
#       Remove child directory
        rmtree(dir_name)
    return None

# Create a Vandermonde matrix for index set created in sdprelaxation class
def CreateVandermonde(X,d):
    m = X.shape[0]
    n = X.shape[1]
    index = sdprelaxation.genpower(n+1,d)[:,1:]
    card = index.shape[0]
    V = np.ones((m,card))
    for row in range(m):
        for col in range(card):
            for k in range(n):
                V[row,col] *= X[row,k]**index[col,k]
    return V

# Evaluate a polynomial with index set created in sdprelaxation class
def EvaluatePoly(p,X,d):
    m = X.shape[0]
    index = sdprelaxation.genpower(n+1,d)[:,1:]
    card = index.shape[0]
    points = np.zeros(m)
    for row in range(m):
        for dim in range(card):
            term = 1
            for k in range(n):
                term *= X[row,k]**index[dim,k]
            points[row] += p[dim]*term
    return points

if __name__=='__main__':
################################# Run DOE #####################################
###############################################################################
#    GenerateDesigns()
#    DOE()
###############################################################################
###############################################################################

############ Compute active variables using Variable Projection ###############
###############################################################################
    N = 30
    d = 2
    n = 2
#   Clean the results data by removing rows with NaN values and scaling between
#   [-1,1]
    data = CDO.CleanData('Results.csv',scale=[-1,1])
#   Create the input and output vals (Cd and Cl)
    input_vals = data.values[:,0:N]
    Cd_vals = data.values[:,N]
    Cl_vals = data.values[:,N+1]
#   Use variable projection algorithm from EQ
    U,R=variable_projection(input_vals,Cl_vals,n,d,gamma=0.1,beta=0.1)
#   Compute active variable 1 and 2
    active1 = np.dot(input_vals,U[:,0])
    active2 = np.dot(input_vals,U[:,1])
###############################################################################
###############################################################################
    
#################### Plot along both active variables #########################
###############################################################################  
#    X, Y, Z = grid(active1, active2, Cd_vals)
#    fig = plt.figure()
#    ax = fig.add_subplot(1,1,1)
#    cax = plt.contourf(X, Y, Z, 20, vmin=-1, vmax=0.5)
#    cax = ax.scatter(active1, active2,c=Cd_vals, marker='o', s=30)
#    cbar = plt.colorbar(cax, extend='neither', spacing='proportional', orientation='vertical', shrink=0.8)
#    ax.set_xlabel('Active variable 1, $\mathbf{XU_{1}}$')
#    ax.set_ylabel('Active variable 2, $\mathbf{XU_{2}}$')
#    plt.xlim([-2.0, 2.0])
#    plt.ylim([-2.0, 2.0])
#    adjust_spines(ax, ['left', 'bottom'])
#    ax.set_title('Drag Coefficient along $\mathbf{U_{1}}$ and $\mathbf{U_{2}}$')
###############################################################################
###############################################################################
    
#################### Plot along both active variable 1 ########################
###############################################################################
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    cax = ax.scatter(active1, Cd_vals, c=Cd_vals, marker='o', s=100, vmin=-1., vmax=1.)
    cbar = plt.colorbar(cax, extend='neither', spacing='proportional', orientation='vertical', shrink=0.8)
    ax.set_xlabel('Active variable 1, $\mathbf{XU_{1}}$')
    ax.set_ylabel('Non-dimensional drag coefficeint, $C_d$')
    plt.xlim([-2.0, 2.0])
    adjust_spines(ax, ['left', 'bottom'])
    ax.set_title('Drag coefficient along $\mathbf{U_{1}}$')