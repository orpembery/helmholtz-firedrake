# Assumes this code is being run from the 'examples' folder. Otherwise,
# add the helmholtz_firedrake folder to your PYTHONPATH
import sys
sys.path.append('../')
from firedrake import *
import numpy as np
from helmholtz_firedrake import problems as hh
from helmholtz_firedrake.utils import h_to_num_cells, nd_indicator

k = 30.0

# Define a mesh that keeps pollution error bounded
num_cells = h_to_num_cells(k**-1.5,2)

L = 1.0

mesh = SquareMesh(num_cells,num_cells,1.0)

# Use piecewise-linear finite elements
V = FunctionSpace(mesh,"CG",1)

# Define the problem
prob = hh.HelmholtzProblem(k,V)

# Use f and g corresponding to a plane wave
prob.f_g_plane_wave()

# Use MUMPS as a direct solver
prob.use_mumps()

# Solve the problem
prob.solve()

# Plot the solution
prob.plot()
