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

mesh = UnitSquareMesh(num_cells,num_cells)

# Use piecewise-linear finite elements
V = FunctionSpace(mesh,"CG",1)

# n = 1 inside a square of size 1/3 x 1/3, and n=0.5 outside
x = SpatialCoordinate(mesh)
square_limits = np.array([[1.0/3.0,2.0/3.0],[1.0/3.0,2.0/3.0]])
n = 0.5 + nd_indicator(x,0.5,square_limits)

# Define the problem
prob = hh.HelmholtzProblem(k,V,n=n)

# Use f and g corresponding to a plane wave (in homogeneous media)
prob.f_g_plane_wave()

# Use MUMPS as a direct solver
prob.use_mumps()

# Solve the problem
prob.solve()

# Plot the solution
prob.plot()
