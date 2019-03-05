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
