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

mesh = SquareMesh(num_cells,num_cells,L)

# Use piecewise-linear finite elements
V = FunctionSpace(mesh,"CG",1)

# n = 1.5 + sin(x*y)
x = SpatialCoordinate(mesh)
square_limits = np.array([[1.0/3.0,2.0/3.0],[1.0/3.0,2.0/3.0]])
n_osc = 30.0
n = 1.5 + sin(n_osc*x[0])*sin(n_osc*x[1])

# Define the problem
prob = hh.HelmholtzProblem(k,V,n=n)


# n is modified so that it decays to 1 outside a square
centre = np.array((0.5,0.5))
width = 0.5
transition_zone_width = 0.2

# Use f and g corresponding to a plane wave (in homogeneous media)
angle = 2.0*np.pi/3.0

prob.sharp_cutoff(np.array([0.5,0.5]),0.6)

prob.plot_n()

prob.f_g_scattered_plane_wave([np.cos(angle),np.sin(angle)])

# Use MUMPS as a direct solver
prob.use_mumps()

# Solve the problem
prob.solve()

# Plot the solution
prob.plot()
