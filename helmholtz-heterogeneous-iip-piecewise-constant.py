# Modified from the firedrake `Simple Helmholtz equation' demo
from firedrake import *
from math import ceil # so that we can define the mesh size
import numpy as np

# Define wavenumber
k = 10.0

# Define number of `pieces' in piecewise constant coefficients - double

coeff_pieces = 2

# Define mesh size to eliminate pollution effect
mesh_size = ceil(k**(1.5))

# Create a mesh
mesh = UnitSquareMesh(mesh_size, mesh_size)

# Define function space for functions - continuous piecewise linear
V = FunctionSpace(mesh, "CG", 1)

# Define trial and test functions on the space
u = TrialFunction(V)
v = TestFunction(V)

# Right-hand side g is the boundary condition given by a plane wave with direction d
x = SpatialCoordinate(mesh)

nu = FacetNormal(mesh)

d = as_vector([1.0/sqrt(2.0),1.0/sqrt(2.0)])

g=1j*k*exp(1j*k*dot(x,d))*(dot(d,nu)-1)

# Define coefficients

def heaviside(x): # x here is a single coordinate of a SpatialCoordinate
  return 0.5 * (sign(real(x)) + 1.0)

def Iab(x,a,b) : # indicator function on [a,b] - x is a single coordinate of a spatial coordinate, 0.0  <= a < b <= 1 are doubles
  return 0.5 * ( heaviside(x-a) + 1.0 - heaviside(x-b) )

n = 1.0 # background

n_values = 10.0 * np.array([[0.1,-0.05],[-0.03,0.08]])

for xii in range(0,coeff_pieces-1):
  for yii in range(0,coeff_pieces-1):
    n += n_values[xii,yii] * Iab(x[0],xii/coeff_pieces,(xii+1)/coeff_pieces) * Iab(x[1],yii/coeff_pieces,(yii+1)/coeff_pieces)

# Plot n

n_func = Function(V)

n_func.interpolate(n)

import matplotlib.pyplot as plt

plot(n_func,num_sample_points=1)

plt.show()

A = as_matrix([[1.0,0.0],[0.0,1.0]])

# Define sesquilinear form and antilinear functional
a = (inner(A * grad(u), grad(v)) - k**2 * inner(real(n) * u,v)) * dx - (1j* k * inner(u,v)) * ds # real(n) is just a failsafe
L =  inner(g,v)*ds

# Define numerical solution
u_h = Function(V)

# Solve using a direct LU solver
solve(a == L, u_h, solver_parameters={'ksp_type': 'gmres', 'pc_type': 'lu'})

# Write solution to a file for visualising
File("helmholtz.pvd").write(u_h)

# Plot the image
try:
  import matplotlib.pyplot as plt
except:
  warning("Matplotlib not imported")

#try:
#  plot(mesh)
#  plt.show()
#except:
#  warning("Dunno")
  
try:
  plot(u_h,num_sample_points=1)
except Exception as e:
  warning("Cannot plot figure. Error msg: '%s'" % e)

try:
  plt.show()
except Exception as e:
  warning("Cannot show figure. Error msg: '%s'" % e)
