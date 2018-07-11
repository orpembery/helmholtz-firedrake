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
  return heaviside(x-a) - heaviside(x-b)

n = 1.0 # background

n_values =  0.1 * np.array([[-1.0,2.0],[-3.0,4.0]]) # confusingly, going along rows corresponds to increasing y, and going down rows corresponds to increasing x

# For each `piece', perturb n by the correct value on that piece
for xii in range(0,coeff_pieces):
  for yii in range(0,coeff_pieces):
    n += n_values[xii,yii] * Iab(x[0],xii/coeff_pieces,(xii+1)/coeff_pieces) * Iab(x[1],yii/coeff_pieces,(yii+1)/coeff_pieces)

# Plot n

V_n = FunctionSpace(mesh,"DG", 0)

n_func = Function(V_n)

n_func.interpolate(n)

import matplotlib.pyplot as plt

plot(n_func,num_sample_points=1)

plt.show()

A = as_matrix([[1.0,0.0],[0.0,1.0]]) # background

np.random.seed(1) # Set random seed

A_values = 0.0 * np.random.random_sample([coeff_pieces,coeff_pieces,2,2]) # because it's easier than manually specifying it, although this doesn't give symmetric A, which is bad - NEED TO FIX

for xii in range(0,coeff_pieces-1):
  for yii in range(0,coeff_pieces-1):
    A += as_matrix(A_values[xii,yii,:,:]) * Iab(x[0],xii/coeff_pieces,(xii+1)/coeff_pieces) * Iab(x[1],yii/coeff_pieces,(yii+1)/coeff_pieces)

# Currently haven't checked whether A is doing what I expect - the code below (in some form) should allow me to check, if I can figure out how to plot A....
   
#V_A = VectorFunctionSpace(mesh, "DG", 0)

#A_func = Function(V_A)

#A_func.interpolate(A)

#A_func_temp = Function(V_n)

#A_func_temp.interpolate(A_func.sub([0,0]))

#plot(A_func_temp,num_sample_points=1)

#plt.show()

# Define sesquilinear form and antilinear functional
a = (inner(A * grad(u), grad(v)) - k**2 * inner(real(n) * u,v)) * dx - (1j* k * inner(u,v)) * ds # real(n) is just a failsafe
L =  inner(g,v)*ds

# Define numerical solution
u_h = Function(V)

# Solve using a direct LU solver
solve(a == L, u_h, solver_parameters={'ksp_type': 'gmres', 'pc_type': 'lu'})

# Write solution to a file for visualising
File("helmholtz-piecewise.pvd").write(u_h)

# Plot the image
try:
  import matplotlib.pyplot as plt
except:
  warning("Matplotlib not imported")

try:
  plot(u_h,num_sample_points=1)
except Exception as e:
  warning("Cannot plot figure. Error msg: '%s'" % e)

try:
  plt.show()
except Exception as e:
  warning("Cannot show figure. Error msg: '%s'" % e)
