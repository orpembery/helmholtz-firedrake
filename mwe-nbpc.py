from firedrake import *
import numpy as np
from functools import reduce
from warnings import warn

k = 30.0 # The wavenumber - real

mesh_condition = 1.5 # h ~ k**mesh_condition) - real

coeff_pieces = 14 # Number of `pieces' the piecewise constant coefficient has in each direction - int

noise_level_A = 0.01 # As for noise_level_n, but for A

# Define mesh size to eliminate pollution effect
mesh_size = np.ceil(k**(mesh_condition)/np.sqrt(2.0))

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

  # Will symmetrise a 2x2 matrix
def symmetrise(A):
    A_lower = np.tril(A,k=-1)
    return np.diagflat(np.diagonal(A).copy()) + A_lower + np.transpose(A_lower)

np.random.seed(1) # Set random seed



### I'm fairly sure the error is somewhere in here ###

# This is a bit cack-handed, and there may well be a better way to do it

# We're going to take the identity matrix, and then add a random perturbation, which is piecewise constant on each subdomain (that is, on each subdomain, A is a random (symmetric) perturbation of the identity matrix
A = as_matrix([[1.0,0.0],[0.0,1.0]])

# Generate the random matrices that will be used to form the peterturbations
A_values = noise_level_A * (2.0 * np.random.random_sample([coeff_pieces**2,2,2]) - 1.0) # Uniform (-1,1) random variates

# Make a list of the matrices
A_values_list = list(A_values)

# Symmetrise all the matrices
A_values_list = [symmetrise(A_dummy) for A_dummy in A_values_list]

# Turn all the matrices into Firedrake `Constant`s
A_values_constant_list = [Constant(A_dummy,domain=mesh) for A_dummy in A_values_list]

# This extracts the relevant element of the list, given a 2-d index - this enables us to, given a subdomain index [xii,yii], extract the correct matrix from the list of matrices
def list_extract(values_list,x_coord,y_coord,coord_length): # The list should contain coord_length**2 elements
  return values_list[x_coord + y_coord * coord_length]

# Form A by looping over all the subdomains
for xii in range(0,coeff_pieces-1):
  for yii in range(0,coeff_pieces-1):
    #A = A + list_extract(A_values_constant_list,xii,yii,coeff_pieces) * Iab(x[0],xii/coeff_pieces,(xii+1)/coeff_pieces) * Iab(x[1],yii/coeff_pieces,(yii+1)/coeff_pieces)
    A = A + Constant(np.array([[0.0,0.0],[0.0,0.0]]))

### Fairly sure error above here ###
    
# Define n
n = 1.0

    
# Define sesquilinear form and antilinear functional for real problem
a = (inner(A * grad(u), grad(v)) - k**2 * inner(real(n) * u,v)) * dx - (1j* k * inner(u,v)) * ds # real(n) is just a failsafe
L =  inner(g,v)*ds

# Define numerical solution
u_h = Function(V)

solve(a==L,u_h,solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

