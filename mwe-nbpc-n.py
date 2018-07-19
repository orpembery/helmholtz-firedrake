from firedrake import *
import numpy as np
#from functools import reduce
#from warnings import warn

k = 10.0 # The wavenumber - real

mesh_condition = 1.5 # h ~ k**mesh_condition) - real

coeff_pieces = 20 # Number of `pieces' the piecewise constant coefficient has in each direction - int

noise_level_n = 0.01 # As for noise_level_n, but for A

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

np.random.seed(1) # Set random seed

# Define A
A = as_matrix(np.array([[1.0,0.0],[0.0,1.0]]))

    
# Define n
n = 1.0

def n_noise(noise_level_n,coeff_pieces):
        n_values =  noise_level_n * (2.0 * np.random.random_sample([coeff_pieces,coeff_pieces]) - 1.0) # Uniform (-1,1) random variates
        # confusingly, going along rows of n_values corresponds to increasing y, and going down rows corresponds to increasing x
        return n_values

n_values_constant = Constant(n_noise(noise_level_n,coeff_pieces),domain=mesh)

# For each `piece', perturb n by the correct value on that piece
for xii in range(0,coeff_pieces):
    for yii in range(0,coeff_pieces):
        n += n_values_constant[xii,yii] * Iab(x[0],xii/coeff_pieces,(xii+1)/coeff_pieces) * Iab(x[1],yii/coeff_pieces,(yii+1)/coeff_pieces)


    
# Define sesquilinear form and antilinear functional for real problem
a = (inner(A * grad(u), grad(v)) - k**2 * inner(real(n) * u,v)) * dx - (1j* k * inner(u,v)) * ds # real(n) is just a failsafe
L =  inner(g,v)*ds

# Define numerical solution
u_h = Function(V)

solve(a==L,u_h,solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

