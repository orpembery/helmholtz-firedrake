from firedrake import *
import numpy as np

# User-changeable parameters

k = 10.0 # The wavenumber - real

mesh_condition = '3/2' # should be '3/2' or '2' depending on whether h \lesssim k^{-3/2} or h \lesssim k^{-2}

coeff_pieces = 10 # Number of `pieces' the piecewise constant coefficient has in each direction - int

n_background = 'constant' # The background with respect to which we precondition. Options are 'constant', 'bad', or 'good', which correspond to the background being 1.0, n jumping down, n jumping up
#FILL IN MORE DETAIL HERE WHEN IT'S DONE

noise_level_n = 0.5 # The size of the noise in n, i.e. ||n-n_0||_{L^\infty} = noise_level_n, when we don't scale with k

A_background = 'constant' # COMMENT THIS LIKE FOR n, BUT I THINK THE JUMPS WILL GO THE OTHER WAY

noise_level_A = 0.1 # As for noise_level_n, but for A

num_repeats = 50 # number of repeats to do



### The user does not need to change anything below this point ###



# Define mesh size to eliminate pollution effect
mesh_size = np.ceil(k**(1.5))

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

# PUT IF IN HERE
n_pre = 1.0 # background

n = n_pre

np.random.seed(1) # Set random seed

# PUT IF IN SO THAT IT SKIPS IF ZERO
n_values =  noise_level_n * (2.0 * np.random.random_sample([coeff_pieces,coeff_pieces]) - 1.0) # Uniform (-1,1) random variates
# confusingly, going along rows of n_values corresponds to increasing y, and going down rows corresponds to increasing x
# NORMALISE THE NOISE

n_values_constant = Constant(n_values,domain=mesh)

# For each `piece', perturb n by the correct value on that piece
for xii in range(0,coeff_pieces):
  for yii in range(0,coeff_pieces):
    n += n_values_constant[xii,yii] * Iab(x[0],xii/coeff_pieces,(xii+1)/coeff_pieces) * Iab(x[1],yii/coeff_pieces,(yii+1)/coeff_pieces)

#IF IN HERE
A_pre =  as_matrix([[1.0,0.0],[0.0,1.0]]) 

A = A_pre


# PUT IF IN SO THAT IT SKIPS IF ZERO
A_values = noise_level_A * (2.0 * np.random.random_sample([coeff_pieces**2,2,2]) - 1.0) # Uniform (-1,1) random variates
# NORMALISE THE NOISE

# We want each 2x2 `piece' of A_values to be an entry in a list, so that we can then turn each of them into a Firedrake `Constant` (I hope that this will mean Firedrake passes them as arguments to the C kernel, as documented on the `Constant` documentation page

A_values_list = list(A_values)

# Will symmetrise a 2x2 matrix
def symmetrise(A):
  A_lower = np.tril(A,k=-1)
  return np.diagflat(np.diagonal(A).copy()) + A_lower + np.transpose(A_lower)

# Symmetrise all the matrices
A_values_list = [symmetrise(A_dummy) for A_dummy in A_values_list]

# Make all the matrices into Firedrake `Constant`s

A_values_constant_list = [Constant(A_dummy,domain=mesh) for A_dummy in A_values_list]

# This extracts the relevant element of the list, given a 2-d index
def list_extract(values_list,x_coord,y_coord,coord_length): # The list should contain coord_length**2 elements
  return values_list[x_coord + y_coord * coord_length]

for xii in range(0,coeff_pieces-1):
  for yii in range(0,coeff_pieces-1):
    A += list_extract(A_values_constant_list,xii,yii,coeff_pieces) * Iab(x[0],xii/coeff_pieces,(xii+1)/coeff_pieces) * Iab(x[1],yii/coeff_pieces,(yii+1)/coeff_pieces)


# Define sesquilinear form and antilinear functional for real problem
a = (inner(A * grad(u), grad(v)) - k**2 * inner(real(n) * u,v)) * dx - (1j* k * inner(u,v)) * ds # real(n) is just a failsafe
L =  inner(g,v)*ds

# Define sesquilinear form for preconditioning problem
a_pre = (inner(A_pre * grad(u), grad(v)) - k**2 * inner(real(n_pre) * u,v)) * dx - (1j* k * inner(u,v)) * ds # real(n_pre) is just a failsafe

# Define numerical solution
u_h = Function(V)

# The following code courtesy of Lawrence Mitchell - it assumes the preconditioner doesn't change - for QMC/MCMC would need to do something else I suspect (i.e. call on Lawrence again :P)

problem = LinearVariationalProblem(a, L, u_h, aP=a_pre, constant_jacobian=False)
solver = LinearVariationalSolver(problem, solver_parameters={"ksp_type": "gmres",
                                                            "mat_type": "aij",
                                                            "pmat_type": "aij",
                                                            "snes_lag_preconditioner": -1,
                                                            "pc_type": "lu",
                                                            "ksp_reuse_preconditioner": True,
                                                            'ksp_norm_type': 'unpreconditioned'})


# Now perform all the experiments
try:
  import matplotlib.pyplot as plt
except:
  warning("Matplotlib not imported")


for repeat_ii in range(0,num_repeats):
    solver.solve()

    # JUST FOR DEBUGGING, CHECK THE SOLUTION IS CHANGING
    # Plot the image

    try:
      plot(u_h,num_sample_points=1)
    except Exception as e:
      warning("Cannot plot figure. Error msg: '%s'" % e)

    try:
      plt.show()
    except Exception as e:
      warning("Cannot show figure. Error msg: '%s'" % e)

    # RECORD NUMBER OF GMRES ITS

    # Create new values of A and n
    n_values_constant.assign(noise_level_n * (2.0 * np.random.random_sample([coeff_pieces,coeff_pieces]) - 1.0))

    [A_this_constant.assign(noise_level_A * (2.0 * np.random.random_sample([2,2]) - 1.0)) for A_this_constant in A_values_constant_list]


    
