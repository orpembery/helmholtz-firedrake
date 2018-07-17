from firedrake import *
import numpy as np
from functools import reduce
from warnings import warn
import subprocess
import datetime
import csv

# Write files that test function definitions


### User-changeable parameters ###

k = 30.0 # The wavenumber - real

mesh_condition = 1.5 # h ~ k**mesh_condition) - real

coeff_pieces = 14 # Number of `pieces' the piecewise constant coefficient has in each direction - int

n_background = 'constant' # The background with respect to which we precondition. Options are 'constant', 'bad', or 'good', which correspond to the background being 1.0, n jumping down, n jumping up
#FILL IN MORE DETAIL HERE WHEN IT'S DONE

noise_level_n = 0.1 # The size of the noise in n, i.e. ||n-n_0||_{L^\infty} = noise_level_n, when we don't scale with k

A_background = 'constant' # COMMENT THIS LIKE FOR n, BUT I THINK THE JUMPS WILL GO THE OTHER WAY

noise_level_A = 0.00 # As for noise_level_n, but for A

num_repeats = 1 # number of repeats to do

### The user does not need to change anything below this point ###

def test_helmholtz_nearby_precon(k,mesh_condition,coeff_pieces,n_background,noise_level_n,A_background,noise_level_A,num_repeats):

  # Define mesh size to eliminate pollution effect
  mesh_size = np.ceil(k**(mesh_condition)/np.sqrt(2.0)) # dividing by sqrt(2.0) because firedrake interprets the mesh size (in UnitSquareMesh as the number of cells in the x and y directions, whereas we want k**msh_condition to be the diameter of the cell

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

  # Set background state
  if n_background == 'constant':
    n_pre = 1.0
  else:
    warn('Currently I''ve only implemented this for a constant background for n. Using n_background = ''constant''')
    n_pre = 1.0

  n = n_pre

  np.random.seed(1) # Set random seed


  if noise_level_n != 0.0:
    def n_noise(noise_level_n,coeff_pieces):
        n_values =  noise_level_n * (2.0 * np.random.random_sample([coeff_pieces,coeff_pieces]) - 1.0) # Uniform (-1,1) random variates
        # confusingly, going along rows of n_values corresponds to increasing y, and going down rows corresponds to increasing x
        return n_values

    n_values_constant = Constant(n_noise(noise_level_n,coeff_pieces),domain=mesh)

    # For each `piece', perturb n by the correct value on that piece
    for xii in range(0,coeff_pieces):
      for yii in range(0,coeff_pieces):
        n += n_values_constant[xii,yii] * Iab(x[0],xii/coeff_pieces,(xii+1)/coeff_pieces) * Iab(x[1],yii/coeff_pieces,(yii+1)/coeff_pieces)

  if A_background == 'constant':
    A_pre =  as_matrix([[1.0,0.0],[0.0,1.0]])
  else:
    warn('Currently I''ve only implemented this for a constant background for A. Using A_background = ''constant''')
    A_pre =  as_matrix([[1.0,0.0],[0.0,1.0]])

  A = A_pre

  if noise_level_A != 0.0:
    warn('Currently I''ve only implemented variable n. Therefore using constant A = identity')
  #  def A_noise(noise_level_A,coeff_pieces): # Generates a list of the values of A on the different subdomains
  #      # Will symmetrise a 2x2 matrix
  #      def symmetrise(A):
  #          A_lower = np.tril(A,k=-1)
  #          return np.diagflat(np.diagonal(A).copy()) + A_lower + np.transpose(A_lower)

  #      A_values = noise_level_A * (2.0 * np.random.random_sample([coeff_pieces**2,2,2]) - 1.0) # Uniform (-1,1) random variates

  #      A_values_list = list(A_values)
        # We want each 2x2 `piece' of A_values to be an entry in a list, so that we can then turn each of them into a Firedrake `Constant` (I hope that this will mean Firedrake passes them as arguments to the C kernel, as documented on the `Constant` documentation page

        # Symmetrise all the matrices
  #      A_values_list = [symmetrise(A_dummy) for A_dummy in A_values_list]


  #      return A_values_list

  #  A_values_list = A_noise(noise_level_A,coeff_pieces)

  #  A_values_constant_list = [Constant(A_dummy,domain=mesh) for A_dummy in A_values_list]

    # This extracts the relevant element of the list, given a 2-d index
  #  def list_extract(values_list,x_coord,y_coord,coord_length): # The list should contain coord_length**2 elements
  #    return values_list[x_coord + y_coord * coord_length]

    # Form A by looping over all the subdomains
  #  for xii in range(0,coeff_pieces-1):
  #    for yii in range(0,coeff_pieces-1):
  #      A += list_extract(A_values_constant_list,xii,yii,coeff_pieces) * Iab(x[0],xii/coeff_pieces,(xii+1)/coeff_pieces) * Iab(x[1],yii/coeff_pieces,(yii+1)/coeff_pieces)


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


  
  # For saving in CSV

  # Get git hash
  git_hash = subprocess.run("git rev-parse HEAD", shell=True, stdout=subprocess.PIPE)
  git_hash_string = git_hash.stdout.decode('UTF-8')[:-1] # help from https://stackoverflow.com/a/6273618

  # For current date and time
  date_time = datetime.datetime(1,1,1) # This initialises the object. I don't understand why this is necessary
  date_time = date_time.utcnow().isoformat()
  
  # Write CSV
  with open('nearby-preconditioning-test-output-' + date_time + '.csv', 'w', newline = '') as csvfile: # from https://docs.python.org/3.5/library/csv.html
    file_writer = csv.writer(csvfile, delimiter = ',', quoting = csv.QUOTE_MINIMAL)
    file_writer.writerow(['Git hash', git_hash_string])
    file_writer.writerow(['Date/Time', date_time]) # Current time in UTC as an ISO string
    file_writer.writerow(['k',k ])
    file_writer.writerow(['mesh_condition',mesh_condition])
    file_writer.writerow(['coeff_pieces',coeff_pieces])
    file_writer.writerow(['n_background',n_background])
    file_writer.writerow(['noise_level_n',noise_level_n])
    file_writer.writerow(['A_background',A_background])
    file_writer.writerow(['noise_level_A',noise_level_A])
    file_writer.writerow(['num_repeats',num_repeats])
    file_writer.writerow(['Iteration number','Number of GMRES iterations'])
      # Now perform all the experiments
    for repeat_ii in range(0,num_repeats):
      solver.solve()

      file_writer.writerow([repeat_ii,solver.snes.ksp.getIterationNumber()])

      # Create new values of A and n
      if noise_level_n != 0.0:
        n_values_constant.assign(n_noise(noise_level_n,coeff_pieces))

        #if noise_level_A != 0.0:
          #[A_dummy.assign(A_noise(noise_level_A,1)[0]) for A_dummy in A_values_constant_list] # All this does is generate correctly normalised realisation of symmetric 2x2 matrices one at a time, and replaces the values of the constants with these generated matrices

  return 'Test Completed'

test_helmholtz_nearby_precon(k,mesh_condition,coeff_pieces,n_background,noise_level_n,A_background,noise_level_A,num_repeats)
