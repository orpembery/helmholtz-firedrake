# Modified from the firedrake `Simple Helmholtz equation' demo and https://www.firedrakeproject.org/demos/saddle_point_systems.py.html
from firedrake import *
from math import ceil # so that we can define the mesh size

# Define wavenumber
k = 10.0

# Define mesh size to eliminate pollution effect
mesh_size = ceil(k**(1.5))

# Create a mesh
mesh = UnitSquareMesh(mesh_size, mesh_size)

# Define function space for functions - continuous piecewise linear
V = FunctionSpace(mesh, "CG", 1)

# Define trial and test functions on the space
u = TrialFunction(V)
v = TestFunction(V)

# Define right-hand side function - Gaussian approximation of point-source - gives circular waves
f = Coefficient(V)
g = Coefficient(V)
x = SpatialCoordinate(mesh)
x_centre = 0.5
y_centre = 0.5
#f.interpolate(exp((-(k/pi)**2)*((x[0]-x_centre)**2 + (x[1]-y_centre)**2)))
#f.interpolate(1.0)

# Right-hand side g is the boundary condition given by a plane wave with direction d
nu = FacetNormal(mesh)

# Unsure if the following is the correct way to allow us to take a dot product with u
d = as_vector([1.0/sqrt(2.0),1.0/sqrt(2.0)])

# Boundary condition
g=1j*k*exp(1j*k*dot(x,d))*(dot(d,nu)-1)

# Define coefficients

# Define function space for coefficients
V_A = TensorFunctionSpace(mesh, "CG", 1, symmetry=True)

A=Coefficient(V_A)

A=as_matrix([[1.0,0.0],[0.0,1.0]])

n = Coefficient(V)

#n=1.0

n_centre=as_vector([0.5,0.5])
n = 0.5+abs(x - n_centre)**2

# Define sesquilinear form and antilinear functional
a = (inner(A * grad(u), grad(v)) - k**2 * inner(real(n) * u,v)) * dx - (1j* k * inner(u,v)) * ds # real(n) is just a failsafe
L =  inner(g,v)*ds#inner(f,v) * dx +

# Define numerical solution
u_h = Function(V)

# Define the problem with respect to which we will precondition

A_pre = Coefficient(V_A)

A_pre = A=as_matrix([[1.0,0.0],[0.0,1.0]])

n_pre = Coefficient(V)

n_pre = 1.0

# Define sesquilinear form and antilinear functional for preconditioning
a_pre = (inner(A_pre * grad(u), grad(v)) - k**2 * inner(real(n_pre) * u,v)) * dx - (1j* k * inner(u,v)) * ds # real(n) is just a failsafe


# Because we're using a preconditioner, we set up the solver in slightly more detail
parameters = {'ksp_type': 'gmres', # use GMRES
              'pc_type': 'lu', # use an LU factorisation of the preconditioning matrix as a precondition (i.e., compute the exact inverse)
              'ksp_norm_type': 'unpreconditioned'} # measure convergence in the unpreconditioned norm

A = assemble(a, mat_type = 'aij') # assemble the monolithic matrix

P = assemble(a_pre, mat_type = 'aij') # same for preconditioning problem

solver = LinearSolver(A,P=P, solver_parameters = parameters)

b = assemble(L) # assemble right-hand side

solver.solve(u_h,b)

# Print GMRES convergence
print(solver.ksp.getIterationNumber())

# Now attempting to build on this for UQ

# Trying to extract the (LU-decomposition) preconditioner
pc_obj = solver.ksp.pc
B_not_needed, P_LU = pc_obj.getOperators() # PETSc operator corresponding to the LU decomposition of P. B is not needed

# Now (just as a test) do the same calculation as above, but this time we pass in the preconditioner LU object

# Write solution to a file for visualising
File("helmholtz.pvd").write(u_h)

# Plot the image
try:
  import matplotlib.pyplot as plt
except:
  warning("Matplotlib not imported")

try:
  plot(u_h)
except Exception as e:
  warning("Cannot plot figure. Error msg: '%s'" % e)

try:
  plt.show()
except Exception as e:
  warning("Cannot show figure. Error msg: '%s'" % e)
