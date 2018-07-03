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
precon_parameters = {'ksp_type': 'preonly', # only do an LU decomposition
                     'pc_type': 'lu'} # use an LU factorisation of the preconditioning matrix as a precondition (i.e., compute the exact inverse)


A_pre = assemble(a_pre, mat_type = 'aij') # same for preconditioning problem

solver_precon = LinearSolver(A_pre, solver_parameters = precon_parameters)

b = assemble(L) # assemble right-hand side

solver_precon.solve(u_h,b) # we're not worried about storing the solution yet

# Trying to extract the (LU-decomposition) preconditioner
pc_obj = solver_precon.ksp
B_not_needed, P_LU = pc_obj.getOperators() # PETSc operator corresponding to the LU decomposition of P. B is not needed

# Now  do the same calculation (with sesquilinear form a) as above, but this time we pass in the preconditioner LU object

A = assemble(a, mat_type = 'aij') # assemble the monolithic matrix

gmres_parameters = {'ksp_type': 'gmres', # use GMRES
                    'ksp_norm_type': 'unpreconditioned'} # measure convergence in the unpreconditioned norm
# I don't think we need any extra operators here to say that we're preconditioning, as it's all being passed in below

solver = LinearSolver(A,solver_parameters = gmres_parameters)

gmres_obj = solver.ksp

A_obj, not_needed = gmres_obj.getOperators() # PETSc operators corresponding to A (and a preconditioner, but it doesn't matter what it is, because we're about to replace it

#A_obj.PetscObjectReference() # The PETSc documentation (http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PCSetOperators.html) says you need to do this as I'm keeping A_obj, but it didn't work, so I'll comment it out, see what happens, and ask Jack

gmres_obj.setOperators(A_obj,P_LU) # This should set the system matrix (or its action) as A_obj and the preconditioner as P_LU

# Note to self, have not used the .pc stuff as I don't think it's needed, but we'll see at runtime

solver.solve(u_h,b) # again, not fussing about storing the solution

# Print GMRES convergence
print(solver.ksp.getIterationNumber())


# I'm not sure the number of GMRES iterations is the same as it was before.

# Aim - tidy up this code, and compare it to the previous version

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

