# Modified from the firedrake `Simple Helmholtz equation' demo and https://www.firedrakeproject.org/demos/saddle_point_systems.py.html
from firedrake import *
from math import ceil # so that we can define the mesh size

# Define wavenumber
k = 10.0

# Define mesh size to eliminate pollution effect
mesh_size = ceil(k**(1.5))

# Create a mesh
mesh = UnitSquareMesh(mesh_size, mesh_size)

# Define function space - continuous piecewise linear
V = FunctionSpace(mesh, "CG", 1)

# Define trial and test functions on the space
u = TrialFunction(V)
v = TestFunction(V)

# Define right-hand side function - boundary condition corresponding to a plane wave
g = Coefficient(V)
x = SpatialCoordinate(mesh)

nu = FacetNormal(mesh)

d = as_vector([1.0/sqrt(2.0),1.0/sqrt(2.0)])

# Boundary condition
g=1j*k*exp(1j*k*dot(x,d))*(dot(d,nu)-1)

# Define coefficients
# Define function space for coefficients
V_A = TensorFunctionSpace(mesh, "CG", 1, symmetry=True)

A=Coefficient(V_A)

A=as_matrix([[1.0,0.0],[0.0,1.0]])

n = Coefficient(V)

n_centre=as_vector([0.5,0.5])
n = 0.5+abs(x - n_centre)**2

# Define sesquilinear form and antilinear functional
a = (inner(A * grad(u), grad(v)) - k**2 * inner(real(n) * u,v)) * dx - (1j* k * inner(u,v)) * ds # real(n) is just a failsafe
L =  inner(g,v)*ds

# Define numerical solution
u_h = Function(V)

# Define the problem with respect to which we will precondition

A_pre = Coefficient(V_A)

A_pre = A=as_matrix([[1.0,0.0],[0.0,1.0]])

n_pre = Coefficient(V)

n_pre = 1.0

# Define sesquilinear form and antilinear functional for preconditioning
a_pre = (inner(A_pre * grad(u), grad(v)) - k**2 * inner(real(n_pre) * u,v)) * dx - (1j* k * inner(u,v)) * ds # real(n) is just a failsafe


# These parameters are only for the initial solve, the main reason for which is to calculate the LU decomposition of the preconditioning problem
precon_parameters = {'ksp_type': 'preonly', # only do an LU decomposition
                     'pc_type': 'lu'}

A_mat_pre = assemble(a_pre, mat_type = 'aij') # assemble preconditioning problem

solver_precon = LinearSolver(A_mat_pre, solver_parameters = precon_parameters)

b = assemble(L) # assemble right-hand side

solver_precon.solve(u_h,b) # we're not fussed about doing anything with the solution at this stage

# This should extract the (operator corresponding to the) LU decomposition of the preconditioning problem
pc_solver_obj = solver_precon.ksp
B_not_needed, P_LU = pc_solver_obj.getOperators() # PETSc operator corresponding to the LU decomposition of P. B is not needed

# Now  do the same calculation as above (but with sesquilinear form a), but this time we pass in the preconditioner LU object

#A_mat = assemble(a, mat_type = 'aij') # assemble the monolithic matrix
A_mat = A_mat_pre # for now, preconditioned problem is the same as solving problem

gmres_parameters = {'ksp_type': 'gmres', # use GMRES
                    'ksp_norm_type': 'unpreconditioned'} # measure convergence in the unpreconditioned norm

solver = LinearSolver(A_mat,solver_parameters = gmres_parameters)

gmres_solver_obj = solver.ksp

A_mat_obj, not_needed = gmres_solver_obj.getOperators() # PETSc operators corresponding to A (and a preconditioner, but it doesn't matter what it is, because we're about to replace it

# The PETSc documentation (http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PCSetOperators.html) says you need to do the following as you're keeping A_obj, but it didn't work. I don't understand why.
#A_obj.PetscObjectReference()

gmres_solver_obj.setOperators(A_mat_obj,P_LU) # This should set the system matrix (or its action) as A_obj and the preconditioner (or its action) as P_LU

solver.solve(u_h,b) # again, not fussing about storing the solution

# Print GMRES convergence
print(solver.ksp.getIterationNumber())

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

# Note: In the PETSc documentation it seems there are two equivalent syntaxes: commands such as KSPgetOperators and PCgetOperators. I'm unclear on what the difference is between them. In the Firedrake tutorial on interfacing with PETSc (https://www.firedrakeproject.org/petsc-interface.html), the PCgetOperators syntax is used.
