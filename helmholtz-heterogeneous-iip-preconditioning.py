# Modified from the firedrake `Simple Helmholtz equation' demo and https://www.firedrakeproject.org/demos/saddle_point_systems.py.html
from firedrake import *
from math import ceil # so that we can define the mesh size

# Define wavenumber
k = 40.0

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

# Boundary condition - a plane wave impedance boundary condition on the `incoming' edges, and a zero impedance condition (i.e. outgoing waves) on the other boundaries
g = conditional(dot(d,nu) < 0.0,1j*k*exp(1j*k*dot(x,d))*(dot(d,nu)-1),0.0)

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


# The following code courtesy of Lawrence Mitchell - it assumes the preconditioner doesn't change - for QMC/MCMC would need to do something else I suspect (i.e. call on Lawrence again :P)

problem = LinearVariationalProblem(a, L, u_h, aP=a_pre, constant_jacobian=False)
solver = LinearVariationalSolver(problem, solver_parameters={"ksp_type": "gmres",
                                                            "mat_type": "aij",
                                                            "pmat_type": "aij",
                                                            "snes_lag_preconditioner": -1,
                                                            "pc_type": "lu",
                                                            "ksp_reuse_preconditioner": True,
                                                            'ksp_norm_type': 'unpreconditioned'})

# If this was going in a loop:
#for i in range(_):
# would also change parameters
#   solver.solve()

solver.solve()

# Print GMRES convergence - currently doesn't work
print(solver.snes.ksp.getIterationNumber())

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
