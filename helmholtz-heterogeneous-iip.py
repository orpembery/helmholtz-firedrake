# Modified from the firedrake `Simple Helmholtz equation' demo
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

# Right-hand side g is the boundary condition given by a plane wave with direction d
x = SpatialCoordinate(mesh)

nu = FacetNormal(mesh)

d = as_vector([1.0/sqrt(2.0),1.0/sqrt(2.0)])

g=1j*k*exp(1j*k*dot(x,d))*(dot(d,nu)-1)

# Define coefficients

# Define function spaces for coefficients
V_A = TensorFunctionSpace(mesh, "DG", 0, symmetry=True)

V_n = FunctionSpace(mesh, "DG", 0)

A=Function(V_A)

A_expr = as_matrix([[1.0,0.0],[0.0,1.0]])

A = A_expr # This isn't right, as A is now just a double

# Experiment to see if I can use a different space for coefficients
n = Function(V)

n.interpolate(Expression(1))

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
