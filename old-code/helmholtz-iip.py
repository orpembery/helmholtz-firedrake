# Modified from the firedrake `Simple Helmholtz equation' demo
from firedrake import *
from math import ceil # so that we can define the mesh size

# Define wavenumber
k = 50.0

# Define mesh size to eliminate pollution effect
mesh_size = ceil(k**(1.5))

# Create a mesh
mesh = UnitSquareMesh(mesh_size, mesh_size)

# Define function space - continuous piecewise linear
V = FunctionSpace(mesh, "CG", 1)

# Define trial and test functions on the space
u = TrialFunction(V)
v = TestFunction(V)

# Define right-hand side function - Gaussian approximation of point-source - gives circular waves
f = Function(V)
g = Function(V)
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



# Define sesquilinear form and antilinear functional
a = (inner(grad(u), grad(v)) - k**2*inner(u,v)) * dx - (1j* k * inner(u,v)) * ds
L =  inner(g,v)*ds#inner(f,v) * dx +

# Define numerical solution
u_h = Function(V)

# Solve using a direct LU solver
solve(a == L, u_h, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

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
