# Modified from the firedrake `Simple Helmholtz equation' demo
from firedrake import *

# Define wavenumber
k = 10.0

# Create a mesh
mesh = UnitSquareMesh(10*k, 10*k)

# Define function space - continuous piecewise linear
V = FunctionSpace(mesh, "CG", 1)

# Define trial and test functions on the space
u = TrialFunction(V)
v = TestFunction(V)

# Define right-hand side function - Gaussian approximation of point-source - gives circular waves
f = Function(V)
x, y = SpatialCoordinate(mesh)
x_centre = 0.5
y_centre = 0.5
f.interpolate(exp((-(10.1/pi)**2)*((x-x_centre)**2 + (y-y_centre)**2)))
#f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))  # not Gaussian atm


# Define sesquilinear form and antilinear functional
a = (inner(grad(u), grad(v)) - k**2*inner(u,v)) * dx - (1j* k * inner(u,v)) * ds
L = (inner(f,v)) * dx

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
