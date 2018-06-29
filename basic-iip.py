from firedrake import *
import math

# Set up mesh to satisfy h lesssim k^(-3.2)

k = 100.0

mesh_size = np.ceil(k^(1.5))

mesh = UnitSquareMesh(mesh_size,mesh_size)

# set up function spaces

V = FunctionSpace(mesh,"CG",1)

u = TrialFunction(V)

v = TestFunction(V)

# define right-hand side

f = function(V)

x,y = SpatialCoordinate(mesh)

x_centre = 0.5
y_centre = 0.5

# Gaussian forcing
f.interpolate(exp(-(k/math.pi)^2*((x-x_centre)^2 + (y-y_centre)^2)))

# Define bilinear form and right-hand side

a = (inner(grad(u),grad(v)) - k*inner(u,v))*dx - jk*(inner(u,v))*ds
L = f * conj(v) * dx

# redefine u to hold the solution

u = Function(V)

# direct solve via LU

solve(a==L, solver_parameters={'ksp_type' : 'preonly',
                               'pc_type' : 'lu'})

# write solution to file
File("helmholtz-iip.pvd").write(u)

# plot with matplotlib

try:
  import matplotlib.pyplot as plt
except:
  warning("Matplotlib not imported")

try:
  plot(u)
except Exception as e:
  warning("Cannot plot figure. Error msg: '%s'" % e)

  # and show it

try:
  plt.show()
except Exception as e:
  warning("Cannot show figure. Error msg: '%s'" % e)
