from firedrake import *
from matplotlib import pyplot as plt

mesh = UnitSquareMesh(2,2)

V = FunctionSpace(mesh,"CG",1)

u = TrialFunction(V)

v = TestFunction(V)

x = SpatialCoordinate(mesh)

f = x[0]

a = inner(u,v)*dx

L = inner(real(f),v)*dx

u_h = Function(V)

solve(a==L,u_h)

plot(u_h,num_sample_points=1)

plt.show()

f = x[1]

L = inner(real(f),v)*dx

solve(a==L,u_h)

plot(u_h,num_sample_points=1)

plt.show()

