from firedrake import *

mesh = UnitSquareMesh(10,10)

V = FunctionSpace(mesh, "DG", 1)

u = TrialFunction(V)

v = TestFunction(V)

x = SpatialCoordinate(mesh)

coeff = conditional(le(real(x[0]),0.5),1.0,conditional(le(real(x[1]),0.5),1.0/2.0,1.0/3.0))

a = inner(real(coeff)*u,v)*dx

L = inner(1.0,v)*dx

u_h = Function(V)

solve(a==L,u_h)

import matplotlib.pyplot as plt
plot(u_h,num_sample_points=1)
plt.show()
