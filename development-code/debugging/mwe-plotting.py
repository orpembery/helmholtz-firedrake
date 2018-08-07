from firedrake import *

mesh = UnitSquareMesh(3,3)

V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)

v = TestFunction(V)

f = Coefficient(V)

f = 1.0

a = inner(u,v) * dx

L = inner(f,v) * dx

u_h = Function(V)

solve(a==L,u_h)

File("mwe.pvd").write(u_h)

import matplotlib.pyplot as plt

plot(mesh)
plt.show()

plot(u_h)
plt.show()
