from firedrake import *

mesh = UnitSquareMesh(3,3)

V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)

v = TestFunction(V)

f = 1.0

x = SpatialCoordinate(mesh)

coeff = sign(x[0]-0.5) # this coeff fails

coeff_real = sign(real(x[0]-0.5)) # this coeff works

a = coeff * inner(u,v) * dx

L = inner(f,v) * dx

u_h = Function(V)

solve(a==L,u_h)
