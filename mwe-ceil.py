from firedrake import *
from numpy import floor

mesh = UnitSquareMesh(10,10)

V = FunctionSpace(mesh,"CG",1)

u = TrialFunction(V)

v = TestFunction(V)

f = 1.0

x = SpatialCoordinate(mesh)

g = sign(10.0*real(x[0]))

a = (g * inner(u,v))*dx

L = inner(f,v) * dx

u_h  = Function(V)

solve(a==L,u_h)
