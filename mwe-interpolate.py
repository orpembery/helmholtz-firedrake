from firedrake import *

mesh = UnitSquareMesh(2,2)

V = FunctionSpace(mesh,"DG",0)

n = Function(V)

n.interpolate(Expression(1))
