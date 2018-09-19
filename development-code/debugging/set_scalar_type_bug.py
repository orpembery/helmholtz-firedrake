from firedrake import *

mesh = UnitSquareMesh(2,2)
V = FunctionSpace(mesh,"CG",1)
func = Function(V)
func.at([0.5,0.5])
