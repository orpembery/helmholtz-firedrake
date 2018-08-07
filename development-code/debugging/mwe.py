from firedrake import *

mesh = UnitSquareMesh(10, 10)

V = FunctionSpace(mesh, "CG", 1)

g = Function(V)

nu = FacetNormal(mesh)

d = as_vector((1.0,0.0))

g.interpolate(dot(nu,d)) # something going wrong with dimension

