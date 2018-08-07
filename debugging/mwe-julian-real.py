from firedrake import *
import ufl

mesh = UnitSquareMesh(10,10)

x = SpatialCoordinate(mesh)

domain = SubDomainData(ufl.And(0.0 <= x[0],x[0] <= 1.0))
