from firedrake import *

omega = Constant(5)
d1 = Constant(1)
d2 = Constant(0)


mesh = UnitSquareMesh(64,64)

FS1 = FunctionSpace(mesh, "CG", 1)
FS2 = FunctionSpace(mesh, "CG", 1)
V = FS1 * FS2

uR, uI = TrialFunctions(V)
vR, vI = TestFunctions(V)

x,y = SpatialCoordinate(mesh)
fR = Function(FS1).interpolate(cos(omega*(x*d1 + y*d2)))
fI = Function(FS2).interpolate(sin(omega*(x*d1 + y*d2)))
