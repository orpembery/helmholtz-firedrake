from firedrake import *

omega = Constant(5)
d1 = Constant(1)
d2 = Constant(0)


mesh = UnitSquareMesh(100,100)

FS1 = FunctionSpace(mesh, "CG", 1)
FS2 = FunctionSpace(mesh, "CG", 1)
V = FS1 * FS2

uR, uI = TrialFunctions(V)
vR, vI = TestFunctions(V)

x,y = SpatialCoordinate(mesh)
fR = Function(FS1).interpolate(cos(omega*(x*d1 + y*d2)))
fI = Function(FS2).interpolate(sin(omega*(x*d1 + y*d2)))

a = (dot(grad(uR),grad(vR)) - omega*omega * uR * vR + dot(grad(uI),grad(vI)) - omega*omega * uI * vI)*dx# - omega * (uI * vR - uR * vI)*ds
L = (fR * vR + fI * vI)*dx

w = Function(V)

#solve(a==L, w,solver_parameters={'ksp_type': 'preonly','pc_type': 'lu'})
#solve(a==L,w)
solve(a==L, w, solver_parameters={"ksp_type": "gmres"})
uR, uI = w.split()

File("initial_helmholtz.pvd").write(uR)
