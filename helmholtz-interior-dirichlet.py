from firedrake import *

omega = Constant(25.0)
#d1 = Constant(sin(pi/3.0))
#d2 = Constant(cos(pi/3.0))
sigma = Constant(1.1)

mesh = UnitSquareMesh(150,150)

V = FunctionSpace(mesh, "CG", 1)


u = TrialFunction(V)
v = TestFunction(V)

x,y = SpatialCoordinate(mesh)
f = Function(V)
f.interpolate(1.0/sqrt(2.0*pi*sigma**2) * exp( -( ( x - 0.5)**2 + (y - 0.5)**2 )/(2.0*sigma**2)))


a = (dot(grad(u),grad(v)) - omega*omega * u * v )*dx
L = f * v * dx

u = Function(V)

solve(a==L, u,solver_parameters={'ksp_type': 'preonly','pc_type': 'lu'})
#solve(a==L,u)
#solve(a==L, u, solver_parameters={"ksp_type": "gmres"})


File("helmholtz-interior-dirichlet.pvd").write(u)
