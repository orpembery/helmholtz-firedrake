from firedrake import *
import helmholtz.problems as hh
import numpy as np

k = 20.0

h = k**(-0.5)

num_mesh = np.ceil(np.sqrt(2.0)/h)

mesh = UnitSquareMesh(num_mesh,num_mesh)

V = FunctionSpace(mesh,"CG",2)

x = SpatialCoordinate(mesh)

#n = (x[0] - 0.5)**2 + (x[1]-0.5)**2

#prob = hh.HelmholtzProblem(k,V,n=n)

prob = hh.HelmholtzProblem(k,V)

prob.f_g_plane_wave()

prob.force_lu()

prob.solve()

prob.plot()
