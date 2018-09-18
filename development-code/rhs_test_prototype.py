import firedrake as fd
import numpy as np

from helmholtz.problems import HelmholtzProblem

N = 100

mesh = fd.UnitSquareMesh(N,N)

V = fd.FunctionSpace(mesh,"CG",1)

k = 10.0

prob = HelmholtzProblem(k,V)

prob.force_lu()

A_right = fd.as_matrix([[1.0,0.0],[0.0,1.0]])

alpha = np.ones(((N+1)**2,1))

prob.set_rhs_nbpc_paper(A_right,alpha)

prob.solve()
