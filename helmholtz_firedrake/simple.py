# Yes, it's a hack.

import sys
print(sys.argv)
sys.argv.append('-log_view')
sys.argv.append('-history')
sys.argv.append('tmp.txt')
print(sys.argv)

from firedrake import *
from helmholtz_firedrake.problems import HelmholtzProblem



mesh = UnitSquareMesh(20,20)

V = FunctionSpace(mesh,"CG",1)

prob = HelmholtzProblem(10.0,V,A_pre=as_matrix([[1.0,0.0],[0.0,1.0]]),n_pre=1.0)

for ii in range(10):

    print(ii)
    
    prob.solve()
