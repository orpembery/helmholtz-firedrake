from firedrake import *
from matplotlib import pyplot as plt
import helmholtz_firedrake as hh
import numpy as np

mesh = UnitSquareMesh(100,100)

k = 10.0

V = FunctionSpace(mesh,"CG",1)

f = 0.0

g = 1.0

class n_simple(object):
    """KISS."""

    def __init__(self):
        self.n = Constant(1.0)

    def sample(self):
        self.n.assign(np.random.random_sample() + 0.5)

class A_simple(object):
    """KISS."""

    def __init__(self):
        self.A = Constant(np.array([[1.0,0.0],[0.0,1.0]]))

    def sample(self):
        self.A.assign(np.array([[1.0,0.0],[0.0,1.0]]))

A_stoch = A_simple()

n_stoch = n_simple()


        
sprob = hh.StochasticHelmholtzProblem(k,V,A_stoch,n_stoch)

sprob.solve()

plot(sprob.u_h)

plt.show()

sprob.sample()

sprob.solve()

plot(sprob.u_h)

plt.show()

sprob.sample()

if False:
    prob = hh.HelmholtzProblem(k=k,V=V,n=n,f=f,g=g)

    prob.solve()

    plot(prob.u_h)

    plt.show()

    n.assign(2.0)

    prob.solve()

    plot(prob.u_h)

    plt.show()

    
    u = TrialFunction(V)

    v = TestFunction(V)

    x = SpatialCoordinate(mesh)

    f = x[0] - x[0]

    a = inner(u,v)*dx

    L = inner(real(f),v)*dx

    u_h = Function(V)

    solve(a==L,u_h)

    plot(u_h,num_sample_points=1)

    plt.show()

    f = x[1]

    L = inner(real(f),v)*dx

    solve(a==L,u_h)

    plot(u_h,num_sample_points=1)

    plt.show()

