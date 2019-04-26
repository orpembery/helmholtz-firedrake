import firedrake as fd
from helmholtz_firedrake.problems import HelmholtzProblem
from helmholtz_firedrake import utils
from matplotlib import pyplot as plt
import numpy as np
import sys

# This isn't in a formal pytest framework because the 'at' functionality
# isn't working yet in complex Firedrake.

mesh = fd.UnitSquareMesh(100,100)

V = fd.FunctionSpace(mesh,"CG",1)

k = 10.0

x = fd.SpatialCoordinate(mesh)

n = x[0] - 0.5

prob = HelmholtzProblem(k,V,n=n)

prob.n_min(0.1)

prob.plot_n()

plt.show()




