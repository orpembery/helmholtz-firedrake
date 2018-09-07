from firedrake import *
from helmholtz import coefficients
import numpy as np

mesh = UnitSquareMesh(10,10)

x_centre = (0.5,0.5)

N = 2

delta = 0.5

series_term_lower = 0.1

series_term_upper = 0.5

r_max = np.sqrt(2.0)/2.0

coeff = coefficients.SmoothNTCoeff(mesh,x_centre,N,delta,series_term_lower,series_term_upper,r_max)
