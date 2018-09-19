from firedrake import *
from helmholtz.coefficients import PiecewiseConstantCoeffGenerator as pw
import numpy as np
from matplotlib import pyplot as plt
mesh1 = UnitSquareMesh(10,10)
mesh2 = UnitSquareMesh(20,20)
mesh3 = UnitSquareMesh(10,10)
mesh4 = UnitSquareMesh(20,20)
np.random.seed(1)
coeff1 = pw(mesh1,5,0.1,1.0,[1])#as_matrix([[1.0,0.0],[0.0,1.0]]),[2,2])
coeff3 = pw(mesh1,5,0.1,1.0,[1])#as_matrix([[1.0,0.0],[0.0,1.0]]),[2,2])
np.random.seed(1)
coeff2 = pw(mesh2,5,0.1,1.0,[1])#as_matrix([[1.0,0.0],[0.0,1.0]]),[2,2])
coeff4 = pw(mesh2,5,0.1,1.0,[1])#as_matrix([[1.0,0.0],[0.0,1.0]]),[2,2])

#print("coeff1")
#print(type(coeff1.coeff))

#print("coeff2")
#print(type(coeff2.coeff))

V1 = FunctionSpace(mesh1,"DG",0)
V2 = FunctionSpace(mesh2,"DG",0)

#V_plot = FunctionSpace(mesh2,"DG",0)

fun1 = Function(V1)

fun2 = Function(V2)

fun3 = Function(V1)

fun4 = Function(V2)

fun1.interpolate(coeff1.coeff)

fun2.interpolate(coeff2.coeff)

fun3.interpolate(coeff3.coeff)

fun4.interpolate(coeff4.coeff)

plot(fun1)

plt.show()

plot(fun2)

plt.show()

plot(fun3)

plt.show()

plot(fun4)

plt.show()
