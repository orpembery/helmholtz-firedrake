import numpy as np
import firedrake as fd
import helmholtz.problems as hh
#from helmholtz.problems import HelmholtzProblem
#from src.helmholtz_firedrake import HelmholtzProblem
from matplotlib import pyplot as plt

k_range = [10.0,12.0,20.0,30.0,40.0]#[10.0,12.0]#[20.0,40.0]
num_levels = 2
    
log_err_L2 = np.empty((num_levels,len(k_range)))
log_err_H1 = np.empty((num_levels,len(k_range)))

tolerance = 0.05

for ii_k in range(len(k_range)):
    k = k_range[ii_k]
    print(k)
    num_points = np.ceil(np.sqrt(2.0) * k**(1.5)) * 2.0**np.arange(float(num_levels))

    log_h = np.log(np.sqrt(2.0) * 1.0 / num_points)
    for ii_points in range(num_levels):
        print(ii_points)
        # Coarsest grid has h ~ k^{-1.5}, and then do uniform refinement
        mesh = fd.UnitSquareMesh(num_points[ii_points],num_points[ii_points])
        V = fd.FunctionSpace(mesh, "CG", 1)

        x = fd.SpatialCoordinate(mesh)
        nu = fd.FacetNormal(mesh)
        d = fd.as_vector([1.0/np.sqrt(2.0),1.0/np.sqrt(2.0)])
        exact_soln = fd.exp(1j * k * fd.dot(x,d))
        f = 0.0
        g = 1j*k*fd.exp(1j*k*fd.dot(x,d))*(fd.dot(d,nu)-1)

        prob = hh.HelmholtzProblem(k,V,f=f,g=g)

        prob.solve()

        log_err_L2[ii_points,ii_k] = np.log(fd.norms.errornorm(exact_soln,prob.u_h,norm_type="L2"))
        log_err_H1[ii_points,ii_k] = np.log(fd.norms.errornorm(exact_soln,prob.u_h,norm_type="H1"))

    #plt.plot(log_h,log_err_L2[:,ii_k])

    #plt.plot(log_h,log_err_H1[:,ii_k])

    #plt.show()

    fit_L2 = np.polyfit(log_h,log_err_L2[:,ii_k],deg=1)[0]

    fit_H1 = np.polyfit(log_h,log_err_H1[:,ii_k],deg=1)[0]

    print(fit_L2)

    print(fit_H1)

    assert abs(fit_L2 - 2.0) <= tolerance

    assert abs(fit_H1 - 1.0) <= tolerance

#print(log_h.shape)

#print(log_err_L2.shape)



