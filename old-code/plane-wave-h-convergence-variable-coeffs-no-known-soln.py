# Modified from the firedrake `Simple Helmholtz equation' demo
# This tests whether we get the correct convergence as we refine the mesh parameter
# Expect order h^2 convergence in L^2 norm and order h convergence in (weighted) H^1 norm
from firedrake import *
import csv # for writing output
from numpy import arange # for iterating over k

with open('planewave-errors-variable-coeffs-no-known-soln.csv', 'w', newline='') as errorfile:
    errorwriter = csv.writer(errorfile, delimiter=',')

    errorwriter.writerow(["L^2 error","H^1 error","Weighted H^1 error"])

    # Define wavenumber
    for k in arange(10.0,50.0,10.0):
        errorwriter.writerow(["k=",k])
        print(k)

        # First do it for the finest mesh

        # Create a mesh
        mesh_size = 2**9
        mesh_fine = UnitSquareMesh(mesh_size, mesh_size)

        # Define function space - continuous piecewise linear
        V_fine = FunctionSpace(mesh_fine, "CG", 1)

        # Define trial and test functions on the space
        u_fine = TrialFunction(V_fine)
        v_fine = TestFunction(V_fine)

        # Define right-hand side function - Gaussian approximation of point-source - gives circular waves
        f_fine = Function(V_fine)
        x_fine = SpatialCoordinate(mesh_fine)
        x_centre = 0.5
        y_centre = 0.5
        f_fine.interpolate(exp((-(k/pi)**2)*((x_fine[0]-x_centre)**2 + (x_fine[1]-y_centre)**2)))


        # Right-hand side g is the boundary condition given by a plane wave with direction d
        nu_fine = FacetNormal(mesh_fine)

        # Unsure if the following is the correct way to allow us to take a dot product with u
        d = as_vector([1.0/sqrt(2.0),1.0/sqrt(2.0)])


        # Define coefficients

        # Define function space for coefficients
        V_A_fine = TensorFunctionSpace(mesh_fine, "CG", 1, symmetry=True)

        A_fine=Coefficient(V_A_fine)

        A_fine=as_matrix([[1.0,0.0],[0.0,1.0]])

        n_fine = Coefficient(V_fine)

        n_centre=as_vector([0.5,0.5])
        n_fine = abs(x_fine - n_centre)**2

        # Define sesquilinear form and antilinear functional
        a_fine = (inner(A_fine * grad(u_fine), grad(v_fine)) - k**2 * inner(real(n_fine) * u_fine,v_fine)) * dx - (1j* k * inner(u_fine,v_fine)) * ds # real(n) is just a failsafe
        L_fine =  inner(f_fine,v_fine) * dx


        # Define numerical solution
        u_fine_h = Function(V_fine)

        # Solve using a direct LU solver
        solve(a_fine == L_fine, u_fine_h, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

        # Then do it for all the other mesh sizes

        for mesh_size_index in list(range(1,9,1)):

          # Create a mesh
            mesh_size = 2**mesh_size_index
            mesh = UnitSquareMesh(mesh_size, mesh_size)

            # Define function space - continuous piecewise linear
            V = FunctionSpace(mesh, "CG", 1)

            # Define trial and test functions on the space
            u = TrialFunction(V)
            v = TestFunction(V)

            # Define right-hand side function - Gaussian approximation of point-source - gives circular waves
            f = Function(V)
            x = SpatialCoordinate(mesh)
            x_centre = 0.5
            y_centre = 0.5
            f.interpolate(exp((-(k/pi)**2)*((x[0]-x_centre)**2 + (x[1]-y_centre)**2)))


            # Right-hand side g is the boundary condition given by a plane wave with direction d
            nu = FacetNormal(mesh)

            # Unsure if the following is the correct way to allow us to take a dot product with u
            d = as_vector([1.0/sqrt(2.0),1.0/sqrt(2.0)])


            # Define coefficients

            # Define function space for coefficients
            V_A = TensorFunctionSpace(mesh, "CG", 1, symmetry=True)

            A=Coefficient(V_A)

            A=as_matrix([[1.0,0.0],[0.0,1.0]])

            n = Coefficient(V)

            n=1

            # Define sesquilinear form and antilinear functional
            a = (inner(A * grad(u), grad(v)) - k**2 * inner(real(n) * u,v)) * dx - (1j* k * inner(u,v)) * ds # real(n) is just a failsafe
            L =  inner(f,v) * dx


            # Define numerical solution
            u_h = Function(V)

            # Solve using a direct LU solver
            solve(a == L, u_h, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

            # Compare to fine solution

            # This next bit is a hack to get them on the same mesh, based on code by Jack Betteridge. I'm not entirely sure how it works
            coords = interpolate(mesh_fine.coordinates, V)     
            u_h_coarse = Function(V_fine)
            u_h_coarse.dat.data[:] = u_h.at(coords.dat.data_ro)

            print(-mesh_size_index)
            #  print(sqrt(assemble(inner(u_true-u_h,u_true-u_h) * dx))) # L^2 error - seems to give a different answer - maybe when errornorm is passed a ufl expression it projects rather than interpolates?
            print(errornorm(u_h_fine,u_h_coarse,norm_type="L2")) # L^2 error
            #  print(sqrt(assemble((inner(grad(u_true-u_h),grad(u_true-u_h)) + inner(u_true-u_h,u_true-u_h))*dx))) # H^1 error 
            print(errornorm(u_h_fine,u_h_coarse,norm_type="H1")) # H^1 error
            print(sqrt(norm(u_h_fine-u_h_coarse,norm_type="H1")**2 + (k**2-1)*norm(u_h_fine-u_h_coarse,norm_type="L2")**2)) # Weighted H^1 error
            errorwriter.writerow(["h=2^",-mesh_size_index])
            errorwriter.writerow([errornorm(u_h_fine,u_h_coarse,norm_type="L2"),errornorm(u_h_fine,u_h_coarse,norm_type="H1"),sqrt(norm(u_h_fine-u_h_coarse,norm_type="H1")**2 + (k**2-1)*norm(u_h_fine-u_h_coarse,norm_type="L2")**2)]) # L^2 norm, H^2 norm, and weighted H^1 norm
            # It looks like the ratio of the L^2 error to the H^1 error decreases with decreasing h, but I think this is consistent(ish) with Wu's paper



