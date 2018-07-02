# Modified from the firedrake `Simple Helmholtz equation' demo
# This tests whether we get the correct convergence as we refine the mesh parameter
# Expect order h^2 convergence in L^2 norm and order h convergence in (weighted) H^1 norm
from firedrake import *
import csv # for writing output
from numpy import arange # for iterating over k

with open('planewave-errors.csv', 'w', newline='') as errorfile:
  errorwriter = csv.writer(errorfile, delimiter=',')

  errorwriter.writerow(["L^2 error","H^1 error","Weighted H^1 error"])

  # Define wavenumber
  for k in arange(10.0,50.0,10.0):
    errorwriter.writerow(["k=",k])
    print(k)
    
    for mesh_size_index in list(range(1,10,1)):

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
      g = Function(V)
      x = SpatialCoordinate(mesh)
      x_centre = 0.5
      y_centre = 0.5
      #f.interpolate(exp((-(k/pi)**2)*((x[0]-x_centre)**2 + (x[1]-y_centre)**2)))
      #f.interpolate(1.0)

      # Right-hand side g is the boundary condition given by a plane wave with direction d
      nu = FacetNormal(mesh)

      # Unsure if the following is the correct way to allow us to take a dot product with u
      d = as_vector([1.0/sqrt(2.0),1.0/sqrt(2.0)])

      # Boundary condition
      g=1j*k*exp(1j*k*dot(x,d))*(dot(d,nu)-1)
      f=0.0*x



      # Define sesquilinear form and antilinear functional
      a = (inner(grad(u), grad(v)) - k**2*inner(u,v)) * dx - (1j* k * inner(u,v)) * ds
      L =  inner(g,v)*ds #inner(f,v) * dx +

      # Define numerical solution
      u_h = Function(V)

      # Solve using a direct LU solver
      solve(a == L, u_h, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

      # Compare to true solution
      u_true = Function(V)
      u_true.interpolate(exp(1j*k*dot(x,d)))

      print(-mesh_size_index)
      #  print(sqrt(assemble(inner(u_true-u_h,u_true-u_h) * dx))) # L^2 error - seems to give a different answer - maybe when errornorm is passed a ufl expression it projects rather than interpolates?
      print(errornorm(exp(1j*k*dot(x,d)),u_h,norm_type="L2")) # L^2 error
      #  print(sqrt(assemble((inner(grad(u_true-u_h),grad(u_true-u_h)) + inner(u_true-u_h,u_true-u_h))*dx))) # H^1 error 
      print(errornorm(exp(1j*k*dot(x,d)),u_h,norm_type="H1")) # H^1 error
      print(sqrt(norm(exp(1j*k*dot(x,d))-u_h,norm_type="H1")**2 + (k**2-1)*norm(exp(1j*k*dot(x,d))-u_h,norm_type="L2")**2)) # Weighted H^1 error
      errorwriter.writerow(["h=2^",-mesh_size_index])
      errorwriter.writerow([errornorm(exp(1j*k*dot(x,d)),u_h,norm_type="L2"),errornorm(exp(1j*k*dot(x,d)),u_h,norm_type="H1"),sqrt(norm(exp(1j*k*dot(x,d))-u_h,norm_type="H1")**2 + (k**2-1)*norm(exp(1j*k*dot(x,d))-u_h,norm_type="L2")**2)]) # L^2 norm, H^2 norm, and weighted H^1 norm
      # It looks like the ratio of the L^2 error to the H^1 error decreases with decreasing h, but I think this is consistent(ish) with Wu's paper



