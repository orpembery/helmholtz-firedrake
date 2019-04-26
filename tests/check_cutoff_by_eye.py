import firedrake as fd
from helmholtz_firedrake import utils
from matplotlib import pyplot as plt
import numpy as np
import sys

# This isn't in a formal pytest framework because the 'at' functionality
# isn't working yet in complex Firedrake.

selection = int(sys.argv[1])

if selection == 0:
    """Tests that the f function works correctly."""
    # I can't tell if this isn't working correctly, or if we just see
    # rounding error. I think it's just rounding error.
    num_cells = 1000

    mesh = fd.UnitIntervalMesh(num_cells)

    V = fd.FunctionSpace(mesh,"CG",1)

    v = fd.Function(V)

    x = fd.SpatialCoordinate(mesh)

    #    for eps in [1.0,0.5,0.1,0.05]:
    v.interpolate(utils.f(x[0]))
    fd.plot(v)
    print("Any NaNs?")
    print(np.any(np.isnan(v.dat.data_ro)))
    #print(v.dat.data_ro)
    plt.show()

if selection == 1:
    """Tests that the 1-d transition function works correctly."""
    # I can't tell if this isn't working correctly, or if we just see
    # rounding error. I think it's just rounding error.
    num_cells = 3000

    mesh = fd.UnitIntervalMesh(num_cells)

    V = fd.FunctionSpace(mesh,"DG",0)

    v = fd.Function(V)

    x = fd.SpatialCoordinate(mesh)

    for eps in [1.0,0.5,0.1,0.05]:
        print('eps =',eps)
        v.interpolate(utils.one_d_transition(x[0],eps))
        fd.plot(v,num_sample_points=1)
        print("Any NaNs?")
        print(np.any(np.isnan(v.dat.data_ro)))
        #print(v.dat.data_ro)
        plt.show()
        for val in [0.0,1.0]:
            print('Where is transition function equal to',str(val),'?')
            v.interpolate(fd.conditional(fd.eq(utils.one_d_transition(x[0],eps),val),1.0,0.0))
            fd.plot(v,num_sample_points=1)
            plt.show()
        
elif selection == 2:
    """Tests that the 1-d cutoff function works correctly."""
    # I can't tell if this isn't working correctly, or if we just see
    # rounding error. I think it's just rounding error.
    num_cells = 1000

    mesh = fd.UnitIntervalMesh(num_cells)

    V = fd.FunctionSpace(mesh,"DG",0)

    v = fd.Function(V)

    x = fd.SpatialCoordinate(mesh)

    y = 0.5

    for w in [0.05,0.1,0.15]:

        for eps in [0.1,0.05]:
            print(w)
            print(eps)
            v.interpolate(utils.one_d_cutoff(x[0],y,w,eps))
            fd.plot(v,num_sample_points=1)
            print("Any NaNs?")
            print(np.any(np.isnan(v.dat.data_ro)))
            plt.show()

elif selection == 3:
    """Tests that the n-d cutoff function works correctly."""
    # I can't tell if this isn't working correctly, or if we just see
    # rounding error. I think it's just rounding error.
    num_cells = 100

    mesh = fd.UnitSquareMesh(num_cells,num_cells)

    V = fd.FunctionSpace(mesh,"DG",0)

    v = fd.Function(V)

    x = fd.SpatialCoordinate(mesh)

    y = (0.5,0.5)

    w_options = [0.1,0.15,0.2]

    eps_options = [0.01,0.05]
    
    for w in [(ii,jj) for ii in w_options for jj in w_options]:
        
        for eps in [(ii,jj) for ii in eps_options for jj in eps_options]:
            print("Width of region")
            print(w)
            print("size of cutoff")
            print(eps)
            v.interpolate(utils.nd_cutoff(x,y,w,eps))
            fd.plot(v,num_sample_points=1)
            print("Any NaNs?")
            print(np.any(np.isnan(v.dat.data_ro)))
            plt.show()

elif selection == 4:
    """Tests that the n-d cutoff function works correctly."""
    # I can't tell if this isn't working correctly, or if we just see
    # rounding error. I think it's just rounding error.
    num_cells = 100

    mesh = fd.UnitSquareMesh(num_cells,num_cells)

    V = fd.FunctionSpace(mesh,"DG",0)

    v = fd.Function(V)

    x = fd.SpatialCoordinate(mesh)

    y = (0.5,0.5)

    w_options = [0.1,0.15,0.2]

    eps_options = [0.01,0.05]
    
    for w in [(ii,jj) for ii in w_options for jj in w_options]:
        
        for eps in [(ii,jj) for ii in eps_options for jj in eps_options]:
            print("Width of region")
            print(w)
            print("size of cutoff")
            print(eps)
            v.interpolate(fd.conditional(fd.eq(utils.nd_cutoff(x,y,w,eps),1.0),1.0,0.0))
            fd.plot(v,num_sample_points=1)
            print("Any NaNs?")
            print(np.any(np.isnan(v.dat.data_ro)))
            plt.show()

if selection == 5:
    """Tests that the f function works correctly."""
    # I can't tell if this isn't working correctly, or if we just see
    # rounding error. I think it's just rounding error.
    num_cells = 1000

    mesh = fd.UnitIntervalMesh(num_cells)

    V = fd.FunctionSpace(mesh,"DG",0)

    v = fd.Function(V)

    x = fd.SpatialCoordinate(mesh)

    #    for eps in [1.0,0.5,0.1,0.05]:
    for val in [0.0,1.0]:
        print('Where is f equal to',str(val),'?')
        v.interpolate(fd.conditional(fd.eq(utils.f(x[0]),val),1.0,0.0))
        fd.plot(v,num_sample_points=1)
        print("Any NaNs?")
        print(np.any(np.isnan(v.dat.data_ro)))
        #print(v.dat.data_ro)
        plt.show()
