import helmholtz.problems as hh
import firedrake as fd
import numpy as np
import nearby_preconditioning.experiments as nbex

# nbpc test - throws right error with 13/14
def test_coeff_definition_error():
    """Test that a coeff with too many pieces is caught."""
    k = 20.0
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)

    num_pieces = 14
    noise_level = 0.5
    
    A_pre = fd.as_matrix([[1.0,2.0],[0.0,1.0]])
    A_stoch = nbex.PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,A_pre,[2,2])

    n_pre = 1.0
    n_stoch = nbex.PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,n_pre,[1])

    f = 1.0
    g = 1.0
    
    GMRES_its=nearby_preconditioning_test(V,k,A_pre,A_stoch,
                                n_pre,n_stoch,f,g,num_repeats)

    # If something's wrong, this should throw an error. If it exits
    # silently, only printing a message, should record nothing

    assert GMRES_its == []

# check if use exact precon always get one GMRES

# check h to mesh points

# generating matrix coeff - check spd and check random (and check scaalras random)
