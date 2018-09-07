import firedrake as fd
import helmholtz.problems as hh
import helmholtz.coefficients as coeff
import helmholtz.utils as hh_utils
import numpy as np

def nearby_preconditioning_experiment(V,k,A_pre,A_stoch,n_pre,n_stoch,f,g,
                                num_repeats):
    """For a given preconditioning Helmholtz problem, performs a test of
    the effectiveness of nearby preconditioning.

    For a given preconditioning Helmholtz problem, and given methods for
    generating realisations of Helmholtz problems with random field
    coefficients, generates realisations of the Helmholtz problems, and
    then uses the preconditioner to perform preconditioned GMRES. Then
    records the number of GMRES iterations needed to acheive
    convergence.

    Parameters:

    V - see HelmholtzProblem

    k - see HelmholtzProblem

    A_pre - see HelmholtzProblem

    A_stoch - see StochasticHelmholtzProblem

    n_pre - see HelmholtzProblem

    n_stoch - see StochasticHelmholtzProblem

    f - see HelmholtzProblem

    g - see HelmholtzProblem

    num_repeats - int, specifying the number of realisations to take.


    Returns: list of ints of length num_repeats, giving the number of
    GMRES iterations for the different realisations.
    """

    prob = hh.StochasticHelmholtzProblem(
        k=k, V=V, A_stoch=A_stoch, n_stoch=n_stoch,
        **{"A_pre": A_pre, "n_pre" : n_pre, "f" : f, "g" : g})

    all_GMRES_its = []

    for ii_repeat in range(num_repeats):
        try:
            prob.solve()
        except RecursionError:
            print("Suffered a Python RecursionError.\
            Have you specified something using a big loop in UFL?\
            Aborting all further solves.")
            break
            
        all_GMRES_its.append(prob.GMRES_its)

        prob.sample()

    return all_GMRES_its

def nearby_preconditioning_piecewise_experiment_set(
        A_pre_type,n_pre_type,num_pieces,seed,num_repeats,
        k_list,h_list,noise_master_level_list,noise_modifier_list,
        save_location):
    """Test nearby preconditioning for a range of parameter values.

    Performs nearby preconditioning tests for a range of values of k,
    the mesh size h, and the size of the random noise (which can be
    specified in terms of k and h). The random noise is piecewise
    constant on a grid unrelated to the finite-element mesh.

    Parameters:

    A_pre_type - string - options are 'constant', giving A_pre =
    [[1.0,0.0],[0.0,1.0]].

    n_pre_type - string - options are 'constant', giving n_pre = 1.0.

    num_pieces - see
        helmholtz.coefficients.PieceWiseConstantCoeffGenerator.

    seed - see StochasticHelmholtzProblem.

    num_repeats - see nearby_preconditioning_test.

    k_list - list of positive floats - the values of k for which we will
    run experiments.

    h_list - list of 2-tuples; in each tuple (call it t) t[0] should be
    a positive float and t[1] should be a float. These specify the
    values of the mesh size h for which we will run experiments. h =
    t[0] * k**t[1].

    noise_master_level_list - list of 2-tuples, where each entry of the
    tuple is a positive float.  This defines the values of base_noise_A
    and base_noise_n to be used in the experiments. Call a given tuple
    t. Then base_noise_A = t[0] and base_noise_n = t[1].

    noise_modifier_list - list of 4-tuples; the entries of each tuple
    should be floats. Call a given tuple t. This modifies the base noise
    so that the L^\infty norms of A and n are less than or equal to
    (respectively) base_noise_A * h**t[0] * k**t[1] and base_noise_n *
    h**t[2] * k**t[3].

    save_location - string specifying the absolute path for the the
    folder in which to save the .csv output files.
    """

    if not(isinstance(A_pre_type,str)):
        raise TypeError("Input A_pre_type should be a string")
    elif A_pre_type is not "constant":
        raise HelmholtzNotImplementedError(
            "Currently only implemented A_pre_type = 'constant'.")

    if not(isinstance(n_pre_type,str)):
        raise TypeError("Input n_pre_type should be a string")
    elif n_pre_type is not "constant":
        raise HelmholtzNotImplementedError(
            "Currently only implemented n_pre_type = 'constant'.")

    if not(isinstance(k_list,list)):
        raise TypeError("Input k_list should be a list.")
    elif any(not(isinstance(k,float)) for k in k_list):
        raise TypeError("Input k_list should be a list of floats.")
    elif any(k <= 0 for k in k_list):
        raise TypeError(
            "Input k_list should be a list of positive floats.")

    if not(isinstance(h_list,list)):
        raise TypeError("Input h_list should be a list.")
    elif any(not(isinstance(h_tuple,tuple)) for h_tuple in h_list):
        raise TypeError("Input h_list should be a list of tuples.")
    elif any(len(h_tuple) is not 2 for h_tuple in h_list):
        raise TypeError("Input h_list should be a list of 2-tuples.")
    elif any(not(isinstance(h_tuple[0],float)) for h_tuple in h_list)\
             or any(h_tuple[0] <= 0 for h_tuple in h_list):
        raise TypeError(
            "The first item of every tuple in h_list\
            should be a positive float.")
    elif any(not(isinstance(h_tuple[1],float)) for h_tuple in h_list):
        raise TypeError(
            "The second item of every tuple in h_list should be a float.")

    if not(isinstance(noise_master_level_list,list)):
        raise TypeError(
            "Input noise_master_level_list should be a list.")
    elif any(not(isinstance(noise_tuple,tuple))
             for noise_tuple in noise_master_level_list):
        raise TypeError(
            "Input noise_master_level_list should be a list of tuples.")
    elif any(len(noise_tuple) is not 2
             for noise_tuple in noise_master_level_list):
        raise TypeError(
            "Input noise_master_level_list should be a list of 2-tuples.")
    elif any(any(not(isinstance(noise_tuple[i],float))
                 for i in range(len(noise_tuple)))
             for noise_tuple in noise_master_level_list):
        raise TypeError(
            "Input noise_master_level_list\
            should be a list of 2-tuples of floats.")

    if not(isinstance(noise_modifier_list,list)):
        raise TypeError("Input noise_modifier_list should be a list.")
    elif any(not(isinstance(mod_tuple,tuple))
             for mod_tuple in noise_modifier_list):
        raise TypeError(
            "Input noise_modifier_list should be a list of tuples.")
    elif any(len(mod_tuple) is not 4 for mod_tuple in noise_modifier_list):
        raise TypeError(
            "Input noise_modifier_list should be a list of 4-tuples.")
    elif any(any(not(isinstance(mod_tuple[i],float))
                 for i in range(len(mod_tuple)))
             for mod_tuple in noise_modifier_list):
        raise TypeError(
            "Input noise_modifier_list\
            should be a list of 4-tuples of floats.")

    if A_pre_type is "constant":
        A_pre = fd.as_matrix([[1.0,0.0],[0.0,1.0]])

    if n_pre_type is "constant":
        n_pre = 1.0
    
    for k in k_list:
        for h_tuple in h_list:
            h = h_tuple[0] * k**h_tuple[1]
            mesh_points = hh_utils.h_to_mesh_points(h)
            mesh = fd.UnitSquareMesh(mesh_points,mesh_points)
            V = fd.FunctionSpace(mesh, "CG", 1)

            f = 0.0
            d = fd.as_vector([1.0/fd.sqrt(2.0),1.0/fd.sqrt(2.0)])
            x = fd.SpatialCoordinate(mesh)
            nu = fd.FacetNormal(mesh)
            g=1j*k*fd.exp(1j*k*fd.dot(x,d))*(fd.dot(d,nu)-1)

            for noise_master in noise_master_level_list:
                A_noise_master = noise_master[0]
                n_noise_master = noise_master[1]

                for modifier in noise_modifier_list:

                    A_modifier = h ** modifier[0] * k**modifier[1]
                    n_modifier = h ** modifier[2] * k**modifier[3]
                    A_noise_level = A_noise_master * A_modifier
                    n_noise_level = n_noise_master * n_modifier
                    A_stoch = coeff.PiecewiseConstantCoeffGenerator(
                        mesh,num_pieces,A_noise_level,A_pre,[2,2])
                    n_stoch = coeff.PiecewiseConstantCoeffGenerator(
                        mesh,num_pieces,n_noise_level,n_pre,[1])
                    np.random.seed(seed)
                    
                    GMRES_its = nearby_preconditioning_experiment(
                        V,k,A_pre,A_stoch,n_pre,n_stoch,f,g,num_repeats)

                    hh_utils.write_GMRES_its(
                        GMRES_its,save_location,
                        {'k' : k,
                         'h_tuple' : h_tuple,
                         'num_pieces' : num_pieces,
                         'A_pre_type' : A_pre_type,
                         'n_pre_type' : n_pre_type,
                         'noise_master' : noise_master,
                         'modifier' : modifier,
                         'num_repeats' : num_repeats
                         }
                        )

def nearby_preconditioning_experiment_gamma(k_range,n_lower_bound,n_var_base,
                                      n_var_k_power_range,num_repeats):
    """Tests the effectiveness of nearby preconditioning for a
    homogeneous but gamma-distributed random refractive index.

    This is an initial version - it holds the mean of the refractive
    index constant = 1, but then changes the variance of the refractive
    index.
    """
    
    for k in k_range:

        num_points = hh_utils.h_to_mesh_points(k**(-1.5))
        
        mesh = fd.UnitSquareMesh(num_points,num_points)

        V = fd.FunctionSpace(mesh, "CG", 1)
        
        for n_var_k_power in n_var_k_power_range:
            print(k)
            print(n_var_k_power)
            n_var = n_var_base * k**n_var_k_power
            
            # Ensure Gamma variates have mean 1 - n_lower_bound and
            # variance n_var
            scale = n_var / (1.0 - n_lower_bound)
            shape = (1.0 - n_lower_bound)**2 / n_var
            
            n_stoch = coeff.GammaConstantCoeffGenerator(
                shape,scale,n_lower_bound)

            n_pre = 1.0
            f = 0.0
            g = 1.0
            
            GMRES_its = nearby_preconditioning_test(
                V,k,A_pre=fd.as_matrix([[1.0,0.0],[0.0,1.0]]),
                A_stoch=None,n_pre=n_pre,n_stoch=n_stoch,
                f=f,g=g,num_repeats=num_repeats)

            save_location =\
                "/home/owen/Documents/code/helmholtz-firedrake/output/testing/"

            info = {"function" : "nearby_preconditioning_test_gamma",
                    "k" : k,
                    "h" : "k**(-1.5)",
                    "n_var_base" : n_var_base,
                    "n_var_k_power" : n_var_k_power,
                    "n_lower_bound" : n_lower_bound,
                    "scale" : scale,
                    "shape" : shape,
                    "f" : f,
                    "g" : g,
                    "n_pre" : n_pre,
                    "num_repeats" : num_repeats
                    }
                    
            
            hh_utils.write_GMRES_its(GMRES_its,save_location,info)

def special_rhs_for_paper_experiment(k_list,num_system,num_rhs,
                                     fine_grid_refinement):
    """Tests to see if the required condition in the paper holds.

    For a variety of different Helmholtz systems, and a variety of
    different special right-hand sides (given by F(v) = (A_rhs
    \grad(\sum_i \alpha_i \phi_i),\grad v)_{L^2}, where the phi_i are
    the basis functions for piecewise-affine finite-elements, records
    the weighted H^1 norm of the finite-element error (approximated by
    taking the solution on a much finer grid) and the norm of the
    right-hand side in (H^1_k)'.

    Parameters:

    k_list - list of positive integers - the values of k for which
    experiments will be done.

    num_system - positive integer - the number of different system
    matrices (i.e. A,n) for which to perform experiments.

    num_rhs - positive integer - the number of different right-hand
    sides (for each different system) for which to perform experiments.

    fine_grid_refinement - positive integer - the `true' solution is
    computed on a very fine grid - this will be a
    fine_grid_refinement-fold uniform refinement of the finest grid used
    in any of the computations. (E.g., if fine_grid_refinement = 3, the
    the grid for computing the `true' solution will have a mesh size
    2^{-3} = 3-times smaller than the finest grid used in any of the
    computations.)
    """

    h_power_list = [-1.5,-2.0]

    seed = 1.0
    
    smallest_mesh_size_num_points =\
        h_to_mesh_points(max(k_list)**min(h_power_list))

    np.random.seed(seed)
        
    for k in k_list:

        for h_power in h_power_list:

            storage = np.full((num_system*num_rhs,2),-1.0)
            
            # Set up problem so that it can be easily altered

            # For now, create a separate class that implements a very simple A and n, so that a more complicated one can always be substituted in later.
            
            for ii_system in range(num_system):

                # Set up `fine' problem
                
                # Sample system and update problem
                
                for ii_rhs in range(num_rhs):

                    # Sample rhs

                    # Compute fine problem

                    # Compute coarse problem

                    # transfer coarse solution to fine mesh

                    # measure weighted H^1 norm of error

                    # measure L^2 norm of rhs as proxy

                    # store error and rhs

                    print("In progress")

            
