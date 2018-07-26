import helmholtz_firedrake as hh

def nearby_preconditioning_test(mesh,V,k,A_pre,A_gen,n_pre,n_gen,f,g,seed,num_repeats):
    """For a given preconditioning Helmholtz problem, performs a test of the effectiveness of nearby preconditioning.

    For a given preconditioning Helmholtz problem, and given methods for generating realisations of Helmholtz problems with random field coefficients, generates realisations of the Helmholtz problems, and then uses the preconditioner to perform preconditioned GMRES. Then records the number of GMRES iterations needed to acheive convergence.

    Parameters:

    mesh - see HelmholtzProblem

    V - see HelmholtzProblem

    k - see HelmholtzProblem

    A_pre - see HelmholtzProblem

    A_gen - see StochasticHelmholtzProblem

    n_pre - see HelmholtzProblem

    n_gen - see StochasticHelmholtzProblem

    f - see HelmholtzProblem

    g - see HelmholtzProblem

    seed - see StochasticHelmholtzProblem

    num_repeats - int, specifying the number of realisations to take.


    Returns a list of ints of length num_repeats, giving the number of GMRES iterations for the different realisations.
    
    """

    prob_pre = hh.HelmholtzProblem(mesh, V, k, A_pre, n_pre, f, g, boundary_condition_type="Impedance", aP=None)

    prob = hh.StochasticHelmholtzProblem(mesh, V, k, A_gen, n_gen, f, g, seed, boundary_condition_type="Impedance", aP=None)

    all_GMRES_its = []

    for ii_repeat in range(num_repeats):
        prob.solve()

        all_GMRES_its.append(prob.GMRES_its)

        prob.resample_coefficients()

    return all_GMRES_its

def nearby_preconditioning_test_set(A_pre_type,n_pre_type,seed,num_repeats,k_list,h_list,noise_master_level_list,noise_modifier_list):
    """Performs many nearby preconditioning tests for a range of parameter values.

    Performs nearby preconditioning tests for a range of values of k, the mesh size h, and the size of the random noise (which can be specified in terms of k and h). The random noise is piecewise constant on a grid unrelated to the finite-element mesh.

    Parameters:

    A_pre_type - string - options are 'constant', giving A_pre = [[1.0,0.0],[0.0,1.0]].

    n_pre_type - string - options are 'constant', giving n_pre = 1.0.

    seed - see StochasticHelmholtzProblem.

    num_repeats - see nearby_preconditioning_test.

    k_list - list of positive floats - the values of k for which we will run experiments.

    h_list - list of 2-tuples; in each tuple (call it t) t[0] should be a positive float and t[1] should be a float. These specify the values of the mesh size h for which we will run experiments. h = t[0] * k**t[1].

    noise_master_level_list - list of 2-tuples, where each entry of the tuple is a positive float.  This defines the values of base_noise_A and base_noise_n to be used in the experiments. Call a given tuple t. The base_noise_A = t[0] and base_noise_n = t[1].

    noise_modifier_list - list of 4-tuples; the entries of each tuple should be floats. Call a given tuple t. This modifies the base noise so that the L^\infty norms of A and n are less than or equal to (respectively) base_noise_A * h**t[0] * k**t[1] and base_noise_n * h**t[2] * k**t[3].
    """

    # Do error hecking - (A_pre_type,n_pre_type,seed,num_repeats,k)list,h_list,noise_master_level_list,noise_modifier_list):

    if not(isinstance(A_pre_type,str)):
        raise UserInputError("Input A_pre_type should be a string")
    elif A_pre_type is not "constant":
        raise HelmholtzNotImplementedError("Currently only implemented A_pre_type = 'constant'.")

    if not(isinstance(n_pre_type,str)):
        raise UserInputError("Input n_pre_type should be a string")
    elif n_pre_type is not "constant":
        raise HelmholtzNotImplementedError("Currently only implemented n_pre_type = 'constant'.")

    if not(isinstance(k_list,list)):
        raise UserInputError("Input k_list should be a list of positive floats.")
    elif any(not(isinstance(k,float) for k in k_list)):
        raise UserInputError("Input k_list should be a list of positive floats.")
    elif any(k <= 0 for k in k_list):
        raise UserInputError("Input k_list should be a list of positive floats.")

    if not(isinstance(h_list,list)):
        raise UserInputError("Input h_list should be a list of 2-tuples.")
    elif any(not(isinstance(h_tuple,tuple) for h_tuple in h_list)):
        raise UserInputError("Input h_list should be a list of 2-tuples.")
    elif any(len(h_tuple) is not 2 for h_tuple in h_list)):
        raise UserInputError("Input h_list should be a list of 2-tuples.")
    elif any(not(isinstance(h_tuple[0],float)) for h_tuple in h_list) or any(h_tuple[0] <= 0 for h_tuple in h_list):
        raise UserInputError("The first item of every tuple in h_list should be a positive float.")
    elif any(not(isinstance(h_tuple[1],float)):
        raise UserInputError("The second item of every tuple in h_list should be a float.")
    
    for ii_k in range(

    
