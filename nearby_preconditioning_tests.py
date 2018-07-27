import helmholtz_firedrake as hh
import numpy as np

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

def nearby_preconditioning_test_set(A_pre_type,n_pre_type,seed,num_repeats,dimension,k_list,h_list,noise_master_level_list,noise_modifier_list):
    """Performs many nearby preconditioning tests for a range of parameter values.

    Performs nearby preconditioning tests for a range of values of k, the mesh size h, and the size of the random noise (which can be specified in terms of k and h). The random noise is piecewise constant on a grid unrelated to the finite-element mesh.

    Parameters:

    A_pre_type - string - options are 'constant', giving A_pre = [[1.0,0.0],[0.0,1.0]].

    n_pre_type - string - options are 'constant', giving n_pre = 1.0.

    seed - see StochasticHelmholtzProblem.

    num_repeats - see nearby_preconditioning_test.

    dimension - either 2 or 3 - the spatial dimension of the problem

    k_list - list of positive floats - the values of k for which we will run experiments.

    h_list - list of 2-tuples; in each tuple (call it t) t[0] should be a positive float and t[1] should be a float. These specify the values of the mesh size h for which we will run experiments. h = t[0] * k**t[1].

    noise_master_level_list - list of 2-tuples, where each entry of the tuple is a positive float.  This defines the values of base_noise_A and base_noise_n to be used in the experiments. Call a given tuple t. The base_noise_A = t[0] and base_noise_n = t[1].

    noise_modifier_list - list of 4-tuples; the entries of each tuple should be floats. Call a given tuple t. This modifies the base noise so that the L^\infty norms of A and n are less than or equal to (respectively) base_noise_A * h**t[0] * k**t[1] and base_noise_n * h**t[2] * k**t[3].
    """

    if not(isinstance(A_pre_type,str)):
        raise hh.UserInputError("Input A_pre_type should be a string")
    elif A_pre_type is not "constant":
        raise HelmholtzNotImplementedError("Currently only implemented A_pre_type = 'constant'.")

    if not(isinstance(n_pre_type,str)):
        raise hh.UserInputError("Input n_pre_type should be a string")
    elif n_pre_type is not "constant":
        raise HelmholtzNotImplementedError("Currently only implemented n_pre_type = 'constant'.")

    if dimension is not 2 or dimension is not 3:
        raise hh.UserInputError("Input dimension must be 2 or 3")

    if not(isinstance(k_list,list)):
        raise hh.UserInputError("Input k_list should be a list.")
    elif any(not(isinstance(k,float) for k in k_list)):
        raise hh.UserInputError("Input k_list should be a list of floats.")
    elif any(k <= 0 for k in k_list):
        raise hh.UserInputError("Input k_list should be a list of positive floats.")

    if not(isinstance(h_list,list)):
        raise hh.UserInputError("Input h_list should be a list.")
    elif any(not(isinstance(h_tuple,tuple) for h_tuple in h_list)):
        raise hh.UserInputError("Input h_list should be a list of tuples.")
    elif any(len(h_tuple) is not 2 for h_tuple in h_list)):
        raise hh.UserInputError("Input h_list should be a list of 2-tuples.")
    elif any(not(isinstance(h_tuple[0],float)) for h_tuple in h_list) or any(h_tuple[0] <= 0 for h_tuple in h_list):
        raise hh.UserInputError("The first item of every tuple in h_list should be a positive float.")
    elif any(not(isinstance(h_tuple[1],float)):
        raise hh.UserInputError("The second item of every tuple in h_list should be a float.")

    if not(isinstance(noise_master_level_list,list)):
        raise hh.UserInputError("Input noise_master_level_list should be a list.")
    elif any(not(isinstance(noise_tuple,tuple)) for noise_tuple in noise_master_level_list):
        raise hh.UserInputError("Input noise_master_level_list should be a list of tuples.")
    elif any(len(noise_tuple) is not 2 for noise_tuple in noise_master_level_list)):
        raise hh.UserInputError("Input noise_master_level_list should be a list of 2-tuples.")
    elif any(any(not(isinstance(noise_tuple[i],float)) for i in range(len(noise_tuple))) for noise_tuple in noise_master_level_list):
        raise hh.UserInputError("Input noise_master_level_list should be a list of 2-tuples of floats.")

    if not(isinstance(noise_modifier_list,list)):
        raise hh.UserInputError("Input noise_modifier_list should be a list.")
    elif any(not(isinstance(mod_tuple,tuple)) for mod_tuple in noise_modifier_list):
        raise hh.UserInputError("Input noise_modifier_list should be a list of tuples.")
    elif any(len(noise_tuple) is not 4 for mod_tuple in noise_modifier_list)):
        raise hh.UserInputError("Input noise_modifier_list should be a list of 4-tuples.")
    elif any(any(not(isinstance(mod_tuple[i],float)) for i in range(len(mod_tuple))) for mod_tuple in noise_modifier_list):
        raise hh.UserInputError("Input noise_modifier_list should be a list of 4-tuples of floats.")


    
    for ii_k in range(k_list):
        for ii_h in range(h_list):
            # Assemble mesh, having calculated size (separate function)
            # Define function space
            for ii_master in range(noise_master_level_list):
                for ii_modifier in range(noise_modifier_list):
                    

    
class A_generator:
    """Does the work of A_gen in StochastichelmholtzProblem for the case of piecewise continuous on some grid."""

# This can alsmost certainly be made better, but is a bit of a hack until they've sorted complex so that I can do things the way Julian does.
    
    def __init__(self,seed,mesh,num_pieces,noise_level_A,A_pre):
        """Initialises."""

        # Set up consts
 
        A_values = noise_level_A * (2.0 * np.random.random_sample([num_pieces**2,2,2]) - 1.0) # Uniform (-1,1) random variates

        A_values_list = list(A_values)
        # We want each 2x2 `piece' of A_values to be an entry in a list, so that we can then turn each of them into a Firedrake `Constant` (I hope that this will mean Firedrake passes them as arguments to the C kernel, as documented on the `Constant` documentation page.)

        # Symmetrise all the matrices
        A_values_list = [self.symmetrise(A_dummy) for A_dummy in A_values_list]

        A_values_list = A_noise(noise_level_A,coeff_pieces)

        A_values_constant_list = [Constant(A_dummy,domain=mesh) for A_dummy in A_values_list]

        # This extracts the relevant element of the list, given a 2-d index
        def list_extract(values_list,x_coord,y_coord,coord_length): # The list should contain coord_length**2 elements
            return values_list[x_coord + y_coord * coord_length]

        self.A = A_pre
        
        # Form A by looping over all the subdomains
        for xii in range(0,coeff_pieces-1):
            for yii in range(0,coeff_pieces-1):
                self.A += list_extract(A_values_constant_list,xii,yii,coeff_pieces) * Iab(x[0],xii/coeff_pieces,(xii+1)/coeff_pieces) * Iab(x[1],yii/coeff_pieces,(yii+1)/coeff_pieces)

    # Will 'symmetrise' a 2x2 matrix by copying the lower left-hand entry into the top right-hand entry.
    def symmetrise(self,A):
        A_lower = np.tril(A,k=-1)
        return np.diagflat(np.diagonal(A).copy()) + A_lower + np.transpose(A_lower)

def heaviside(x): # x here is a single coordinate of a SpatialCoordinate
    return 0.5 * (sign(real(x)) + 1.0)

def Iab(x,a,b) : # indicator function on [a,b] - x is a single coordinate of a spatial coordinate, 0.0  <= a < b <= 1 are doubles
    return heaviside(x-a) - heaviside(x-b)
