import firedrake as fd
import helmholtz.problems as hh
import numpy as np
import subprocess
import datetime
import csv

def nearby_preconditioning_test(V,k,A_pre,A_stoch,n_pre,n_stoch,f,g,
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

def nearby_preconditioning_test_set(
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

    num_pieces - see PieceWiseConstantCoeffGenerator.

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

    save_location - string specifying the subfolder of the current
    folder in which to save the .csv output files.
    """

    if not(isinstance(A_pre_type,str)):
        raise hh.UserInputError("Input A_pre_type should be a string")
    elif A_pre_type is not "constant":
        raise HelmholtzNotImplementedError(
            "Currently only implemented A_pre_type = 'constant'.")

    if not(isinstance(n_pre_type,str)):
        raise hh.UserInputError("Input n_pre_type should be a string")
    elif n_pre_type is not "constant":
        raise HelmholtzNotImplementedError(
            "Currently only implemented n_pre_type = 'constant'.")

    if not(isinstance(k_list,list)):
        raise hh.UserInputError("Input k_list should be a list.")
    elif any(not(isinstance(k,float)) for k in k_list):
        raise hh.UserInputError("Input k_list should be a list of floats.")
    elif any(k <= 0 for k in k_list):
        raise hh.UserInputError(
            "Input k_list should be a list of positive floats.")

    if not(isinstance(h_list,list)):
        raise hh.UserInputError("Input h_list should be a list.")
    elif any(not(isinstance(h_tuple,tuple)) for h_tuple in h_list):
        raise hh.UserInputError("Input h_list should be a list of tuples.")
    elif any(len(h_tuple) is not 2 for h_tuple in h_list):
        raise hh.UserInputError("Input h_list should be a list of 2-tuples.")
    elif any(not(isinstance(h_tuple[0],float)) for h_tuple in h_list)\
             or any(h_tuple[0] <= 0 for h_tuple in h_list):
        raise hh.UserInputError(
            "The first item of every tuple in h_list\
            should be a positive float.")
    elif any(not(isinstance(h_tuple[1],float)) for h_tuple in h_list):
        raise hh.UserInputError(
            "The second item of every tuple in h_list should be a float.")

    if not(isinstance(noise_master_level_list,list)):
        raise hh.UserInputError(
            "Input noise_master_level_list should be a list.")
    elif any(not(isinstance(noise_tuple,tuple))
             for noise_tuple in noise_master_level_list):
        raise hh.UserInputError(
            "Input noise_master_level_list should be a list of tuples.")
    elif any(len(noise_tuple) is not 2
             for noise_tuple in noise_master_level_list):
        raise hh.UserInputError(
            "Input noise_master_level_list should be a list of 2-tuples.")
    elif any(any(not(isinstance(noise_tuple[i],float))
                 for i in range(len(noise_tuple)))
             for noise_tuple in noise_master_level_list):
        raise hh.UserInputError(
            "Input noise_master_level_list\
            should be a list of 2-tuples of floats.")

    if not(isinstance(noise_modifier_list,list)):
        raise hh.UserInputError("Input noise_modifier_list should be a list.")
    elif any(not(isinstance(mod_tuple,tuple))
             for mod_tuple in noise_modifier_list):
        raise hh.UserInputError(
            "Input noise_modifier_list should be a list of tuples.")
    elif any(len(mod_tuple) is not 4 for mod_tuple in noise_modifier_list):
        raise hh.UserInputError(
            "Input noise_modifier_list should be a list of 4-tuples.")
    elif any(any(not(isinstance(mod_tuple[i],float))
                 for i in range(len(mod_tuple)))
             for mod_tuple in noise_modifier_list):
        raise hh.UserInputError(
            "Input noise_modifier_list\
            should be a list of 4-tuples of floats.")

    if A_pre_type is "constant":
        A_pre = fd.as_matrix([[1.0,0.0],[0.0,1.0]])

    if n_pre_type is "constant":
        n_pre = 1.0
    
    for k in k_list:
        for h_tuple in h_list:
            h = h_tuple[0] * k**h_tuple[1]
            mesh_points = h_to_mesh_points(h)
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
                    A_stoch = PieceWiseConstantCoeffGenerator(
                        mesh,num_pieces,A_noise_level,A_pre,[2,2])
                    n_stoch = PieceWiseConstantCoeffGenerator(
                        mesh,num_pieces,n_noise_level,n_pre,[1])
                    np.random.seed(seed)
                    
                    GMRES_its = nearby_preconditioning_test(
                        V,k,A_pre,A_stoch,n_pre,n_stoch,f,g,num_repeats)

                    write_GMRES_its(
                        GMRES_its,k,h_tuple,num_pieces,A_pre_type,n_pre_type,
                        noise_master,modifier,num_repeats,save_location)
def h_to_mesh_points(h):
    """Given a mesh size h, computes the arguments to Firedrake's
    UnitSquareMesh that will give (at most) that mesh size.

    Parameter:

    h - positive float - the mesh size.
    """
    return np.ceil(np.sqrt(2.0)/h)

def write_GMRES_its(GMRES_its,k,h_tuple,num_pieces,A_pre_type,n_pre_type,
                    noise_master,modifier,num_repeats,save_location):
    """Writes the number of GMRES iterations, and other information, to
    a .csv file.

    Parameters:

    GMRES_its - list of positive ints of length num_repeats (output of
    nearby_preconditioning_test).

    k - see nearby_preconditioning_test.

    h_tuple - 2-tuple - an element of the list h_list in
    nearby_preconditioning_test_set.

    num_pieces - see nearby_preconditioning_test_set.

    A_pre_type - see nearby_preconditioning_test_set.

    n_pre_type - see nearby_preconditioning_test_set.

    noise_master - 2-tuple - element of the list noise_master_level_list
    in nearby_preconditioning_test_set.

    modifier - 4-tuple - element of the list noise_modifier_list in
    nearby_preconditioning_test_set.

    num_repeats - see nearby_preconditioning_test.

    save_location - see nearby_preconditioning_test_set.
    """

    # Get git hash
    git_hash = subprocess.run("git rev-parse HEAD", shell=True,
                              stdout=subprocess.PIPE)
    # help from https://stackoverflow.com/a/6273618
    git_hash_string = git_hash.stdout.decode('UTF-8')[:-1]

    # Get current date and time
    # This initialises the object. I don't understand why this is
    # necessary.
    date_time = datetime.datetime(1,1,1)
    date_time = date_time.utcnow().isoformat()
  
    # Write CSV
    # from https://docs.python.org/3.5/library/csv.html
    with open(save_location + 'nearby-preconditioning-test-output-'
              + date_time + '.csv', 'w', newline = '') as csvfile:
        file_writer = csv.writer(csvfile, delimiter = ',',
                                 quoting = csv.QUOTE_MINIMAL)
        file_writer.writerow(['Git hash', git_hash_string])
        # Current time in UTC as an ISO string
        file_writer.writerow(['Date/Time', date_time])
        file_writer.writerow(['k',k ])
        file_writer.writerow(['h_tuple',h_tuple])
        file_writer.writerow(['num_pieces',num_pieces])
        file_writer.writerow(['A_pre_type',A_pre_type])
        file_writer.writerow(['n_pre_type',n_pre_type])
        file_writer.writerow(['noise_master',noise_master])
        file_writer.writerow(['modifier',modifier])
        file_writer.writerow(['num_repeats',num_repeats])
        file_writer.writerow(
            ['Experiment number','Number of GMRES iterations'])

        for ii in range(len(GMRES_its)):
            file_writer.writerow([ii,GMRES_its[ii]])

class PiecewiseConstantCoeffGenerator(object):
    """Does the work of A_stoch and n_stoch in
    StochasticHelmholtzProblem for the case of piecewise continuous on
    some grid.

    This can almost certainly be made better, but is a bit of a hack
    until they've sorted complex so that I can do things the way Julian
    does.

    Attribute:

    coeff - a UFL expression for a piecewise constant (scalar- or
    matrix-valued) coefficient, implemented using Firedrake Constants.

    Method:

    sample - randomly updates coeff by randomly sampling around the
    known background given by the input argument coeff_pre. Samples have
    (entrywise) L^\infty norm <= noise_level almost surely.
    """

    def __init__(self,mesh,num_pieces,noise_level,coeff_pre,coeff_dims):
        """Initialises a piecewise-constant random coefficient, where
        the coefficient is peicewise-constant on a num_pieces x
        num_pieces grid.

        Parameters:

        mesh - a Firedrake mesh object.

        num_pieces - int - the number of `pieces' in each direction for
        the piecewise-constant coefficients. Empirically must be <= 13,
        or get a recursion error.

        noise_level - positive float - the level of the random noise.

        coeff_pre - a UFL expression with dimension coeff_dims - the
        background around which we take piecewise-constant random
        perturbations.

        coeff_dims - a list, either [1] or [2,2] - the `dimension' of
        the coefficient (i.e., either scalar-valued or 2x2 matrix
        valued).
        """

        self._coeff_dims = coeff_dims
        """Dimension of the coefficient - [1] or [2,2]."""

        self._num_pieces = num_pieces
        """Number of pieces in each direction\
        for random coefficients."""

        self._noise_level = noise_level
        """Magnitude of random noise."""
                                                  
        self._coeff_initialise(mesh,coeff_pre)
        """Create realisation of random coefficient."""

        self.sample()
        

    def _list_extract(self,values_list,x_coord,y_coord,coord_length):
        """If values_list were put into a coord_length x
        coord_length array, extracts the item at position
        (x_coord,y_coord).
        
        Parameters:
        
        values_list - list of length coord_length**2.

        x_coord, y_coord - int in range(coord_length).

        coord_length - int.
        """
        return values_list[x_coord + y_coord * coord_length]

    def _heaviside(self,x):
        """Defines the heaviside step function in UFL.

        Parameters:

        x - single coordinate of a UFL SpatialCoordinate.
        """
        return 0.5 * (fd.sign(fd.real(x)) + 1.0)

    def _Iab(self,x,a,b) :
        """Indicator function on [a,b] in UFL.

        Parameters:

        x - single coordinate of a UFL SpatialCoordinate.
        
        a, b - floats in [0,1].
        """
        return self._heaviside(x-a) - self._heaviside(x-b)

    def _coeff_initialise(self,mesh,coeff_pre):
        """Initialises self.coeff equal to coeff_pre, but sets up
        Firedrake Constant structure to allow for sampling.

        Parameters:

        mesh - a Firedrake mesh.

        coeff_pre - see init.
        """

        self._coeff_values = []
        
        for ii in range(self._num_pieces**2):
            if self._coeff_dims == [2,2]:
                self._coeff_values.append(np.array([[0.0,0.0],[0.0,0.0]]))
            elif self._coeff_dims == [1]:
                self._coeff_values.append(np.array(0.0))
            else:
                raise NotImplementedError(
                          "Have only implemented real- and\
                          matrix-valued coefficients")
                
        self._coeff_values = [fd.Constant(coeff_dummy,domain=mesh)
                              for coeff_dummy in self._coeff_values]
        
        # Form coeff by looping over all the subdomains
        x = fd.SpatialCoordinate(mesh)

        self.coeff = coeff_pre
        
        for xii in range(self._num_pieces):
            for yii in range(self._num_pieces):
                self.coeff +=\
                self._list_extract(self._coeff_values,xii,yii,
                                   self._num_pieces)\
                * self._Iab(x[0],xii/self._num_pieces,(xii+1)/
                            self._num_pieces)\
                * self._Iab(x[1],yii/self._num_pieces,(yii+1)/self._num_pieces)

    def sample(self):
        """Samples the coefficient coeff."""
        [coeff_dummy.assign(self._generate_matrix_coeff())
         for coeff_dummy in self._coeff_values]

    def _generate_matrix_coeff(self):
        """Will generate a random float or a 2x2 s.p.d. numpy array.

        Floats are Unif(-self._noise_level,self._noise_level), matrices
        are generated so that the entrywise L^\infty norm is <=
        self._noise_level almost surely.

        For matrices, uses the fact that [[a,b],[b,c]] is
        positive-definite iff a>0 and b**2 < a*c.
        """

        if self._coeff_dims == [1]:
            coeff = self._noise_level*(
                2.0 * np.random.random_sample(self._coeff_dims) - 1.0)
        elif self._coeff_dims == [2,2]:
            a = np.random.random_sample(1)
            c = np.random.random_sample(1)
            b = np.sqrt(a*c) * np.random.random_sample(1)

            coeff = self._noise_level * np.array([[a,b],[b,c]])
        return coeff