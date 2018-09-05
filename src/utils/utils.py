def h_to_mesh_points(h):
    """Converts a mesh size to a number of points giving that mesh size.

    Given a mesh size h, computes the arguments to Firedrake's
    UnitSquareMesh that will give (at most) that mesh size in 2D.

    Parameter:

    h - positive float - the mesh size.
    """
    return np.ceil(np.sqrt(2.0)/h)

def write_GMRES_its(GMRES_its,save_location,info):
    """Writes the number of GMRES iterations, and other information, to
    a .csv file.

    Parameters:

    save_location - see nearby_preconditioning_test_set.

    GMRES_its - list of positive ints of length num_repeats (output of
    nearby_preconditioning_test).

    info - a dict containing all of the other information to be written
    to the file.

    The output csv file will have the filename
    'nearby-preconditioning-test-output-date_time.csv, where date_time
    is the date and time. The rows of the file will consist of the hash
    of the current git commit, then the date and time, then all of the
    entries of info (where the value first column will be the key, and
    the value in the second column will be the value in the dict),
    followed by the GMRES iterations (repeat number in the first column,
    number of GMRES iterations in the second).
    """

    # Check save_location is actually a directory path
    assert save_location[-1] == "/"
    
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

        # All other properties
        for ii in info.keys():
            file_writer.writerow([ii,info[ii]])

        for ii in range(len(GMRES_its)):
            file_writer.writerow([ii,GMRES_its[ii]])
        print(save_location)
