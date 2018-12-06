import numpy as np
import subprocess
import datetime
import csv
from firedrake import norm

def h_to_mesh_points(h):
    """Converts a mesh size to a number of points giving that mesh size.

    Given a mesh size h, computes the arguments to Firedrake's
    UnitSquareMesh that will give (at most) that mesh size in 2D.

    Parameter:

    h - positive float - the mesh size.

    Output:

    positive int - the number of points to use in both the x- and
    y-directions.
    """
    return np.ceil(np.sqrt(2.0)/h)

def mesh_points_to_h(num_points_x,num_points_y):
    """Converts the number of points in a Firedrake 2-D UnitSquareMesh
    into the mesh size.

    Parameters:

    num_points_x - positive int - the number of points in the
    x-direction.

    num_points_y - positive int - the number of points in the
    y-direction.

    Output:

    positive float - the mesh size.
    """

    return np.sqrt(1.0/(float(num_points_x-1)**2)
                   + 1.0/(float(num_points_y-1)**2))

def write_repeats_to_csv(data,save_location,name_string,info):
    """Writes the results of a number of experiments, to a .csv file.

    Parameters:

    data - a numpy array containing the numerical data to be written to
    the csv. Each row of the array should correspond to a different
    repeat of the experiment. Note that the rows of data will be
    numbered automatically, so there's no need to include these numbers
    as a separate column in data itself.

    save_location - string containing the absolute path to the directory
    in which the csv will be saved.

    name_string - string containing the beginning of the filename for
    the csv. The csv file will then have the filename given by
    name_string + date_time + '.csv', where date_time is a string
    containing the date and time.

    info - a dict containing all of the other information to be written
    to the file. None of the keys of the dict should be the integer 0.

    The rows of the file will consist of the hash of the current git
    commit, then the date and time, then all of the entries of info
    (where the value first column will be the key, and the value in the
    second column will be the value in the dict), followed by the
    contents of data (the row number in the first column, and then the
    columns of data in the subsequent columns.

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
    with open(save_location + name_string + date_time + '.csv',
              'w', newline = '') as csvfile:
        file_writer = csv.writer(csvfile, delimiter = ',',
                                 quoting = csv.QUOTE_NONNUMERIC)
        file_writer.writerow(['Git hash', git_hash_string])

        # Current time in UTC as an ISO string
        file_writer.writerow(['Date/Time', date_time])

        # All other properties
        for ii in info.keys():
            file_writer.writerow([ii,info[ii]])

        for ii in range(data.shape[0]):

            if len(data.shape) == 1:
                file_writer.writerow([ii,data[ii]])
            else:            
                file_writer.writerow(np.concatenate((np.array([ii]),data[ii,:])))

def read_repeats_from_csv(save_location):
    """Reads repeats, and metadata from a csv file.

    This function assumes the csv file has been saved by
    write_repeats_to_csv.

    Parameters:

    save_location - string, specifying the location of the csv file
    (including the filename).

    Output:

    a tuple (info,data), where info is a dict, containing the
    information that would have been passed into write_repeats_to_csv as
    thei nput argument 'info', and data is a numpy array containing the
    data that would have been passed in to write_repeats_to_csv as
    'data'.

    """

    info = dict()
    
    # adapted from https://docs.python.org/3.5/library/csv.html
    with open(save_location,
               newline = '') as csvfile:
        file_reader = csv.reader(csvfile,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
        reached_GMRES = False
        for row in file_reader:
            print(row)
            if row[0] == 0:
                reached_GMRES = True
                data = np.array(row)
                # ABOVE MAY NOT WORK if row is very long or array method doesn't work
                if reached_GMRES == False:
                    # Technically from https://stackoverflow.com/a/1024851
                    info[row[0]] = row[1]
                else:
                    np.concatenate((data,row))

    return (info,data)
            

    # go down the rows putting stuff into the dict until the first column is 0

    # Then great a numpy array of the correct width, call it data
    
    # Then switch to putting it into data

def write_GMRES_its(GMRES_its,save_location,info):

    """Writes the number of GMRES iterations, and other information, to
    a .csv file.

    Parameters:

    save_location - see write_repeats_to_csv.

    GMRES_its - list of positive ints of length num_repeats (output of
    nearby_preconditioning_test).

    info - see write_repeats_to_csv

    The output csv file will have the filename
    'nearby-preconditioning-test-output-date_time.csv, where date_time
    is the date and time. The rows of the file will consist of the hash
    of the current git commit, then the date and time, then all of the
    entries of info (where the value first column will be the key, and
    the value in the second column will be the value in the dict),
    followed by the GMRES iterations (repeat number in the first column,
    number of GMRES iterations in the second).
    """

    write_repeats_to_csv(GMRES_its,save_location,
                         'nearby-preconditioning-test-output-',info)

def norm_weighted(u,k):
    """Computes the weighted H^1 norm of u.

    Inputs:

    u - a Firedrake Function

    k - positive real - the wavenumber.

    Output:

    positive real - the weighted H^1 norm of u.
    """

    return np.sqrt(norm(u,norm_type="H1")**2.0\
                            + (k**2.0 - 1.0)*norm(u,norm_type="L2")**2.0)
