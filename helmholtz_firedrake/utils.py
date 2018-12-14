import numpy as np
import subprocess
import datetime
import csv
from firedrake import norm, sign, real
import pandas as pd

def h_to_num_cells(h,d):
    """Converts a mesh size to a number of cells giving that mesh size.

    Given a mesh size h, computes the arguments to Firedrake's
    UnitSquareMesh or UnitCubeMesh that will give (at most) that mesh size in 2D.

    Parameters:

    h - positive float - the mesh size.

    d - 2 or 3 - the geometric dimension of the mesh

    Output:

    positive int - the number of cells to use in both the x- and
    y-directions.
    """
    return np.ceil(np.sqrt(float(d))/h)

def num_cells_to_h(num_cells_tuple,d):
    """Converts the number of points in a Firedrake UnitSquareMesh or
    UnitCubeMesh into the mesh size.

    Parameters:

    num_cells_tuple - tuple of length d of positive ints - the number of
    cells in each direction.

    d - 2 or 3.

    Output:

    positive float - the mesh size.

    """

    assert len(num_cells_tuple) == d
    
    # This is a bit of hack, may be a cleaner way to do it
    return np.sqrt(np.array(
        [1.0/float(num_cells**2) for num_cells in num_cells_tuple]
    ).sum())

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
                file_writer.writerow(np.concatenate((np.array([ii]),
                                                     data[ii,:])))

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

        file_reader = csv.reader(csvfile,delimiter=',',
                                 quoting=csv.QUOTE_NONNUMERIC)
        reached_GMRES = False

        for row in file_reader:

            if row[0] == 0:
                reached_GMRES = True
                data = np.array(row)

            if reached_GMRES == False:
                # Technically from
                # https://stackoverflow.com/a/1024851
                info[row[0]] = row[1]
            # Next condition needed because we've already added row 0 to
            # the data
            elif row[0] == 0:
                pass
            else:
                data = np.vstack((data,row))

    return (info,data)
            
def csv_list_to_dataframe(csv_list,names_list):
    """Writes content from a list of csv files to a Pandas DataFrame.

    Given a list of csv files (written by write_repeats_to_csv) extracts
    all of the written data and places it in the rows of Pandas
    DataFrame, and multiindexes the rows by the information
    corresponding to each file given by the fields named in
    names_list.

    Parameters:

    csv_list - list of strings, giving the locations (including
    filenames) of the csv files to read in.

    names_list - list of strings, giving the fields from each of the csv
    files that will be used to index the resulting DataFrame.

    Output - Pandas DataFrame, the rows of which correspond to each of
    the csv files read in. The rows are indexed by the fields given in
    names_list.
    """

    labels = []

    for csv_file in csv_list:
        this_output = read_repeats_from_csv(csv_file)

        labels.append([this_output[0][key] for key in names_list])
        
        # Need to extend the output array or the array about to be added
        # if they're not the same size. Issues arise if the output array
        # hasn't been created yet.
        try:
            current_size = output_array.shape[1]
                   
            new_size = this_output[1].shape[0]
            if current_size < new_size:
                output_array = np.hstack((output_array,
                                          np.full((output_array.shape[0],
                                                   new_size-current_size),
                                                  np.nan)))
            elif new_size < current_size:
                # Concatenation is confusing, as this_output has the data in
                # columns, but we're putting the data into rows.

                this_output[1] = np.hstack(this_output[1],
                                           np.full((current_size-new_size,
                                                    this_output[1].shape[1]),
                                                   np.nan))

            output_array = np.vstack((output_array,this_output[1][:,1:].transpose()))

        except UnboundLocalError:
            output_array = this_output[1][:,1:].transpose()

        
    # Following adapted from
    # https://pandas.pydata.org/pandas-docs/stable/advanced.html
    index = pd.MultiIndex.from_tuples(labels,names=names_list)

    return pd.DataFrame(output_array,index=index)
    
def write_GMRES_its(GMRES_its,save_location,info):

    """Writes the number of GMRES iterations, and other information, to
    a .csv file.

    Parameters:

    save_location - see write_repeats_to_csv.

    GMRES_its - 1-dimensional numpy array of positive ints.

    info - see write_repeats_to_csv

    The output csv file will have the filename
    info-as-strings-date_time.csv, where info_as_strings will consist of
    all of the keys in the dictionary info, followed by their values
    (for example, if info = {'k' : 10, 'h' : 0.1} then info_as_strings
    will be k-10--h-0.1--). date_time is the date and time. The rows of
    the file will consist of the hash of the current git commit, then
    the date and time, then all of the entries of info (where the value
    first column will be the key, and the value in the second column
    will be the value in the dict), followed by the GMRES iterations
    (repeat number in the first column, number of GMRES iterations in
    the second).
    """

    info_as_strings = ''

    for key in sorted(info.keys()):
        info_as_strings = info_as_strings + key + '-' + str(info[key]) + '--'
    
    write_repeats_to_csv(GMRES_its,save_location,
                         info_as_strings,info)

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

def bounded_error_mesh_size(p):
    """Gives the exponent for the mesh size that keeps relative error
    bounded for p-FEs.

    It is not proven that this mesh size is sufficient to keep the
    relative error bounded for degree p finite-elements, but it seems to
    work.

    Parameter:

    p - positive int - the degree of the finite-element basis functions.

    Output:

    exponent - positive float - number such that the relative error in
    the finite-element method is bounded if $h \sim k^exponent$.
    """

    return (p+2)/(p+1)

def heaviside(x):
    """Defines the heaviside step function in UFL.

    Parameters:

    x - single coordinate of a UFL SpatialCoordinate.
    """
    return 0.5 * (sign(real(x)) + 1.0)

def Iab(x,a,b) :
    """Indicator function on [a,b] in UFL.

    Parameters:

    x - single coordinate of a UFL SpatialCoordinate.

    a, b - floats, a < b.
    """
    return heaviside(x-a) - heaviside(x-b)

def nd_indicator(x,val,loc_array):
    """Define the indicator function on cubes in n-dimensions on a Mesh.

    Parameters:

    x - a Firedrake SpatialCoordinate

    val - float or Firedrake Constant - the value to take on the cube. 

    loc_tuple - a d x 2 numpy array of floats, where d is the geometric
    dimension of the mesh. The jth row gives the limits of the cube in
    dimension j.
    """

    d = x.ufl_domain().geometric_dimension()

    for ii in range(d):
        val = val * Iab(x[ii],loc_array[ii,0],loc_array[ii,1])

    return val

def flatiter_hack(constant_array,coords):
    """What follows is a hack - the flatiter seems to miss the first
         entry, and then try and do an extra one at the end (which
         fails). So when it fails, we do the first one.
    """
    try:
        constant_array[coords]
    except IndexError:
        coords = tuple(np.zeros(len(coords),dtype=int))

    return coords
