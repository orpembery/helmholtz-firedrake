# Plots the result of lots of nearby preconditioning test runs
import subprocess
import numpy as np
import csv
import matplotlib.pyplot as plt

# Count # of csv files
num_csv = subprocess.run("ls | grep -c 'nearby.*csv'", shell=True, stdout=subprocess.PIPE)
num_csv = num_csv.stdout.decode('UTF-8')[:-1] # help from https://stackoverflow.com/a/6273618
num_csv = int(num_csv) # from https://stackoverflow.com/questions/379906/how-do-i-parse-a-string-to-a-float-or-int-in-python#379910

# Figure out how many rows are needed = 2 + num_repeats (k, n_noise, all the repeats)
all_csv = subprocess.run("ls | grep 'nearby.*csv'", shell=True, stdout=subprocess.PIPE)
all_csv_list = all_csv.stdout.decode('UTF-8')[:-1] # help from https://stackoverflow.com/a/6273618
all_csv_list = all_csv_list.splitlines() # https://stackoverflow.com/questions/13169725/how-to-convert-a-string-that-has-newline-characters-in-it-into-a-list-in-python#13169786

# Read in first csv file to figure out how many repeats
this_csv = []
with open(all_csv_list[0]) as csvfile:
    csv_reader = csv.reader(csvfile,delimiter=",")
    for row in csv_reader:
        this_csv.append(row[1])

num_repeats_index = 9 # user-defined  - this_csv[num_repeats_index] should be num_repeats
num_repeats = int(this_csv[num_repeats_index])

num_needed_fields = 4 + num_repeats # again, user specified

# Create empty numpy array
results = np.empty([num_needed_fields,num_csv])

# Now extract all the relevant information - again, indicies are user-specified
for ii in range(len(all_csv_list)):
    with open(all_csv_list[ii]) as csvfile:
        csv_reader = csv.reader(csvfile,delimiter=",")
        row_index = 0
        for row in csv_reader:
            #print(row_index)
            # this is a bit hacky at the moment
            if row_index == 2: # k
                results[0,ii] = row[1]
            elif row_index == 8: # noise level
                results[1,ii] = row[1]
                # leave extra rows to fill in later
            elif row_index >= 11: # GMRES Its
                results[row_index-7,ii] = row[1]
            row_index = row_index + 1

# add in extra rows so that it's easy (I hope) to extract the results for a given master_noise_level
# to clarify (using python indexing), row [1] gives the noise_level, row[2] gives the noise level * k and row [3] gives the noise level / (h**2 * )
# Therefore, if you want to know which columns correspond to noise 0.1/k, you just do results[2,:]==0.1
for mesh_condition in [1.5,2.0]:

    for ii in range(len(results[0,:])):
        k = results[0,ii]
        results[2,ii] = k * results[1,ii]

        num_mesh_cells = np.ceil(k**(mesh_condition) * np.sqrt(2.0))

        mesh_size_along_edge = 1.0 / num_mesh_cells
             
        mesh_size = mesh_size_along_edge * np.sqrt(2.0)
        
        results[3,ii] = results[1,ii] / (mesh_size**2 * k)

    # for each master_noise_level
    for master_noise_level in [0.1,0.2,0.3,0.4,0.5]:

        # for each of the three possibilities
        for power_of_k_index in [1,2,3]: # i.e. each of the three noise possibilities

            # extract the results
            relevant_columns = abs(results[power_of_k_index,:] - master_noise_level) <= 1e-10
            to_plot_columns = results[:,relevant_columns]
            # put all the columns of GMRES iterations in one LONG column
            just_gmres_its = to_plot_columns[4:,:]
            gmres_its_to_plot = just_gmres_its.reshape((just_gmres_its.size,1),order="F")

            # get all the corresponding values of k
            just_k = to_plot_columns[0,:]
            just_k = np.tile(just_k,(just_gmres_its.shape[0],1))
            k_to_plot = just_k.reshape((just_k.size,1),order="F")
            # and plot them
            if power_of_k_index == 1:
                plot_label = 'noise ~ 1'
            elif power_of_k_index == 2:
                plot_label = 'noise ~ k^{-1}'
            elif power_of_k_index == 3:
                plot_label = 'noise ~ h^{2}k'
            plt.plot(k_to_plot,gmres_its_to_plot, 'o',label=plot_label)
                    
            plt.legend()
            plt.xlabel('k')
            plt.ylabel('No. of GMRES iterations')
            title_string = 'h ~ k^-' + str(mesh_condition) + ' and Noise level multiplied by ' + str(master_noise_level)
            plt.title(title_string)

        plt.show()

