import nearby_preconditioning.experiments as nbpc

A_pre_type = 'constant'

n_pre_type = 'constant'

num_pieces = 3

seed = 1

num_repeats = 2

k_list = [10.0]

h_list = [(1.0,-1.0)]#,(1.0,-1.5),(1.0,-2.0)]

noise_master_level_list = [(0.1,0.0)]

noise_modifier_list = [(0.0,0.0,0.0,0.0)]#,(1.0,0.0,0.0,0.0)]

save_location = '/home/owen/code/helmholtz-firedrake/output-to-copy-to-x/nbpc-paper/initial-test-of-A-condition/'

nbpc.nearby_preconditioning_piecewise_experiment_set(A_pre_type,n_pre_type,num_pieces,seed,num_repeats,k_list,h_list,noise_master_level_list,noise_modifier_list,save_location)
