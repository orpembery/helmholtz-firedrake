import nearby_preconditioning_tests as nbpc

A_pre_type = 'constant'

n_pre_type = 'constant'

num_pieces = 2

seed = 1

num_repeats = 2

k_list = [10.0]

h_list = [(1.0,-1.5)]

noise_master_level_list = [(0.1,0.1)]

noise_modifier_list = [(0.0,-1.0,0.0,-1.0)]

save_location = 'output/testing/'

nbpc.nearby_preconditioning_test_set(A_pre_type,n_pre_type,num_pieces,seed,num_repeats,k_list,h_list,noise_master_level_list,noise_modifier_list,save_location)
