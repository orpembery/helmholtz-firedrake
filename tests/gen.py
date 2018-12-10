import helmholtz_firedrake.utils as hhu
import numpy as np

info1 = {'k':10.0,'blarg':7}

info2 = {'k':10.0,'blarg':8}

data1 = np.array([[6],[3],[7],[4],[2],[7]])

data2 = np.array([[7],[2],[6],[3],[7],[23],[8]])

name1 = 'no-1-'

name2 = 'no-2-'

save_location = '/home/owen/code/helmholtz-firedrake/tmp/'

hhu.write_repeats_to_csv(data1,save_location,name1,info1)

hhu.write_repeats_to_csv(data2,save_location,name2,info2)
