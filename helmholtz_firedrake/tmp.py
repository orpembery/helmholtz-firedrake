import numpy as np

dave = np.array([[[1,3],[4,6]],[[7,9],[10,12]]])



print(dave)

num_pieces = 2

with np.nditer(dave,op_flags=['readwrite']) as it:

    for x in it:

        x[...] = 7

print(dave)
#     print(it[0])
    # print(np.array((it.multi_index,1+np.array(it.multi_index)),dtype='float').T/float(num_pieces))


a = np.arange(6).reshape(2,3)
print(a)
with np.nditer(a, op_flags=['readwrite']) as it:
    for x in it:
        x = 2*x

print(a)

