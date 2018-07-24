These were the output files from the first ever run using the firedrake implementation, with variable n only.

The mesh size is wrong! In defining the number of points, I wrote k**(mesh_size) / sqrt(2), whereas it should have been k ** (mesh_sze) * sqrt(2), i.e., there should be double the number of points. I don't think it'll matter too much, but would want to rerun these.