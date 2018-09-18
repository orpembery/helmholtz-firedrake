elts = 40

mesh = PeriodicIntervalMesh(elts, 2)
mesh_output = PeriodicIntervalMesh(elts*10, 2)

V = FunctionSpace(mesh, "DG", degree)
V_output = FunctionSpace(mesh_output, "DG", 1)
W = VectorFunctionSpace(mesh_output, V_output.ufl_element())
coords = interpolate(mesh_output.coordinates, W)

u = Function(V, name="u")
u_out = Function(V_output, name="u")

u_out.dat.data[:] = u.at(coords.dat.data_ro)
