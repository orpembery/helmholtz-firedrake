actuator_domains = []

sub_dx = 1.0 / NUM_ACTUATORS

sub_dy = sub_dx

if DIM == 3:

    sub_dz = sub_dx

    for i in range(0, int(NUM_ACTUATORS)):

        for j in range(0, int(NUM_ACTUATORS)):

            for k in range(0, int(NUM_ACTUATORS)):

                sub_xa = i * sub_dx

                sub_xb = (i + 1) * sub_dx

                sub_ya = j * sub_dy

                sub_yb = (j + 1) * sub_dy

                sub_za = k * sub_dz

                sub_zb = (k + 1) * sub_dz

                actuator_domains.append(

                    SubDomainData(

                        reduce(ufl.And,

                               [x[0] >= sub_xa,

                                x[0] <= sub_xb,

                                x[1] >= sub_ya,

                                x[1] <= sub_yb,

                                x[2] >= sub_za,

                                x[2] <= sub_zb])

                    )

                )



else:

    for i in range(0, int(NUM_ACTUATORS)):

        for j in range(0, int(NUM_ACTUATORS)):

            sub_xa = i * sub_dx

            sub_xb = (i + 1) * sub_dx

            sub_ya = j * sub_dy

            sub_yb = (j + 1) * sub_dy

            actuator_domains.append(

                SubDomainData(

                    reduce(ufl.And,

                           [x[0] >= sub_xa,

                            x[0] <= sub_xb,

                            x[1] >= sub_ya,

                            x[1] <= sub_yb])

                )

            )



marker = Function(FunctionSpace(mesh, "DG", 0))

for i, actuator in enumerate(actuator_domains):

    marker.interpolate(Constant(i + 1), subset=actuator)

    File(RESULT_DIR + 'subdomains.pvd').write(marker)
