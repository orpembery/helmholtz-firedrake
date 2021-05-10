# Summary #
This code allows one to solve the Helmholtz equation using the Firedrake numerical PDE software.

It is used in the PhD thesis [The Helmholtz Equation in Heterogeneous and Random Media: Analysis and Numerics](https://researchportal.bath.ac.uk/en/studentTheses/the-helmholtz-equation-in-heterogeneous-and-random-media-analysis) (release 'Thesis version') and the paper [Analysis of a Helmholtz preconditioning problem motivated by uncertainty quantification](https://arxiv.org/abs/2005.13390) (release 'Paper version').

See [this repository](https://github.com/orpembery/thesis) for more information on the usage of the code in the thesis, as well as instructions on how to download and use the code.

Major changes between releases:
- In 'Thesis version' GMRES is restarted after 30 interations; in 'Paper version' GMRES is restarted after 500 iterations (whilst giving a warning).
- In 'Thesis version' GMRES monitors convergence in the standard 2-norm; in 'Paper version' the [preconditioned norm](https://firedrakeproject.org/solving-interface.html#setting-solver-tolerances) is used.
---
For any queries, contact Owen Pembery on opembery 'at' gmail 'dot' com.