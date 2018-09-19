#!/bin/bash

# Default modules to load
module load git/2.5.1
module load intel/mpi/64/18.0.128
module load intel/mkl/64/11.3.3
module load boost/intel/1.57.0
module load htop
module load cmake/3.5.2
module load intel/compiler/64/18.0.128
module load slurm/16.05.3

MAIN=/home/a/orp20/scratch/firedrake-complex

export PETSC_ARCH=arch-python-linux-x86_64
export PETSC_DIR=${MAIN}/petsc
export PYTHON_DIR=${MAIN}/python3
export PATH="${PYTHON_DIR}/bin:$PATH"

export MPICC=mpiicc
export MPICXX=mpiicpc
export MPIF90=mpiifort
export INTEL_LICENSE_FILE=/cm/shared/licenses/intel/
