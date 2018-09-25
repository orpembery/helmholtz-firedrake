#!/bin/bash

###
# Setup
###

# Default modules to load
module load git/2.5.1
module load intel/mpi/64/18.0.128
module load intel/mkl/64/11.3.3
module load boost/intel/1.57.0
module load htop
module load cmake/3.5.2
module load intel/compiler/64/18.0.128
module load slurm #/16.05.3
#module load hdf5/gcc/1.8.17

# Set main to be working directory
MAIN=`pwd`

# Load python2
module load python/2.7.8

###
# Download and install PETSc
###
git clone https://github.com/firedrakeproject/petsc.git

export PETSC_ARCH=arch-python-linux-x86_64
unset PETSC_DIR
cd ./petsc

# Configure PETSc
# remove --prefix??
# Last two options weren't in Jack's script, but were passed by the complex install script
./configure --with-shared-libraries=1 --with-debugging=0 --with-c2html=0 --with-cc=mpiicc --with-cxx=mpiicpc --with-fc=mpiifort --download-fblaslapack --download-eigen --with-fortran-bindings=0 --download-chaco --download-metis --download-parmetis --download-scalapack --download-hypre --download-mumps --download-netcdf --download-hdf5 --download-pnetcdf # --download-exodusii --with-scalar-type=complex

# Build PETSc
make -j 17 PETSC_DIR=${MAIN}/petsc PETSC_ARCH=arch-python-linux-x86_64 all

# REMOVE???
#make -j 17 PETSC_DIR=${MAIN}/petsc PETSC_ARCH=arch-python-linux-x86_64 install
# CHECK???
make -j 17 PETSC_DIR=${MAIN}/petsc PETSC_ARCH=arch-python-linux-x86_64 check
#make -j 17 PETSC_DIR=${MAIN}/petsc PETSC_ARCH=arch-python-linux-x86_64 streams # This checks the scaling and is optional


# Set PETSc directory
export PETSC_DIR=${MAIN}/petsc
cd ..

###
# Download and install python3
###
wget https://www.python.org/ftp/python/3.6.5/Python-3.6.5.tgz
mkdir python3
tar -xzvf Python-3.6.5.tgz -C ./python3 --strip-components=1
unset PYTHON_DIR
export PYTHON_DIR=${MAIN}/python3
cd ./python3

# Configure
# try with --with-tcltk-includes='-I/apps/python/intel/2018.1.023/intelpython3/include'
# and --with-tcltk-libs='-L/apps/python/intel/2018.1.023/intelpython3/lib -ltcl8.6 -ltk8.6'
./configure --enable-shared --enable-ipv6 --with-ensurepip=yes --prefix=${PYTHON_DIR} CPPFLAGS=-I${PYTHON_DIR}/include LDFLAGS="-L${PYTHON_DIR}/lib -Wl,-rpath=${PYTHON_DIR}/lib,--no-as-needed" CFLAGS="-Wformat -Wformat-security -D_FORTIFY_SOURCE=2 -fstack-protector -O3 -fpic -fPIC" PKG_CONFIG_PATH=${PYTHON_DIR}/lib/pkgconfig --enable-optimizations

# Build Python3
make -j 17
make -j 17 install

# Add python to path
export PATH="${PYTHON_DIR}/bin:$PATH"
cd ..

# Remove references to inbuilt python - crashes otherwise
unset PYTHONHOME

# May need:
python3 -m ensurepip

###
# Download and install firedrake
###
wget https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install

export MPICC=mpiicc
export MPICXX=mpiicpc
export MPIF90=mpiifort

module unload python/2.7.8
# remove???
#unset PETSC_ARCH

export INTEL_LICENSE_FILE=/cm/shared/licenses/intel/
### This line doesn't work, python3 complains about not being able to find encodings module, but then runs fine in terminal after
python3 firedrake -install --mpicc=mpiicc --mpicxx=mpiicpc --mpif90=mpiifort --no-package-manager --disable-ssh --honour-petsc-dir

# Add paths to .bashrc (only do this once!)
# echo PETSC_DIR=${MAIN}/petsc >> ~/.bashrc
# echo PATH="${PYTHON_DIR}/bin:$PATH" >> ~.bashrc
