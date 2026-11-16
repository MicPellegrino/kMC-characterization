#!/bin/bash

# Probably not needed for Symmetrix (inference)
# pip3 install torch torchvision
# pip3 install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu13
# pip3 install cupy-cuda13x
# pip3 install mace-torch

cd symmetrix/symmetrix
pip install .
cd ../..

LAMMPS_FOLDER="lammps-30Sep2026"

cd symmetrix/pair_symmetrix
./install.sh "../../${LAMMPS_FOLDER}"
cd ../..

rm -rf "${LAMMPS_FOLDER}/build-symmetrix"
mkdir -p "${LAMMPS_FOLDER}/build-symmetrix"
cd "${LAMMPS_FOLDER}/build-symmetrix"

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=20 \
    -D CMAKE_C_COMPILER=${CC} \
    -D CMAKE_CXX_COMPILER=${CXX} \
    -D CMAKE_Fortran_COMPILER=ftn \
    -D CMAKE_CXX_STANDARD_REQUIRED=ON \
    -D CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -march=native -ffast-math" \
    -D SYMMETRIX_KOKKOS=on \
    -D SYMMETRIX_SPHERICART_CUDA=on \
    -D PKG_KOKKOS=on \
    -D Kokkos_ENABLE_CUDA=on \
    -D Kokkos_ENABLE_OPENMP=on \
    -D Kokkos_ARCH_SPR=on \
    -D Kokkos_ARCH_AMPERE86=on \
    -D Kokkos_PREC=mixed \
    -D Kokkos_ENABLE_SERIAL=on \
    -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=on \
    -D FFT_KOKKOS=CUFFT \
    -D BUILD_OMP=ON \
    -D BUILD_MPI=on \
    -D PKG_PYTHON=on \
    -D BUILD_LIB=on \
    -D BUILD_SHARED_LIBS=on \
    -D PKG_OPENMP=on \
    -D PKG_GPU=off \
    -D PKG_MOLECULE=on \
    -D PKG_MANYBODY=on \
    -D PKG_KSPACE=on \
    -D PKG_RIGID=on \
    -D PKG_REAXFF=on \
    -D PKG_FEP=on \
    -D PKG_REPLICA=on \
    -D PKG_EXTRA-DUMP=on \
    -D PKG_EXTRA-FIX=on \
    -D PKG_EXTRA-COMPUTE=on \
    -D PKG_EXTRA-PAIR=on \
    -D PKG_EXTRA-COMMAND=on \
    -D PKG_MC=on \
    -D PKG_MEAM=on \
    -D CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS=OFF \
    -D CMAKE_CXX_USE_RESPONSE_FILE_FOR_LIBRARIES=OFF \
    ../cmake

ulimit -S -s 65536

make -j 56
make install-python
cd ../..

# Install other libraries
pip3 install ase scipy matplotlib mpi4py cython
pip3 install -U ovito
