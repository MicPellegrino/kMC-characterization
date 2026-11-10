#!/bin/bash

cd symmetrix/pair_symmetrix
./install.sh ../../lammps-2Sep2026
cd ../..

mkdir lammps-2Sep2026/build-symmetrix
cd lammps-2Sep2026/build-symmetrix

cmake \
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
    ../cmake

make -j 56

make install-python
