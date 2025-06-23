#!/bin/bash

mkdir lammps-12Jun2025/build-cuda
cd lammps-12Jun2025/build-cuda

cmake \
    -D BUILD_MPI=on \
    -D BUILD_LIB=on \
    -D BUILD_SHARED_LIBS=on \
    -D PKG_OPENMP=on \
    -D PKG_GPU=on \
    -D GPU_API=cuda \
    -D PKG_MOLECULE=on \
    -D PKG_MANYBODY=on \
    -D PKG_KSPACE=on \
    -D PKG_RIGID=on \
    -D PKG_REAXFF=on \
    -D PKG_EXTRA-DUMP=on \
    -D PKG_EXTRA-FIX=on \
    -D PKG_MC=on \
    -D PKG_MEAM=on \
    -D CMAKE_INSTALL_PREFIX=$(pwd) \
    ../cmake

make -j 56

make install-python
