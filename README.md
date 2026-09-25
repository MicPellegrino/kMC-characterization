Tested on `Python 3.10.0` and LAMMPS `22Jul2025`.
List of packages:
```
Package                Version
---------------------- ---------
lammps                 2025.7.22
numpy                  1.26.4
ovito                  3.11.1
packaging              26.3
pip                    26.2.1
PySide6                6.7.3
setuptools             84.0.0
shiboken6              6.7.3
traits                 7.1.0
WarrenCowleyParameters 3.0.1
wheel                  0.48.0
```
Ovito is installed after creating a `mamba` venv with the desired Python version:
```
mamba env create -n kmc python=3.10.0
```
activating it:
```
mamba activate kmc
```
and running:
```
mamba install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.11.1
```
Other packages you may want to install:
```
matplotlib
mpi4py
ase
scipy
cython
```
I suggest using `pip` to install additional packages.

TODO
- Include a `fix` to compute the MSD of PVD atoms (not sure if it's correct or useful);
- Include support for MACE via ML-IAP interface
