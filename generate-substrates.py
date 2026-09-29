import os
from substrate import *
import lammps
from lammps_wrapper import LammpsBuild
import numpy as np
import numpy.random as rng
from mpi4py import MPI
import sys

# Global variables
DEFAULT_NATIVE_GPU_FLAGS="-pk gpu 1 -sf gpu"
DEFAULT_KOKKOS_GPU_FLAGS="-k on g 1 -sf kk"
DEFAULT_SEED = 99999
DEFAULT_NX = 31
DEFAULT_NY = 31
DEFAULT_NZ = 7

comm = MPI.COMM_WORLD
idproc = comm.Get_rank()
nprocs = comm.Get_size()

lmp_build = LammpsBuild()

# TODO: support NNPs

# For now we assume equiatomic composition (HEA)
# TODO: support general stoichiometry
class Alloy :
    def __init__(self, ntypes, typelist, phase, a):
        self.ntypes = ntypes
        self.typelist = typelist
        self.phase = phase
        self.a = a

def generate_substrate(lmp_build,
    name,
    alloy,
    nx,
    ny,
    ns,
    dLx,
    ff_name,
    ff_type,
    ff_lib=None,
    seed=1234,
    orient='100',
    tout=50,
    nsteps=1000,
    flags=None) :

    if flags==None :
        lmp_cmdargs = ' '.join(sys.argv[1:])
    else :
        lmp_cmdargs = flags
    lmp = lammps.lammps(cmdargs=lmp_cmdargs.split(),comm=comm)

    lmp_header(lmp)
    lmp_lattice(lmp,alloy.a,nx,ny,ns,alloy.phase,orient)
    lmp_box(lmp,alloy.ntypes,dLx)
    if ff_type == 'eam' or ff_type == 'eam/alloy' or ff_type == 'eam/fs' :
        lmp_potential_eam(lmp,ff_name,alloy.typelist,ff_type)
    if ff_type == 'meam' or  ff_type == 'meam/ms' :
        lmp_potential_meam(lmp,ff_name,ff_lib,alloy.typelist,ff_type)
    lmp_energy_min(lmp)
    lmp_md_output(lmp,tout=tout)
    lmp_relaxation(lmp,nsteps=nsteps,seed=seed)
    lmp.command(f"write_data {name}")

    lmp.close()
    if lmp_build.has_kokkos_cuda_support :
        lmp.lib.lammps_kokkos_finalize()


os.system("mkdir -p substrates")
alloys = dict()
# alloys['Al'] = Alloy(1,['Al'],'fcc',4.05)
# alloys['Mo'] = Alloy(1,['Mo'],'bcc',3.15)
# alloys['Ni'] = Alloy(1,['Ni'],'fcc',3.52)
# alloys['AlTi'] = Alloy(2,['Al','Ti'],'fcc',4.05)
# alloys['AlTi_bcc'] = Alloy(2,['Al','Ti'],'bcc',3.179)
alloys['CoFeNi_fcc'] = Alloy(3,['Co','Fe','Ni'],'fcc',3.58)

# Simulation box parameters
dLz = 10.0
# fftype = 'eam/alloy'
# ffname = 'test/CuAgAuNiPdPtAlPbFeMoTaWMgCoTiZr_Zhou04.eam.alloy'
# ffname = 'test/FeNiCrCoCu-with-ZBL.eam.alloy'
fftype = 'meam'
ffname = 'test/CoNiCrFeMn-meam/CoNiCrFeMn.meam'
fflib = 'test/CoNiCrFeMn-meam/library.meam Co Ni Cr Fe Mn'

### NB! BCC has less atoms per unit cell (and so on...) ###

for an in alloys.keys() :
    if alloys[an].phase=='fcc' :
        nx_ref = DEFAULT_NX
        ny_ref = DEFAULT_NY
        ns_ref = DEFAULT_NZ
    elif alloys[an].phase=='bcc' :
        nx_ref = int(np.round((2**(1/3))*31))
        ny_ref = int(np.round((2**(1/3))*31))
        ns_ref = int(np.round((2**(1/3))*7))
    else :
        if idproc == 0 :
            print("!! Only FCC and BCC supported at the moment !!")
    # Generate 100 substrate
    nx = nx_ref
    ny = ny_ref
    ns = ns_ref
    name = 'substrates/'+an+'_100.data'
    generate_substrate(lmp_build,
        name,
        alloys[an],
        nx,
        ny,
        ns,
        dLz,
        ffname,
        ff_type=fftype,
        ff_lib=fflib,
        seed=rng.randint(DEFAULT_SEED),
        orient='100')
    # Generate 110 substrate
    nx = nx_ref
    ny = int(np.round(ny_ref/np.sqrt(2)))
    ns = int(np.round(ns_ref/np.sqrt(2)))
    name = 'substrates/'+an+'_110.data'
    generate_substrate(lmp_build,
        name,
        alloys[an],
        nx,
        ny,
        ns,
        dLz,
        ffname,
        ff_type=fftype,
        ff_lib=fflib,
        seed=rng.randint(DEFAULT_SEED),
        orient='110')
    # Generate 111 substrate
    nx = int(np.round(nx_ref/np.sqrt(2)))
    ny = int(np.round(1.5*ny_ref/np.sqrt(6)))
    ns = int(np.round(ns_ref/np.sqrt(3)))
    name = 'substrates/'+an+'_111.data'
    generate_substrate(lmp_build,
        name,
        alloys[an],
        nx,
        ny,
        ns,
        dLz,
        ffname,
        ff_type=fftype,
        ff_lib=fflib,
        seed=rng.randint(DEFAULT_SEED),
        orient='111')

MPI.Finalize()