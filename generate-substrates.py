import os
from substrate import *
import lammps
import numpy as np
import numpy.random as rng
from mpi4py import MPI

class Alloy :
    def __init__(self, ntypes, typelist, phase, a):
        self.ntypes = ntypes
        self.typelist = typelist
        self.phase = phase
        self.a = a

def generate_substrate(name,
    alloy,
    nx,
    ny,
    ns,
    dLx,
    ff_name,
    ff_flavour,
    seed,
    orient='100',
    tout=50,
    nsteps=1000,
    GPU=True) :

    if GPU :
        lmp = lammps.lammps(cmdargs=['-pk','gpu','1','-sf','gpu'])
    else :
        lmp = lammps.lammps()
    lmp_header(lmp)
    lmp_lattice(lmp,alloy.a,nx,ny,ns,alloy.phase,orient)
    lmp_box(lmp,alloy.ntypes,dLx)
    lmp_potential_eam(lmp,ff_name,alloy.typelist,flavour=ff_flavour)
    lmp_energy_min(lmp)
    lmp_md_output(lmp,tout=tout)
    lmp_relaxation(lmp,nsteps=nsteps,seed=seed)
    lmp.command(f"write_data {name}")
    lmp.close()


os.system("mkdir substrates")
alloys = dict()
alloys['Al'] = Alloy(1,['Al'],'fcc',4.05)
alloys['Mo'] = Alloy(1,['Mo'],'bcc',3.15)
alloys['Ni'] = Alloy(1,['Ni'],'fcc',3.52)

# Simulation box parameters
dLz = 10.0
ffname = 'test/CuAgAuNiPdPtAlPbFeMoTaWMgCoTiZr_Zhou04.eam.alloy'

### NB! BCC has less atoms per unit cell (and so on...) ###

for an in alloys.keys() :
    print(alloys[an])
    if alloys[an].phase=='fcc' :
        nx_ref = 31
        ny_ref = 31
        ns_ref = 7
    elif alloys[an].phase=='bcc' :
        nx_ref = int(np.round((2**(1/3))*31))
        ny_ref = int(np.round((2**(1/3))*31))
        ns_ref = int(np.round((2**(1/3))*7))
    else :
        print("!! Only FCC and BCC supported at the moment !!")
    # Generate 100 substrate
    nx = nx_ref
    ny = ny_ref
    ns = ns_ref
    name = 'substrates/'+an+'_100.data'
    generate_substrate(name,
        alloys[an],
        nx,
        ny,
        ns,
        dLz,
        ffname,
        ff_flavour='eam/alloy',
        seed=rng.randint(99999),
        orient='100',
        GPU=False)
    # Generate 110 substrate
    nx = nx_ref
    ny = int(np.round(ny_ref/np.sqrt(2)))
    ns = int(np.round(ns_ref/np.sqrt(2)))
    name = 'substrates/'+an+'_110.data'
    generate_substrate(name,
        alloys[an],
        nx,
        ny,
        ns,
        dLz,
        ffname,
        ff_flavour='eam/alloy',
        seed=rng.randint(99999),
        orient='110',
        GPU=False)
    # Generate 111 substrate
    nx = int(np.round(nx_ref/np.sqrt(2)))
    ny = int(np.round(1.5*ny_ref/np.sqrt(6)))
    ns = int(np.round(ns_ref/np.sqrt(3)))
    name = 'substrates/'+an+'_111.data'
    generate_substrate(name,
        alloys[an],
        nx,
        ny,
        ns,
        dLz,
        ffname,
        ff_flavour='eam/alloy',
        seed=rng.randint(99999),
        orient='111',
        GPU=False)
