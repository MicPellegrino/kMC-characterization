import lammps
import lammps_wrapper as lmp_wrap
import numpy as np
from random_distributions import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
me = comm.Get_rank()
nprocs = comm.Get_size()

### TODO: all these inputs should be in a separate file (like a Gromacs .mdp) ###

Ed = 10
Na = 5000

# m = 26.982 # Al
# m = 95.94 # Mo
m = 58.693 # Ni

na_sub = 1
na_ada = 1
frac_list = [1.0]
substrate_file = "substrates/Ni_100.data"
ff_file = "test/CuAgAuNiPdPtAlPbFeMoTaWMgCoTiZr_Zhou04.eam.alloy"
sub_an_list = ['Ni']
ada_an_list = ['Ni']
xlowf = 0
xuppf = 125.55
ylowf = 0
yuppf = 125.55
zlowf = -40.5
zuppf = 8.2
xlowi = 0
xuppi = 125.55
ylowi = 0
yuppi = 125.55
zlowi = 40
zuppi = 50
T = 300

### ------------------------------------------------------------------------- ###

# Arrays containing the initial velocities and positions of PVD atoms.
# TODO: generate an array for the atomtype
if me==0 :
    vx, vy, vz, vabs = velocity_distribution(Ed,m,Na)
    xr, yr = plane_uniform(0,125.55,0,125.55,Na)
    type_list = np.arange(na_sub+1,na_sub+na_ada+1)
    atype_vec = gen_atype_vector(type_list,frac_list,Na)
else :
    vx = np.empty(Na)
    vy = np.empty(Na)
    vz = np.empty(Na)
    vabs = np.empty(Na)
    xr = np.empty(Na)
    yr = np.empty(Na)
    atype_vec = np.empty(Na)
comm.Bcast(vx, root=0)
comm.Bcast(vy, root=0)
comm.Bcast(vz, root=0)
comm.Bcast(vabs, root=0)
comm.Bcast(xr, root=0)
comm.Bcast(yr, root=0)
comm.Bcast(atype_vec, root=0)

# TODO: 'cmdargs' should be passed as input when calling the script from the cmd line
lmp = lammps.lammps()
# lmp = lammps.lammps(cmdargs=['-pk','gpu','1','-sf','gpu'])

# Defining units and boundary conditions
lmp_wrap.lammps_units(lmp)

# Defining output frequency
lmp_wrap.lammps_nstout(lmp)

# Initial substrate configuration and system topology
lmp_wrap.lammps_topology(lmp, substrate_file, ff_file, sub_an_list, ada_an_list, 
    na_sub=na_sub, na_ada=na_ada)

# Freezing some of the lower layers of the substrate to prevent downward motion
lmp_wrap.lammps_freeze(lmp, xlowf, xuppf, ylowf, yuppf, zlowf, zuppf)

# Saving pre
lmp.command("write_data collisions_pre.data")

# Defining LAMMPS output
lmp_wrap.lammps_dump(lmp)

# Molecular Dynamics fixes
lmp_wrap.lammps_md(lmp, T)

# Previous attempts
# lmp.command("region inflow sphere 63.0 63.0 55.0 1.0")
# lmp.command("region inflow block 0 125.55 0 125.55 50 68.85")

# Defining inflow region
lmp_wrap.lammps_inflow(lmp, xlowi, xuppi, ylowi, yuppi, zlowi, zuppi)

# TODO: case where there is more than 1 atomtype for adatoms
# The atom type to pass to 'create_atoms' should be selected from a vector
for n in range(Na) :
    lmp.command(f"create_atoms 2 single {xr[n]} {yr[n]} 45.0 group adatoms")
    lmp.command("run 0 post no")
    lmp.command(f"velocity newatom set 0.0 0.0 {-vabs[n]}")
    lmp.command("run ${nrun}")

# Saving after
lmp.command("write_data collisions_post.data")

MPI.Finalize()
