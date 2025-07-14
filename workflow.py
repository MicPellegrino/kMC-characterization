import lammps
import lammps_wrapper as lmp_wrap
import numpy as np
from random_distributions import uniform_unit_hemisphere, kinetic_energy, velocity_distribution, plane_uniform
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

T = 300


# Arrays containing the initial velocities and positions 
# for the PVD atoms
if me==0 :
    vx, vy, vz, vabs = velocity_distribution(Ed,m,Na)
    xr, yr = plane_uniform(0,125.55,0,125.55,Na) 
else :
    vx = np.empty(Na)
    vy = np.empty(Na)
    vz = np.empty(Na)
    vabs = np.empty(Na)
    xr = np.empty(Na)
    yr = np.empty(Na)
comm.Bcast(vx, root=0)
comm.Bcast(vy, root=0)
comm.Bcast(vz, root=0)
comm.Bcast(vabs, root=0)
comm.Bcast(xr, root=0)
comm.Bcast(yr, root=0)

# TODO: 'cmdargs' should be passed as input when calling the script from the cmd line
lmp = lammps.lammps()
# lmp = lammps.lammps(cmdargs=['-pk','gpu','1','-sf','gpu'])

# Defining units and boundary conditions
lmp_wrap.lammps_units(lmp)

# Defining output frequency
lmp_wrap.lammps_nstout(lmp)

# Initial substrate configuration and system topology
lmp_wrap.lammps_topology(lmp, substrate_file, ff_file, sub_an_list, ada_an_list)

# Freezing some of the lower layers of the substrate to prevent downward motion
lmp_wrap.lammps_freeze(lmp, xlowf, xuppf, ylowf, yuppf, zlowf, zuppf)

# Saving pre
lmp.command("write_data collisions_pre.data")

# Defining LAMMPS output
lmp_wrap.lammps_dump(lmp)

# Molecular Dynamics fixes
lmp_wrap.lammps_md(lmp, T)

# lmp.command("region inflow sphere 63.0 63.0 55.0 1.0")
# lmp.command("region inflow block 0 125.55 0 125.55 50 68.85")
lmp.command("region inflow block 0 125.55 0 125.55 40 50")
lmp.command("group newatom dynamic adatoms region inflow")
for n in range(Na) :
    lmp.command(f"create_atoms 2 single {xr[n]} {yr[n]} 45.0 group adatoms")
    lmp.command("run 0 post no")
    lmp.command(f"velocity newatom set 0.0 0.0 {-vabs[n]}")
    lmp.command("run ${nrun}")

# Saving after
lmp.command("write_data collisions_post.data")

MPI.Finalize()
