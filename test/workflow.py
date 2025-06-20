import lammps
import numpy as np
from random_distributions import uniform_unit_hemisphere, kinetic_energy, velocity_distribution, plane_uniform
from mpi4py import MPI

comm = MPI.COMM_WORLD
me = comm.Get_rank()
nprocs = comm.Get_size()

Ed = 10
Na = 5000
m_Al = 26.982

if me==0 :
    vx, vy, vz, vabs = velocity_distribution(Ed,m_Al,Na)
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

##### LAMMPS run ##### 
lmp = lammps.lammps()
# lmp = lammps.lammps(cmdargs=['-pk','gpu','1','-sf','gpu'])

substrate_file = "Al_100_relax.data"

initialization_commands="""
units metal 
dimension 3 
boundary p p p
atom_style atomic
atom_modify map array
"""
lmp.commands_string(initialization_commands)

# Most of these should be dynamic
output_variables="""
variable tout equal 125
variable ndump equal 25
variable nevery equal 25
variable nrepeat equal 10
variable nfreq equal 250
variable nrun equal 250
"""
lmp.commands_string(output_variables)

lmp.command(f"read_data {substrate_file}")

topology="""
group substrate type 1
group adatoms type 2
include "potential.lammps"
"""
lmp.commands_string(topology)

# Freeze some of the lower layers of the substrate to prevent downward motion
freeze="""
region lowsub block 0 125.55 0 125.55 -40.5 8.2
group frozen region lowsub
velocity frozen set 0.0 0.0 0.0
fix myFreeze frozen setforce 0.0 0.0 0.0
"""
lmp.commands_string(freeze)

# Saving pre
lmp.command("write_data collisions_pre.data")

# Dumping impacting atoms in .dump files and substarte in .dcd file
output_definitions="""
variable pea_avg equal "pe/atoms"
thermo ${tout}
thermo_style custom step v_pea_avg pe temp lx ly lz press
variable dummyMol atom "gmask(substrate)+2.0*gmask(adatoms)+3.0*gmask(frozen)"
dump myDcd substrate dcd ${ndump} substrate.dcd
dump myDump adatoms custom ${ndump} collisions.dump id type x y z xu yu zu vx vy vz v_dummyMol
fix avePe adatoms ave/time ${nevery} ${nrepeat} ${nfreq} v_pea_avg ave one file pe.dat
"""
lmp.commands_string(output_definitions)

md_fixes="""
fix myMD1 substrate nvt temp 300.0 300.0 1.0
fix myMD2 adatoms nve
"""
lmp.commands_string(md_fixes)

# lmp.command("region inflow sphere 63.0 63.0 55.0 1.0")
# lmp.command("region inflow block 0 125.55 0 125.55 50 68.85")
lmp.command("region inflow block 0 125.55 0 125.55 38 42")
lmp.command("group newatom dynamic adatoms region inflow")
for n in range(Na) :
    lmp.command(f"create_atoms 2 single {xr[n]} {yr[n]} 40.0 group adatoms")
    lmp.command("run 0 post no")
    lmp.command(f"velocity newatom set 0.0 0.0 {-vabs[n]}")
    lmp.command("run ${nrun}")

# Saving after
lmp.command("write_data collisions_post.data")

MPI.Finalize()
