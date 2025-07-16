import lammps
import lammps_wrapper as lmp_wrap
import numpy as np
from random_distributions import *
from parser import load_input_file
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
me = comm.Get_rank()
nprocs = comm.Get_size()

### TODO: I/O should only be from rank 0 ###
params = load_input_file("input.txt")
Ed = params["Ed"]
Na = params["Na"]
m = params["m"]
na_sub = params["na_sub"]
na_ada = params["na_ada"]
frac_list = params["frac_list"]
substrate_file = params["substrate_file"]
ff_file = params["ff_file"]
sub_an_list = params["sub_an_list"]
ada_an_list = params["ada_an_list"]
xlowf = params["xlowf"]
xuppf = params["xuppf"]
ylowf = params["ylowf"]
yuppf = params["yuppf"]
zlowf = params["zlowf"]
zuppf = params["zuppf"]
xlowi = params["xlowi"]
xuppi = params["xuppi"]
ylowi = params["ylowi"]
yuppi = params["yuppi"]
zlowi = params["zlowi"]
zuppi = params["zuppi"]
T = params["T"]

# Arrays containing the initial velocities and positions of PVD atoms.
if me==0 :
    vx, vy, vz, vabs = velocity_distribution(Ed,m,Na)
    xr, yr = plane_uniform(0,125.55,0,125.55,Na)
    type_list = np.arange(na_sub+1,na_sub+na_ada+1)
    atype_vec = gen_atype_vector(type_list,frac_list,Na)
else :
    vx = np.empty(Na,dtype=np.float64)
    vy = np.empty(Na,dtype=np.float64)
    vz = np.empty(Na,dtype=np.float64)
    vabs = np.empty(Na,dtype=np.float64)
    xr = np.empty(Na,dtype=np.float64)
    yr = np.empty(Na,dtype=np.float64)
    atype_vec = np.empty(Na,dtype=int)
comm.Bcast(vx, root=0)
comm.Bcast(vy, root=0)
comm.Bcast(vz, root=0)
comm.Bcast(vabs, root=0)
comm.Bcast(xr, root=0)
comm.Bcast(yr, root=0)
comm.Bcast(atype_vec, root=0)

# LAMMPS 'cmdargs' is passed as input when calling the script from the cmd line
lmp_cmdargs = ' '.join(sys.argv[1:])
lmp = lammps.lammps(cmdargs=lmp_cmdargs.split())

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

# Simulation of the actual coating process
lmp_wrap.lammps_coat(lmp, Na, atype_vec, xr, yr, vabs, zgen=45)

# Saving after
lmp.command("write_data collisions_post.data")

MPI.Finalize()
