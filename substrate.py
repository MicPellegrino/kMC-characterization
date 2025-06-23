import lammps
import numpy as np
from mpi4py import MPI

#####################################################################

def lmp_header(lmp) :
    
    """ Typical header for simulations with metals """

    commands="""
    units metal 
    dimension 3 
    boundary p p p
    atom_style atomic
    atom_modify map array
    """
    lmp.commands_string(commands)

def lmp_lattice(lmp, a, nx, ny, ns, phase='fcc', orient='100') :

    """ Definition of the crystal lattice """

    assert phase=='fcc' or phase=='bcc', "only FCC and BCC phases available at the moment"
    if orient=='100' :
        command_lattice=f"lattice {phase} {a} orient x 1 0 0 orient y 0 1 0 orient z 0 0 1"
    elif orient=='110' :
        command_lattice=f"lattice {phase} {a} orient x 0 0 1 orient y 1 1 0 orient z -1 1 0"
    elif orient=='111' :
        command_lattice=f"lattice {phase} {a} orient x 1 -1 0 orient y 1 1 -2 orient z 1 1 1"
    else :
        print("!! Error: only 100, 110 and 111 orientation available at the moment !!")
        command_lattice=""
    lmp.command(command_lattice)
    command_region=f"region box block 0 {nx} 0 {ny} 0 {ns} units lattice"
    lmp.command(command_region)

def lmp_box(lmp, ntypes, dLx) :
    
    """ Creation of the system box """

    commands=f"""
    create_box {ntypes} box
    create_atoms 1 box
    change_box all z delta {-dLx} {dLx} boundary p p p
    lattice none 1.0
    """
    lmp.commands_string(commands)

def lmp_potential_eam(lmp, ffname, type_names, flavour='eam/alloy') :

    """ Definition of the interatomic potential (Embedded Atom Model) """

    command_style=f"pair_style {flavour}"
    lmp.command(command_style)

    command_pair=f"pair_coeff * * {ffname}"
    for n in type_names :
        command_pair += (' '+n)
    lmp.command(command_pair)

    # In principle these should not be touched!
    commands="""
    neighbor 2.0 bin
    neigh_modify delay 0 every 1 check yes
    """

#####################################################################

if __name__ == "__main__" :

    ### TESTING SUBSTRATE GENERATION ###
    lmp = lammps.lammps()
    lmp_header(lmp)
    lmp_lattice(lmp,a=4.05,nx=31,ny=31,ns=7,phase='fcc',orient='100')
    # lmp_lattice(lmp,a=4.05,nx=31,ny=int(31/np.sqrt(2)),ns=int(7/np.sqrt(2)),phase='fcc',orient='110')
    # lmp_lattice(lmp,a=4.05,nx=int(31/np.sqrt(2)),ny=int(1.5*31/np.sqrt(6)),ns=int(7/np.sqrt(3)),phase='fcc',orient='111')
    lmp_box(lmp,ntypes=2,dLx=10.0)
    lmp_potential_eam(lmp,'test/CuAgAuNiPdPtAlPbFeMoTaWMgCoTiZr_Zhou04.eam.alloy',['Al', 'Al'],flavour='eam/alloy')
    lmp.command("write_data test1.data")