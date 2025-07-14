import lammps

def lammps_units(lmp) :

    """ Default units and b.c. for metallic systems """

    initialization_commands="""
    units metal 
    dimension 3 
    boundary p p p
    atom_style atomic
    atom_modify map array
    """
    lmp.commands_string(initialization_commands)

def lammps_topology(lmp, substrate_file) :

    lmp.command(f"read_data {substrate_file} extra/atom/types 1")

    topology="""
    group substrate type 1
    group adatoms type 2
    variable ffname string "test/CuAgAuNiPdPtAlPbFeMoTaWMgCoTiZr_Zhou04.eam.alloy"
    pair_style eam/alloy 
    pair_coeff * * ${ffname} Mo Mo
    neighbor 2.0 bin
    neigh_modify delay 0 every 1 check yes
    """
    lmp.commands_string(topology)