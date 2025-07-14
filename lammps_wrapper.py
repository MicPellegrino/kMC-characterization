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

def lammps_nstout(lmp, tout=125, ndump=25, nevery=25, nrepeat=10, nfreq=250, nrun=250) :

    # Most of these should be dynamic
    output_variables=f"""
    variable tout equal {tout}
    variable ndump equal {ndump}
    variable nevery equal {nevery}
    variable nrepeat equal {nrepeat}
    variable nfreq equal {nfreq}
    variable nrun equal {nrun}
    """
    lmp.commands_string(output_variables)

def lammps_topology(lmp, substrate_file, na_sub=1, na_ada=1) :

    """ Define inital coonfiguration, atom types and groups, force field """

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