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

    """ Define variables for output frequencies """
    
    output_variables=f"""
    variable tout equal {tout}
    variable ndump equal {ndump}
    variable nevery equal {nevery}
    variable nrepeat equal {nrepeat}
    variable nfreq equal {nfreq}
    variable nrun equal {nrun}
    """
    lmp.commands_string(output_variables)

def lammps_topology(lmp, substrate_file, ff_file, sub_an_list, ada_an_list, 
    ff_style='eam/alloy', na_sub=1, na_ada=1) :

    """ Define inital coonfiguration, atom types and groups, force field """

    lmp.command(f"read_data {substrate_file} extra/atom/types {na_ada}")

    types_sub = ""
    for t in range(1,na_sub+1) :
        types_sub += (str(t)+" ")
    types_ada = ""
    for t in range(na_sub+1,na_sub+na_ada+1) :
        types_ada += (str(t)+" ")

    type_names_string = ""
    for a in sub_an_list :
        type_names_string += (a+" ")
    for a in ada_an_list :
        type_names_string += (a+" ")

    topology=f"""
    group substrate type {types_sub}
    group adatoms type {types_ada}
    pair_style {ff_style} 
    pair_coeff * * {ff_file} {type_names_string}
    neighbor 2.0 bin
    neigh_modify delay 0 every 1 check yes
    """
    lmp.commands_string(topology)

def lammps_freeze(lmp, xlow, xupp, ylow, yupp, zlow, zupp) :

    """ Define the portion of the substrate that will be frozen """

    freeze=f"""
    region lowsub block {xlow} {xupp} {ylow} {yupp} {zlow} {zupp}
    group frozen region lowsub
    velocity frozen set 0.0 0.0 0.0
    fix myFreeze frozen setforce 0.0 0.0 0.0
    """
    lmp.commands_string(freeze)

# TODO: output file names should be given as input
# Also, remove the LAMMPS variables for output frequencies, 
# it's cleaner to keep all variables in Python
def lammps_dump(lmp) :

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