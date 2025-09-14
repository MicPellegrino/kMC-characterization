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

# TODO: Support for MEAM and NNPs
def lammps_topology(lmp, substrate_file, ff_file, sub_an_list, ada_an_list, 
    ff_style='eam/alloy', na_sub=1, na_ada=1) :

    """ Define inital configuration, atom types and groups, force field """

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

def lammps_dump(lmp, 
    sub_traj='substrate.dcd', coll_traj='collisions.dump', tout=125, ndump=25, nevery=25, nrepeat=10, nfreq=250) :

    """ Defining the output observables and trajectories """

    # Dumping impacting atoms in .dump files and substarte in .dcd file
    output_definitions=f"""
    variable pea_avg equal "pe/atoms"
    thermo {tout}
    thermo_style custom step v_pea_avg pe temp lx ly lz press
    variable dummyMol atom "gmask(substrate)+2.0*gmask(adatoms)+3.0*gmask(frozen)"
    dump myDcd substrate dcd {ndump} {sub_traj}
    dump myDump adatoms custom {ndump} {coll_traj} id type x y z xu yu zu vx vy vz v_dummyMol
    fix avePe adatoms ave/time {nevery} {nrepeat} {nfreq} v_pea_avg ave one file pe.dat
    """
    lmp.commands_string(output_definitions)

def lammps_md(lmp, T, Tdamp=1.0) :

    """ Defining MD fixes """

    md_fixes=f"""
    fix myMD1 substrate nvt temp {T} {T} 1.0
    fix myMD2 adatoms nve
    """
    lmp.commands_string(md_fixes)

def lammps_inflow(lmp, xlow, xupp, ylow, yupp, zlow, zupp) :

    """ Define region where to generate incoming atoms """

    inflow=f"""
    region inflow block {xlow} {xupp} {ylow} {yupp} {zlow} {zupp}
    group newatom dynamic adatoms region inflow
    """
    lmp.commands_string(inflow)

def lammps_coat(lmp, Na, atype_vec, xr, yr, vabs, zgen, 
    nrun=250) :
    
    for n in range(Na) :
        lmp.command(f"create_atoms {atype_vec[n]} single {xr[n]} {yr[n]} {zgen} group adatoms")
        lmp.command("run 0 post no")
        lmp.command(f"velocity newatom set 0.0 0.0 {-vabs[n]}")
        lmp.command(f"run {nrun}")