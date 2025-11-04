# Library of Ovito wrappers to perform basic analysis (PTM, density profiles, ...)

from ovito.io import *
from ovito.modifiers import *
import numpy as np

class Counts:
    def __init__(self):
        self.OTHER = 0
        self.FCC = 0
        self.HCP = 0
        self.BCC = 0
        self.ICO = 0
        self.SC = 0
        self.CUBIC_DIAMOND = 0
        self.HEX_DIAMOND = 0
        self.GRAPHENE = 0

def polyhedral_template_matching(conf_data,conf_dump=None) :
    
    if conf_dump == None :
        pipeline_ptm = import_file(conf_data)
    else :
        # TODO: How to properly append the .data file?
        pipeline_ptm = import_file(conf_dump)
    pipeline_ptm.modifiers.append(PolyhedralTemplateMatchingModifier())
    
    fractions = dict()
    fractions['OTHER'] = []
    fractions['FCC'] = []
    fractions['HCP'] = []
    fractions['BCC'] = []
    fractions['ICO'] = []
    fractions['SC'] = []
    fractions['CUBIC_DIAMOND'] = []
    fractions['HEX_DIAMOND'] = []
    fractions['GRAPHENE'] = []

    for data in pipeline_ptm.frames:
        n_other = data.attributes['PolyhedralTemplateMatching.counts.OTHER']
        f_other = n_other / data.particles.count
        fractions['OTHER'].append(f_other)
        n_fcc = data.attributes['PolyhedralTemplateMatching.counts.FCC']
        f_fcc = n_fcc / data.particles.count
        fractions['FCC'].append(f_fcc)
        n_hcp = data.attributes['PolyhedralTemplateMatching.counts.HCP']
        f_hcp = n_hcp / data.particles.count
        fractions['HCP'].append(f_hcp)
        n_bcc = data.attributes['PolyhedralTemplateMatching.counts.BCC']
        f_bcc = n_bcc / data.particles.count
        fractions['BCC'].append(f_bcc)
        n_ico = data.attributes['PolyhedralTemplateMatching.counts.ICO']
        f_ico = n_ico / data.particles.count
        fractions['ICO'].append(f_ico)
        n_sc = data.attributes['PolyhedralTemplateMatching.counts.SC']
        f_sc = n_sc / data.particles.count
        fractions['SC'].append(f_sc)
        n_cubicdiamond = data.attributes['PolyhedralTemplateMatching.counts.CUBIC_DIAMOND']
        f_cubicdiamond = n_cubicdiamond / data.particles.count
        fractions['CUBIC_DIAMOND'].append(f_cubicdiamond)
        n_hexdiamond = data.attributes['PolyhedralTemplateMatching.counts.HEX_DIAMOND']
        f_hexdiamond = n_hexdiamond / data.particles.count
        fractions['HEX_DIAMOND'].append(f_hexdiamond)
        n_graphene = data.attributes['PolyhedralTemplateMatching.counts.GRAPHENE']
        f_graphene = n_graphene / data.particles.count
        fractions['GRAPHENE'].append(f_graphene)

    return fractions


def density_profile(conf_data,conf_dump=None,nbins=20) :
    
    if conf_dump == None :
        pipeline_ptm = import_file(conf_data)
    else :
        # TODO: How to properly append the .data file?
        pipeline_ptm = import_file(conf_dump)

    bin_profiles = []
    bin_centres = []
    for data in pipeline_ptm.frames:
        # Assuming it's a cubic cell (i.e. non triclinic)
        box_xx = data.cell[0,0]
        box_yy = data.cell[1,1]
        box_zz = data.cell[2,2]
        centre_z = data.cell[2,3]
        dz = box_zz/nbins
        dV = box_xx*box_yy*dz
        bin_centres.append(np.linspace(0.5*dz,box_zz-0.5*dz,nbins))
        p_types = set()
        for pt in data.particles["Particle Type"][...] :
            p_types.add(pt)
        bin_vals = dict()
        for pt in p_types :
            z_coord_pt = data.particles.positions[data.particles['Particle Type']==pt][:,2]
            ntype = len(z_coord_pt)
            bin_vals[pt] = np.zeros(nbins)
            for n in range(ntype) :
                z = z_coord_pt[n]-centre_z
                z = z % box_zz
                assert (z>=0 and z<box_zz), "Particle outside the simulation box!"
                k = int(z/dz)
                bin_vals[pt][k] += 1.0
            bin_vals[pt] /= dV
        bin_profiles.append(bin_vals)

    return bin_profiles, bin_centres

### TESTS ###

def test_polyhedral_template_matching() :
    
    import matplotlib.pyplot as plt 
    conf_data = "test/test-files/cofeni.data"
    conf_dump = "test/test-files/cofeni.dump"
    print("Testing without trajectory")
    polyhedral_template_matching(conf_data)
    print("Testing with trajectory")
    fractions = polyhedral_template_matching(conf_data,conf_dump)
    for ptm_type in fractions.keys() :
        plt.plot(fractions[ptm_type], label=ptm_type)
    plt.legend()
    plt.xlabel('frame')
    plt.ylabel('fraction')
    plt.show()

def test_density_profile() :
    
    import matplotlib.pyplot as plt 
    conf_data = "test/test-files/cofeni.data"
    conf_dump = "test/test-files/cofeni.dump"
    print("Testing without trajectory")
    bin_profiles, bin_centres = density_profile(conf_data)
    plt.plot(bin_centres[0],bin_profiles[0][1], label='type 1')
    plt.plot(bin_centres[0],bin_profiles[0][2], label='type 2')
    plt.plot(bin_centres[0],bin_profiles[0][3], label='type 3')
    plt.legend()
    plt.xlabel('z [Å]')
    plt.ylabel(r'number density [1/Å$^3$]')
    plt.show()
    print("Testing with trajectory")
    density_profile(conf_data,conf_dump)

if __name__ == "__main__" :

    # test_polyhedral_template_matching()
    test_density_profile()