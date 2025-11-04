# Library of Ovito wrappers to perform basic analysis (PTM, density profiles, ...)

from ovito.io import *
from ovito.modifiers import *

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
        # ...
        n_fcc = data.attributes['PolyhedralTemplateMatching.counts.FCC']
        f_fcc = n_fcc / data.particles.count
        # f_fcc_vec.append(f_fcc)
        n_hcp = data.attributes['PolyhedralTemplateMatching.counts.HCP']
        f_hcp = n_hcp / data.particles.count
        # f_hcp_vec.append(f_hcp)
        n_bcc = data.attributes['PolyhedralTemplateMatching.counts.BCC']
        f_bcc = n_bcc / data.particles.count
        # f_bcc_vec.append(f_bcc)
        n_ico = data.attributes['PolyhedralTemplateMatching.counts.ICO']
        f_ico = n_ico / data.particles.count
        # ...
        n_sc = data.attributes['PolyhedralTemplateMatching.counts.SC']
        f_sc = n_sc / data.particles.count
        # ...
        n_cubicdiamond = data.attributes['PolyhedralTemplateMatching.counts.CUBIC_DIAMOND']
        f_cubicdiamond = n_cubicdiamond / data.particles.count
        # ...
        n_hexdiamond = data.attributes['PolyhedralTemplateMatching.counts.HEX_DIAMOND']
        f_hexdiamond = n_hexdiamond / data.particles.count
        # ...
        n_graphene = data.attributes['PolyhedralTemplateMatching.counts.GRAPHENE']
        f_graphene = n_graphene / data.particles.count
        # ...

### TESTS ###

def test_polyhedral_template_matching() :
    
    conf_data = "test/test-files/cofeni.data"
    conf_dump = "test/test-files/cofeni.dump"
    print("Testing without trajectory")
    polyhedral_template_matching(conf_data)
    print("Testing with trajectory")
    polyhedral_template_matching(conf_data,conf_dump)

if __name__ == "__main__" :

    test_polyhedral_template_matching()