# Library of Ovito wrappers to perform basic analysis (PTM, density profiles, ...)

from ovito.modifiers import *
from ovito.modifiers import *

def polyhedral_template_matching(conf_data,conf_dump=None) :
    if conf_dump == None :
        pipeline_ptm = import_file(conf_data)
    else :
        pipeline_ptm = import_file([conf_dump,conf_data])

### TESTS ###

def test_polyhedral_template_matching() :
    
    conf_data = "test/test-files/cofeni.data"
    conf_dump = "test/test-files/cofeni.dump"
    print("Testing without trajectory")
    polyhedral_template_matching(conf_data)
    print("Testing with trajectory")
    polyhedral_template_matching(conf_data,conf_dump)