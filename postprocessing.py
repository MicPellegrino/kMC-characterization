# Example postprocessing script

from analysis import *
import matplotlib.pyplot as plt

conf_data = "collisions_post.data"
conf_dump = "coarse.dump"

dt_coarse = 0.001*5000

fractions = polyhedral_template_matching(conf_data,conf_dump)
bin_profiles, bin_centres = density_profile(conf_data,conf_dump,nbins=50)

for ptm_type in fractions.keys() :
    n = len(fractions[ptm_type])
    plt.plot(dt_coarse*np.linspace(0,n-1,n), fractions[ptm_type], label=ptm_type)
plt.legend()
plt.xlabel('time [ps]')
plt.ylabel('fraction')
plt.show()

plt.plot(bin_centres[-1],bin_profiles[-1][1], label='type 1')
plt.plot(bin_centres[-1],bin_profiles[-1][2], label='type 2')
plt.plot(bin_centres[-1],bin_profiles[-1][3], label='type 3')
plt.legend()
plt.xlabel('z [Å]')
plt.ylabel(r'number density [1/Å$^3$]')
plt.show()