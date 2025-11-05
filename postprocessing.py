# Example postprocessing script

from analysis import *
import matplotlib.pyplot as plt

FS_LABEL = 25
FS_LEGEND = 20
FS_TICKS = 15

conf_data = "collisions_post.data"
conf_dump = "coarse.dump"

t_in = 0
t_fin = 500
dt_coarse = 0.001*5000

print("Performing Polyhedral Template Matching...")
fractions = polyhedral_template_matching(conf_data,conf_dump)
print("Performing density profile binning...")
bin_profiles, bin_centres = density_profile(conf_data,conf_dump,nbins=100)

for ptm_type in fractions.keys() :
    n = len(fractions[ptm_type])
    t = dt_coarse*np.linspace(0,n-1,n)
    plt.plot(t, fractions[ptm_type], lw=4, label=ptm_type)
plt.legend(fontsize=FS_LEGEND)
plt.xlabel('time [ps]',fontsize=FS_LABEL)
plt.ylabel('fraction',fontsize=FS_LABEL)
plt.xlim([t_in,t_fin])
plt.xticks(fontsize=FS_TICKS)
plt.yticks(fontsize=FS_TICKS)
plt.show()

plt.plot(bin_centres[-1],bin_profiles[-1][1], lw=4, label='Al')
plt.plot(bin_centres[-1],bin_profiles[-1][2], lw=4, label='Ti')
plt.plot(bin_centres[-1],bin_profiles[-1][3], lw=4, label='Mo')
plt.legend(fontsize=FS_LEGEND)
plt.xlabel('z [Å]',fontsize=FS_LABEL)
plt.ylabel(r'number density [1/Å$^3$]',fontsize=FS_LABEL)
plt.xlim([bin_centres[-1][0],bin_centres[-1][-1]])
plt.xticks(fontsize=FS_TICKS)
plt.yticks(fontsize=FS_TICKS)
plt.show()