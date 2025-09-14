import numpy as np
import numpy.random as rng

# Global conversion factors
CONV1 = 1.661 # [1e27 amu->Kg]
CONV2 = 1.602 # [1e19 eV->J]
CFSR = np.sqrt(CONV2/CONV1)
SI2METAL = 1e2 # [m/s->A/ps]

# Tranformation of kinetic energy due to collisions
# kTg = [eV], nc = [], wm = []
ekin_f = lambda ekin, kTg, nc, wm : (ekin-kTg)*np.exp(nc*np.log(1-0.5*wm))+kTg

###########################################

def uniform_unit_hemisphere(N) :
    x = rng.normal(size=N)
    y = rng.normal(size=N)
    z = rng.normal(size=N)
    s = np.sqrt(x*x+y*y+z*z)
    xs = x/s 
    ys = y/s 
    zs = -abs(z/s)
    return xs, ys, zs

###########################################

def kinetic_energy(Ed,N,a_cut=100) :
    a = 0.3*Ed
    b1 = 2*a*((a+a_cut)**2)/(a_cut**2)
    b2 = ((a+a_cut)**2)/(a_cut**2)
    f = lambda e : b1*x/((e+a)**3) # x<a_cut
    F = lambda x : b2*(x*x)/((a+x)**2)
    b3 = (a_cut**2)/((a+a_cut)**2)
    Fm1 = lambda x : a*(np.sqrt(b3*x)+b3*x)/(1-b3*x)
    u = rng.uniform(0,1,N)
    return Fm1(u)

###########################################

def velocity_distribution(Ed,m,N) :
    ek = kinetic_energy(Ed,N)
    prefac = SI2METAL*CFSR*np.sqrt(2*ek/m)
    xs, ys, zs = uniform_unit_hemisphere(N)
    vx = prefac*xs
    vy = prefac*ys
    vz = prefac*zs
    return vx, vy, vz, prefac

###########################################

def plane_uniform(xlow,xupp,ylow,yupp,N) :
    x = (xupp-xlow)*rng.uniform(0,1,N)+xlow
    y = (yupp-ylow)*rng.uniform(0,1,N)+ylow
    return x, y

###########################################

def gen_atype_vector(type_list,frac_list,N) :
    nf_list = []
    for f in frac_list :
        nf_list.append(int(np.round(N*f)))
    indices = np.arange(N)
    atype_vec = type_list[0]*np.ones(N,dtype=int)
    for i in range(len(nf_list)) :
        idx = np.arange(len(indices))
        c = rng.choice(idx,nf_list[i],replace=False)
        atype_vec[indices[c]] = type_list[i]
        indices = np.delete(indices,c)
    return atype_vec

###########################################

# The default atomic mass for the collision gas is the one of Argon (39.948u)
def velocity_per_type(Ed,m_vec,N,type_list,atype_vec,kTg=0,nc=0,mg=39.948) :
    
    vx = np.zeros(N)
    vy = np.zeros(N)
    vz = np.zeros(N)
    prefac = np.zeros(N)

    for i in range(len(type_list)) :
        
        Ni = np.sum(atype_vec==type_list[i])
        idx = np.argwhere(atype_vec==type_list[i])
        idx = idx.ravel()   # Sometimes numpy is really stupid...
        ek = kinetic_energy(Ed,Ni)

        # Collisions with gas before getting to the substrate:
        wm = 4*(mg*m_vec[i])/((mg+m_vec[i])**2)
        ek = ekin_f(ek,kTg,nc,wm)

        prefac_i = SI2METAL*CFSR*np.sqrt(2*ek/m_vec[i])
        xs, ys, zs = uniform_unit_hemisphere(Ni)

        vx[idx] = prefac_i*xs
        vy[idx] = prefac_i*ys
        vz[idx] = prefac_i*zs
        prefac[idx] = prefac_i

    return vx, vy, vz, prefac

###########################################

### TESTS ###

def test_uniform_hemisphere() :
    import matplotlib.pyplot as plt
    N = 1000
    xs, ys, zs = uniform_unit_hemisphere(N)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs, 'ro')
    plt.show()

def test_kinetic_energy_distribution() :
    import matplotlib.pyplot as plt
    N = 50000
    Ed = 10
    a_cut=100
    u_F = kinetic_energy(Ed,N,a_cut=a_cut)
    a = 0.3*Ed
    b = 2*a*((a+a_cut)**2)/(a_cut**2)
    f = lambda e : b*e/((e+a)**3) # x<a_cut
    E = np.linspace(0,100,1000)
    dE = E[1]-E[0]
    plt.plot(E, f(E)/np.sum(f(E)*dE),'k-',linewidth=3)
    plt.hist(u_F,bins=int(np.sqrt(N)),density=True,alpha=0.85)
    plt.xlabel(r'$\varepsilon$ [eV]',fontsize=25)
    plt.ylabel(r'$\rho(\varepsilon)$ []',fontsize=25)
    plt.xticks(fontsize=20)    
    plt.yticks(fontsize=20)
    plt.show()

def test_al_atom_velocity() :
    import matplotlib.pyplot as plt
    Ed = 10
    N = 1000
    m_Al = 26.982 # [amu]
    vx, vy, vz, _ = velocity_distribution(Ed,m_Al,N)
    soa = np.vstack((vx,vy,vz))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    origin = np.zeros_like(vx)
    ax.quiver(origin,origin,origin,soa[0],soa[1],soa[2],length=1,normalize=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_vel = max(np.max(vx),np.max(vy))
    ax.set_xlim([-max_vel,max_vel])
    ax.set_ylim([-max_vel,max_vel])
    ax.set_zlim([np.min(vz),0])
    plt.show()

def test_generate_atomtypes_multiple() :
    N = 1000
    type_list=[3,4,5]
    frac_list=[0.333,0.333,0.334]
    v = gen_atype_vector(type_list,frac_list,N)
    print(np.sum(v==3))
    print(np.sum(v==4))
    print(np.sum(v==5))

def test_generate_atomtypes_single() :
    N = 1000
    type_list=[2]
    frac_list=[1.0]
    v = gen_atype_vector(type_list,frac_list,N)
    print(np.sum(v==2))

def test_velocity_per_type(kTg=0,nc=0) :
    import matplotlib.pyplot as plt
    Ed = 10
    N = 300000
    type_list=[3,4,5]
    frac_list=[0.333,0.333,0.334]
    m_vec = [26.982,95.94,58.693]   # Al, Mo, Ti
    atype_vec = gen_atype_vector(type_list,frac_list,N)
    idx_3 = np.argwhere(atype_vec==3)
    idx_3 = idx_3.ravel()
    idx_4 = np.argwhere(atype_vec==4)
    idx_4 = idx_4.ravel()
    idx_5 = np.argwhere(atype_vec==5)
    idx_5 = idx_5.ravel()
    vx, vy, vz, vabs = velocity_per_type(Ed,m_vec,N,type_list,atype_vec,kTg,nc)
    plt.hist(vabs[idx_3],bins=int(np.sqrt(N//3)),density=True,label='Al',alpha=0.75)
    plt.hist(vabs[idx_4],bins=int(np.sqrt(N//3)),density=True,label='Mo',alpha=0.75)
    plt.hist(vabs[idx_5],bins=int(np.sqrt(N//3)),density=True,label='Ni',alpha=0.75)
    plt.xlabel('A/ps')
    plt.legend()
    plt.show()
    ekin_3 = 0.5*m_vec[0]*vabs[idx_3]*vabs[idx_3]/(SI2METAL*CFSR)**2
    ekin_4 = 0.5*m_vec[1]*vabs[idx_4]*vabs[idx_4]/(SI2METAL*CFSR)**2
    ekin_5 = 0.5*m_vec[2]*vabs[idx_5]*vabs[idx_5]/(SI2METAL*CFSR)**2
    plt.hist(ekin_3,bins=int(np.sqrt(N//3)),density=True,label='Al',alpha=0.75)
    plt.hist(ekin_4,bins=int(np.sqrt(N//3)),density=True,label='Mo',alpha=0.75)
    plt.hist(ekin_5,bins=int(np.sqrt(N//3)),density=True,label='Ni',alpha=0.75)
    plt.xlabel('eV')
    plt.legend()
    plt.show()

if __name__ == "__main__" :

    # test_uniform_hemisphere()
    # test_kinetic_energy_distribution()
    # test_al_atom_velocity()
    # test_generate_atomtypes_multiple()
    # test_generate_atomtypes_single()
    test_velocity_per_type(kTg=0.067,nc=2)
