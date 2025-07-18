import numpy as np
import numpy.random as rng

# Global conversion factors
CONV1 = 1.661 # [amu->Kg]
CONV2 = 1.602 # [eV->J]
CFSR = np.sqrt(CONV2/CONV1)

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

# TODO: generalize to multiple atomtypes
def velocity_distribution(Ed,m,N) :
    ek = kinetic_energy(Ed,N)
    prefac = (1e2)*CFSR*np.sqrt(2*ek/m)
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

if __name__ == "__main__" :

    import matplotlib.pyplot as plt

    # TEST: uniform hemisphere
    N = 1000
    xs, ys, zs = uniform_unit_hemisphere(N)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs, 'ro')
    plt.show()

    # TEST: kinetic energy distribution
    Ed = 10
    a_cut=100
    u_F = kinetic_energy(Ed,N,a_cut=a_cut)
    a = 0.3*Ed
    b = 2*a*((a+a_cut)**2)/(a_cut**2)
    f = lambda e : b*e/((e+a)**3) # x<a_cut
    E = np.linspace(0,100,1000)
    dE = E[1]-E[0]
    plt.plot(E, f(E)/np.sum(f(E)*dE))
    plt.plot(E, np.zeros(len(E)))
    plt.hist(u_F,bins=int(np.sqrt(N)),density=True)
    plt.show()

    # TEST: Al atoms velocity
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

    # TEST: generating vectors of atomtypes
    N = 1000
    type_list=[3,4,5]
    frac_list=[0.333,0.333,0.334]
    # type_list=[2]
    # frac_list=[1.0]
    v = gen_atype_vector(type_list,frac_list,N)
    print(np.sum(v==3))
    print(np.sum(v==4))
    print(np.sum(v==5))
    # print(np.sum(v==2))
