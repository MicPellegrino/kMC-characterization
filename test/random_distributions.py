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

def kinetic_energy(Ed,N) :
    a = 0.3*Ed
    f = lambda e : 2*a*e/((e+a)**3)
    F = lambda x : (x*x)/((a+x)**2)
    Fm1 = lambda x : a*(np.sqrt(x)+x)/(1-x)
    u = rng.uniform(0,1,N)
    return Fm1(u)

###########################################

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
    u_F = kinetic_energy(Ed,N)
    a = 0.3*Ed
    f = lambda e : 2*a*e/((e+a)**3)
    E = np.linspace(0,200,2000)
    dE = E[1]-E[0]
    plt.plot(E, f(E)/np.sum(f(E)*dE))
    plt.plot(E, np.zeros(len(E)))
    plt.hist(u_F[u_F<200],bins=int(np.sqrt(N)),density=True)
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
