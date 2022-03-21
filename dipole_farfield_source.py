import numpy as np
import lumapi
import scipy.constants
import time

def create_source(xsize, ysize, theta, phi, wavelengths, depth, n=2.4, z=0, grid = 20e-9, create_file = True):
    x = np.linspace(-xsize/2, xsize/2, xsize/grid)
    y = np.linspace(-ysize/2, ysize/2, ysize/grid)
    xv, yv, zv, wlv = np.meshgrid(x, y, np.array(z), wavelengths, indexing = 'ij')

    const = 1

    Ix = const*np.sin(theta)*np.cos(phi)
    Iy = const*np.sin(theta)*np.sin(phi)
    Iz = const*np.cos(theta)

    #d is depth of source, z is location of input plane
    height = depth + zv

    #Distance from source to points on plane
    rv = np.sqrt(np.square(xv) + np.square(yv) + np.square(height))
    thetav = np.arccos(height/rv)
    phiv = np.arctan2(yv, xv)

    beta = 2*np.pi/(wlv/n)
    omega = beta*(3e8/n)

    #Calculate A using eqn 6.94 in Balanis
    N = np.exp(-1j*beta*rv)/rv

    Ax = N*Ix
    Ay = N*Iy
    Az = N*Iz

    #Convert to spherical coordinates
    Ar = np.sin(thetav)*np.cos(phiv)*Ax + np.sin(thetav)*np.sin(phiv)*Ay + np.cos(thetav)*Az
    Atheta = np.cos(thetav)*np.cos(phiv)*Ax + np.cos(thetav)*np.sin(thetav)*Ay - np.sin(thetav)*Az
    Aphi = -np.sin(phiv)*Ax + np.cos(phiv)*Ay

    #Calculate E using far field approximation using eq 6.101a in Balanis
    Er = 0
    Etheta = -1j*omega*Atheta
    Ephi = -1j*omega*Aphi

    #Calculate H as well
    impedance = 377/n
    Hr = 0
    Htheta = -Ephi/impedance
    Hphi = Etheta/impedance

    #Convert to cartesian coordinates
    Ex = np.cos(thetav)*np.cos(phiv)*Etheta - np.sin(phiv)*Ephi
    Ey = np.cos(thetav)*np.sin(phiv)*Etheta + np.cos(phiv)*Ephi
    Ez = -np.sin(thetav)*Etheta

    Hx = np.cos(thetav)*np.cos(phiv)*Htheta - np.sin(phiv)*Hphi
    Hy = np.cos(thetav)*np.sin(phiv)*Htheta + np.cos(phiv)*Hphi
    Hz = -np.sin(thetav)*Htheta

    E = np.stack((Ex, Ey, Ez), axis=-1)
    H = np.stack((Hx, Hy, Hz), axis=-1)

    fields = {'E' : E, 'H' : H, 'x': x, 'y': y, 'z':z, 'wl':wavelengths}

    if create_file:
        fdtd = lumapi.FDTD()
        lumapi.putMatrix(fdtd.handle, 'x', x)
        lumapi.putMatrix(fdtd.handle, 'y', y)
        lumapi.putMatrix(fdtd.handle, 'z', z*np.ones(1))
        lumapi.putMatrix(fdtd.handle, 'f', np.divide(scipy.constants.speed_of_light,wavelengths))
        lumapi.putMatrix(fdtd.handle, 'Ex', Ex)
        lumapi.putMatrix(fdtd.handle, 'Ey', Ey)
        lumapi.putMatrix(fdtd.handle, 'Ez', Ez)
        lumapi.putMatrix(fdtd.handle, 'Hx', Hx)
        lumapi.putMatrix(fdtd.handle, 'Hy', Hy)
        lumapi.putMatrix(fdtd.handle, 'Hz', Hz)

        fdtd.eval("EM = rectilineardataset('EM fields', x, y, z);")
        fdtd.eval("EM.addparameter('lambda', c/f, 'f', f);")
        fdtd.eval("EM.addattribute('E', Ex, Ey, Ez);")
        fdtd.eval("EM.addattribute('H', Hx, Hy, Hz);")
        fdtd.eval("matlabsave('sourcefile.mat', EM);")
        fdtd.close()

    return fields
